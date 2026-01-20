#!/usr/bin/env python3.10
"""
lemma.py — lemmatizer (stanza/spaCy) and lemma caches for files/chunks.

Exported API:
- extract_lemmas(text: str) -> FrozenSet[str]
- FILE_LEMMAS: Dict[str, FrozenSet[str]]
- CHUNK_LEMMA_CACHE: Dict[str, FrozenSet[str]]
- CHUNK_LEMMA_CACHE_FILE: Path
- LEMMA_INDEX_FILE: Path
- save_chunk_lemma_cache(...)
- load_saved_lemmas()
- update_file_lemmas_async(docs, stored_hashes, new_hashes)
- LEMMA_POOL (ThreadPoolExecutor) — for controlled shutdown if needed

Notes:
- No model downloads or heavy initialization at import time.
- Model init is lazy. Optional stanza resources auto-download can happen on first use.
- Cache files are stored under cfg.CACHE_PATH.
"""

from __future__ import annotations

# Stdlib
import asyncio
import functools
import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

# Third-party
import spacy
import stanza
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# Local
from asketmc_bot import config as cfg

log = logging.getLogger("asketmc.lemma")
log.setLevel(logging.DEBUG if getattr(cfg, "DEBUG", False) else logging.INFO)

# ─────────────────────────────────────────────────────────
# Caches (stored under cfg.CACHE_PATH)
# ─────────────────────────────────────────────────────────
CHUNK_LEMMA_CACHE_FILE: Path = cfg.CACHE_PATH / "chunk_lemma_index.json"
LEMMA_INDEX_FILE: Path = cfg.CACHE_PATH / "lemma_index.json"

CHUNK_LEMMA_CACHE: Dict[str, FrozenSet[str]] = {}
FILE_LEMMAS: Dict[str, FrozenSet[str]] = {}

# ─────────────────────────────────────────────────────────
# Pool and locks
# ─────────────────────────────────────────────────────────
_LEMMA_LOCK = threading.Lock()
_MODELS_INIT_LOCK = threading.Lock()
LEMMA_POOL = ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8))

# ─────────────────────────────────────────────────────────
# Lazy model initialization
# ─────────────────────────────────────────────────────────
_STANZA_NLP_RU: Optional[stanza.Pipeline] = None
_SPACY_EN = None

# Tracks readiness separately (important for cache invalidation).
_SPACY_READY = False
_STANZA_READY = False

_MODELS_INIT_TRIED = False
_LAST_INIT_ATTEMPT_TS = 0.0
_INIT_RETRY_COOLDOWN_SEC = float(getattr(cfg, "LEMMA_INIT_RETRY_COOLDOWN_SEC", 300.0))

# Degraded-mode reporting (one-time)
_DEGRADED_WARNED = False

# Cache epoch: changes whenever model availability changes.
# Used to prevent "empty lemma" results from being cached forever.
_MODELS_EPOCH = 0

# Stanza resources location + auto-download switch (lazy, on first use).
_STANZA_DIR: Path = Path(
    getattr(cfg, "STANZA_RESOURCES_DIR", cfg.CACHE_PATH / "stanza")
).resolve()
_STANZA_AUTO_DOWNLOAD: bool = bool(getattr(cfg, "STANZA_AUTO_DOWNLOAD", True))


def _invalidate_all_lemma_caches(reason: str) -> None:
    """Clear all lemma-related caches and persisted cache files."""
    global _MODELS_EPOCH
    _MODELS_EPOCH += 1

    try:
        _extract_lemmas_cached.cache_clear()
    except Exception:
        pass

    CHUNK_LEMMA_CACHE.clear()
    FILE_LEMMAS.clear()

    for fp in (LEMMA_INDEX_FILE, CHUNK_LEMMA_CACHE_FILE):
        try:
            if fp.exists():
                fp.unlink()
        except Exception:
            pass

    log.info(
        "[LEMMA_CACHE] invalidated (epoch=%d) reason=%s",
        _MODELS_EPOCH,
        reason,
    )


def _notify_degraded_once() -> None:
    global _DEGRADED_WARNED
    if _DEGRADED_WARNED:
        return
    _DEGRADED_WARNED = True
    log.warning(
        "[LEMMATIZER] Lemmatization is in degraded mode: spaCy EN model and/or stanza RU resources are unavailable. "
        "Lemmas may be empty until models are available."
    )


def _lang_of(text: str) -> str:
    """Best-effort language detection. Returns 'en' or 'ru'."""
    try:
        detected_lang = detect(text)
    except LangDetectException:
        detected_lang = "ru"
    except Exception:
        detected_lang = "ru"
    return "en" if detected_lang == "en" else "ru"


def _looks_like_torch_weights_only_error(err: Exception) -> bool:
    """Heuristic to detect PyTorch 2.6+ 'weights_only' compatibility failures surfaced via stanza."""
    msg = str(err)
    needles = (
        "Weights only load failed",
        "WeightsUnpickler error",
        "torch.serialization.add_safe_globals",
        "weights_only",
        "Unsupported global",
    )
    return any(n in msg for n in needles)


def _torch_allowlist_stanza_safe_globals() -> bool:
    """
    Allowlist numpy globals required by torch weights-only unpickler (PyTorch 2.6+).
    Only use if checkpoints are trusted.
    """
    try:
        import numpy as np  # local import
        import torch  # local import
    except Exception as exc:
        log.warning("[LEMMATIZER] torch/numpy import failed: %s", exc)
        return False

    ser = getattr(torch, "serialization", None)
    add = getattr(ser, "add_safe_globals", None) if ser else None
    if not callable(add):
        log.warning(
            "[LEMMATIZER] torch.serialization.add_safe_globals not available; cannot relax weights_only safely"
        )
        return False

    try:
        allow = [np.core.multiarray._reconstruct, np.ndarray, np.dtype]
        if hasattr(np.core.multiarray, "scalar"):
            allow.append(np.core.multiarray.scalar)
        add(allow)
        return True
    except Exception as exc:
        log.warning("[LEMMATIZER] torch safe-globals allowlist failed: %s", exc)
        return False


class _TorchLoadWeightsOnlyFalse:
    """
    Narrow monkeypatch: force torch.load(weights_only=False) during stanza pipeline init.
    Only use if checkpoints are trusted.
    """

    def __enter__(self):
        import torch  # local import

        self._torch = torch
        self._orig = torch.load

        def _patched(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return self._orig(*args, **kwargs)

        torch.load = _patched
        return self

    def __exit__(self, exc_type, exc, tb):
        self._torch.load = self._orig
        return False


def _try_init_models(*, force_retry: bool = False) -> Tuple[bool, bool]:
    """
    Lazily initialize spaCy EN + stanza RU.

    Returns (spacy_ready, stanza_ready).
    May trigger stanza resource download (lazy) if STANZA_AUTO_DOWNLOAD=True.

    Thread-safety:
    - Single initializer under _MODELS_INIT_LOCK.

    Retry behavior:
    - If init previously failed, allow throttled retry (cooldown) or explicit force_retry.
    """
    global _STANZA_NLP_RU, _SPACY_EN, _MODELS_INIT_TRIED, _LAST_INIT_ATTEMPT_TS
    global _SPACY_READY, _STANZA_READY

    now = time.time()

    if (
        _MODELS_INIT_TRIED
        and (now - _LAST_INIT_ATTEMPT_TS) < _INIT_RETRY_COOLDOWN_SEC
        and not force_retry
    ):
        return _SPACY_READY, _STANZA_READY

    with _MODELS_INIT_LOCK:
        now = time.time()
        if (
            _MODELS_INIT_TRIED
            and (now - _LAST_INIT_ATTEMPT_TS) < _INIT_RETRY_COOLDOWN_SEC
            and not force_retry
        ):
            return _SPACY_READY, _STANZA_READY

        _LAST_INIT_ATTEMPT_TS = now
        _MODELS_INIT_TRIED = True

        prev_spacy = _SPACY_READY
        prev_stanza = _STANZA_READY

        # ---- spaCy EN ----
        spacy_en = None
        try:
            spacy_en = spacy.load("en_core_web_sm")
            _SPACY_READY = True
            log.info("[LEMMATIZER] spaCy en_core_web_sm ready")
        except Exception as e:
            _SPACY_READY = False
            log.warning("[LEMMATIZER] spaCy model not available: %s", e)

        # ---- stanza RU ----
        stanza_ru = None
        _STANZA_DIR.mkdir(parents=True, exist_ok=True)

        def _build_stanza_pipeline() -> stanza.Pipeline:
            return stanza.Pipeline(
                lang="ru",
                processors="tokenize,pos,lemma",
                use_gpu=False,
                verbose=False,
                dir=str(_STANZA_DIR),
            )

        try:
            stanza_ru = _build_stanza_pipeline()
            _STANZA_READY = True
            log.info("[LEMMATIZER] stanza ru pipeline ready (dir=%s)", _STANZA_DIR)
        except Exception as e:
            _STANZA_READY = False

            if _looks_like_torch_weights_only_error(e):
                log.warning(
                    "[LEMMATIZER] stanza ru pipeline blocked by torch serialization restrictions; "
                    "trying safe-globals allowlist (dir=%s). Error=%s",
                    _STANZA_DIR,
                    e,
                )

                if _torch_allowlist_stanza_safe_globals():
                    try:
                        stanza_ru = _build_stanza_pipeline()
                        _STANZA_READY = True
                        log.info(
                            "[LEMMATIZER] stanza ru pipeline ready after torch safe-globals allowlist (dir=%s)",
                            _STANZA_DIR,
                        )
                    except Exception as e2:
                        _STANZA_READY = False
                        log.warning(
                            "[LEMMATIZER] stanza still blocked after safe-globals allowlist; "
                            "trying torch.load(weights_only=False) (dir=%s). Error=%s",
                            _STANZA_DIR,
                            e2,
                        )

                if not _STANZA_READY:
                    try:
                        with _TorchLoadWeightsOnlyFalse():
                            stanza_ru = _build_stanza_pipeline()
                        _STANZA_READY = True
                        log.info(
                            "[LEMMATIZER] stanza ru pipeline ready after forcing torch.load(weights_only=False) (dir=%s)",
                            _STANZA_DIR,
                        )
                    except Exception as e3:
                        _STANZA_READY = False
                        log.warning(
                            "[LEMMATIZER] stanza ru pipeline blocked even after weights_only=False; "
                            "stanza disabled (dir=%s). Error=%s",
                            _STANZA_DIR,
                            e3,
                        )
            else:
                log.warning("[LEMMATIZER] stanza pipeline not available: %s", e)

                if _STANZA_AUTO_DOWNLOAD:
                    try:
                        log.info(
                            "[LEMMATIZER] downloading stanza resources for ru (model_dir=%s)",
                            _STANZA_DIR,
                        )
                        stanza.download(
                            "ru",
                            model_dir=str(_STANZA_DIR),
                            processors="tokenize,pos,lemma",
                            verbose=False,
                        )
                        stanza_ru = _build_stanza_pipeline()
                        _STANZA_READY = True
                        log.info(
                            "[LEMMATIZER] stanza ru pipeline ready after download (dir=%s)",
                            _STANZA_DIR,
                        )
                    except Exception as e2:
                        _STANZA_READY = False
                        log.warning("[LEMMATIZER] stanza download/init failed: %s", e2)

        _SPACY_EN = spacy_en
        _STANZA_NLP_RU = stanza_ru

        if (_SPACY_READY and not prev_spacy) or (_STANZA_READY and not prev_stanza):
            _invalidate_all_lemma_caches(
                reason=f"models_became_available spacy={_SPACY_READY} stanza={_STANZA_READY}"
            )

        return _SPACY_READY, _STANZA_READY


@functools.lru_cache(maxsize=10_000)
def _extract_lemmas_cached(text: str, epoch: int) -> FrozenSet[str]:
    """Cached lemmatization; epoch prevents degraded empty results from sticking forever."""
    _ = epoch  # part of cache key; not used directly

    if not text or len(text) < 3:
        return frozenset()

    spacy_ready, stanza_ready = _try_init_models(force_retry=False)
    if not spacy_ready and not stanza_ready:
        _notify_degraded_once()
        return frozenset()

    lang = _lang_of(text)

    if lang == "en":
        if not spacy_ready or _SPACY_EN is None:
            return frozenset()
        try:
            doc = _SPACY_EN(text)
            lemmas = {
                tok.lemma_.lower()
                for tok in doc
                if tok.is_alpha and not tok.is_stop and len(tok.text) > 2
            }
            return frozenset(lemmas)
        except Exception:
            return frozenset()

    if not stanza_ready or _STANZA_NLP_RU is None:
        return frozenset()

    try:
        with _LEMMA_LOCK:
            doc = _STANZA_NLP_RU(text)
    except Exception:
        return frozenset()

    try:
        lemmas = {
            w.lemma.lower()
            for s in doc.sentences
            for w in s.words
            if w.lemma
            and len(w.lemma) > 2
            and w.upos in cfg.GOOD_POS
            and w.lemma.lower() not in cfg.STOP_WORDS
        }
        return frozenset(lemmas)
    except Exception:
        return frozenset()


def extract_lemmas(text: str) -> FrozenSet[str]:
    """Public API: lemmatize text with epoch-aware caching."""
    return _extract_lemmas_cached(text, _MODELS_EPOCH)


def _chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_lemmas_for_chunk(text: str) -> FrozenSet[str]:
    h = _chunk_hash(text)
    lemmas = CHUNK_LEMMA_CACHE.get(h)
    if lemmas is None:
        try:
            lemmas = extract_lemmas(text)
        except Exception:
            lemmas = frozenset()
        CHUNK_LEMMA_CACHE[h] = lemmas
    return lemmas


def save_chunk_lemma_cache(chunk_cache_file: Path = CHUNK_LEMMA_CACHE_FILE) -> None:
    data = {k: list(v) for k, v in CHUNK_LEMMA_CACHE.items()}
    try:
        chunk_cache_file.parent.mkdir(parents=True, exist_ok=True)
        chunk_cache_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("[LEMMA_CACHE] Chunk cache saved: %s (keys=%d)", chunk_cache_file, len(data))
    except Exception as e:
        log.error("[LEMMA_CACHE] Save error: %s", e)


def _persist_lemmas(lemma_file: Optional[Path] = None) -> None:
    lemma_file = lemma_file or LEMMA_INDEX_FILE
    try:
        lemma_file.parent.mkdir(parents=True, exist_ok=True)
        dump = {k: list(v) for k, v in FILE_LEMMAS.items()}
        lemma_file.write_text(
            json.dumps(dump, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("[LEMMA_CACHE] File-lemmas saved: %s (files=%d)", lemma_file, len(FILE_LEMMAS))
    except Exception as e:
        log.error("[LEMMA_CACHE] Persist lemmas error: %s", e)


def load_saved_lemmas() -> None:
    if not LEMMA_INDEX_FILE.exists():
        log.info("[LEMMA_CACHE] Lemma cache not found: %s", LEMMA_INDEX_FILE)
        return
    try:
        data = json.loads(LEMMA_INDEX_FILE.read_text("utf-8"))
        if isinstance(data, dict):
            for fname, lst in data.items():
                if isinstance(lst, list):
                    FILE_LEMMAS[fname] = frozenset(lst)
        log.info("[LEMMA_CACHE] Loaded file-lemmas: %d files", len(FILE_LEMMAS))
    except Exception as e:
        log.warning("[LEMMA_CACHE] Load lemmas error: %s", e)


def _read_file(fp: Path) -> str:
    try:
        return fp.read_text("utf-8", "ignore")
    except Exception:
        return ""


async def _compute_and_store_lemmas(fp: Path) -> None:
    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(LEMMA_POOL, _read_file, fp)
    try:
        FILE_LEMMAS[fp.name] = extract_lemmas(text)
        log.debug("[LEMMA_CACHE] %s: %d lemmas", fp.name, len(FILE_LEMMAS[fp.name]))
    except Exception:
        FILE_LEMMAS[fp.name] = frozenset()
        log.debug("[LEMMA_CACHE] %s: set empty (error)", fp.name)


async def update_file_lemmas_async(
    docs: List[Path],
    stored_hashes: Dict[str, str],
    new_hashes: Dict[str, str],
) -> List[Path]:
    changed = [d for d in docs if stored_hashes.get(d.name) != new_hashes.get(d.name)]
    if not changed:
        log.info("[LEMMA_CACHE] No changed files")
        return []

    tasks = [asyncio.create_task(_compute_and_store_lemmas(d)) for d in changed]
    await asyncio.gather(*tasks)
    _persist_lemmas()
    log.info("[LEMMA_CACHE] Updated file-lemmas for %d files", len(changed))
    return changed
