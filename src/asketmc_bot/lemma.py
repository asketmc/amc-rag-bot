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
- Model init is lazy and failure-tolerant.
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
from typing import Dict, FrozenSet, List, Optional

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
_MODELS_INIT_TRIED = False
_MODELS_AVAILABLE = False

# Degraded-mode reporting / retry gate
_DEGRADED_WARNED = False
_LAST_INIT_ATTEMPT_TS = 0.0
_INIT_RETRY_COOLDOWN_SEC = float(getattr(cfg, "LEMMA_INIT_RETRY_COOLDOWN_SEC", 300.0))


def _try_init_models(*, force_retry: bool = False) -> bool:
    """
    Lazily initialize stanza + spaCy models.

    No downloads are triggered here.
    If models are missing/unavailable, returns False and lemmatization degrades to empty sets.

    Thread-safety:
    - Single initializer under _MODELS_INIT_LOCK.

    Retry behavior:
    - If init previously failed, allow a throttled retry (cooldown) or explicit force_retry.
    """
    global _STANZA_NLP_RU, _SPACY_EN, _MODELS_INIT_TRIED, _MODELS_AVAILABLE
    global _LAST_INIT_ATTEMPT_TS

    now = time.time()

    if _MODELS_INIT_TRIED and _MODELS_AVAILABLE:
        return True

    if _MODELS_INIT_TRIED and not _MODELS_AVAILABLE and not force_retry:
        if (now - _LAST_INIT_ATTEMPT_TS) < _INIT_RETRY_COOLDOWN_SEC:
            return False

    with _MODELS_INIT_LOCK:
        now = time.time()

        if _MODELS_INIT_TRIED and _MODELS_AVAILABLE:
            return True

        if _MODELS_INIT_TRIED and not _MODELS_AVAILABLE and not force_retry:
            if (now - _LAST_INIT_ATTEMPT_TS) < _INIT_RETRY_COOLDOWN_SEC:
                return False

        _LAST_INIT_ATTEMPT_TS = now

        spacy_en = None
        stanza_ru = None

        # spaCy EN
        try:
            spacy_en = spacy.load("en_core_web_sm")
            log.info("[LEMMATIZER] spaCy en_core_web_sm ready")
        except Exception as e:
            spacy_en = None
            log.warning("[LEMMATIZER] spaCy model not available: %s", e)

        # stanza RU (expects resources already installed in stanza's default dir)
        try:
            stanza_ru = stanza.Pipeline(
                lang="ru",
                processors="tokenize,pos,lemma",
                use_gpu=False,
                verbose=False,
            )
            log.info("[LEMMATIZER] stanza ru pipeline ready")
        except Exception as e:
            stanza_ru = None
            log.warning("[LEMMATIZER] stanza resources/pipeline not available: %s", e)

        models_available = (spacy_en is not None) or (stanza_ru is not None)
        if not models_available:
            log.warning(
                "[LEMMATIZER] No NLP models available; lemmatization degraded (empty lemma sets). "
                "Will retry after cooldown=%.0fs.",
                _INIT_RETRY_COOLDOWN_SEC,
            )

        _SPACY_EN = spacy_en
        _STANZA_NLP_RU = stanza_ru
        _MODELS_AVAILABLE = models_available
        _MODELS_INIT_TRIED = True

        return _MODELS_AVAILABLE


def _notify_degraded_once() -> None:
    global _DEGRADED_WARNED
    if _DEGRADED_WARNED:
        return
    _DEGRADED_WARNED = True
    log.warning(
        "[LEMMATIZER] Lemmatization is running in degraded mode (no NLP models). "
        "Ensure spaCy model 'en_core_web_sm' and/or stanza Russian resources are installed. "
        "Lemmas will be empty until models become available."
    )


# ─────────────────────────────────────────────────────────
# Lemmatization
# ─────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=10_000)
def _extract_lemmas(text: str) -> FrozenSet[str]:
    # Fast path: skip empty / too short
    if not text or len(text) < 3:
        return frozenset()

    # Best-effort init; if failed, emit a single warning and occasionally retry via _try_init_models.
    if not _try_init_models():
        _notify_degraded_once()
        # Throttled retry gate lives in _try_init_models itself, so no extra logic here.
        return frozenset()

    # Language detection (best-effort)
    try:
        detected_lang = detect(text)
    except LangDetectException:
        detected_lang = "ru"
    except Exception:
        detected_lang = "ru"

    lang = "en" if detected_lang == "en" else "ru"

    if lang == "en":
        if _SPACY_EN is None:
            # If spaCy specifically missing, try a forced retry occasionally (still throttled by lock/cooldown).
            _try_init_models(force_retry=False)
            return frozenset()
        try:
            doc = _SPACY_EN(text)
            lemmas = {
                tok.lemma_.lower()
                for tok in doc
                if tok.is_alpha and not tok.is_stop and len(tok) > 2
            }
            return frozenset(lemmas)
        except Exception:
            return frozenset()

    # ru
    if _STANZA_NLP_RU is None:
        # If stanza specifically missing, allow cooldown-based retries.
        _try_init_models(force_retry=False)
        return frozenset()

    try:
        # Stanza pipeline is not thread-safe in many setups; serialize calls.
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


# Public alias
extract_lemmas = _extract_lemmas


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
        for fname, lst in data.items():
            FILE_LEMMAS[fname] = frozenset(lst)
        log.info("[LEMMA_CACHE] Loaded file-lemmas: %d files", len(FILE_LEMMAS))
    except Exception as e:
        log.warning("[LEMMA_CACHE] Load lemmas error: %s", e)


# ─────────────────────────────────────────────────────────
# Async update of FILE_LEMMAS for changed files
# ─────────────────────────────────────────────────────────
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
