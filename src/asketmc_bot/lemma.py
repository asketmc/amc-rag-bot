#!/usr/bin/env python3.10
"""
lemma.py — лемматизатор (stanza/spaCy) и кэш лемм для файлов/чанков.

Экспортируемое API:
- extract_lemmas(text: str) -> FrozenSet[str]
- FILE_LEMMAS: Dict[str, FrozenSet[str]]
- CHUNK_LEMMA_CACHE: Dict[str, FrozenSet[str]]
- CHUNK_LEMMA_CACHE_FILE: Path
- LEMMA_INDEX_FILE: Path
- save_chunk_lemma_cache(...)
- load_saved_lemmas()
- update_file_lemmas_async(docs, stored_hashes, new_hashes)
- LEMMA_POOL (ThreadPoolExecutor) — для корректного shutdown в core
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, FrozenSet, List

# Third-party
import stanza
import spacy
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# Local
import config as cfg

log = logging.getLogger("asketmc.lemma")
log.setLevel(logging.DEBUG if getattr(cfg, "DEBUG", False) else logging.INFO)

# ─────────────────────────────────────────────────────────
# Инициализация моделей (stanza + spaCy)
# ─────────────────────────────────────────────────────────
log.info("[LEMMATIZER] Preparing models...")
try:
    stanza.download('ru', verbose=False)
    log.info("[LEMMATIZER] stanza ru downloaded/ready")
except Exception as e:
    log.exception("[LEMMATIZER] stanza download error: %s", e)
    raise SystemExit(1)

try:
    STANZA_NLP_RU = stanza.Pipeline(
        lang='ru',
        processors='tokenize,pos,lemma',
        use_gpu=False,
        verbose=False,
    )
    log.info("[LEMMATIZER] stanza pipeline ready")
except Exception as e:
    log.exception("[LEMMATIZER] stanza pipeline error: %s", e)
    raise SystemExit(1)

try:
    SPACY_EN = spacy.load("en_core_web_sm")
    log.info("[LEMMATIZER] spaCy en_core_web_sm ready")
except Exception as e:
    log.exception("[LEMMATIZER] spaCy load error: %s", e)
    raise SystemExit(1)

# Пул и мьютекс
_LEMMA_LOCK = threading.Lock()
LEMMA_POOL = ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8))
log.info("[LEMMATIZER] ThreadPoolExecutor started (max_workers=%d)", getattr(LEMMA_POOL, "_max_workers", 0))

# ─────────────────────────────────────────────────────────
# Лемматизация
# ─────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=10_000)
def _extract_lemmas(text: str) -> FrozenSet[str]:
    try:
        detected_lang = detect(text)
    except LangDetectException:
        detected_lang = "ru"
    except Exception:
        detected_lang = "ru"

    lang = "en" if detected_lang == "en" else "ru"

    if lang == "en":
        try:
            doc = SPACY_EN(text)
            lemmas = {
                tok.lemma_.lower()
                for tok in doc
                if tok.is_alpha and not tok.is_stop and len(tok) > 2
            }
            return frozenset(lemmas)
        except Exception:
            return frozenset()

    # ru
    try:
        with _LEMMA_LOCK:
            doc = STANZA_NLP_RU(text)
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

# Публичный алиас
extract_lemmas = _extract_lemmas

# ─────────────────────────────────────────────────────────
# Кэш лемм: для чанков и для файлов
# ─────────────────────────────────────────────────────────
CHUNK_LEMMA_CACHE_FILE = Path("rag_cache/chunk_lemma_index.json")
CHUNK_LEMMA_CACHE: Dict[str, FrozenSet[str]] = {}

FILE_LEMMAS: Dict[str, FrozenSet[str]] = {}
LEMMA_INDEX_FILE = Path("rag_cache/lemma_index.json")

def _chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_lemmas_for_chunk(text: str) -> FrozenSet[str]:
    """
    (Опционально — если когда-либо потребуется напрямую)
    """
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
        chunk_cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("[LEMMA_CACHE] Chunk cache saved: %s (keys=%d)", chunk_cache_file, len(data))
    except Exception as e:
        log.error("[LEMMA_CACHE] Save error: %s", e)

def _persist_lemmas(lemma_file: Path = None) -> None:
    lemma_file = lemma_file or LEMMA_INDEX_FILE
    try:
        lemma_file.parent.mkdir(parents=True, exist_ok=True)
        dump = {k: list(v) for k, v in FILE_LEMMAS.items()}
        lemma_file.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
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
# Асинхронное обновление FILE_LEMMAS по изменившимся файлам
# ─────────────────────────────────────────────────────────
def _read_file(fp: Path) -> str:
    try:
        return fp.read_text("utf-8", "ignore")
    except Exception:
        return ""

async def _compute_and_store_lemmas(fp: Path) -> None:
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, _read_file, fp)
    try:
        FILE_LEMMAS[fp.name] = extract_lemmas(text)
        log.debug("[LEMMA_CACHE] %s: %d lemmas", fp.name, len(FILE_LEMMAS[fp.name]))
    except Exception:
        FILE_LEMMAS[fp.name] = frozenset()
        log.debug("[LEMMA_CACHE] %s: set empty (error)", fp.name)

async def update_file_lemmas_async(
    docs: List[Path],
    stored_hashes: Dict[str, str],
    new_hashes: Dict[str, str]
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
