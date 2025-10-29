#!/usr/bin/env python3.10
"""
index_builder.py — конфигурация эмбеддеров и сборка/загрузка индекса LlamaIndex.

Экспортируемое API:
- build_index() -> VectorStoreIndex  (async)

Модуль:
- настраивает Settings.embed_model (BAAI/bge-m3) и SentenceSplitter
- пересчитывает/загружает индекс
- проставляет леммы в метаданных чанков, используя lemma.py
- поддерживает кэш хэшей документов и кэш лемм (FILE_LEMMAS)
"""

from __future__ import annotations

# Stdlib
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, List

# Third-party
import torch
from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Local
import config as cfg
from lemma import (
    extract_lemmas,
    FILE_LEMMAS,
    load_saved_lemmas,
    update_file_lemmas_async,
    save_chunk_lemma_cache,
)

log = logging.getLogger("asketmc.index")

# ─────────────────────────────────────────────────────────
# Embeddings configuration
# ─────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_LOG_EVERY = getattr(cfg, "EMBED_LOG_EVERY", 1000)

class LoggingBGE(HuggingFaceEmbedding):
    _counter: int = 0
    __slots__ = ()
    def _get_text_embedding(self, text: str):
        t0 = time.time()
        vec = super()._get_text_embedding(text)
        type(self)._counter += 1
        if type(self)._counter % EMBED_LOG_EVERY == 0:
            logging.getLogger("asketmc.embed").info(
                "EMB %d | %s | %s… | %.3fs",
                type(self)._counter,
                self._target_device,
                text[:120].replace("\n", " "),
                time.time() - t0,
            )
        return vec

# Настройка глобальных Settings LlamaIndex — один раз при импорте
Settings.embed_model = LoggingBGE("BAAI/bge-m3", normalize=True, device=DEVICE)
Settings.node_parser = SentenceSplitter(
    chunk_size=getattr(cfg, "CHUNK_SIZE", 512),
    chunk_overlap=getattr(cfg, "CHUNK_OVERLAP", 128),
    include_metadata=False,
    paragraph_separator="\n\n",
)

# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def _doc_hash(fp: Path) -> str:
    """SHA-256 файла по 1 МБ чанкам."""
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# ─────────────────────────────────────────────────────────
# Build / Load index
# ─────────────────────────────────────────────────────────
async def build_index() -> VectorStoreIndex:
    """
    Собирает или грузит индекс документов из cfg.DOCS_PATH.
    - Валидирует хэши (cfg.HASH_FILE)
    - Загружает кэш FILE_LEMMAS и актуализирует его для изменившихся файлов
    - Проставляет леммы каждому чанк-узлу в metadata["lemmas"]
    - Сохраняет индекс и хэши, а также кэш чанков (через lemma.save_chunk_lemma_cache)
    """
    docs: List[Path] = list(cfg.DOCS_PATH.glob("*"))
    hashes: Dict[str, str] = {d.name: _doc_hash(d) for d in docs}

    if cfg.HASH_FILE.exists():
        try:
            stored: Dict[str, str] = json.loads(cfg.HASH_FILE.read_text("utf-8"))
        except Exception:
            stored = {}
    else:
        stored = {}

    # 1) Кэш лемм для файлов
    load_saved_lemmas()
    try:
        await update_file_lemmas_async(docs, stored, hashes)
    except Exception as e:
        log.warning("[build_index] update_file_lemmas_async failed: %s", e)

    # 2) Индекс: загрузка или сборка
    idx: Optional[VectorStoreIndex] = None
    if cfg.CACHE_PATH.exists() and stored == hashes:
        try:
            idx = load_index_from_storage(StorageContext.from_defaults(persist_dir=str(cfg.CACHE_PATH)))
            log.info("[build_index] Loaded index from cache: %s", cfg.CACHE_PATH)
        except Exception as e:
            log.warning("[build_index] Failed to load cached index: %s", e)
            idx = None

    if idx is None:
        ll_docs: List[Document] = []
        for fp in docs:
            try:
                text = fp.read_text("utf-8", "ignore")
                ll_docs.append(Document(text=text, metadata={"file_name": fp.name}))
            except Exception as e:
                log.warning("[build_index] Read error %s: %s", fp, e)
        idx = VectorStoreIndex.from_documents(ll_docs)
        log.info("[build_index] Built new index from %d docs", len(ll_docs))

    # 3) Проставление лемм в metadata каждого узла
    for node in idx.docstore.docs.values():
        fname = node.metadata.get("file_name")
        if not fname:
            continue
        lem = FILE_LEMMAS.get(fname)
        if lem is None:
            # подстраховка: если файла нет в FILE_LEMMAS, считаем из содержимого узла
            lem = extract_lemmas(node.get_content())
        node.metadata["lemmas"] = list(lem)

    # 4) Persist index, hashes, chunk-lemma-cache
    try:
        idx.storage_context.persist(str(cfg.CACHE_PATH))
        cfg.HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
        cfg.HASH_FILE.write_text(json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8")
        save_chunk_lemma_cache()
        log.info("[build_index] Persisted index and hashes.")
    except Exception as e:
        log.warning("[build_index] Persist failed: %s", e)

    return idx
