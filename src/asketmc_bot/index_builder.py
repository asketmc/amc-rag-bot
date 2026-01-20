#!/usr/bin/env python3.10
"""
index_builder.py — embeddings configuration and LlamaIndex index build/load.

Exported API:
- build_index() -> VectorStoreIndex  (async)

Responsibilities:
- configures LlamaIndex Settings.embed_model (BAAI/bge-m3) and SentenceSplitter
- loads or rebuilds index based on document hashes
- maintains file-lemma cache via asketmc_bot.lemma
- annotates nodes with metadata["lemmas"]
"""

from __future__ import annotations

# Stdlib
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Third-party
try:
    import torch  # type: ignore
except Exception:  # ImportError + binary load errors
    torch = None  # type: ignore[assignment]

from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Local (src-layout safe)
from asketmc_bot import config as cfg
from asketmc_bot.lemma import (
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
def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "Torch is not available. Install extra dependencies, e.g. "
            "`pip install -e '.[gpu]'` (or move torch to an optional extra)."
        )


_require_torch()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore[union-attr]
EMBED_LOG_EVERY = getattr(cfg, "EMBED_LOG_EVERY", 1000)


class LoggingBGE(HuggingFaceEmbedding):
    _counter: int = 0
    __slots__ = ()

    def _get_text_embedding(self, text: str):
        t0 = time.perf_counter()
        vec = super()._get_text_embedding(text)
        type(self)._counter += 1
        if type(self)._counter % EMBED_LOG_EVERY == 0:
            logging.getLogger("asketmc.embed").info(
                "EMB %d | %s | %s… | %.3fs",
                type(self)._counter,
                self._target_device,
                text[:120].replace("\n", " "),
                time.perf_counter() - t0,
                )
        return vec


_SETTINGS_CONFIGURED = False


def _configure_llamaindex_settings() -> None:
    global _SETTINGS_CONFIGURED
    if _SETTINGS_CONFIGURED:
        return

    Settings.embed_model = LoggingBGE("BAAI/bge-m3", normalize=True, device=DEVICE)
    Settings.node_parser = SentenceSplitter(
        chunk_size=getattr(cfg, "CHUNK_SIZE", 512),
        chunk_overlap=getattr(cfg, "CHUNK_OVERLAP", 128),
        include_metadata=False,
        paragraph_separator="\n\n",
    )
    _SETTINGS_CONFIGURED = True


# ─────────────────────────────────────────────────────────
# Document helpers
# ─────────────────────────────────────────────────────────
def _make_document(fp: Path) -> Optional[Document]:
    """
    Create a Document with a stable id_ so we can delete/update incrementally.
    Using filename as id_ assumes DOCS_PATH has unique filenames.
    """
    try:
        text = fp.read_text("utf-8", "ignore")
    except Exception as e:
        log.warning("[build_index] Read error %s: %s", fp, e)
        return None
    return Document(text=text, metadata={"file_name": fp.name}, id_=fp.name)


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def _doc_hash(fp: Path) -> str:
    """SHA-256 of a file read by 1MB chunks."""
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_stored_hashes() -> Dict[str, str]:
    if not cfg.HASH_FILE.exists():
        return {}
    try:
        return json.loads(cfg.HASH_FILE.read_text("utf-8"))
    except Exception:
        return {}


def _compute_hashes_incremental(docs: List[Path], stored: Dict[str, str]) -> Dict[str, str]:
    """
    Reduce hashing cost:
    - If HASH_FILE exists and is newer than doc mtime, reuse stored hash.
    - Otherwise compute sha256.
    """
    hashes: Dict[str, str] = {}
    hash_file_mtime_ns = cfg.HASH_FILE.stat().st_mtime_ns if cfg.HASH_FILE.exists() else 0

    for fp in docs:
        try:
            mtime_ns = fp.stat().st_mtime_ns
        except Exception:
            mtime_ns = 0

        if fp.name in stored and mtime_ns and mtime_ns <= hash_file_mtime_ns:
            hashes[fp.name] = stored[fp.name]
        else:
            hashes[fp.name] = _doc_hash(fp)

    return hashes


def _plan_doc_changes(stored: Dict[str, str], current: Dict[str, str]) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Return (added, removed, modified) doc ids (filenames).
    """
    stored_keys = set(stored.keys())
    current_keys = set(current.keys())

    added = current_keys - stored_keys
    removed = stored_keys - current_keys
    modified = {k for k in (stored_keys & current_keys) if stored.get(k) != current.get(k)}

    return added, removed, modified


def _load_index_if_present() -> Optional[VectorStoreIndex]:
    if not cfg.CACHE_PATH.exists():
        return None
    try:
        return load_index_from_storage(StorageContext.from_defaults(persist_dir=str(cfg.CACHE_PATH)))
    except Exception as e:
        log.warning("[build_index] Failed to load cached index: %s", e)
        return None


# ─────────────────────────────────────────────────────────
# Build / Load index
# ─────────────────────────────────────────────────────────
async def build_index() -> VectorStoreIndex:
    """
    Build or load index of documents from cfg.DOCS_PATH.
    - Validates hashes (cfg.HASH_FILE)
    - Loads FILE_LEMMAS cache and updates it for changed files
    - Annotates nodes with metadata["lemmas"]
    - Persists index, hashes, and chunk lemma cache
    """
    _configure_llamaindex_settings()

    docs: List[Path] = [p for p in cfg.DOCS_PATH.glob("*") if p.is_file()]
    stored: Dict[str, str] = _load_stored_hashes()
    hashes: Dict[str, str] = _compute_hashes_incremental(docs, stored)

    # 1) File-lemma cache
    load_saved_lemmas()
    try:
        await update_file_lemmas_async(docs, stored, hashes)
    except Exception as e:
        log.warning("[build_index] update_file_lemmas_async failed: %s", e)

    # 2) Index: load if present, then apply incremental changes
    idx: Optional[VectorStoreIndex] = _load_index_if_present()
    if idx is not None:
        log.info("[build_index] Loaded index from cache: %s", cfg.CACHE_PATH)

        added, removed, modified = _plan_doc_changes(stored, hashes)
        if added or removed or modified:
            log.info(
                "[build_index] Incremental update | added=%d removed=%d modified=%d",
                len(added),
                len(removed),
                len(modified),
            )

            # removals first
            for doc_id in sorted(removed):
                try:
                    idx.delete_ref_doc(doc_id, delete_from_docstore=True)
                except Exception as e:
                    log.warning("[build_index] delete_ref_doc failed (%s): %s", doc_id, e)

            # updates
            for doc_id in sorted(modified):
                fp = cfg.DOCS_PATH / doc_id
                doc = _make_document(fp)
                if doc is None:
                    continue
                try:
                    idx.update_ref_doc(doc)
                except Exception as e:
                    log.warning("[build_index] update_ref_doc failed (%s): %s", doc_id, e)

            # inserts
            for doc_id in sorted(added):
                fp = cfg.DOCS_PATH / doc_id
                doc = _make_document(fp)
                if doc is None:
                    continue
                try:
                    idx.insert(doc)
                except Exception as e:
                    log.warning("[build_index] insert failed (%s): %s", doc_id, e)

    # cache missing or load failed -> build from scratch
    if idx is None:
        ll_docs: List[Document] = []
        for fp in docs:
            doc = _make_document(fp)
            if doc is not None:
                ll_docs.append(doc)

        idx = VectorStoreIndex.from_documents(ll_docs)
        log.info("[build_index] Built new index from %d docs", len(ll_docs))

    # 3) Annotate lemmas in node metadata
    for node in idx.docstore.docs.values():
        fname = node.metadata.get("file_name")
        if not fname:
            continue
        lem = FILE_LEMMAS.get(fname)
        if lem is None:
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
