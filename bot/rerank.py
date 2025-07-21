#!/usr/bin/env python3.10
"""
rerank.py – production-ready version
All parameters are taken from config.py.
CPU/GPU switch, thread-safe initialization,
limited ThreadPoolExecutor, safetensors loading,
explicit resource release and race condition protection.
Any inference or initialization error returns an empty list (fail-safe).
In DEBUG mode, maximum logging at all rerank stages.
"""

from __future__ import annotations

import sys
import logging
import inspect
import time
import asyncio
from typing import List, Tuple, Protocol, Optional
from concurrent.futures import ThreadPoolExecutor

import torch
import sentence_transformers
from sentence_transformers import CrossEncoder
from llama_index.core.schema import NodeWithScore

import config as cfg

# ──────────────────────────────────────────────────────────
# sentence-transformers/CrossEncoder diagnostics
# ──────────────────────────────────────────────────────────
print("sentence_transformers version:", sentence_transformers.__version__)
print("CrossEncoder imported from:", CrossEncoder.__module__)

cross_sig = inspect.signature(CrossEncoder.__init__)
print("CrossEncoder __init__ signature:", cross_sig)
if "trust_remote_code" not in cross_sig.parameters:
    print("WARNING: CrossEncoder does not support trust_remote_code! Possible package conflict or outdated version.")

# ──────────────────────────────────────────────────────────
# Logging configuration
# ──────────────────────────────────────────────────────────
log = logging.getLogger("asketmc.rerank")
log.setLevel(logging.DEBUG if getattr(cfg, "DEBUG", False) else logging.INFO)
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"))
    log.addHandler(h)

log.info("[INIT] rerank.py starting, torch.cuda: %s", torch.cuda.is_available())
log.info("[INIT] CrossEncoder from: %s", CrossEncoder.__module__)
log.info("[INIT] CrossEncoder signature: %s", cross_sig)

# ──────────────────────────────────────────────────────────
# Global objects
# ──────────────────────────────────────────────────────────
_reranker: Optional[CrossEncoder] = None
_executor: Optional[ThreadPoolExecutor] = None
_init_lock = asyncio.Lock()

log.debug("[GLOBAL] Global objects initialized: _reranker=None, _executor=None, _init_lock created.")

def log_global_state(note: str = ""):
    log.info("[GLOBAL STATE] %s _reranker=%r _executor=%r _init_lock=%r",
             note, _reranker, _executor, _init_lock)

log_global_state("After declaration")

# ──────────────────────────────────────────────────────────
# Typing for node.get_content / node.text
# ──────────────────────────────────────────────────────────
class NodeLike(Protocol):
    def get_content(self) -> str: ...
    text: str

# ──────────────────────────────────────────────────────────
# Global objects and thread/async safety
# ──────────────────────────────────────────────────────────
log.debug("[GLOBAL] Global objects initialized: _reranker=None, _executor=None, _init_lock created (id=%s).", id(_init_lock))

def log_global_state(note: str = ""):
    log.info(
        "[GLOBAL STATE] %s | _reranker: %s (id=%s, type=%s) | _executor: %s (id=%s, type=%s) | _init_lock: id=%s (locked=%s, type=%s)",
        note,
        repr(_reranker),
        id(_reranker),
        type(_reranker).__name__ if _reranker else None,
        repr(_executor),
        id(_executor),
        type(_executor).__name__ if _executor else None,
        id(_init_lock),
        _init_lock.locked(),
        type(_init_lock).__name__,
    )

log_global_state("After declaration")

# For controlling cleanup/initialization
def reset_globals():
    global _reranker, _executor
    log.info("[RESET] Resetting global objects reranker/executor")
    _reranker = None
    if _executor:
        _executor.shutdown(wait=True)
        log.info("[RESET] Executor shutdown complete")
        _executor = None
    log_global_state("After reset_globals()")

# ──────────────────────────────────────────────────────────
# Utility functions and thread-safe reranker with extended logging
# ──────────────────────────────────────────────────────────

def _choose_device() -> str:
    device = str(getattr(cfg, "RERANKER_DEVICE", "cpu")).lower()
    log.debug("[_choose_device] RERANKER_DEVICE=%r", device)
    if device == "cuda":
        cuda_ok = torch.cuda.is_available()
        log.debug("[_choose_device] torch.cuda.is_available() = %s", cuda_ok)
        if cuda_ok:
            return "cuda"
        log.error("[_choose_device] RERANKER_DEVICE='cuda', but GPU is unavailable.")
        raise RuntimeError("RERANKER_DEVICE='cuda', but GPU is unavailable.")
    if device != "cpu":
        log.error("[_choose_device] Invalid RERANKER_DEVICE=%r", device)
        raise ValueError("RERANKER_DEVICE must be 'cpu' or 'cuda'.")
    return "cpu"

async def init_reranker(force: bool = False) -> None:
    """
    Thread-safe CrossEncoder initialization.
    Sets up reranker and thread pool. Restarts if `force=True`.
    """
    global _reranker, _executor
    async with _init_lock:
        model_name = getattr(cfg, "RERANKER_MODEL_NAME", "BAAI/bge-reranker-large")
        max_len = getattr(cfg, "MAX_LEN", 512)
        workers = getattr(cfg, "EXECUTOR_WORKERS", 4)
        log.info(
            "[init_reranker] called with force=%s, model_name=%s, max_len=%d, workers=%d",
            force, model_name, max_len, workers
        )
        if _reranker is not None and not force:
            log.info("[init_reranker] Already initialized, force=False, skipping.")
            return
        try:
            if _executor:
                log.info("[init_reranker] Shutting down previous executor...")
                _executor.shutdown(wait=False)
                _executor = None
            if _reranker:
                if torch.cuda.is_available():
                    mem_before = torch.cuda.memory_allocated()
                log.info("[init_reranker] Deleting previous reranker...")
                del _reranker
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    mem_after = torch.cuda.memory_allocated()
                    log.info(
                        "[init_reranker] CUDA memory cleared: was %.2f MB, now %.2f MB",
                        mem_before / (1024 ** 2), mem_after / (1024 ** 2)
                    )
                _reranker = None
            device = _choose_device()
            log.info(
                "[init_reranker] Creating CrossEncoder on device %r, model: %s, max_len=%d",
                device, model_name, max_len
            )
            _reranker = CrossEncoder(
                model_name,
                device=device,
                max_length=max_len,
            )
            _executor = ThreadPoolExecutor(max_workers=workers)
            log.info(
                "[init_reranker] CrossEncoder '%s' (id=%r) loaded on %r, max_len=%d, workers=%d.",
                model_name, id(_reranker), device, max_len, workers
            )
        except Exception as ex:
            log.exception("[init_reranker] CrossEncoder initialization error: %s", ex)
            raise

async def shutdown_reranker() -> None:
    """
    Releases thread pool and GPU memory.
    """
    global _reranker, _executor
    async with _init_lock:
        log.info("[shutdown_reranker] called.")
        try:
            if _executor:
                log.info("[shutdown_reranker] Shutting down executor...")
                _executor.shutdown(wait=False)
                _executor = None
            if _reranker:
                if torch.cuda.is_available():
                    mem_before = torch.cuda.memory_allocated()
                log.info("[shutdown_reranker] Deleting reranker...")
                del _reranker
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    mem_after = torch.cuda.memory_allocated()
                    log.info(
                        "[shutdown_reranker] CUDA memory cleared: was %.2f MB, now %.2f MB",
                        mem_before / (1024 ** 2), mem_after / (1024 ** 2)
                    )
                _reranker = None
            log.info("[shutdown_reranker] Reranker and executor successfully released.")
        except Exception as ex:
            log.exception("[shutdown_reranker] Resource release error: %s", ex)

def _filter_pairs(query: str, nodes: List[NodeWithScore]) -> Tuple[List[List[str]], List[NodeWithScore]]:
    """
    Trims candidate list and forms (query, document) pairs,
    skipping empty documents.
    """
    cand_nodes: List[NodeWithScore] = nodes[:getattr(cfg, "RERANK_INPUT_K", 20)]
    pairs: List[List[str]] = []
    filtered_nodes: List[NodeWithScore] = []

    for idx, n in enumerate(cand_nodes):
        content = ""
        if hasattr(n.node, "get_content"):
            content = n.node.get_content() or ""
        elif hasattr(n.node, "text"):
            content = n.node.text or ""

        log.debug(
            "[_filter_pairs] idx=%d, node_id=%r, content_len=%d, content_sample=%r",
            idx, getattr(n.node, "id", "n/a"), len(content), content[:50]
        )

        if content.strip():
            pairs.append([query, content])
            filtered_nodes.append(n)

    log.debug("[_filter_pairs] selected %d / %d candidates", len(filtered_nodes), len(cand_nodes))
    return pairs, filtered_nodes

async def rerank(
    query: str,
    nodes: List[NodeWithScore],
) -> List[NodeWithScore]:
    """
    Hardware-agnostic (CPU/GPU) reranker for top-k nodes.
    Returns the top cfg.RERANK_OUTPUT_K elements.
    Any error yields an empty list.
    Extended logging: model name, batch_size, top_k, sample of query and document.
    """
    global _reranker, _executor
    model_name = getattr(cfg, "RERANKER_MODEL_NAME", "BAAI/bge-reranker-large")
    max_len = getattr(cfg, "MAX_LEN", 512)
    input_k = getattr(cfg, "RERANK_INPUT_K", 20)
    output_k = getattr(cfg, "RERANK_OUTPUT_K", 5)
    batch_size = getattr(cfg, "BATCH_SIZE", 16)
    log.debug(
        "[rerank] query_len=%d nodes=%d model=%s max_len=%d input_k=%d output_k=%d batch_size=%d",
        len(query), len(nodes), model_name, max_len, input_k, output_k, batch_size
    )
    if _reranker is None or _executor is None:
        log.warning("[rerank] Reranker not initialized — attempting initialization.")
        try:
            await init_reranker()
        except Exception as ex:
            log.error("[rerank] Reranker initialization failed: %s", ex)
            return []
    if not nodes:
        log.debug("[rerank] No input nodes, returning empty list")
        return []
    if len(query) > getattr(cfg, "QUERY_MAX_CHARS", 2048):
        log.warning("[rerank] Query exceeds %d character limit.", getattr(cfg, "QUERY_MAX_CHARS", 2048))
        raise ValueError(f"query exceeds {getattr(cfg, 'QUERY_MAX_CHARS', 2048)} characters")
    pairs, valid_nodes = _filter_pairs(query, nodes)
    if not pairs:
        # Log input and sample docs
        log.warning(
            "[rerank] no pairs after filtering. query=%r, sample_node_ids=%r",
            query[:120], [getattr(n.node, "id", "n/a") for n in nodes[:3]]
        )
        for idx, n in enumerate(nodes[:3]):
            if hasattr(n.node, "get_content"):
                sample_content = n.node.get_content()
            else:
                sample_content = str(n.node)
            log.debug("[rerank] Empty doc idx=%d id=%r sample=%r", idx, getattr(n.node, "id", "n/a"), sample_content[:100])
        return []

    loop = asyncio.get_running_loop()

    def _predict() -> List[float]:
        t0 = time.perf_counter()
        try:
            log.info(
                "[_predict] Model: %s | input_k=%d | batch_size=%d | valid_nodes=%d",
                model_name, input_k, batch_size, len(valid_nodes)
            )
            scores = _reranker.predict(
                pairs,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            dt = time.perf_counter() - t0
            log.info(
                "[_predict] Rerank completed in %.3f sec for %d items, model: %s (id=%r)",
                dt, len(pairs), model_name, id(_reranker)
            )
            log.debug("[_predict] scores=%r", scores)
            return scores
        except Exception as ex:
            log.exception("[_predict] CrossEncoder inference error: %s", ex)
            raise

    try:
        scores: List[float] = await loop.run_in_executor(_executor, _predict)
    except Exception as ex:
        log.error("[rerank] Rerank executor failed: %s", ex)
        return []

    ranked: List[Tuple[NodeWithScore, float]] = sorted(
        zip(valid_nodes, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    log.debug(
        "[rerank] ranked_top=[%s]",
        ", ".join(
            "(id=%r, score=%.6f, file=%r)" % (
                getattr(n.node, "id", "n/a"),
                s,
                n.node.metadata.get("file_name", "n/a"),
            )
            for n, s in ranked[:output_k]
        )
    )

    if not ranked:
        # If still empty, log input, config, and sample
        log.warning(
            "[rerank] Empty ranking result. query=%r, model=%s, batch_size=%d, top_k=%d",
            query[:120], model_name, batch_size, output_k
        )
        for idx, n in enumerate(nodes[:3]):
            if hasattr(n.node, "get_content"):
                sample_content = n.node.get_content()
            else:
                sample_content = str(n.node)
            log.debug("[rerank] Sample doc idx=%d id=%r sample=%r", idx, getattr(n.node, "id", "n/a"), sample_content[:100])
        return []

    return [n for n, _ in ranked[:output_k]]

# ──────────────────────────────────────────────────────────
