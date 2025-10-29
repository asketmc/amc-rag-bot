#!/usr/bin/env python3.10
"""Production-grade CrossEncoder reranker (async-safe, CPU/GPU).

Public API:
    * await init_reranker(force: bool = False) -> None
    * await shutdown_reranker() -> None
    * await rerank(query: str, nodes: List[NodeWithScore]) -> List[NodeWithScore]

Features:
    - Config-driven with safe defaults (taken from `config`).
    - Async/thread-safe initialization guarded by an asyncio.Lock.
    - Bounded ThreadPoolExecutor for CPU-bound inference.
    - CPU/GPU selection with CUDA availability checks.
    - Inference timeout and retries.
    - Strict input validation (regex + max length).
    - No eval/exec/shell, no unsafe deserialization.
    - Structured logging; DEBUG is verbose in development.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import torch
from llama_index.core.schema import NodeWithScore
from sentence_transformers import CrossEncoder

import config as cfg

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

_LOG = logging.getLogger("asketmc.rerank")
_LOG.setLevel(logging.DEBUG if getattr(cfg, "DEBUG", False) else logging.INFO)
if not _LOG.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    )
    _LOG.addHandler(_handler)

# -----------------------------------------------------------------------------
# Configuration defaults
# -----------------------------------------------------------------------------

_DEFAULT_ALLOWED_RE = (
    r"^[\w\s\.\,\:\;\!\?\(\)\-\[\]\{\}'\"/@#%&\*\+\=\<\>\$€£…©®™°|\\`~№"
    r"А-Яа-яЁё]+$"
)
ALLOWED_CHARS_REGEX = getattr(cfg, "ALLOWED_CHARS_REGEX", _DEFAULT_ALLOWED_RE)
_QUERY_MAX_CHARS = int(getattr(cfg, "QUERY_MAX_CHARS", 2048))

_RERANKER_MODEL_NAME = getattr(cfg, "RERANKER_MODEL_NAME", "BAAI/bge-reranker-large")
_RERANK_INPUT_K = int(getattr(cfg, "RERANK_INPUT_K", 20))
_RERANK_OUTPUT_K = int(getattr(cfg, "RERANK_OUTPUT_K", 5))
_MAX_LEN = int(getattr(cfg, "MAX_LEN", 512))
_BATCH_SIZE = int(getattr(cfg, "BATCH_SIZE", 16))
_RERANKER_DEVICE = str(getattr(cfg, "RERANKER_DEVICE", "cpu")).lower()
_EXECUTOR_WORKERS = int(getattr(cfg, "EXECUTOR_WORKERS", 4))
_PREDICT_TIMEOUT_SEC = float(getattr(cfg, "PREDICT_TIMEOUT_SEC", 120.0))
_PREDICT_RETRIES = int(getattr(cfg, "PREDICT_RETRIES", 1))  # 0 means no retries

# -----------------------------------------------------------------------------
# Globals (lazy-initialized; protected by _INIT_LOCK)
# -----------------------------------------------------------------------------

_RERANKER: Optional[CrossEncoder] = None
_EXECUTOR: Optional[ThreadPoolExecutor] = None
_INIT_LOCK = asyncio.Lock()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

_ALLOWED_RE = re.compile(ALLOWED_CHARS_REGEX, re.UNICODE)


def _sanitize_input(text: str, *, max_len: int = _QUERY_MAX_CHARS) -> str:
    """Validate and normalize user input."""
    if not isinstance(text, str):
        raise ValueError("query must be a string")
    s = text.strip()
    if not s:
        raise ValueError("query is empty")
    if len(s) > max_len:
        raise ValueError(f"query exceeds {max_len} characters")
    if not _ALLOWED_RE.fullmatch(s):
        raise ValueError("query contains disallowed characters")
    return s


def _choose_device() -> str:
    """Return device string based on config and CUDA availability."""
    if _RERANKER_DEVICE == "cpu":
        return "cpu"
    if _RERANKER_DEVICE == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("RERANKER_DEVICE='cuda' but CUDA is not available")
    raise ValueError("RERANKER_DEVICE must be 'cpu' or 'cuda'")


def _node_text(n: NodeWithScore) -> str:
    """Extract textual content from a NodeWithScore safely."""
    node = getattr(n, "node", n)
    if hasattr(node, "get_content"):
        try:
            return node.get_content() or ""
        except Exception:
            return ""
    if hasattr(node, "text"):
        return getattr(node, "text", "") or ""
    return ""


def _filter_pairs(
    query: str, nodes: List[NodeWithScore]
) -> Tuple[List[List[str]], List[NodeWithScore]]:
    """Return (query, doc) pairs and corresponding nodes, skipping empty docs."""
    cand_nodes: List[NodeWithScore] = nodes[:_RERANK_INPUT_K]
    pairs: List[List[str]] = []
    filtered_nodes: List[NodeWithScore] = []

    for idx, n in enumerate(cand_nodes):
        content = _node_text(n)
        if content.strip():
            pairs.append([query, content])
            filtered_nodes.append(n)
        else:
            _LOG.debug(
                "[_filter_pairs] skip empty doc idx=%d id=%r",
                idx,
                getattr(getattr(n, "node", n), "id", "n/a"),
            )

    _LOG.debug(
        "[_filter_pairs] selected %d/%d candidates",
        len(filtered_nodes),
        len(cand_nodes),
    )
    return pairs, filtered_nodes


# -----------------------------------------------------------------------------
# Lifecycle
# -----------------------------------------------------------------------------

async def init_reranker(force: bool = False) -> None:
    """Initialize CrossEncoder and thread pool safely; re-init on force=True."""
    global _RERANKER, _EXECUTOR
    async with _INIT_LOCK:
        if _RERANKER is not None and _EXECUTOR is not None and not force:
            _LOG.info("[init_reranker] already initialized; skip (force=False)")
            return

        if _EXECUTOR is not None:
            _LOG.info("[init_reranker] shutting down previous executor")
            _EXECUTOR.shutdown(wait=True)
            _EXECUTOR = None
        if _RERANKER is not None:
            try:
                if torch.cuda.is_available():
                    mem_before = torch.cuda.memory_allocated()
                del _RERANKER
                _RERANKER = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    mem_after = torch.cuda.memory_allocated()
                    _LOG.info(
                        "[init_reranker] CUDA cache cleared: %.2f -> %.2f MB",
                        mem_before / (1024 ** 2),
                        mem_after / (1024 ** 2),
                    )
            except Exception as exc:  # pragma: no cover
                _LOG.warning(
                    "[init_reranker] error during previous model cleanup: %s", exc
                )

        device = _choose_device()
        _LOG.info(
            "[init_reranker] loading CrossEncoder model=%s device=%s max_length=%d "
            "workers=%d",
            _RERANKER_MODEL_NAME,
            device,
            _MAX_LEN,
            _EXECUTOR_WORKERS,
        )
        _RERANKER = CrossEncoder(
            _RERANKER_MODEL_NAME,
            device=device,
            max_length=_MAX_LEN,
        )
        _EXECUTOR = ThreadPoolExecutor(max_workers=_EXECUTOR_WORKERS)
        _LOG.info(
            "[init_reranker] CrossEncoder ready (id=%r) on %s; executor workers=%d",
            id(_RERANKER),
            device,
            _EXECUTOR_WORKERS,
        )


async def shutdown_reranker() -> None:
    """Release thread pool and GPU memory (if any)."""
    global _RERANKER, _EXECUTOR
    async with _INIT_LOCK:
        _LOG.info("[shutdown_reranker] called")
        try:
            if _EXECUTOR is not None:
                _EXECUTOR.shutdown(wait=True)
                _EXECUTOR = None
            if _RERANKER is not None:
                try:
                    if torch.cuda.is_available():
                        mem_before = torch.cuda.memory_allocated()
                    del _RERANKER
                    _RERANKER = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        mem_after = torch.cuda.memory_allocated()
                        _LOG.info(
                            "[shutdown_reranker] CUDA cache cleared: %.2f -> %.2f MB",
                            mem_before / (1024 ** 2),
                            mem_after / (1024 ** 2),
                        )
                except Exception as exc:  # pragma: no cover
                    _LOG.warning(
                        "[shutdown_reranker] error during model cleanup: %s", exc
                    )
        finally:
            _LOG.info("[shutdown_reranker] resources released")


# -----------------------------------------------------------------------------
# Rerank
# -----------------------------------------------------------------------------

async def rerank(query: str, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
    """Score candidate nodes with a CrossEncoder and return top-K results.

    Args:
        query: User query string.
        nodes: Candidate nodes to rerank.

    Returns:
        Top `cfg.RERANK_OUTPUT_K` nodes sorted by descending score.
        Returns an empty list on inference failure.

    Raises:
        ValueError: If the query violates input policy.
    """
    global _RERANKER, _EXECUTOR

    query = _sanitize_input(query, max_len=_QUERY_MAX_CHARS)

    if not nodes:
        _LOG.debug("[rerank] empty input nodes -> []")
        return []

    if _RERANKER is None or _EXECUTOR is None:
        _LOG.info("[rerank] reranker not initialized; attempting init")
        await init_reranker(force=False)
        if _RERANKER is None or _EXECUTOR is None:
            _LOG.error("[rerank] init failed -> []")
            return []

    pairs, valid_nodes = _filter_pairs(query, nodes)
    if not pairs:
        _LOG.warning("[rerank] no valid documents after filtering -> []")
        return []

    loop = asyncio.get_running_loop()

    def _predict_once() -> List[float]:
        t0 = time.perf_counter()
        scores = _RERANKER.predict(
            pairs,
            batch_size=_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        dt = time.perf_counter() - t0
        _LOG.info(
            "[_predict_once] model=%s items=%d batch=%d took=%.3fs",
            _RERANKER_MODEL_NAME,
            len(pairs),
            _BATCH_SIZE,
            dt,
        )
        try:
            return scores.tolist()  # type: ignore[attr-defined]
        except Exception:
            return list(scores)

    attempt = 0
    last_err: Optional[Exception] = None
    while attempt <= _PREDICT_RETRIES:
        attempt += 1
        try:
            fut = loop.run_in_executor(_EXECUTOR, _predict_once)
            scores: List[float] = await asyncio.wait_for(
                fut, timeout=_PREDICT_TIMEOUT_SEC
            )
            ranked: List[Tuple[NodeWithScore, float]] = sorted(
                zip(valid_nodes, scores), key=lambda x: x[1], reverse=True
            )
            if not ranked:
                _LOG.warning("[rerank] empty result after scoring -> []")
                return []
            top_k = _RERANK_OUTPUT_K
            out = [n for n, _ in ranked[:top_k]]
            _LOG.debug(
                "[rerank] top%d ids=%s",
                top_k,
                [getattr(getattr(n, "node", n), "id", "n/a") for n in out],
            )
            return out
        except asyncio.TimeoutError as exc:
            last_err = exc
            _LOG.warning(
                "[rerank] inference timeout after %.1fs (attempt %d/%d)",
                _PREDICT_TIMEOUT_SEC,
                attempt,
                _PREDICT_RETRIES + 1,
            )
        except Exception as exc:
            last_err = exc
            _LOG.exception(
                "[rerank] inference error on attempt %d/%d: %s",
                attempt,
                _PREDICT_RETRIES + 1,
                exc,
            )
        if attempt <= _PREDICT_RETRIES:
            await asyncio.sleep(min(0.25 * attempt, 1.0))

    _LOG.error(
        "[rerank] giving up after %d attempts -> [] (%s)",
        _PREDICT_RETRIES + 1,
        last_err,
    )
    return []
