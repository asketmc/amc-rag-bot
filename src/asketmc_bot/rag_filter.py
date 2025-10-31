from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from typing import FrozenSet, List, Optional, Sequence, Set, Tuple

from llama_index.core.schema import NodeWithScore
import config as cfg

__all__ = ["get_filtered_nodes", "build_context", "purge_filter_cache"]

rag_log = logging.getLogger("asketmc.rag")

def _cfg(name: str, default):
    return getattr(cfg, name, default)

TOP_K: int = int(_cfg("TOP_K", 8))
FILTER_ALPHA: float = float(_cfg("FILTER_ALPHA", 0.5))
SCORE_RELATIVE_THRESHOLD: float = float(_cfg("SCORE_RELATIVE_THRESHOLD", 0.7))
LEMMA_MATCH_RATIO: float = float(_cfg("LEMMA_MATCH_RATIO", 0.10))
CONTEXT_BONUS_PER_INTERSECTION: float = float(_cfg("CONTEXT_BONUS_PER_INTERSECTION", 0.1))
FILTER_CACHE_MAX_SIZE: int = int(_cfg("FILTER_CACHE_MAX_SIZE", 256))

def _safe_score(v: Optional[float]) -> float:
    try:
        return float(v) if v is not None else 0.0
    except Exception:
        return 0.0

def _node_id(n: NodeWithScore) -> str:
    node = n.node
    return (
        getattr(node, "node_id", None)
        or getattr(node, "id_", None)
        or node.metadata.get("node_id")
        or node.metadata.get("doc_id")
        or node.metadata.get("file_name")
        or f"node@{id(node)}"
    )

def _metadata_lemmas(n: NodeWithScore) -> FrozenSet[str]:
    lem = n.node.metadata.get("lemmas", [])
    try:
        return frozenset(str(x) for x in lem)
    except Exception:
        return frozenset()

def _cache_key(qlem: FrozenSet[str], raw_nodes: Sequence[NodeWithScore]) -> str:
    q_part = " ".join(sorted(qlem))
    fp_items: List[str] = []
    for n in raw_nodes[:64]:
        fp_items.append(f"{_node_id(n)}:{_safe_score(n.score):.6f}")
    fp = "|".join(fp_items)
    return hashlib.sha256(f"{q_part}||{fp}".encode("utf-8")).hexdigest()

def _content_from_node(n: NodeWithScore) -> str:
    node = n.node
    try:
        if hasattr(node, "get_content"):
            return node.get_content(metadata_mode="none") or ""
        return getattr(node, "text", "") or ""
    except Exception:
        return ""

_FILTER_CACHE: "OrderedDict[str, Tuple[NodeWithScore, ...]]" = OrderedDict()
_FILTER_CACHE_LOCK = asyncio.Lock()

def _cache_get(key: str) -> Optional[List[NodeWithScore]]:
    tpl = _FILTER_CACHE.get(key)
    if tpl is None:
        return None
    _FILTER_CACHE.move_to_end(key, last=True)
    return list(tpl)

def _cache_put(key: str, nodes: List[NodeWithScore]) -> None:
    _FILTER_CACHE[key] = tuple(nodes)
    _FILTER_CACHE.move_to_end(key, last=True)
    while len(_FILTER_CACHE) > FILTER_CACHE_MAX_SIZE:
        _FILTER_CACHE.popitem(last=False)

def purge_filter_cache() -> None:
    _FILTER_CACHE.clear()

async def _filter_nodes(raw_nodes: List[NodeWithScore], qlem: FrozenSet[str]) -> List[NodeWithScore]:
    if not raw_nodes:
        return []
    scores = [_safe_score(n.score) for n in raw_nodes]
    max_score = max(scores) if scores else 1.0
    if max_score <= 0.0:
        max_score = 1.0
    alpha = FILTER_ALPHA
    strict: List[Tuple[NodeWithScore, float]] = []
    for n in raw_nodes:
        s = _safe_score(n.score)
        lemmas = _metadata_lemmas(n)
        inter = len(qlem & lemmas)
        rel_score = s / max_score
        ratio = inter / (len(qlem) or 1)
        weight = s + alpha * ratio
        if (rel_score >= SCORE_RELATIVE_THRESHOLD) or (ratio >= LEMMA_MATCH_RATIO):
            strict.append((n, weight))
    if strict:
        strict.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in strict[:TOP_K]]
    fallback: List[Tuple[NodeWithScore, float, int, float]] = []
    for n in raw_nodes:
        s = _safe_score(n.score)
        lemmas = _metadata_lemmas(n)
        inter = len(qlem & lemmas)
        if inter <= 0:
            continue
        ratio = inter / (len(qlem) or 1)
        weight = s + alpha * ratio
        fallback.append((n, weight, inter, s))
    fallback.sort(key=lambda x: x[1], reverse=True)
    return [n for n, _, _, _ in fallback[:TOP_K]]

async def get_filtered_nodes(raw_nodes: List[NodeWithScore], qlem: FrozenSet[str]) -> List[NodeWithScore]:
    key = _cache_key(qlem, raw_nodes)
    async with _FILTER_CACHE_LOCK:
        cached = _cache_get(key)
    if cached is not None:
        return cached
    nodes = await _filter_nodes(raw_nodes, qlem)
    async with _FILTER_CACHE_LOCK:
        _cache_put(key, nodes)
    return list(nodes)

def build_context(nodes: List[NodeWithScore], qlem: FrozenSet[str], char_limit: int) -> str:
    beta = CONTEXT_BONUS_PER_INTERSECTION
    scored: List[Tuple[NodeWithScore, float]] = []
    for n in nodes:
        s = _safe_score(n.score)
        inter = len(qlem & _metadata_lemmas(n))
        weight = s + beta * float(inter)
        scored.append((n, weight))
    scored.sort(key=lambda x: x[1], reverse=True)
    parts: List[str] = []
    seen_hashes: Set[str] = set()
    total = 0
    sep = "\n---\n"
    sep_len = len(sep)
    for n, _ in scored:
        txt = (_content_from_node(n) or "").strip()
        if not txt:
            continue
        h = hashlib.sha256(txt.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            continue
        prospective = total + (sep_len if parts else 0) + len(txt)
        if prospective > max(0, int(char_limit)):
            break
        seen_hashes.add(h)
        if parts:
            parts.append(sep)
            total += sep_len
        parts.append(txt)
        total += len(txt)
    return "".join(parts)
