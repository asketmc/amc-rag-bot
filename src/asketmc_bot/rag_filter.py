# rag_filter.py — фильтрация кандидатов и сборка контекста для RAG
from __future__ import annotations

from typing import List, FrozenSet, Tuple, Dict, Set
import logging

from llama_index.core.schema import NodeWithScore
import config as cfg

# Логгер раздела RAG
rag_log = logging.getLogger("asketmc.rag")

# Кэш результатов фильтрации:
#   key: str (sha256 от отсортированного множества лемм запроса)
#   val: List[NodeWithScore]
FILTER_CACHE: Dict[str, List[NodeWithScore]] = {}

def _cache_key(qlem: FrozenSet[str]) -> str:
    """
    Вычисляет ключ кэша для множества лемм запроса.
    """
    import hashlib
    key = hashlib.sha256(" ".join(sorted(qlem)).encode("utf-8")).hexdigest()
    rag_log.debug("[rag_filter._cache_key] qlem=%r -> key=%s", list(qlem), key)
    return key

async def _filter_nodes(
    raw_nodes: List[NodeWithScore], qlem: FrozenSet[str]
) -> List[NodeWithScore]:
    """
    Жёсткая фильтрация и ранжирование по относительному скору
    и пересечению лемм с запросом.
    """
    rag_log.info("[rag_filter._filter_nodes] raw_nodes=%d, qlem=%r", len(raw_nodes), list(qlem))
    if not raw_nodes:
        rag_log.warning("[rag_filter._filter_nodes] empty raw_nodes -> []")
        return []

    max_score = max(n.score for n in raw_nodes)
    rag_log.debug("[rag_filter._filter_nodes] max_score=%.6f", max_score)

    alpha = getattr(cfg, "FILTER_ALPHA", 0.5)
    strict: List[Tuple[NodeWithScore, float]] = []
    for idx, n in enumerate(raw_nodes):
        lemmas = frozenset(n.node.metadata.get("lemmas", []))
        inter = len(qlem & lemmas)
        rel_score = n.score / (max_score or 1.0)
        ratio = inter / (len(qlem) or 1)
        weight = n.score + alpha * ratio
        rag_log.debug(
            "[rag_filter._filter_nodes] idx=%d file=%r score=%.6f rel=%.3f inter=%d ratio=%.3f weight=%.3f",
            idx, n.node.metadata.get("file_name", "n/a"), n.score, rel_score, inter, ratio, weight
        )
        if (
            rel_score >= getattr(cfg, "SCORE_RELATIVE_THRESHOLD", 0.7)
            or ratio >= getattr(cfg, "LEMMA_MATCH_RATIO", 0.1)
        ):
            strict.append((n, weight))

    if strict:
        strict.sort(key=lambda x: -x[1])
        for i, (n, w) in enumerate(strict[: cfg.TOP_K]):
            rag_log.debug(
                "[rag_filter._filter_nodes] TOP#%d file=%r score=%.6f weight=%.3f",
                i, n.node.metadata.get("file_name", "n/a"), n.score, w
            )
        return [n for n, _ in strict[: cfg.TOP_K]]

    # Fallback: хотя бы одно пересечение лемм
    fallback: List[Tuple[NodeWithScore, float, int, float]] = []
    for idx, n in enumerate(raw_nodes):
        lemmas = frozenset(n.node.metadata.get("lemmas", []))
        inter = len(qlem & lemmas)
        ratio = inter / (len(qlem) or 1)
        weight = n.score + alpha * ratio
        if inter > 0:
            fallback.append((n, weight, inter, n.score))
        rag_log.debug(
            "[rag_filter._filter_nodes:fallback] idx=%d file=%r score=%.6f inter=%d weight=%.3f",
            idx, n.node.metadata.get("file_name", "n/a"), n.score, inter, weight
        )

    fallback.sort(key=lambda x: -x[1])
    for i, (n, w, inter, score) in enumerate(fallback[: cfg.TOP_K]):
        rag_log.debug(
            "[rag_filter._filter_nodes:fallback] TOP#%d file=%r score=%.6f inter=%d weight=%.3f",
            i, n.node.metadata.get("file_name", "n/a"), score, inter, w
        )
    return [n for n, _, _, _ in fallback[: cfg.TOP_K]]

async def get_filtered_nodes(
    raw_nodes: List[NodeWithScore], qlem: FrozenSet[str]
) -> List[NodeWithScore]:
    """
    Кэшируемая обёртка над _filter_nodes.
    """
    key = _cache_key(qlem)
    if key in FILTER_CACHE:
        rag_log.info("[rag_filter.get_filtered_nodes] cache hit key=%s nodes=%d", key, len(FILTER_CACHE[key]))
        return FILTER_CACHE[key]
    rag_log.info("[rag_filter.get_filtered_nodes] cache miss key=%s -> filter", key)
    nodes = await _filter_nodes(raw_nodes, qlem)
    FILTER_CACHE[key] = nodes
    return nodes

def _content_from_node(n: NodeWithScore) -> str:
    txt = getattr(n.node, "get_content", lambda: "")()
    rag_log.debug("[rag_filter._content_from_node] file=%r len=%d",
                  n.node.metadata.get("file_name", "n/a"), len(txt))
    return txt

def build_context(
    nodes: List[NodeWithScore], qlem: FrozenSet[str], char_limit: int
) -> str:
    """
    Сборка итогового контекста из уникальных чанков (до лимита символов).
    """
    rag_log.info("[rag_filter.build_context] nodes=%d char_limit=%d", len(nodes), char_limit)
    beta = 0.1
    scored: List[Tuple[NodeWithScore, float]] = []
    for n in nodes:
        lemmas = frozenset(n.node.metadata.get("lemmas", []))
        inter = len(qlem & lemmas)
        weight = n.score + beta * inter
        scored.append((n, weight))
        rag_log.debug(
            "[rag_filter.build_context] file=%r score=%.6f inter=%d weight=%.6f",
            n.node.metadata.get("file_name", "n/a"), n.score, inter, weight
        )
    scored.sort(key=lambda x: x[1], reverse=True)

    parts: List[str] = []
    seen_hashes: Set[int] = set()
    total = 0
    for n, _ in scored:
        txt = _content_from_node(n).strip()
        if not txt:
            continue
        h = hash(txt)
        if h in seen_hashes:
            continue
        if total + len(txt) > char_limit:
            rag_log.info("[rag_filter.build_context] limit reached total=%d", total)
            break
        seen_hashes.add(h)
        parts.append(txt)
        total += len(txt) + 4
    rag_log.info("[rag_filter.build_context] parts=%d total=%d", len(parts), total)
    return "\n---\n".join(parts)
