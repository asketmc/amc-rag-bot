#!/usr/bin/env python3.10
"""
rag_langchain.py — LangGraph integration for the Asketmc RAG Discord bot.

Design goals:
- No side effects at import time (no global graph/pipeline objects).
- Dependency injection (no imports from `main` that create cycles).
- PEP 8/257 compliant, typed, idempotent nodes.
- Structured logging; deterministic behavior; clear guards and fallbacks.
- Safe import behavior when optional deps (langgraph/langsmith) are not installed.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Awaitable, Callable, Dict, List, Set, TypedDict, cast

from asketmc_bot import config as cfg

log = logging.getLogger("asketmc.rag.langgraph")

# --- Optional dependency: langgraph -----------------------------------------
# Keep module importable even when langgraph isn't installed (dev/CI environments).
try:
    # pylint: disable=import-error
    from langgraph.graph import END, START, StateGraph  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    END = "__end__"  # type: ignore[assignment]
    START = "__start__"  # type: ignore[assignment]
    StateGraph = Any  # type: ignore[misc,assignment]


class RagState(TypedDict, total=False):
    """Graph state passed between LangGraph nodes."""
    question: str
    lemmas: Set[str]
    retrieved_nodes: List[Any]
    reranked_nodes: List[Any]
    filtered_nodes: List[Any]
    context: str
    answer: str


class RagDeps(TypedDict):
    """
    Dependencies required by the RAG pipeline.
    Provide concrete callables and objects to avoid circular imports.
    """

    extract_lemmas: Callable[[str], Set[str]]
    retriever: Any  # must expose `aretrieve(q: str) -> Awaitable[List[Any]]`
    rerank: Callable[[str, List[Any]], Awaitable[List[Any]]]
    get_filtered_nodes: Callable[[List[Any], Set[str]], Awaitable[List[Any]]]
    build_context: Callable[[List[Any], Set[str], int], str]

    # Must accept sys_prompt + ctx_txt + q (discord user question).
    # `timeout_sec` may be supported by implementation.
    query_model: Callable[..., Awaitable[tuple[str, bool]]]

    sys_prompt: str
    ctx_len_local: int
    ctx_len_remote: int


def _maybe_init_langsmith() -> None:
    """
    Initialize LangSmith tracing if enabled via env (LANGSMITH_TRACING=1/true/yes).

    This function must not raise; tracing is always best-effort.
    """
    val = (os.getenv("LANGSMITH_TRACING") or "").strip().lower()
    if val not in {"1", "true", "yes"}:
        return

    try:
        # pylint: disable=import-error,import-outside-toplevel
        import langsmith  # type: ignore[import-not-found,unused-import] # noqa: F401
        log.info("LangSmith tracing enabled (module present).")
    except Exception as exc:  # pragma: no cover
        log.warning("LangSmith tracing requested but not available: %s", exc)


def _require_non_empty_question(state: RagState, *, node_name: str) -> str:
    question = (state.get("question") or "").strip()
    if not question:
        raise ValueError(f"Empty question for {node_name} node.")
    return question


def _lemmas_node(deps: RagDeps) -> Callable[[RagState], Dict[str, Any]]:
    def _impl(state: RagState) -> Dict[str, Any]:
        question = _require_non_empty_question(state, node_name="lemmas")
        lemmas = deps["extract_lemmas"](question)
        return {"lemmas": set(lemmas) if lemmas else set()}

    return _impl


def _retrieve_node(deps: RagDeps) -> Callable[[RagState], Awaitable[Dict[str, Any]]]:
    async def _impl(state: RagState) -> Dict[str, Any]:
        question = _require_non_empty_question(state, node_name="retrieve")
        retrieved = await deps["retriever"].aretrieve(question)
        return {"retrieved_nodes": list(retrieved) if retrieved else []}

    return _impl


def _rerank_node(deps: RagDeps) -> Callable[[RagState], Awaitable[Dict[str, Any]]]:
    async def _impl(state: RagState) -> Dict[str, Any]:
        question = _require_non_empty_question(state, node_name="rerank")
        retrieved = state.get("retrieved_nodes") or []
        if not retrieved:
            return {"reranked_nodes": []}

        ranked = await deps["rerank"](question, retrieved)
        # Keep stable fallback to original ordering if reranker returns empty/None.
        return {"reranked_nodes": list(ranked) if ranked else list(retrieved)}

    return _impl


def _filter_node(deps: RagDeps) -> Callable[[RagState], Awaitable[Dict[str, Any]]]:
    async def _impl(state: RagState) -> Dict[str, Any]:
        lemmas = state.get("lemmas") or set()
        candidates = state.get("reranked_nodes") or state.get("retrieved_nodes") or []
        filtered = await deps["get_filtered_nodes"](candidates, cast(Set[str], lemmas))
        return {"filtered_nodes": list(filtered) if filtered else []}

    return _impl


def _context_node(
        deps: RagDeps,
        *,
        use_remote_ctx: bool,
) -> Callable[[RagState], Dict[str, Any]]:
    def _impl(state: RagState) -> Dict[str, Any]:
        lemmas = state.get("lemmas") or set()
        nodes = state.get("filtered_nodes") or []
        limit = deps["ctx_len_remote"] if use_remote_ctx else deps["ctx_len_local"]
        ctx_txt = deps["build_context"](nodes, cast(Set[str], lemmas), int(limit))
        return {"context": ctx_txt}

    return _impl


def _llm_node(deps: RagDeps) -> Callable[[RagState], Awaitable[Dict[str, Any]]]:
    async def _impl(state: RagState) -> Dict[str, Any]:
        question = _require_non_empty_question(state, node_name="llm")
        context = (state.get("context") or "").strip()
        sys_prompt = (deps.get("sys_prompt") or "").strip()

        timeout_total = int(getattr(cfg, "HTTP_TIMEOUT_TOTAL", 240))

        # Prefer calling with keyword args matching LLMClient.query_model signature.
        try:
            text, _used_fallback = await deps["query_model"](
                sys_prompt=sys_prompt,
                ctx_txt=context,
                q=question,
                timeout_sec=timeout_total,
            )
        except TypeError:
            # Fallback for simpler callable signatures
            text, _used_fallback = await deps["query_model"](sys_prompt, context, question, timeout_total)

        answer = (text or "").strip() or "No answer."
        return {"answer": answer}

    return _impl


def build_rag_graph(
        deps: RagDeps,
        *,
        use_remote_ctx: bool = False,
) -> Any:
    """
    Build a LangGraph StateGraph for the RAG pipeline (no side effects, not compiled).

    Returns:
        StateGraph-like object (langgraph is optional at import time).
    """
    if StateGraph is Any:  # pragma: no cover
        raise RuntimeError("langgraph is not installed; cannot build graph.")

    graph = StateGraph(RagState)

    graph.add_node("lemmas", _lemmas_node(deps))
    graph.add_node("retrieve", _retrieve_node(deps))
    graph.add_node("rerank", _rerank_node(deps))
    graph.add_node("filter", _filter_node(deps))
    graph.add_node("context", _context_node(deps, use_remote_ctx=use_remote_ctx))
    graph.add_node("llm", _llm_node(deps))

    graph.add_edge(START, "lemmas")
    graph.add_edge("lemmas", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "filter")
    graph.add_edge("filter", "context")
    graph.add_edge("context", "llm")
    graph.add_edge("llm", END)

    return graph


async def run_rag_pipeline(
        question: str,
        deps: RagDeps,
        *,
        use_remote_ctx: bool = False,
        enable_tracing: bool = False,
) -> str:
    """
    Convenience runner to build, compile, and invoke the graph once.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if enable_tracing:
        _maybe_init_langsmith()

    graph = build_rag_graph(deps, use_remote_ctx=use_remote_ctx)
    app = graph.compile()

    init_state: RagState = {"question": question.strip()}
    output: RagState = await app.ainvoke(init_state)

    answer = (output.get("answer") or "").strip()
    return answer or "No answer."


__all__ = [
    "RagState",
    "RagDeps",
    "build_rag_graph",
    "run_rag_pipeline",
]
