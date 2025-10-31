#!/usr/bin/env python3.10
"""
rag_langchain.py — LangGraph/LangChain integration for the Asketmc RAG Discord bot.

Design goals (senior-grade):
- No side effects at import time (no global graph/pipeline objects).
- Dependency injection (no imports from `main` that create cycles).
- PEP 8/257 compliant, typed, idempotent nodes.
- Structured logging; deterministic behavior; clear guards and fallbacks.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, TypedDict

import asyncio
import logging
import os

from langgraph.graph import StateGraph, node, State

import config as cfg  # safe to import after .env is loaded by the actual entrypoint


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

log = logging.getLogger("asketmc.rag.langgraph")


# ──────────────────────────────────────────────────────────────────────────────
# Types & Dependency Injection
# ──────────────────────────────────────────────────────────────────────────────


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
    # Retrieval / processing
    extract_lemmas: Callable[[str], Set[str]]
    retriever: Any  # must expose `aretrieve(q: str) -> Awaitable[List[Any]]`
    rerank: Callable[[str, List[Any]], Awaitable[List[Any]]]
    get_filtered_nodes: Callable[[List[Any], Set[str]], Awaitable[List[Any]]]
    build_context: Callable[[List[Any], Set[str], int], str]

    # LLM querying
    query_model: Callable[
        [Optional[List[Dict[str, str]]], Optional[str], Optional[str], Optional[str], int],
        Awaitable[Tuple[str, bool]],
    ]

    # Prompting / limits
    sys_prompt: str
    ctx_len_local: int
    ctx_len_remote: int


# ──────────────────────────────────────────────────────────────────────────────
# Optional LangSmith tracing
# ──────────────────────────────────────────────────────────────────────────────


def _maybe_init_langsmith() -> None:
    """Initialize LangSmith tracing if enabled via env (LANGSMITH_TRACING=1/true/yes)."""
    val = (os.getenv("LANGSMITH_TRACING") or "").strip().lower()
    if val in {"1", "true", "yes"}:
        try:
            import langsmith

            langsmith.init()
            log.info("LangSmith tracing enabled.")
        except Exception as exc:  # pragma: no cover
            log.warning("LangSmith tracing requested but failed to initialize: %s", exc)


# ──────────────────────────────────────────────────────────────────────────────
# Node factories (closed over dependencies)
# ──────────────────────────────────────────────────────────────────────────────


def _lemmas_node(deps: RagDeps):
    @node
    def _impl(state: RagState) -> RagState:
        """Extract normalized lemmas from the user question."""
        question = (state.get("question") or "").strip()
        if not question:
            raise ValueError("Empty question for lemmas_node.")
        state["lemmas"] = deps["extract_lemmas"](question)
        return state

    return _impl


def _retrieve_node(deps: RagDeps):
    @node
    async def _impl(state: RagState) -> RagState:
        """Retrieve candidate nodes with the configured retriever."""
        question = (state.get("question") or "").strip()
        if not question:
            raise ValueError("Empty question for retrieve_node.")
        retrieved = await deps["retriever"].aretrieve(question)
        state["retrieved_nodes"] = retrieved or []
        return state

    return _impl


def _rerank_node(deps: RagDeps):
    @node
    async def _impl(state: RagState) -> RagState:
        """Rerank retrieved nodes with the provided reranker."""
        retrieved = state.get("retrieved_nodes") or []
        question = (state.get("question") or "").strip()
        if not retrieved:
            state["reranked_nodes"] = []
            return state
        ranked = await deps["rerank"](question, retrieved)
        state["reranked_nodes"] = ranked or retrieved
        return state

    return _impl


def _filter_node(deps: RagDeps):
    @node
    async def _impl(state: RagState) -> RagState:
        """Filter/rerestrict nodes based on lemmas and domain rules."""
        lemmas = state.get("lemmas") or set()
        candidates = state.get("reranked_nodes") or state.get("retrieved_nodes") or []
        filtered = await deps["get_filtered_nodes"](candidates, lemmas)
        state["filtered_nodes"] = filtered or []
        return state

    return _impl


def _context_node(deps: RagDeps, *, use_remote_ctx: bool = False):
    @node
    def _impl(state: RagState) -> RagState:
        """Build bounded context string from filtered nodes."""
        lemmas = state.get("lemmas") or set()
        nodes = state.get("filtered_nodes") or []
        limit = deps["ctx_len_remote"] if use_remote_ctx else deps["ctx_len_local"]
        ctx_txt = deps["build_context"](nodes, lemmas, int(limit))
        state["context"] = ctx_txt
        return state

    return _impl


def _llm_node(deps: RagDeps):
    @node
    async def _impl(state: RagState) -> RagState:
        """Call LLM (OpenRouter with fallback) using system prompt + built context."""
        question = (state.get("question") or "").strip()
        context = state.get("context") or ""
        if not question:
            raise ValueError("Empty question for llm_node.")

        # Compose prompt deterministically
        prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER:"

        # Prefer remote call with messages; also pass structured args for fallback path.
        messages = [
            {"role": "system", "content": deps["sys_prompt"]},
            {"role": "user", "content": prompt},
        ]
        text, _used_fallback = await deps["query_model"](
            messages, deps["sys_prompt"], context, question, 240
        )
        state["answer"] = (text or "").strip() or "No answer."
        return state

    return _impl


# ──────────────────────────────────────────────────────────────────────────────
# Graph builder & public API
# ──────────────────────────────────────────────────────────────────────────────


def build_rag_graph(
    deps: RagDeps,
    *,
    use_remote_ctx: bool = False,
) -> StateGraph:
    """
    Build a LangGraph graph for the RAG pipeline (no side effects, not compiled).

    Args:
        deps: Injected dependencies (retriever, functions, prompts, limits).
        use_remote_ctx: If True, use remote context limit; else local.

    Returns:
        Uncompiled StateGraph to be compiled by caller.
    """
    graph = StateGraph(State)  # LangGraph accepts a base state type; we enforce keys via RagState
    graph.add_node("lemmas", _lemmas_node(deps))
    graph.add_node("retrieve", _retrieve_node(deps))
    graph.add_node("rerank", _rerank_node(deps))
    graph.add_node("filter", _filter_node(deps))
    graph.add_node("context", _context_node(deps, use_remote_ctx=use_remote_ctx))
    graph.add_node("llm", _llm_node(deps))

    graph.add_edge("lemmas", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "filter")
    graph.add_edge("filter", "context")
    graph.add_edge("context", "llm")

    graph.set_entry_point("lemmas")
    graph.set_exit_point("llm")
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

    Args:
        question: User question string.
        deps: Injected dependencies (`RagDeps`).
        use_remote_ctx: Whether to use remote context limit.
        enable_tracing: If True, attempt to init LangSmith (overrides env).

    Returns:
        Final answer string.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if enable_tracing:
        _maybe_init_langsmith()

    graph = build_rag_graph(deps, use_remote_ctx=use_remote_ctx)
    pipeline = graph.compile()

    # Build initial state deterministically
    init_state: RagState = {"question": question.strip()}

    output: RagState = await pipeline.invoke(init_state)
    answer = (output.get("answer") or "").strip()
    return answer or "No answer."


__all__ = [
    "RagState",
    "RagDeps",
    "build_rag_graph",
    "run_rag_pipeline",
]
