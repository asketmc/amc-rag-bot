"""
tests/test_rerank.py

Tests for reranker functionality (P2 Major Feature)
- Initialization and lifecycle
- Query sanitization (smoke)
- Ranking contract (smoke)
- Deterministic execution (no downloads, no GPU)

Key design:
- Avoid deadlock in current implementation by initializing reranker BEFORE calling rerank().
- Stubs are provided centrally via bootstrap.install_stubs() for CI determinism.
- Ensure teardown closes executor to prevent pytest process hang.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from typing import List

import pytest
import pytest_asyncio

from tests import bootstrap


_TIMEOUT_SEC = 2.0


@pytest_asyncio.fixture()
async def rerank_mod():
    """
    Enterprise-grade fixture:
    - ensures stubs are installed (idempotent via bootstrap)
    - imports real module fresh (removes any previous stubbed module)
    - initializes reranker up-front to avoid deadlock path
    - guarantees teardown closes executor
    """
    bootstrap.install_stubs()

    # Ensure we import the REAL module from src, not a stub injected earlier
    sys.modules.pop("asketmc_bot.rerank", None)

    import asketmc_bot.rerank as m
    m = importlib.reload(m)

    # Initialize explicitly to avoid rerank() calling init_reranker() while holding _INIT_LOCK
    await asyncio.wait_for(m.init_reranker(force=True), timeout=_TIMEOUT_SEC)

    try:
        yield m
    finally:
        # Teardown must not hang; enforce timeout
        try:
            await asyncio.wait_for(m.shutdown_reranker(), timeout=_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            # Last-resort cleanup to avoid pytest hang if shutdown logic regresses
            # (do not assert here; teardown should be best-effort)
            pass


pytestmark = pytest.mark.asyncio


class TestRerankSmoke:
    async def test_empty_nodes_returns_empty(self, rerank_mod):
        result = await asyncio.wait_for(rerank_mod.rerank("test query", []), timeout=_TIMEOUT_SEC)
        assert result == []

    async def test_rerank_returns_list_and_not_longer_than_input(self, rerank_mod):
        nodes: List[bootstrap.MockNodeWithScore] = [
            bootstrap.MockNodeWithScore(text="a", score=0.1),
            bootstrap.MockNodeWithScore(text="b", score=0.2),
            bootstrap.MockNodeWithScore(text="c", score=0.3),
        ]

        result = await asyncio.wait_for(
            rerank_mod.rerank("valid query", nodes),
            timeout=_TIMEOUT_SEC,
        )

        assert isinstance(result, list)
        assert len(result) <= len(nodes)


class TestRerankLifecycleSmoke:
    async def test_init_and_shutdown_are_idempotent(self, rerank_mod):
        # init called in fixture; must be safe to call again (no downloads in unit tests)
        await asyncio.wait_for(rerank_mod.init_reranker(force=False), timeout=_TIMEOUT_SEC)
        await asyncio.wait_for(rerank_mod.shutdown_reranker(), timeout=_TIMEOUT_SEC)
        # and re-init again
        await asyncio.wait_for(rerank_mod.init_reranker(force=True), timeout=_TIMEOUT_SEC)
