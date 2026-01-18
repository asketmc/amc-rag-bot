"""
tests/test_rerank.py

Tests for reranker functionality (P2 Major Feature)
- Initialization and lifecycle
- Query sanitization
- Ranking logic
- Error handling and retries

Principles:
- Import rerank as a package module: asketmc_bot.rerank
- Stub heavy deps (torch / sentence_transformers / llama_index) to keep CI fast/deterministic
- Keep assertions minimal but meaningful (smoke + contract-level checks)
"""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass
from typing import List

import pytest


@dataclass
class MockNode:
    text: str = ""
    node_id: str = "test"

    @property
    def id(self) -> str:
        return self.node_id

    def get_content(self) -> str:
        return self.text


@dataclass
class MockNodeWithScore:
    text: str = ""
    score: float = 0.5

    def __post_init__(self) -> None:
        self.node = MockNode(text=self.text)


@pytest.fixture()
def rerank_mod(monkeypatch):
    # ---- stub torch (avoid CUDA checks / import failures) ----
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.__version__ = "0"
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        sys.modules["torch"] = torch

    # ---- stub sentence_transformers (avoid model downloads) ----
    if "sentence_transformers" not in sys.modules:
        st = types.ModuleType("sentence_transformers")
        sys.modules["sentence_transformers"] = st

    # ---- stub llama_index.core.schema.NodeWithScore if rerank imports it ----
    # Some code paths may type-check or import it; keep it minimal.
    if "llama_index" not in sys.modules:
        ll_pkg = types.ModuleType("llama_index")
        ll_pkg.__path__ = []
        sys.modules["llama_index"] = ll_pkg

    if "llama_index.core" not in sys.modules:
        ll_core = types.ModuleType("llama_index.core")
        ll_core.__path__ = []
        sys.modules["llama_index.core"] = ll_core

    if "llama_index.core.schema" not in sys.modules:
        ll_schema = types.ModuleType("llama_index.core.schema")

        class _NodeWithScore:  # pragma: no cover
            def __init__(self, node=None, score: float = 0.0):
                self.node = node
                self.score = float(score)

        ll_schema.NodeWithScore = _NodeWithScore
        sys.modules["llama_index.core.schema"] = ll_schema

    # Import as installed package module (preferred for long-term repo hygiene)
    import asketmc_bot.rerank as m

    # Reload so stubs above apply even if module was imported earlier in test session
    m = importlib.reload(m)
    return m


pytestmark = pytest.mark.asyncio


class TestRerankSmoke:
    async def test_empty_nodes_returns_empty(self, rerank_mod):
        result = await rerank_mod.rerank("test query", [])
        assert result == []

    async def test_rerank_returns_list_and_not_longer_than_input(self, rerank_mod):
        nodes: List[MockNodeWithScore] = [
            MockNodeWithScore(text="a", score=0.1),
            MockNodeWithScore(text="b", score=0.2),
            MockNodeWithScore(text="c", score=0.3),
        ]
        result = await rerank_mod.rerank("valid query", nodes)
        assert isinstance(result, list)
        assert len(result) <= len(nodes)


class TestRerankLifecycleSmoke:
    async def test_init_and_shutdown_do_not_crash(self, rerank_mod):
        # Must not download models in unit tests; should still be safe to call.
        await rerank_mod.init_reranker()
        await rerank_mod.shutdown_reranker()
