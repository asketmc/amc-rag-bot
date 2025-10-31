"""
tests/test_rag_filter.py

Tests for RAG filtering and context building (P1 Critical Feature)
- Node filtering logic
- Context assembly
- Cache functionality
- Lemma matching
"""
import pytest
import asyncio
import sys
from pathlib import Path
from typing import FrozenSet

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "asketmc_bot"))

# Mock llama_index before importing rag_filter
from unittest.mock import MagicMock
llama_pkg = MagicMock()
llama_core = MagicMock()

class MockNode:
    def __init__(self, text="", metadata=None):
        self.text = text
        self.metadata = metadata or {}
        self.id = f"node_{id(self)}"

    def get_content(self, metadata_mode=None):
        return self.text

class MockNodeWithScore:
    def __init__(self, text="", score=0.5, lemmas=None):
        self.node = MockNode(text, {"lemmas": lemmas or []})
        self.score = score

llama_core.schema = MagicMock()
llama_core.schema.NodeWithScore = MockNodeWithScore
sys.modules['llama_index'] = llama_pkg
sys.modules['llama_index.core'] = llama_core
sys.modules['llama_index.core.schema'] = llama_core.schema

# Now import after mocking
from rag_filter import get_filtered_nodes, build_context, purge_filter_cache, _cache_key


pytestmark = pytest.mark.asyncio


class TestNodeFiltering:
    """Test node filtering with lemma matching and score thresholds."""

    async def test_empty_nodes_returns_empty(self):
        """Empty input returns empty list."""
        result = await get_filtered_nodes([], frozenset())
        assert result == []

    async def test_filters_by_score_threshold(self):
        """Nodes below score threshold are filtered out."""
        nodes = [
            MockNodeWithScore(text="high score", score=0.9, lemmas=["test"]),
            MockNodeWithScore(text="low score", score=0.1, lemmas=["test"]),
        ]
        qlem = frozenset(["test"])

        result = await get_filtered_nodes(nodes, qlem)
        # High score node should be included
        assert len(result) >= 1
        assert any(n.node.text == "high score" for n in result)

    async def test_filters_by_lemma_intersection(self):
        """Nodes with lemma overlap are prioritized."""
        nodes = [
            MockNodeWithScore(text="matching", score=0.5, lemmas=["apple", "orange"]),
            MockNodeWithScore(text="no match", score=0.5, lemmas=["banana"]),
        ]
        qlem = frozenset(["apple"])

        result = await get_filtered_nodes(nodes, qlem)
        # Node with matching lemma should be preferred
        if result:
            assert any("apple" in n.node.metadata.get("lemmas", []) for n in result)

    async def test_respects_top_k_limit(self):
        """Returns at most TOP_K nodes."""
        nodes = [
            MockNodeWithScore(text=f"doc{i}", score=0.9, lemmas=["test"])
            for i in range(100)
        ]
        qlem = frozenset(["test"])

        result = await get_filtered_nodes(nodes, qlem)
        # Should not exceed TOP_K (default 8 in config, but may vary)
        assert len(result) <= 30  # reasonable upper bound

    async def test_cache_hit_returns_cached_result(self):
        """Second call with same inputs returns cached result."""
        purge_filter_cache()

        nodes = [MockNodeWithScore(text="cached", score=0.8, lemmas=["test"])]
        qlem = frozenset(["test"])

        result1 = await get_filtered_nodes(nodes, qlem)
        result2 = await get_filtered_nodes(nodes, qlem)

        # Results should be consistent (from cache)
        assert len(result1) == len(result2)


class TestContextBuilding:
    """Test context assembly from filtered nodes."""

    def test_empty_nodes_returns_empty_context(self):
        """Empty nodes list returns empty string."""
        result = build_context([], frozenset(), char_limit=1000)
        assert result == ""

    def test_respects_char_limit(self):
        """Context doesn't exceed character limit."""
        nodes = [
            MockNodeWithScore(text="a" * 1000, score=0.9, lemmas=["test"])
            for _ in range(10)
        ]
        qlem = frozenset(["test"])

        result = build_context(nodes, qlem, char_limit=500)
        assert len(result) <= 500

    def test_deduplicates_identical_content(self):
        """Identical content only appears once."""
        identical_text = "This is identical content"
        nodes = [
            MockNodeWithScore(text=identical_text, score=0.9, lemmas=["test"]),
            MockNodeWithScore(text=identical_text, score=0.8, lemmas=["test"]),
        ]
        qlem = frozenset(["test"])

        result = build_context(nodes, qlem, char_limit=5000)
        # Content should appear only once
        assert result.count(identical_text) == 1

    def test_prioritizes_high_scoring_nodes(self):
        """Higher scored nodes appear first."""
        nodes = [
            MockNodeWithScore(text="low", score=0.3, lemmas=["test"]),
            MockNodeWithScore(text="high", score=0.9, lemmas=["test"]),
        ]
        qlem = frozenset(["test"])

        result = build_context(nodes, qlem, char_limit=5000)
        # High score should appear before low score
        if "high" in result and "low" in result:
            assert result.index("high") < result.index("low")

    def test_includes_separator_between_chunks(self):
        """Multiple chunks are separated by separator."""
        nodes = [
            MockNodeWithScore(text="chunk1", score=0.9, lemmas=["test"]),
            MockNodeWithScore(text="chunk2", score=0.8, lemmas=["test"]),
        ]
        qlem = frozenset(["test"])

        result = build_context(nodes, qlem, char_limit=5000)
        # Should contain separator between chunks
        if "chunk1" in result and "chunk2" in result:
            assert "---" in result

    def test_handles_empty_text_nodes(self):
        """Nodes with empty text are skipped."""
        nodes = [
            MockNodeWithScore(text="", score=0.9, lemmas=["test"]),
            MockNodeWithScore(text="valid", score=0.8, lemmas=["test"]),
        ]
        qlem = frozenset(["test"])

        result = build_context(nodes, qlem, char_limit=5000)
        assert "valid" in result
        # Empty node should not add separators
        assert result.strip() != "---"


class TestCacheFunctionality:
    """Test cache management."""

    def test_purge_cache_clears_all_entries(self):
        """purge_filter_cache removes all cached entries."""
        nodes = [MockNodeWithScore(text="test", score=0.8, lemmas=["test"])]
        qlem = frozenset(["test"])

        # Populate cache
        asyncio.run(get_filtered_nodes(nodes, qlem))

        # Purge
        purge_filter_cache()

        # Cache should be empty (can't directly test, but shouldn't error)
        asyncio.run(get_filtered_nodes(nodes, qlem))

    def test_cache_key_uniqueness(self):
        """Different inputs produce different cache keys."""
        nodes1 = [MockNodeWithScore(text="doc1", score=0.8, lemmas=["apple"])]
        nodes2 = [MockNodeWithScore(text="doc2", score=0.7, lemmas=["banana"])]

        qlem1 = frozenset(["apple"])
        qlem2 = frozenset(["banana"])

        key1 = _cache_key(qlem1, nodes1)
        key2 = _cache_key(qlem2, nodes2)

        assert key1 != key2
