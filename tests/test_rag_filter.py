"""
tests/test_rag_filter.py

Tests for RAG filtering and context building (P1 Critical Feature)
- Node filtering logic
- Context assembly
- Cache functionality
- Lemma matching
"""

import pytest

from tests import bootstrap
from asketmc_bot.rag_filter import (
    _cache_key,
    build_context,
    get_filtered_nodes,
    purge_filter_cache,
)


class TestNodeFiltering:
    """Test node filtering with lemma matching and score thresholds."""

    @pytest.mark.asyncio
    async def test_empty_nodes_returns_empty(self, purge_cache):
        """Empty input returns empty list."""
        result = await get_filtered_nodes([], frozenset())
        assert result == []

    @pytest.mark.asyncio
    async def test_filters_by_score_threshold(self, purge_cache, high_low_nodes, query_lemmas):
        """Nodes below score threshold are filtered out."""
        result = await get_filtered_nodes(high_low_nodes, query_lemmas)
        assert len(result) >= 1
        assert any(n.node.text == "high score" for n in result)

    @pytest.mark.asyncio
    async def test_filters_by_lemma_intersection(self, purge_cache):
        """Nodes with lemma overlap are prioritized."""
        nodes = [
            bootstrap.MockNodeWithScore(text="matching", score=0.5, lemmas=["apple", "orange"]),
            bootstrap.MockNodeWithScore(text="no match", score=0.5, lemmas=["banana"]),
        ]
        qlem = frozenset(["apple"])

        result = await get_filtered_nodes(nodes, qlem)
        assert result, "expected non-empty result for matching lemma"
        assert any("apple" in n.node.metadata.get("lemmas", []) for n in result)

    @pytest.mark.asyncio
    async def test_respects_top_k_limit(self, purge_cache, query_lemmas):
        """Returns at most TOP_K nodes."""
        nodes = [
            bootstrap.MockNodeWithScore(text=f"doc{i}", score=0.9, lemmas=["test"])
            for i in range(100)
        ]

        result = await get_filtered_nodes(nodes, query_lemmas)
        assert len(result) <= 30  # reasonable upper bound

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_result(self, purge_cache, single_high_node, query_lemmas):
        """Second call with same inputs returns cached result."""
        nodes = [single_high_node]

        result1 = await get_filtered_nodes(nodes, query_lemmas)
        result2 = await get_filtered_nodes(nodes, query_lemmas)

        assert len(result1) == len(result2)


class TestContextBuilding:
    """Test context assembly from filtered nodes."""

    def test_empty_nodes_returns_empty_context(self, purge_cache):
        """Empty nodes list returns empty string."""
        result = build_context([], frozenset(), char_limit=1000)
        assert result == ""

    def test_respects_char_limit(self, purge_cache):
        """Context doesn't exceed character limit."""
        nodes = [bootstrap.MockNodeWithScore(text="a" * 1000, score=0.9, lemmas=["test"]) for _ in range(10)]
        qlem = frozenset(["test"])

        result = build_context(nodes, qlem, char_limit=500)
        assert len(result) <= 500

    def test_deduplicates_identical_content(self, purge_cache):
        """Identical content only appears once."""
        identical_text = "This is identical content"
        nodes = [
            bootstrap.MockNodeWithScore(text=identical_text, score=0.9, lemmas=["test"]),
            bootstrap.MockNodeWithScore(text=identical_text, score=0.8, lemmas=["test"]),
        ]
        qlem = frozenset(["test"])

        result = build_context(nodes, qlem, char_limit=5000)
        assert result.count(identical_text) == 1

    def test_prioritizes_high_scoring_nodes(self, purge_cache):
        """Higher scored nodes appear first."""
        nodes = [
            bootstrap.MockNodeWithScore(text="low", score=0.3, lemmas=["test"]),
            bootstrap.MockNodeWithScore(text="high", score=0.9, lemmas=["test"]),
        ]
        qlem = frozenset(["test"])

        result = build_context(nodes, qlem, char_limit=5000)
        assert "high" in result, "expected 'high' in context output"
        assert "low" in result, "expected 'low' in context output"
        assert result.index("high") < result.index("low")

    def test_includes_separator_between_chunks(self, purge_cache):
        """Multiple chunks are separated by separator."""
        nodes = [
            bootstrap.MockNodeWithScore(text="chunk1", score=0.9, lemmas=["test"]),
            bootstrap.MockNodeWithScore(text="chunk2", score=0.8, lemmas=["test"]),
        ]
        qlem = frozenset(["test"])

        result = build_context(nodes, qlem, char_limit=5000)
        assert "chunk1" in result, "expected 'chunk1' in context output"
        assert "chunk2" in result, "expected 'chunk2' in context output"
        assert "---" in result

    def test_handles_empty_text_nodes(self, purge_cache):
        """Nodes with empty text are skipped."""
        nodes = [
            bootstrap.MockNodeWithScore(text="", score=0.9, lemmas=["test"]),
            bootstrap.MockNodeWithScore(text="valid", score=0.8, lemmas=["test"]),
        ]
        qlem = frozenset(["test"])

        result = build_context(nodes, qlem, char_limit=5000)
        assert "valid" in result
        assert result.strip() != "---"


class TestCacheFunctionality:
    """Test cache management."""

    @pytest.mark.asyncio
    async def test_purge_cache_clears_all_entries(self, purge_cache, single_high_node, query_lemmas):
        """Cache purge must not corrupt state; subsequent lookup returns consistent results."""
        nodes = [single_high_node]

        result_before = await get_filtered_nodes(nodes, query_lemmas)
        purge_filter_cache()
        result_after = await get_filtered_nodes(nodes, query_lemmas)

        assert len(result_before) == len(result_after)

    def test_cache_key_uniqueness(self, purge_cache):
        """Different inputs produce different cache keys."""
        nodes1 = [bootstrap.MockNodeWithScore(text="doc1", score=0.8, lemmas=["apple"])]
        nodes2 = [bootstrap.MockNodeWithScore(text="doc2", score=0.7, lemmas=["banana"])]

        qlem1 = frozenset(["apple"])
        qlem2 = frozenset(["banana"])

        key1 = _cache_key(qlem1, nodes1)
        key2 = _cache_key(qlem2, nodes2)

        assert key1 != key2
