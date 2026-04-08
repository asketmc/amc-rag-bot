# FILE: tests/conftest.py
from __future__ import annotations

from collections.abc import Generator

import pytest

from tests import bootstrap


@pytest.fixture(scope="session", autouse=True)
def bootstrap_env() -> Generator[None, None, None]:
    """Configure sys.path and dependency stubs for the test session."""
    bootstrap.configure()
    yield
    # teardown placeholder: cleanup session-level resources here if needed


# ---------------------------------------------------------------------------
# rag_filter helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def purge_cache():
    """Reset rag_filter cache around a test to avoid state leakage."""
    from asketmc_bot.rag_filter import purge_filter_cache

    purge_filter_cache()
    yield
    purge_filter_cache()


@pytest.fixture
def query_lemmas():
    """Minimal lemma set sufficient to satisfy lemma-match logic in filter tests."""
    return frozenset(["test"])


@pytest.fixture
def single_high_node():
    """Single above-threshold node for positive-path and cache-hit scenarios."""
    return bootstrap.MockNodeWithScore(text="cached", score=0.8, lemmas=["test"])


@pytest.fixture
def high_low_nodes():
    """Two nodes straddling the score threshold to exercise filter boundary logic."""
    return [
        bootstrap.MockNodeWithScore(text="high score", score=0.9, lemmas=["test"]),
        bootstrap.MockNodeWithScore(text="low score", score=0.1, lemmas=["test"]),
    ]


# ---------------------------------------------------------------------------
# LLM client helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def llm_config():
    """Shared LLMConfig for LLM-client unit tests."""
    from asketmc_bot.llm_client import LLMConfig

    return LLMConfig(
        api_url="https://test.example/api",
        or_model="test/model",
        or_max_tokens=128,
        openrouter_api_key="test_key",
        ollama_url="http://localhost:11434/api/generate",
        local_model="test:local",
        http_conn_limit=2,
        or_retries=2,
        http_timeout_total=10,
        breaker_base_block_sec=1,
        breaker_max_block_sec=5,
    )

