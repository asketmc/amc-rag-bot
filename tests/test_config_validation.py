"""
tests/test_config_validation.py

Tests for configuration loading and validation (P1 Critical Feature)
- Path resolution
- Required settings presence
- Type validation
- Default values
"""
import pytest
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "asketmc_bot"))

from asketmc_bot import config as cfg
class TestPathConfiguration:
    """Test that all critical paths are configured."""

    def test_project_root_exists(self):
        """PROJECT_ROOT points to valid directory."""
        assert cfg.PROJECT_ROOT.exists()
        assert cfg.PROJECT_ROOT.is_dir()

    def test_pkg_root_exists(self):
        """PKG_ROOT points to valid directory."""
        assert cfg.PKG_ROOT.exists()
        assert cfg.PKG_ROOT.is_dir()

    def test_var_root_created(self):
        """VAR_ROOT is created if it doesn't exist."""
        assert cfg.VAR_ROOT.is_dir()

    def test_cache_path_accessible(self):
        """CACHE_PATH directory exists or can be created."""
        assert cfg.CACHE_PATH.is_dir()

    def test_data_root_accessible(self):
        """DATA_ROOT is accessible."""
        assert cfg.DATA_ROOT.is_dir()

    def test_all_paths_absolute(self):
        """All path configurations are absolute."""
        paths = [
            cfg.PROJECT_ROOT,
            cfg.PKG_ROOT,
            cfg.VAR_ROOT,
            cfg.LOG_DIR,
            cfg.CACHE_PATH,
            cfg.DATA_ROOT,
            cfg.DOCS_PATH,
            cfg.PROMPTS_DIR,
        ]
        for p in paths:
            assert p.is_absolute(), f"{p} is not absolute"


class TestModelConfiguration:
    """Test model and RAG parameter configurations."""

    def test_top_k_is_positive_int(self):
        """TOP_K is a positive integer."""
        assert isinstance(cfg.TOP_K, int)
        assert cfg.TOP_K > 0

    def test_chunk_size_is_positive(self):
        """CHUNK_SIZE is positive."""
        assert isinstance(cfg.CHUNK_SIZE, int)
        assert cfg.CHUNK_SIZE > 0

    def test_chunk_overlap_less_than_chunk_size(self):
        """CHUNK_OVERLAP is less than CHUNK_SIZE."""
        assert cfg.CHUNK_OVERLAP < cfg.CHUNK_SIZE

    def test_context_limits_are_positive(self):
        """Context length limits are positive integers."""
        assert isinstance(cfg.CTX_LEN_REMOTE, int)
        assert isinstance(cfg.CTX_LEN_LOCAL, int)
        assert cfg.CTX_LEN_REMOTE > 0
        assert cfg.CTX_LEN_LOCAL > 0

    def test_reranker_config_valid(self):
        """Reranker configuration is valid."""
        assert isinstance(cfg.RERANKER_MODEL_NAME, str)
        assert len(cfg.RERANKER_MODEL_NAME) > 0
        assert isinstance(cfg.RERANK_INPUT_K, int)
        assert cfg.RERANK_INPUT_K > 0
        assert isinstance(cfg.RERANK_OUTPUT_K, int)
        assert cfg.RERANK_OUTPUT_K > 0
        assert cfg.RERANK_OUTPUT_K <= cfg.RERANK_INPUT_K

    def test_batch_size_positive(self):
        """BATCH_SIZE is positive."""
        assert isinstance(cfg.BATCH_SIZE, int)
        assert cfg.BATCH_SIZE > 0

    def test_max_len_positive(self):
        """MAX_LEN is positive."""
        assert isinstance(cfg.MAX_LEN, int)
        assert cfg.MAX_LEN > 0


class TestRuntimeConfiguration:
    """Test runtime limits and safety constraints."""

    def test_http_conn_limit_positive(self):
        """HTTP_CONN_LIMIT is positive."""
        assert isinstance(cfg.HTTP_CONN_LIMIT, int)
        assert cfg.HTTP_CONN_LIMIT > 0

    def test_http_timeout_positive(self):
        """HTTP_TIMEOUT_TOTAL is positive."""
        assert isinstance(cfg.HTTP_TIMEOUT_TOTAL, int)
        assert cfg.HTTP_TIMEOUT_TOTAL > 0

    def test_retries_non_negative(self):
        """OR_RETRIES is non-negative."""
        assert isinstance(cfg.OR_RETRIES, int)
        assert cfg.OR_RETRIES >= 0

    def test_breaker_block_times_valid(self):
        """Circuit breaker block times are valid."""
        assert isinstance(cfg.OPENROUTER_BLOCK_SEC, int)
        assert isinstance(cfg.OPENROUTER_BLOCK_MAX_SEC, int)
        assert cfg.OPENROUTER_BLOCK_SEC > 0
        assert cfg.OPENROUTER_BLOCK_MAX_SEC >= cfg.OPENROUTER_BLOCK_SEC

    def test_user_cooldown_positive(self):
        """USER_COOLDOWN is positive."""
        assert isinstance(cfg.USER_COOLDOWN, int)
        assert cfg.USER_COOLDOWN > 0

    def test_max_question_len_positive(self):
        """MAX_QUESTION_LEN is positive."""
        assert isinstance(cfg.MAX_QUESTION_LEN, int)
        assert cfg.MAX_QUESTION_LEN > 0

    def test_allowed_channels_is_set(self):
        """ALLOWED_CHANNELS is a set of ints."""
        assert isinstance(cfg.ALLOWED_CHANNELS, set)
        for channel_id in cfg.ALLOWED_CHANNELS:
            assert isinstance(channel_id, int)

    def test_admin_ids_is_set(self):
        """ADMIN_IDS is a set of ints."""
        assert isinstance(cfg.ADMIN_IDS, set)
        for admin_id in cfg.ADMIN_IDS:
            assert isinstance(admin_id, int)

    def test_allowed_chars_regex_compiles(self):
        """ALLOWED_CHARS regex is valid."""
        import re
        assert isinstance(cfg.ALLOWED_CHARS, re.Pattern)
        # Test it matches valid input
        assert cfg.ALLOWED_CHARS.match("test query 123")

    def test_request_semaphore_valid(self):
        """REQUEST_SEMAPHORE is properly configured."""
        import asyncio
        assert isinstance(cfg.REQUEST_SEMAPHORE, asyncio.Semaphore)


class TestLemmaConfiguration:
    """Test lemmatization and filtering configuration."""

    def test_good_pos_is_set(self):
        """GOOD_POS is a non-empty set."""
        assert isinstance(cfg.GOOD_POS, set)
        assert len(cfg.GOOD_POS) > 0
        for pos in cfg.GOOD_POS:
            assert isinstance(pos, str)

    def test_stop_words_is_set(self):
        """STOP_WORDS is a non-empty set."""
        assert isinstance(cfg.STOP_WORDS, set)
        assert len(cfg.STOP_WORDS) > 0
        for word in cfg.STOP_WORDS:
            assert isinstance(word, str)

    def test_lemma_match_ratio_valid(self):
        """LEMMA_MATCH_RATIO is between 0 and 1."""
        assert isinstance(cfg.LEMMA_MATCH_RATIO, float)
        assert 0.0 <= cfg.LEMMA_MATCH_RATIO <= 1.0

    def test_score_threshold_valid(self):
        """SCORE_RELATIVE_THRESHOLD is between 0 and 1."""
        assert isinstance(cfg.SCORE_RELATIVE_THRESHOLD, float)
        assert 0.0 <= cfg.SCORE_RELATIVE_THRESHOLD <= 1.0


class TestConfigGetConf:
    """Test the get_conf helper function."""

    def test_get_existing_value(self):
        """get_conf returns existing config value."""
        result = cfg.get_conf("TOP_K", 999)
        assert result == cfg.TOP_K
        assert result != 999

    def test_get_nonexistent_returns_default(self):
        """get_conf returns default for non-existent key."""
        result = cfg.get_conf("NONEXISTENT_KEY_XYZ", "default_value")
        assert result == "default_value"

    def test_get_with_type_conversion(self):
        """get_conf converts type when specified."""
        result = cfg.get_conf("TOP_K", "0", typ=str)
        assert isinstance(result, str)
