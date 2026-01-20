"""
tests/test_query_model_unit.py

Unit tests for query_model() decision logic.
The production implementation lives in asketmc_bot.llm_client.LLMClient.query_model.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys as _sys

if str(SRC) not in _sys.path:
    _sys.path.insert(0, str(SRC))

from asketmc_bot.llm_client import LLMClient, LLMConfig  # noqa: E402


@pytest.fixture
def config() -> LLMConfig:
    return LLMConfig(
        api_url="https://test.example/api",
        or_model="test/model",
        or_max_tokens=64,
        openrouter_api_key="test_key",
        ollama_url="http://localhost:11434/api/generate",
        local_model="test:local",
        http_conn_limit=1,
        or_retries=1,
        http_timeout_total=5,
        breaker_base_block_sec=1,
        breaker_max_block_sec=4,
    )


@pytest.mark.asyncio
async def test_validation_error_requires_messages_or_components(config):
    client = LLMClient(config)

    with pytest.raises(ValueError, match="requires either"):
        await client.query_model(messages=None, sys_prompt=None, ctx_txt=None, q=None)

    await client.close()


@pytest.mark.asyncio
async def test_remote_success_returns_text_without_fallback(config):
    client = LLMClient(config)

    with (
        patch.object(client, "_call_openrouter", new_callable=AsyncMock) as mock_or,
        patch.object(client, "call_local_llm", new_callable=AsyncMock) as mock_local,
    ):
        mock_or.return_value = ("REMOTE_OK", None)

        txt, used_fb = await client.query_model(sys_prompt="S", ctx_txt="C", q="Q")

        assert txt == "REMOTE_OK"
        assert used_fb is False
        mock_local.assert_not_awaited()
        mock_or.assert_awaited_once()

    await client.close()


@pytest.mark.asyncio
async def test_remote_failure_triggers_fallback_and_blocks_subsequent_remote(config):
    client = LLMClient(config)

    with (
        patch.object(client, "_call_openrouter", new_callable=AsyncMock) as mock_or,
        patch.object(client, "call_local_llm", new_callable=AsyncMock) as mock_local,
    ):
        mock_or.return_value = (None, "transient")
        mock_local.return_value = "LOCAL_OK"

        txt1, fb1 = await client.query_model(sys_prompt="S", ctx_txt="C", q="Q")
        assert txt1 == "LOCAL_OK"
        assert fb1 is True
        assert await client.is_remote_blocked() is True
        mock_or.assert_awaited_once()
        mock_local.assert_awaited_once()

        mock_or.reset_mock()
        mock_local.reset_mock()

        txt2, fb2 = await client.query_model(sys_prompt="S", ctx_txt="C", q="Q")
        assert txt2 == "LOCAL_OK"
        assert fb2 is True
        mock_or.assert_not_awaited()
        mock_local.assert_awaited_once()

    await client.close()
