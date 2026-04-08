"""
tests/test_query_model_unit.py

Unit tests for query_model() decision logic.
The production implementation lives in asketmc_bot.llm_client.LLMClient.query_model.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from asketmc_bot.llm_client import LLMClient, LLMConfig


@pytest.mark.asyncio
async def test_validation_error_requires_messages_or_components(llm_config):
    client = LLMClient(llm_config)

    with pytest.raises(ValueError, match="requires either"):
        await client.query_model(messages=None, sys_prompt=None, ctx_txt=None, q=None)

    await client.close()


@pytest.mark.asyncio
async def test_remote_success_returns_text_without_fallback(llm_config):
    client = LLMClient(llm_config)

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
async def test_remote_failure_triggers_fallback_and_blocks_subsequent_remote(llm_config):
    client = LLMClient(llm_config)

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
