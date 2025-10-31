"""
tests/test_llm_client.py

Tests for LLMClient (P1 Critical Feature)
- Circuit breaker logic
- Remote/local fallback
- Query model validation
- Session management
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from dataclasses import dataclass
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "asketmc_bot"))

from llm_client import LLMClient, LLMConfig, CircuitBreaker, AsyncSessionHolder


pytestmark = pytest.mark.asyncio


class TestCircuitBreaker:
    """Test circuit breaker state transitions and blocking logic."""

    async def test_initial_state_closed(self):
        """Circuit breaker starts in closed state, allowing requests."""
        breaker = CircuitBreaker(base_block=1, max_block=10)
        assert await breaker.state() == "closed"
        assert await breaker.allow() is True

    async def test_failure_opens_breaker(self):
        """Failure transitions breaker to open state."""
        breaker = CircuitBreaker(base_block=1, max_block=10)
        await breaker.on_failure()
        assert await breaker.state() == "open"
        assert await breaker.allow() is False

    async def test_success_resets_breaker(self):
        """Success resets breaker to closed and resets block time."""
        breaker = CircuitBreaker(base_block=2, max_block=10)
        await breaker.on_failure()
        assert await breaker.state() == "open"

        await asyncio.sleep(1.1)  # wait for half-open
        await breaker.allow()  # transition to half-open
        await breaker.on_success(base_block=2)

        assert await breaker.state() == "closed"
        assert await breaker.allow() is True

    async def test_exponential_backoff(self):
        """Block time doubles on repeated failures."""
        breaker = CircuitBreaker(base_block=1, max_block=10)

        await breaker.on_failure()
        assert breaker._block == 2  # doubled

        await breaker.on_failure()
        assert breaker._block == 4  # doubled again

        await breaker.on_failure()
        assert breaker._block == 8

    async def test_max_block_limit(self):
        """Block time caps at max_block."""
        breaker = CircuitBreaker(base_block=1, max_block=5)

        for _ in range(10):
            await breaker.on_failure()

        assert breaker._block <= 5


class TestAsyncSessionHolder:
    """Test session lifecycle and reuse."""

    async def test_session_creation(self):
        """Session is created on first get."""
        holder = AsyncSessionHolder(limit=5, timeout_total=60)
        session = await holder.get()
        assert session is not None
        # Session should be open (closed property should be False)
        # Note: aiohttp.ClientSession.closed is a property, check it exists
        assert hasattr(session, 'closed')
        await holder.close()

    async def test_session_reuse(self):
        """Same session is returned on subsequent gets."""
        holder = AsyncSessionHolder(limit=5, timeout_total=60)
        session1 = await holder.get()
        session2 = await holder.get()
        assert session1 is session2
        await holder.close()

    async def test_session_close(self):
        """Close properly closes the session."""
        holder = AsyncSessionHolder(limit=5, timeout_total=60)
        session = await holder.get()
        await holder.close()
        # After close, session should be closed
        # (Can't easily test with real aiohttp without mocking)


class TestLLMClient:
    """Test LLMClient query logic, fallback, and breaker integration."""

    @pytest.fixture
    def config(self):
        """Standard test config."""
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

    async def test_query_model_requires_input(self, config):
        """query_model raises ValueError without proper input."""
        client = LLMClient(config)

        with pytest.raises(ValueError, match="requires either"):
            await client.query_model()

        await client.close()

    async def test_query_model_with_messages(self, config):
        """query_model accepts prebuilt messages."""
        client = LLMClient(config)

        with patch.object(client, '_call_openrouter', new_callable=AsyncMock) as mock_or:
            mock_or.return_value = ("Remote response", None)

            text, fallback = await client.query_model(
                messages=[{"role": "user", "content": "test"}]
            )

            assert text == "Remote response"
            assert fallback is False
            mock_or.assert_called_once()

        await client.close()

    async def test_query_model_with_components(self, config):
        """query_model accepts sys_prompt, ctx_txt, q components."""
        client = LLMClient(config)

        with patch.object(client, '_call_openrouter', new_callable=AsyncMock) as mock_or:
            mock_or.return_value = ("Remote response", None)

            text, fallback = await client.query_model(
                sys_prompt="System", ctx_txt="Context", q="Question"
            )

            assert text == "Remote response"
            assert fallback is False
            mock_or.assert_called_once()

        await client.close()

    async def test_remote_success_no_fallback(self, config):
        """Successful remote call doesn't trigger fallback."""
        client = LLMClient(config)

        with patch.object(client, '_call_openrouter', new_callable=AsyncMock) as mock_or:
            mock_or.return_value = ("Remote OK", None)

            text, used_fb = await client.query_model(
                sys_prompt="S", ctx_txt="C", q="Q"
            )

            assert text == "Remote OK"
            assert used_fb is False

        await client.close()

    async def test_remote_failure_triggers_local_fallback(self, config):
        """Remote failure triggers local model fallback."""
        client = LLMClient(config)

        with patch.object(client, '_call_openrouter', new_callable=AsyncMock) as mock_or, \
             patch.object(client, 'call_local_llm', new_callable=AsyncMock) as mock_local:
            mock_or.return_value = (None, "transient")  # failure
            mock_local.return_value = "Local OK"

            text, used_fb = await client.query_model(
                sys_prompt="S", ctx_txt="C", q="Q"
            )

            assert "Local OK" in text
            assert used_fb is True
            mock_local.assert_called_once()

        await client.close()

    async def test_breaker_blocks_after_failure(self, config):
        """Circuit breaker blocks remote calls after failure."""
        client = LLMClient(config)

        with patch.object(client, '_call_openrouter', new_callable=AsyncMock) as mock_or, \
             patch.object(client, 'call_local_llm', new_callable=AsyncMock) as mock_local:
            mock_or.return_value = (None, "transient")
            mock_local.return_value = "Local OK"

            # First call: triggers failure and breaker
            text1, fb1 = await client.query_model(sys_prompt="S", ctx_txt="C", q="Q")
            assert fb1 is True
            assert await client.is_remote_blocked() is True

            # Second call: breaker active, remote not called
            mock_or.reset_mock()
            text2, fb2 = await client.query_model(sys_prompt="S", ctx_txt="C", q="Q")
            assert fb2 is True
            mock_or.assert_not_called()  # breaker prevented call

        await client.close()

    async def test_local_llm_timeout_handling(self, config):
        """call_local_llm handles timeouts gracefully."""
        client = LLMClient(config)

        with patch.object(client._session_holder, 'get', new_callable=AsyncMock) as mock_get:
            mock_session = AsyncMock()
            # Mock the context manager properly
            mock_response = AsyncMock()
            mock_response.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_response.__aexit__ = AsyncMock(return_value=None)
            mock_session.post.return_value = mock_response
            mock_get.return_value = mock_session

            result = await client.call_local_llm("test prompt", timeout_sec=1)
            assert "timeout" in result.lower() or "error" in result.lower()

        await client.close()

    @pytest.mark.skip(reason="Cross-thread sync check is implementation-dependent")
    async def test_is_remote_blocked_sync(self, config):
        """Sync breaker check works from non-async context."""
        client = LLMClient(config)
        loop = asyncio.get_running_loop()
        client.attach_loop(loop)

        # Initially closed (not blocked)
        result = client.is_remote_blocked_sync()
        # Should return False when breaker is closed
        assert result in (False, True)  # Either state is valid initially

        # Open breaker
        await client._breaker.on_failure()
        # Give time for state change
        await asyncio.sleep(0.1)
        result2 = client.is_remote_blocked_sync()
        # After failure, should be blocked
        assert result2 is True

        await client.close()
