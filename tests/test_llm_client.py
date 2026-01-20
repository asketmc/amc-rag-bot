"""
tests/test_llm_client.py

Tests for LLMClient (P1 Critical Feature)
- Circuit breaker logic
- Remote/local fallback
- Query model validation
- Session management
- Cross-thread sync breaker check
"""
from __future__ import annotations

import asyncio
import queue
import threading
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in asyncio.sys.path:  # type: ignore[attr-defined]
    # Avoid importing sys at module scope into a test that also uses asyncio.sys in type-checkers.
    import sys as _sys

    if str(SRC) not in _sys.path:
        _sys.path.insert(0, str(SRC))

import asketmc_bot.llm_client as llm_mod  # noqa: E402
from asketmc_bot.llm_client import (  # noqa: E402
    AsyncSessionHolder,
    CircuitBreaker,
    LLMClient,
    LLMConfig,
)


@pytest.fixture
def config() -> LLMConfig:
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


class _FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.t = float(start)

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_initial_state_closed(self):
        breaker = CircuitBreaker(base_block=1, max_block=10)
        assert await breaker.state() == "closed"
        assert await breaker.allow() is True

    @pytest.mark.asyncio
    async def test_failure_opens_breaker_and_blocks_until_timeout(self, monkeypatch):
        clock = _FakeClock()
        monkeypatch.setattr(llm_mod.time, "time", clock.now)

        breaker = CircuitBreaker(base_block=1, max_block=10)

        await breaker.on_failure()
        assert await breaker.state() == "open"
        assert await breaker.allow() is False

        clock.advance(1.01)
        assert await breaker.allow() is True
        assert await breaker.state() == "half_open"

    @pytest.mark.asyncio
    async def test_success_resets_breaker(self, monkeypatch):
        clock = _FakeClock()
        monkeypatch.setattr(llm_mod.time, "time", clock.now)

        breaker = CircuitBreaker(base_block=2, max_block=10)

        await breaker.on_failure()
        assert await breaker.state() == "open"

        clock.advance(2.01)
        assert await breaker.allow() is True
        assert await breaker.state() == "half_open"

        await breaker.on_success(base_block=2)
        assert await breaker.state() == "closed"
        assert await breaker.allow() is True

    @pytest.mark.asyncio
    async def test_exponential_backoff_respects_max(self, monkeypatch):
        clock = _FakeClock()
        monkeypatch.setattr(llm_mod.time, "time", clock.now)

        breaker = CircuitBreaker(base_block=1, max_block=5)

        await breaker.on_failure()
        assert breaker._block == 2

        await breaker.on_failure()
        assert breaker._block == 4

        await breaker.on_failure()
        assert breaker._block == 5

        await breaker.on_failure()
        assert breaker._block == 5


class TestAsyncSessionHolder:
    @pytest.mark.asyncio
    async def test_session_creation_and_reuse(self):
        holder = AsyncSessionHolder(limit=5, timeout_total=60)

        mock_session = AsyncMock()
        mock_session.closed = False
        mock_session.close = AsyncMock(return_value=None)

        with (
            patch("asketmc_bot.llm_client.ClientSession", return_value=mock_session) as cs,
            patch("asketmc_bot.llm_client.TCPConnector") as tc,
            patch("asketmc_bot.llm_client.ClientTimeout") as ct,
        ):
            s1 = await holder.get()
            s2 = await holder.get()

            assert s1 is mock_session
            assert s2 is mock_session
            cs.assert_called_once()
            tc.assert_called_once()
            ct.assert_called_once()

        await holder.close()
        mock_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_session_recreated_if_closed(self):
        holder = AsyncSessionHolder(limit=5, timeout_total=60)

        s1 = AsyncMock()
        s1.closed = False
        s1.close = AsyncMock(return_value=None)

        s2 = AsyncMock()
        s2.closed = False
        s2.close = AsyncMock(return_value=None)

        with (
            patch("asketmc_bot.llm_client.ClientSession", side_effect=[s1, s2]) as cs,
            patch("asketmc_bot.llm_client.TCPConnector"),
            patch("asketmc_bot.llm_client.ClientTimeout"),
        ):
            a = await holder.get()
            assert a is s1

            s1.closed = True
            b = await holder.get()
            assert b is s2

            assert cs.call_count == 2

        await holder.close()


class TestLLMClient:
    @pytest.mark.asyncio
    async def test_query_model_requires_input(self, config):
        client = LLMClient(config)
        with pytest.raises(ValueError, match="requires either"):
            await client.query_model()
        await client.close()

    @pytest.mark.asyncio
    async def test_remote_success_no_fallback(self, config):
        client = LLMClient(config)

        with (
            patch.object(client, "_call_openrouter", new_callable=AsyncMock) as mock_or,
            patch.object(client, "call_local_llm", new_callable=AsyncMock) as mock_local,
        ):
            mock_or.return_value = ("Remote OK", None)

            text, used_fb = await client.query_model(sys_prompt="S", ctx_txt="C", q="Q")

            assert text == "Remote OK"
            assert used_fb is False
            mock_local.assert_not_awaited()

        await client.close()

    @pytest.mark.asyncio
    async def test_remote_failure_triggers_local_fallback_and_breaker(self, config):
        client = LLMClient(config)

        with (
            patch.object(client, "_call_openrouter", new_callable=AsyncMock) as mock_or,
            patch.object(client, "call_local_llm", new_callable=AsyncMock) as mock_local,
        ):
            mock_or.return_value = (None, "transient")
            mock_local.return_value = "Local OK"

            text, used_fb = await client.query_model(sys_prompt="S", ctx_txt="C", q="Q")

            assert text == "Local OK"
            assert used_fb is True
            assert await client.is_remote_blocked() is True

            mock_or.reset_mock()
            mock_local.reset_mock()

            text2, used_fb2 = await client.query_model(sys_prompt="S", ctx_txt="C", q="Q")

            assert text2 == "Local OK"
            assert used_fb2 is True
            mock_or.assert_not_awaited()
            mock_local.assert_awaited()

        await client.close()

    @pytest.mark.asyncio
    async def test_local_llm_timeout_handling(self, config):
        client = LLMClient(config)

        class _TimeoutCM:
            async def __aenter__(self):
                raise asyncio.TimeoutError()

            async def __aexit__(self, exc_type, exc, tb):
                return False

        with patch.object(client._session_holder, "get", new_callable=AsyncMock) as mock_get:
            mock_session = Mock()
            mock_session.post = Mock(return_value=_TimeoutCM())
            mock_get.return_value = mock_session

            result = await client.call_local_llm("test prompt", timeout_sec=1)
            assert "timeout" in result.lower()

        await client.close()

    def test_is_remote_blocked_sync_cross_thread(self, config):
        """
        Deterministic cross-thread test:
        - Run an event loop in a background thread
        - Create client inside that loop thread
        - Call is_remote_blocked_sync() from the main thread
        """
        q: queue.Queue[tuple[LLMClient, asyncio.AbstractEventLoop]] = queue.Queue()

        def _loop_thread() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            client = LLMClient(config)
            client.attach_loop(loop)

            q.put((client, loop))
            loop.run_forever()

            loop.close()

        t = threading.Thread(target=_loop_thread, daemon=True)
        t.start()

        client, loop = q.get(timeout=2.0)

        try:
            assert client.is_remote_blocked_sync() is False

            asyncio.run_coroutine_threadsafe(client._breaker.on_failure(), loop).result(timeout=2.0)
            assert client.is_remote_blocked_sync() is True

            asyncio.run_coroutine_threadsafe(client.close(), loop).result(timeout=2.0)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=2.0)
