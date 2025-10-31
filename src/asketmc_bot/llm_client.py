# src/asketmc_bot/llm_client.py
#!/usr/bin/env python3.10
"""LLM client for OpenRouter (remote) and Ollama (local) with a circuit breaker.

- Manages a shared aiohttp session.
- Retries remote calls with backoff and JSON hardening.
- Falls back to a local model when the remote path fails or is blocked.
- Exposes breaker state for fast, synchronous checks from other threads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from aiohttp import ClientSession, TCPConnector, ClientTimeout


# ── Config DTO ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LLMConfig:
    """Configuration required by the LLM client."""

    api_url: str
    or_model: str
    or_max_tokens: int
    openrouter_api_key: str
    ollama_url: str
    local_model: str
    http_conn_limit: int
    or_retries: int
    http_timeout_total: int
    breaker_base_block_sec: int
    breaker_max_block_sec: int


# ── Async HTTP session holder ─────────────────────────────────────────────────
class AsyncSessionHolder:
    """Create and reuse a single aiohttp ClientSession with a connection limit."""

    def __init__(self, *, limit: int, timeout_total: int) -> None:
        self._lock = asyncio.Lock()
        self._session: Optional[ClientSession] = None
        self._limit = int(limit)
        self._timeout_total = int(timeout_total)

    async def get(self) -> ClientSession:
        """Return an open session, creating it if needed."""
        async with self._lock:
            if self._session is None or self._session.closed:
                self._session = ClientSession(
                    connector=TCPConnector(limit=self._limit),
                    timeout=ClientTimeout(total=self._timeout_total),
                )
            return self._session

    async def close(self) -> None:
        """Close the session if open."""
        async with self._lock:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None


# ── Circuit Breaker ───────────────────────────────────────────────────────────
class CircuitBreaker:
    """Closed/Open/Half-open breaker with exponential backoff."""

    def __init__(self, base_block: int, max_block: int) -> None:
        self._state = "closed"
        self._open_until = 0.0
        self._block = max(1, int(base_block))
        self._max_block = max(self._block, int(max_block))
        self._lock = asyncio.Lock()

    async def allow(self) -> bool:
        """Return True if a remote call is allowed now."""
        async with self._lock:
            now = time.time()
            if self._state == "open":
                if now >= self._open_until:
                    self._state = "half_open"
                    return True
                return False
            return True

    async def on_success(self, *, base_block: int) -> None:
        """Reset breaker after a successful remote call."""
        async with self._lock:
            self._state = "closed"
            self._block = max(1, int(base_block))

    async def on_failure(self) -> None:
        """Open breaker and schedule the next allow after the current block."""
        async with self._lock:
            self._state = "open"
            self._open_until = time.time() + self._block
            self._block = min(self._block * 2, self._max_block)

    async def state(self) -> str:
        """Return breaker state: closed | half_open | open."""
        async with self._lock:
            return self._state


# ── LLM Client ────────────────────────────────────────────────────────────────
class LLMClient:
    """High-level client that wraps remote (OpenRouter) and local (Ollama) calls."""

    def __init__(self, config: LLMConfig, logger: Optional[logging.Logger] = None) -> None:
        self._cfg = config
        self._log = logger or logging.getLogger("llm")
        self._session_holder = AsyncSessionHolder(
            limit=config.http_conn_limit, timeout_total=config.http_timeout_total
        )
        self._breaker = CircuitBreaker(
            base_block=config.breaker_base_block_sec, max_block=config.breaker_max_block_sec
        )
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ---- Loop attachment for cross-thread sync helpers -----------------------
    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Attach the running loop to enable sync helpers from other threads."""
        self._loop = loop

    # ---- Public helpers -------------------------------------------------------
    async def is_remote_blocked(self) -> bool:
        """Async check if remote path is currently blocked by the breaker."""
        return (await self._breaker.state()) == "open"

    def is_remote_blocked_sync(self) -> bool:
        """Thread-safe check usable from non-async contexts."""
        if self._loop is None:
            return False
        fut = asyncio.run_coroutine_threadsafe(self._breaker.state(), self._loop)
        try:
            return fut.result(timeout=0.5) == "open"
        except Exception:
            return False

    async def close(self) -> None:
        """Close underlying resources."""
        await self._session_holder.close()

    # ---- Core API -------------------------------------------------------------
    async def query_model(
        self,
        messages: Optional[List[dict]] = None,
        sys_prompt: Optional[str] = None,
        ctx_txt: Optional[str] = None,
        q: Optional[str] = None,
        timeout_sec: int = 240,
    ) -> Tuple[str, bool]:
        """Query remote model with local fallback. Returns (text, used_fallback)."""
        if messages is None:
            if not (sys_prompt and ctx_txt and q):
                raise ValueError(
                    "query_model requires either `messages` or (`sys_prompt`, `ctx_txt`, `q`)."
                )
            messages = [
                {"role": "system", "content": sys_prompt.strip()},
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{ctx_txt.strip()}\n\nQUESTION: {q.strip()}\nANSWER:",
                },
            ]

        used_fallback = False

        if await self._breaker.allow():
            text, category = await self._call_openrouter(messages)
            if text is not None:
                await self._breaker.on_success(base_block=self._cfg.breaker_base_block_sec)
                return text, used_fallback
            await self._breaker.on_failure()
            used_fallback = True
        else:
            used_fallback = True

        # Fallback to local model
        if sys_prompt is None or ctx_txt is None or q is None:
            sys_prompt = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
            user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            prompt_text = f"{sys_prompt or ''}\n\n{user or ''}"
        else:
            prompt_text = (
                f"{sys_prompt.strip()}\n\nCONTEXT:\n{ctx_txt.strip()}\n\nQUESTION: {q.strip()}\nANSWER:"
            )
        text = await self.call_local_llm(prompt_text, timeout_sec=timeout_sec)
        return text, used_fallback

    async def call_local_llm(self, prompt_text: str, timeout_sec: Optional[int] = None) -> str:
        """Call local Ollama model with a raw prompt."""
        session = await self._session_holder.get()
        try:
            async with session.post(
                self._cfg.ollama_url,
                json={"model": self._cfg.local_model, "prompt": prompt_text, "stream": False},
                timeout=ClientTimeout(total=timeout_sec or self._cfg.http_timeout_total),
            ) as resp:
                raw = await resp.text()
                try:
                    data = json.loads(raw)
                except Exception:
                    return "⚠️ Local LLM returned non-JSON response."
                return (data or {}).get("response", "❌ No response.")
        except asyncio.TimeoutError:
            return "⚠️ Local LLM did not respond (timeout)."
        except Exception as exc:
            return f"⚠️ Local LLM error: {exc}"

    # ---- Internals ------------------------------------------------------------
    async def _call_openrouter(self, messages: List[dict]) -> Tuple[Optional[str], Optional[str]]:
        """Call OpenRouter with retries; return (text, error_category).

        error_category ∈ {None, 'auth', 'transient', 'other'}.
        """
        session = await self._session_holder.get()
        last_exc: Optional[BaseException] = None

        for attempt in range(1, self._cfg.or_retries + 1):
            try:
                async with session.post(
                    self._cfg.api_url,
                    json={
                        "model": self._cfg.or_model,
                        "messages": messages,
                        "max_tokens": self._cfg.or_max_tokens,
                    },
                    headers={"Authorization": f"Bearer {self._cfg.openrouter_api_key}"},
                ) as resp:
                    if resp.status == 401:
                        self._log.error("OpenRouter unauthorized (401)")
                        return None, "auth"
                    if resp.status in {429, 500, 502, 503, 504}:
                        raise RuntimeError(f"HTTP {resp.status}")

                    raw = await resp.text()
                    try:
                        data: Any = json.loads(raw)
                    except Exception:
                        self._log.warning("OpenRouter non-JSON response: %s", raw[:500])
                        return None, "other"

                    msg = (
                        (((data or {}).get("choices") or [{}])[0].get("message") or {}).get("content")
                    )
                    if not msg:
                        self._log.warning("OpenRouter empty content: %s", str(data)[:500])
                        return None, "other"
                    return msg, None
            except Exception as exc:
                last_exc = exc
                wait = min(2**attempt, 10) + 0.1 * attempt
                self._log.warning(
                    "OpenRouter attempt %s/%s failed: %s; backoff %.1fs",
                    attempt,
                    self._cfg.or_retries,
                    exc,
                    wait,
                )
                if attempt < self._cfg.or_retries:
                    await asyncio.sleep(wait)

        if last_exc is not None:
            self._log.error("OpenRouter retries exhausted: %s", last_exc)
        return None, "transient"
