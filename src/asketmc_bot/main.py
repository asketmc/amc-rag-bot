#!/usr/bin/env python3.10
"""
Main entry point for the Asketmc RAG bot (core).

This module prepares the runtime environment and vector index, and exposes
core callables for the Discord bot via dependency injection (DI):

- make_generate_rag_answer: factory that returns an async RAG prompt builder
- query_model: calls a remote model (OpenRouter) with a local fallback (Ollama)
- call_local_llm: direct call to the local LLM

The Discord bot itself is started from `discord_bot.py` and receives the
dependencies defined here.
"""

from __future__ import annotations

# Standard library
import asyncio
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Awaitable, Callable, List, Optional, Tuple

# Third-party
import torch  # used for visibility at startup; safe if CUDA not present
from aiohttp import ClientSession, TCPConnector, ClientTimeout
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex

# Local modules
import config as cfg
from rag_filter import get_filtered_nodes, build_context
from rerank import init_reranker, rerank, shutdown_reranker
from lemma import extract_lemmas, LEMMA_POOL
from index_builder import build_index

# ──────────────────────────────────────────────────────────────────────────────
# Startup diagnostics and .env loading
# ──────────────────────────────────────────────────────────────────────────────

print(f"[STARTUP] Python: {sys.version}")
print(f"[STARTUP] Script: {__file__}")
print(f"[STARTUP] Working dir: {os.getcwd()}")
print(f"[STARTUP] torch: {torch.__version__}, cuda: {torch.cuda.is_available()}")

# Ensure the log directory exists before configuring logging.
try:
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
except Exception as exc:  # pragma: no cover
    print(f"[STARTUP] Error creating LOG_DIR: {exc}", file=sys.stderr)
    sys.exit(1)


def setup_logging(debug: bool = False) -> None:
    """Configure structured logging for the process (files + stdout).

    Args:
        debug: If True, enable DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    app_handler = RotatingFileHandler(
        cfg.LOG_DIR / "app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )
    err_handler = RotatingFileHandler(
        cfg.LOG_DIR / "error.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )
    err_handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    for handler in (app_handler, err_handler, stream_handler):
        handler.setFormatter(formatter)
        root.addHandler(handler)


setup_logging(getattr(cfg, "DEBUG", False))
log = logging.getLogger("asketmc.app")
embed_log = logging.getLogger("asketmc.embed")
rag_log = logging.getLogger("asketmc.rag")

# ──────────────────────────────────────────────────────────────────────────────
# Environment configuration (.env loading)
# ──────────────────────────────────────────────────────────────────────────────
import os
from pathlib import Path
from dotenv import load_dotenv

# Determine the absolute project root (.../LLM)
ROOT = Path(__file__).resolve().parents[2]

# Optional explicit override via environment variable
explicit = os.getenv("ASKETMC_DOTENV")

# Load order (first match wins):
#   1. Explicit path via ASKETMC_DOTENV (highest priority)
#   2. LLM/.env.local  – developer overrides (never used in CI/Prod)
#   3. LLM/.env        – canonical project configuration
candidates = [
    Path(explicit) if explicit else None,
    ROOT / ".env.local",
    ROOT / ".env",
]

for env_path in candidates:
    if env_path and env_path.exists():
        # override=False ensures system/CI variables take precedence.
        load_dotenv(env_path, override=False)
        break

@dataclass(frozen=True)
class Settings:
    """Immutable application settings derived from config and environment."""

    discord_token: str
    openrouter_api_key: str
    api_url: str
    or_model: str
    or_max_tokens: int
    ollama_url: str
    local_model: str
    http_conn_limit: int
    or_retries: int
    or_block_sec: int
    ctx_len_remote: int
    ctx_len_local: int
    top_k: int
    http_timeout_total: int


def _cfg_or(name: str, default: Any) -> Any:
    """Helper to read attributes from `config` with a default."""
    return getattr(cfg, name, default)


def load_settings() -> Settings:
    """Load and validate application settings from environment and config."""
    discord_token = os.getenv("DISCORD_TOKEN")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    if not discord_token:
        sys.exit("DISCORD_TOKEN missing")
    if not openrouter_api_key:
        sys.exit("OPENROUTER_API_KEY missing")

    return Settings(
        discord_token=discord_token,
        openrouter_api_key=openrouter_api_key,
        api_url=_cfg_or("API_URL", "https://openrouter.ai/api/v1/chat/completions"),
        or_model=_cfg_or("OR_MODEL", "openrouter/auto"),
        or_max_tokens=int(_cfg_or("OR_MAX_TOKENS", 512)),
        ollama_url=_cfg_or("OLLAMA_URL", "http://localhost:11434/api/generate"),
        local_model=_cfg_or("LOCAL_MODEL", "llama3:8b"),
        http_conn_limit=int(_cfg_or("HTTP_CONN_LIMIT", 5)),
        or_retries=int(_cfg_or("OR_RETRIES", 3)),
        or_block_sec=int(_cfg_or("OPENROUTER_BLOCK_SEC", 900)),
        ctx_len_remote=int(_cfg_or("CTX_LEN_REMOTE", 20_000)),
        ctx_len_local=int(_cfg_or("CTX_LEN_LOCAL", 12_000)),
        top_k=int(_cfg_or("TOP_K", 16)),
        http_timeout_total=int(_cfg_or("HTTP_TIMEOUT_TOTAL", 240)),
    )


SETTINGS = load_settings()

# ──────────────────────────────────────────────────────────────────────────────
# HTTP / LLM API (OpenRouter / Ollama)
# ──────────────────────────────────────────────────────────────────────────────


class AsyncSessionHolder:
    """Lazily creates and reuses a single aiohttp ClientSession with a connection limit."""

    def __init__(self, limit: int):
        self._lock = asyncio.Lock()
        self._session: Optional[ClientSession] = None
        self._limit = limit

    async def get(self) -> ClientSession:
        """Return a live session (create if missing/closed)."""
        async with self._lock:
            if self._session is None or self._session.closed:
                self._session = ClientSession(
                    connector=TCPConnector(limit=self._limit),
                    timeout=ClientTimeout(total=SETTINGS.http_timeout_total),
                )
            return self._session

    async def close(self) -> None:
        """Close the session if it is open."""
        async with self._lock:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None


session_holder = AsyncSessionHolder(limit=SETTINGS.http_conn_limit)

# OpenRouter circuit breaker: after a failure window, skip remote calls.
_openrouter_until = 0.0
_openrouter_lock = asyncio.Lock()


def is_openrouter_blocked() -> bool:
    """Return True if the OpenRouter circuit-breaker window is active."""
    return time.time() < _openrouter_until


async def _block_openrouter_window(seconds: int) -> None:
    """Extend the OpenRouter circuit-breaker window by `seconds`."""
    global _openrouter_until
    async with _openrouter_lock:
        _openrouter_until = max(_openrouter_until, time.time() + seconds)


async def _call_openrouter(messages: List[dict]) -> Optional[str]:
    """Call OpenRouter with retries and basic error handling.

    Args:
        messages: OpenAI-compatible messages array.

    Returns:
        Model text if successful, otherwise None.
    """
    session = await session_holder.get()
    retries = SETTINGS.or_retries
    for attempt in range(1, retries + 1):
        try:
            async with session.post(
                SETTINGS.api_url,
                json={"model": SETTINGS.or_model, "messages": messages, "max_tokens": SETTINGS.or_max_tokens},
                headers={"Authorization": f"Bearer {SETTINGS.openrouter_api_key}"},
            ) as resp:
                if resp.status == 401:
                    log.error("OpenRouter unauthorized (401)")
                    return None
                if resp.status in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"HTTP {resp.status}")
                data = await resp.json(content_type=None)
                choice = (data or {}).get("choices", [{}])[0]
                msg = (choice or {}).get("message", {})
                text = msg.get("content")
                if not text:
                    log.warning("OpenRouter returned empty content: %s", data)
                    return None
                return text
        except Exception as exc:
            wait = min(2**attempt, 10) + (0.1 * attempt)  # backoff with jitter
            log.warning(
                "OpenRouter attempt %s/%s failed: %s; sleeping %.1fs",
                attempt,
                retries,
                exc,
                wait,
            )
            if attempt < retries:
                await asyncio.sleep(wait)
    return None


async def call_local_llm(prompt_text: str, timeout_sec: Optional[int] = None) -> str:
    """Call the local LLM (Ollama).

    Args:
        prompt_text: Full prompt string.
        timeout_sec: Optional total timeout override.

    Returns:
        Model text or an error message.
    """
    session = await session_holder.get()
    try:
        async with session.post(
            SETTINGS.ollama_url,
            json={"model": SETTINGS.local_model, "prompt": prompt_text, "stream": False},
            timeout=ClientTimeout(total=timeout_sec or SETTINGS.http_timeout_total),
        ) as resp:
            data = await resp.json(content_type=None)
            return (data or {}).get("response", "❌ No response.")
    except asyncio.TimeoutError:
        return "⚠️ Local LLM did not respond (timeout)."
    except Exception as exc:
        return f"⚠️ Local LLM error: {exc}"


async def query_model(
    messages: Optional[List[dict]] = None,
    sys_prompt: Optional[str] = None,
    ctx_txt: Optional[str] = None,
    q: Optional[str] = None,
    timeout_sec: int = 240,
) -> Tuple[str, bool]:
    """Query a remote model via OpenRouter with a local fallback.

    One of:
      * Provide `messages` (OpenAI-style), or
      * Provide (`sys_prompt`, `ctx_txt`, `q`) to build messages.

    Args:
        messages: Pre-built messages for the remote call.
        sys_prompt: System prompt text (used if `messages` is None).
        ctx_txt: Context text (used if `messages` is None).
        q: User question (used if `messages` is None).
        timeout_sec: Fallback local-LLM timeout.

    Returns:
        Tuple of (text, used_fallback) where `used_fallback` is True if local
        model was used.
    """
    # Build messages if not provided.
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

    # Try OpenRouter if not in the block window.
    if not is_openrouter_blocked():
        text = await _call_openrouter(messages)
        if text is not None:
            return text, used_fallback
        used_fallback = True
        await _block_openrouter_window(SETTINGS.or_block_sec)
    else:
        used_fallback = True

    # Fallback: local LLM via a plain prompt string.
    if sys_prompt is None or ctx_txt is None or q is None:
        # If called with `messages`, rebuild a reasonable plain prompt.
        sys_prompt = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        prompt_text = (sys_prompt or "") + "\n\n" + (user or "")
    else:
        prompt_text = (
            sys_prompt.strip()
            + "\n\nCONTEXT:\n"
            + ctx_txt.strip()
            + "\n\nQUESTION: "
            + q.strip()
            + "\nANSWER:"
        )

    text = await call_local_llm(prompt_text, timeout_sec=timeout_sec)
    return text, used_fallback


# ──────────────────────────────────────────────────────────────────────────────
# RAG helpers
# ──────────────────────────────────────────────────────────────────────────────


def make_generate_rag_answer(
    retriever: Any,
) -> Callable[[str, str, bool], Awaitable[Tuple[str, Any, Any]]]:
    """Create a RAG prompt generator bound to a specific retriever.

    Args:
        retriever: Async retriever that exposes `aretrieve(query)`.

    Returns:
        Async function (q, sys_prompt, use_remote) -> (prompt, nodes, ctx_txt).
    """

    async def _generate(q: str, sys_prompt: str, use_remote: bool) -> Tuple[str, Any, Any]:
        qlem = extract_lemmas(q)
        raw_nodes = await retriever.aretrieve(q)
        reranked_nodes = await rerank(q, raw_nodes)
        nodes = await get_filtered_nodes(reranked_nodes or raw_nodes, qlem)
        if not nodes:
            return "⚠️ Not enough data.", None, None
        char_limit = SETTINGS.ctx_len_remote if use_remote else SETTINGS.ctx_len_local
        ctx_txt = build_context(nodes, qlem, char_limit)
        prompt = (
            sys_prompt.strip()
            + "\n\nCONTEXT:\n"
            + ctx_txt.strip()
            + "\n\nQUESTION: "
            + q.strip()
            + "\nANSWER:"
        )
        return prompt, nodes, ctx_txt

    return _generate


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

index: Optional["VectorStoreIndex"] = None


async def _core_shutdown() -> None:
    """Gracefully release core resources (reranker, HTTP session, lemma pool)."""
    try:
        await shutdown_reranker()
    finally:
        try:
            await session_holder.close()
        finally:
            LEMMA_POOL.shutdown(wait=True)
            logging.getLogger("asketmc.app").info("Core shutdown complete")


async def _start_discord_bot_in_executor(
    discord_module: Any,
    *,
    token: str,
    index: Any,
    retriever: Any,
    generate_rag_answer: Callable[[str, str, bool], Awaitable[Tuple[str, Any, Any]]],
) -> None:
    """Run a blocking `discord_module.run_bot(...)` in a thread executor."""
    loop = asyncio.get_running_loop()
    # Pass kwargs via lambda to avoid functools.partial with many args.
    await loop.run_in_executor(
        None,
        lambda: discord_module.run_bot(
            token=token,
            index=index,
            retriever=retriever,
            generate_rag_answer=generate_rag_answer,
            query_model=query_model,
            call_local_llm=call_local_llm,
            build_index=build_index,
            is_openrouter_blocked=is_openrouter_blocked,
            on_core_shutdown=_core_shutdown,
        ),
    )


async def _main() -> None:
    """Async main: build the index, initialize components, then start the bot."""
    global index
    log.info("Building document index...")
    index = await build_index()
    retriever = index.as_retriever(similarity_top_k=SETTINGS.top_k)
    await init_reranker()
    log.info("Index built. Starting Discord bot...")

    # Late import to avoid circular dependencies.
    import discord_bot as discord_module

    generate_rag_answer = make_generate_rag_answer(retriever)

    # Register graceful shutdown handlers where supported (Unix).
    loop = asyncio.get_running_loop()

    def _on_signal(signame: str) -> None:
        log.warning("Received %s, initiating shutdown...", signame)

    for sig in ("SIGINT", "SIGTERM"):
        if hasattr(signal, sig):
            try:
                loop.add_signal_handler(getattr(signal, sig), _on_signal, sig)
            except NotImplementedError:
                # Windows Proactor loop does not support signal handlers.
                log.info(
                    "Signal handlers not supported on this platform; relying on KeyboardInterrupt."
                )
                break

    try:
        # Prefer an async entrypoint if the module provides one (best practice).
        if hasattr(discord_module, "start_bot_async"):
            await discord_module.start_bot_async(
                token=SETTINGS.discord_token,
                index=index,
                retriever=retriever,
                generate_rag_answer=generate_rag_answer,
                query_model=query_model,
                call_local_llm=call_local_llm,
                build_index=build_index,
                is_openrouter_blocked=is_openrouter_blocked,
                on_core_shutdown=_core_shutdown,
            )
        else:
            # Fallback: run blocking `run_bot()` inside a thread executor to avoid
            # "asyncio.run() cannot be called from a running event loop".
            await _start_discord_bot_in_executor(
                discord_module,
                token=SETTINGS.discord_token,
                index=index,
                retriever=retriever,
                generate_rag_answer=generate_rag_answer,
            )
    except KeyboardInterrupt:
        log.warning("KeyboardInterrupt received, shutting down...")
    finally:
        await _core_shutdown()


if __name__ == "__main__":
    try:
        # Optional: on Windows, enable selector policy for broader compatibility.
        if sys.platform == "win32":  # pragma: no cover
            try:
                import asyncio as _asyncio  # local alias to avoid shadowing

                _asyncio.set_event_loop_policy(_asyncio.WindowsSelectorEventLoopPolicy())
            except Exception:
                pass

        asyncio.run(_main())
    except KeyboardInterrupt:
        logging.getLogger("asketmc.app").warning("Interrupted by user.")
    except Exception as exc:  # pragma: no cover
        logging.getLogger("asketmc.app").critical("Fatal error in main entry: %s", exc)
        raise
