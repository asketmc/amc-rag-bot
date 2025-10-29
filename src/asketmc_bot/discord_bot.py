#!/usr/bin/env python3.10
"""
Discord integration: bot, commands, and lifecycle.

This module is independent from `main.py`. All dependencies are injected via
`run_bot(...)` (blocking) or `start_bot_async(...)` (async) to avoid circular
imports and to keep the code testable and modular.

Key features:
- Channel and admin checks implemented with `commands.check` (do not alter
  command signatures).
- Centralized input sanitization and flood control (per-user cooldown).
- Consistent query path for remote (OpenRouter) and local (Ollama) models.
- Graceful shutdown: closes HTTP session, calls core shutdown hook, closes bot.

PEP 8 and PEP 257 compliant.
"""

from __future__ import annotations

# Standard library
import asyncio
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

# Third-party
import discord
from aiohttp import ClientSession, TCPConnector
from discord.ext import commands

# Local
import config as cfg
from rag_filter import FILTER_CACHE

# ─────────────────────────────────────────────────────────────
# Logging (Discord section)
# ─────────────────────────────────────────────────────────────

disc_log = logging.getLogger("asketmc.discord")
disc_log.setLevel(logging.DEBUG if getattr(cfg, "DEBUG", False) else logging.INFO)

# ─────────────────────────────────────────────────────────────
# DI container: runtime state and injected dependencies
# ─────────────────────────────────────────────────────────────

RagGenFn = Callable[[str, str, bool], Awaitable[Tuple[str, Any, Any]]]
QueryModelFn = Callable[..., Awaitable[Tuple[str, bool]]]
CallLocalFn = Callable[[str], Awaitable[str]]
BuildIndexFn = Callable[[], Awaitable[Any]]
IsBlockedFn = Callable[[], bool]
ShutdownFn = Callable[[], Awaitable[None]]


@dataclass
class AppDeps:
    """Injected dependencies and shared runtime state."""
    index: Any
    retriever: Any
    generate_rag_answer: RagGenFn
    query_model: QueryModelFn
    call_local_llm: CallLocalFn
    build_index: BuildIndexFn
    is_openrouter_blocked: IsBlockedFn
    on_core_shutdown: Optional[ShutdownFn] = None


_STATE: Optional[AppDeps] = None
_user_last: Dict[int, float] = {}  # simple per-user cooldown map

# ─────────────────────────────────────────────────────────────
# HTTP session (if needed by future commands)
# ─────────────────────────────────────────────────────────────


class AsyncSessionHolder:
    """Lazily creates and reuses a single aiohttp ClientSession."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._session: Optional[ClientSession] = None
        disc_log.debug("[AsyncSessionHolder] initialized")

    async def get(self) -> ClientSession:
        """Return a live session (create if missing/closed)."""
        async with self._lock:
            if self._session is None or self._session.closed:
                disc_log.info("[AsyncSessionHolder] creating aiohttp session")
                self._session = ClientSession(connector=TCPConnector(limit=cfg.HTTP_CONN_LIMIT))
            return self._session

    async def close(self) -> None:
        """Close the session if it is open."""
        async with self._lock:
            if self._session and not self._session.closed:
                disc_log.info("[AsyncSessionHolder] closing aiohttp session")
                await self._session.close()
                self._session = None


_SESSION_HOLDER = AsyncSessionHolder()

# ─────────────────────────────────────────────────────────────
# Message utilities and input sanitization
# ─────────────────────────────────────────────────────────────

# Basic guard against prompt-injection patterns and Discord mentions.
INJECTION_RE = re.compile(
    r"(?is)(?:^|\s)(?:assistant|system)\s*:|```|</?sys>|###\s*(?:assistant|system)"
)


def sanitize(text: str) -> str:
    """Basic Discord-safe sanitizer and prompt-injection guard.

    Replaces '@' with a zero-width joiner to avoid unintended mentions and
    strips a few known prompt-injection markers.
    """
    text = text.replace("@", "@\u200b")
    return INJECTION_RE.sub(" ", text)


def split_message(text: str, limit: int = 2000) -> List[str]:
    """Split a long message into Discord-sized chunks."""
    chunks: List[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        idx = text.rfind("\n", 0, limit)
        if idx == -1:
            idx = limit
        chunks.append(text[:idx])
        text = text[idx:].lstrip("\n")
    return chunks


async def send_long(ctx: commands.Context, text: str, limit: int = 1900) -> None:
    """Send large text in multiple messages."""
    for chunk in split_message(text, limit):
        await ctx.send(chunk)

# ─────────────────────────────────────────────────────────────
# Access control checks (do not alter command signatures)
# ─────────────────────────────────────────────────────────────


def check_admin_only() -> Callable[[commands.Context], bool]:
    """Allow command only for users listed in cfg.ADMIN_IDS."""
    def predicate(ctx: commands.Context) -> bool:
        if ctx.author.id not in cfg.ADMIN_IDS:
            raise commands.CheckFailure("not_admin")
        return True
    return commands.check(predicate)


def check_channel_allowed() -> Callable[[commands.Context], bool]:
    """Allow command only in channels listed in cfg.ALLOWED_CHANNELS."""
    def predicate(ctx: commands.Context) -> bool:
        if ctx.channel.id not in cfg.ALLOWED_CHANNELS:
            raise commands.CheckFailure("not_allowed_channel")
        return True
    return commands.check(predicate)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def _require_state() -> AppDeps:
    """Return the DI state or raise if the module was not initialized."""
    if _STATE is None:
        raise RuntimeError("Discord module is not initialized. Call run_bot(...) or start_bot_async(...).")
    return _STATE


def _check_cooldown(user_id: int) -> bool:
    """Return True if the user is not sending messages too frequently."""
    now = time.monotonic()
    last = _user_last.get(user_id, 0.0)
    if now - last < cfg.USER_COOLDOWN:
        return False
    _user_last[user_id] = now
    return True

# ─────────────────────────────────────────────────────────────
# Command core
# ─────────────────────────────────────────────────────────────


async def _ask_rag(ctx: commands.Context, q_raw: str, sys_prompt_path: str) -> None:
    """Shared RAG handler used by several commands.

    Steps:
    - Sanitize and validate user input.
    - Enforce per-user cooldown and channel restrictions (via checks).
    - Build RAG prompt/context through DI-provided generator.
    - Query remote model with fallback to local model.
    """
    deps = _require_state()
    disc_log.info("[ask_rag] user=%s(%d) ch=%d", ctx.author.name, ctx.author.id, ctx.channel.id)

    q = sanitize(q_raw.strip().replace("\n", " "))
    if len(q) > cfg.MAX_QUESTION_LEN or not cfg.ALLOWED_CHARS.match(q):
        await ctx.send("❌ Invalid query format.")
        return

    if not _check_cooldown(ctx.author.id):
        await ctx.send("⏳ Please wait before sending another query.")
        return

    await ctx.send("🔍 Thinking…")

    try:
        async with cfg.REQUEST_SEMAPHORE:
            use_remote = not deps.is_openrouter_blocked()  # use remote if allowed
            sys_prompt_text = Path(sys_prompt_path).read_text("utf-8")

            # Build prompt/context via DI function (applies filters/limits).
            prompt, nodes, ctx_txt = await deps.generate_rag_answer(q, sys_prompt_text, use_remote)
            if not nodes:
                await ctx.send("⚠️ Not enough data.")
                return

            # Query model in a single, consistent way.
            answer, used_fallback = await deps.query_model(
                sys_prompt=sys_prompt_text,
                ctx_txt=ctx_txt,
                q=q,
            )

            if used_fallback:
                await send_long(ctx, "⚠️ OpenRouter unavailable, local model used.")
            await send_long(ctx, answer or "❌ No answer.")
    except discord.HTTPException as http_exc:
        await send_long(ctx, f"⚠️ Answer too long: {http_exc}")
    except Exception as exc:  # pragma: no cover
        disc_log.exception("[_ask_rag] Unexpected error: %r", exc)
        await ctx.send(f"❌ Error: {exc}")

# ─────────────────────────────────────────────────────────────
# Bot construction and command registration
# ─────────────────────────────────────────────────────────────

_bot: Optional[commands.Bot] = None


def _build_bot() -> commands.Bot:
    """Create a Discord bot with all intents."""
    intents = discord.Intents.all()
    return commands.Bot(command_prefix="!", intents=intents)


def _register_commands(bot: commands.Bot) -> None:
    """Attach command handlers to the bot."""

    @bot.command(name="strict", help="Answer using strict factual QA prompt and RAG.")
    @check_channel_allowed()
    async def cmd_strict(ctx: commands.Context, *, q: str) -> None:
        await _ask_rag(ctx, q, cfg.PROMPT_STRICT)

    @bot.command(name="think", help="Answer using reasoning/verbose QA prompt and RAG.")
    @check_channel_allowed()
    async def cmd_think(ctx: commands.Context, *, q: str) -> None:
        await _ask_rag(ctx, q, cfg.PROMPT_REASON)

    @bot.command(name="local", help="Answer only with local LLM; cloud is skipped.")
    @check_channel_allowed()
    async def cmd_local_llm(ctx: commands.Context, *, q: str) -> None:
        deps = _require_state()
        q_s = sanitize(q.strip().replace("\n", " "))
        if len(q_s) > cfg.MAX_QUESTION_LEN or not cfg.ALLOWED_CHARS.match(q_s):
            await ctx.send("❌ Invalid query format.")
            return
        if not _check_cooldown(ctx.author.id):
            await ctx.send("⏳ Please wait before sending another query.")
            return

        await ctx.send("🧠 Thinking locally…")
        try:
            async with cfg.REQUEST_SEMAPHORE:
                sys_prompt_text = Path(cfg.PROMPT_STRICT).read_text("utf-8")
                prompt, nodes, _ = await deps.generate_rag_answer(q_s, sys_prompt_text, use_remote=False)
                if not nodes:
                    await ctx.send("⚠️ Not enough data.")
                    return
                answer = await deps.call_local_llm(prompt)
                await send_long(ctx, answer or "❌ No answer.")
        except discord.HTTPException as http_exc:
            await send_long(ctx, f"⚠️ Answer too long: {http_exc}")
        except Exception as exc:  # pragma: no cover
            disc_log.exception("[cmd_local_llm] error: %r", exc)
            await ctx.send(f"❌ Error: {exc}")

    @bot.command(name="status", help="Show bot/index/cache status.")
    async def cmd_status(ctx: commands.Context) -> None:
        deps = _require_state()
        try:
            docs = len(deps.index.docstore.docs)  # type: ignore[attr-defined]
        except Exception:
            docs = -1
        await ctx.send(
            f"🧠 Documents: {docs}\n"
            f"💾 Cache: {'yes' if cfg.CACHE_PATH.exists() else 'no'}\n"
            f"🌐 OpenRouter: {'blocked' if deps.is_openrouter_blocked() else 'ok'}"
        )

    @bot.command(name="reload_index", help="Rebuild the index (admin only).")
    @check_admin_only()
    async def cmd_reload_index(ctx: commands.Context) -> None:
        deps = _require_state()
        try:
            FILTER_CACHE.clear()
            new_index = await deps.build_index()
            new_retriever = new_index.as_retriever(similarity_top_k=cfg.TOP_K)
            deps.index = new_index
            deps.retriever = new_retriever
            await ctx.send("✅ Index reloaded.")
        except Exception as exc:  # pragma: no cover
            disc_log.exception("[reload_index] error: %r", exc)
            await ctx.send(f"❌ Error: {exc}")

    @bot.command(name="multy", help="Multi-step analysis using the strict prompt.")
    @check_channel_allowed()
    async def cmd_multy(ctx: commands.Context, *, q: str) -> None:
        deps = _require_state()
        await ctx.send("🔎 Running multi-step analysis…")
        try:
            sys_prompt_text = Path(cfg.PROMPT_STRICT).read_text("utf-8")
            prompt, nodes, ctx_txt = await deps.generate_rag_answer(
                q.strip(), sys_prompt_text, use_remote=not deps.is_openrouter_blocked()
            )
            if not nodes:
                await ctx.send("⚠️ Not enough data.")
                return
            answer, _ = await deps.query_model(
                sys_prompt=sys_prompt_text,
                ctx_txt=ctx_txt,
                q=q.strip(),
            )
            await send_long(ctx, answer or "❌ No answer.")
        except Exception as exc:  # pragma: no cover
            disc_log.exception("[cmd_multy] error: %r", exc)
            await ctx.send(f"❌ Error: {exc}")

    @bot.command(name="stop", help="Stop the bot (admin only).")
    @check_admin_only()
    async def cmd_stop(ctx: commands.Context) -> None:
        await ctx.send("🛑 Shutting down bot...")
        await _shutdown(bot)

    @bot.event
    async def on_ready() -> None:
        disc_log.info("Bot started as %s (ID: %s)", bot.user, getattr(bot.user, "id", "?"))
        try:
            ch = bot.get_channel(next(iter(cfg.ALLOWED_CHANNELS)))
            if ch:
                await ch.send("✅ Bot started and ready.")
            else:
                disc_log.warning("[on_ready] Allowed channel not found.")
        except Exception as exc:
            disc_log.error("[on_ready] Failed to send startup notification: %s", exc)

    @bot.event
    async def on_command_error(ctx: commands.Context, error: Exception) -> None:
        """Unified command error handler for checks and usage errors."""
        if isinstance(error, commands.CheckFailure):
            msg = str(error)
            if msg == "not_admin":
                await ctx.send("❌ Access denied.")
            elif msg == "not_allowed_channel":
                await ctx.send("❌ This command is not allowed in this channel.")
            else:
                await ctx.send("❌ You are not allowed to use this command here.")
            return

        if isinstance(error, commands.MissingRequiredArgument):
            await ctx.send("❌ Invalid command usage.")
            return

        disc_log.exception("[on_command_error] Unhandled: %r", error)
        await ctx.send(f"❌ Error: {error}")

# ─────────────────────────────────────────────────────────────
# Shutdown
# ─────────────────────────────────────────────────────────────


async def _shutdown(bot: commands.Bot) -> None:
    """Gracefully shut down: notify channel, close HTTP session, call core shutdown, close bot."""
    try:
        ch = bot.get_channel(next(iter(cfg.ALLOWED_CHANNELS)))  # type: ignore[arg-type]
        if ch:
            await ch.send("🛑 Bot stopped.")
        else:
            disc_log.warning("[shutdown] Allowed channel not found.")
    except Exception as exc:
        disc_log.error("[shutdown] Failed to send shutdown notification: %s", exc)

    try:
        await _SESSION_HOLDER.close()
    finally:
        deps = _require_state()
        if deps.on_core_shutdown:
            try:
                await deps.on_core_shutdown()
            except Exception as exc:  # pragma: no cover
                disc_log.exception("[shutdown] on_core_shutdown error: %s", exc)
        await bot.close()

# ─────────────────────────────────────────────────────────────
# Start APIs
# ─────────────────────────────────────────────────────────────


def run_bot(
    *,
    token: str,
    index: Any,
    retriever: Any,
    generate_rag_answer: RagGenFn,
    query_model: QueryModelFn,
    call_local_llm: CallLocalFn,
    build_index: BuildIndexFn,
    is_openrouter_blocked: IsBlockedFn,
    on_core_shutdown: Optional[ShutdownFn] = None,
) -> None:
    """Blocking entrypoint (discord.py will create its own event loop).

    Use this in a dedicated thread/process, or as the top-level entry of a
    synchronous program. If your host code already has an event loop,
    prefer `start_bot_async(...)`.
    """
    global _STATE, _bot
    _STATE = AppDeps(
        index=index,
        retriever=retriever,
        generate_rag_answer=generate_rag_answer,
        query_model=query_model,
        call_local_llm=call_local_llm,
        build_index=build_index,
        is_openrouter_blocked=is_openrouter_blocked,
        on_core_shutdown=on_core_shutdown,
    )
    _bot = _build_bot()
    _register_commands(_bot)
    _bot.run(token, log_handler=None)


async def start_bot_async(
    *,
    token: str,
    index: Any,
    retriever: Any,
    generate_rag_answer: RagGenFn,
    query_model: QueryModelFn,
    call_local_llm: CallLocalFn,
    build_index: BuildIndexFn,
    is_openrouter_blocked: IsBlockedFn,
    on_core_shutdown: Optional[ShutdownFn] = None,
) -> None:
    """Async entrypoint: run Discord bot inside an existing event loop.

    This avoids `asyncio.run()` conflicts (recommended when the host program is
    already async). Mirrors the signature of `run_bot(...)`.
    """
    global _STATE, _bot
    _STATE = AppDeps(
        index=index,
        retriever=retriever,
        generate_rag_answer=generate_rag_answer,
        query_model=query_model,
        call_local_llm=call_local_llm,
        build_index=build_index,
        is_openrouter_blocked=is_openrouter_blocked,
        on_core_shutdown=on_core_shutdown,
    )
    _bot = _build_bot()
    _register_commands(_bot)
    try:
        await _bot.start(token)
    finally:
        # Ensure we run graceful shutdown logic even if `start()` raises.
        await _shutdown(_bot)
