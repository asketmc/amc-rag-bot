#!/usr/bin/env python3.10
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

import discord
from aiohttp import ClientSession, TCPConnector
from discord.ext import commands

import config as cfg
from rag_filter import purge_filter_cache

disc_log = logging.getLogger("asketmc.discord")
disc_log.setLevel(logging.DEBUG if getattr(cfg, "DEBUG", False) else logging.INFO)

RagGenFn = Callable[..., Awaitable[str]]
QueryModelFn = Callable[..., Awaitable[tuple[str, bool]]]
CallLocalFn = Callable[[str], Awaitable[str]]
BuildIndexFn = Callable[[], Awaitable[Any]]
IsBlockedFn = Callable[[], bool]
ShutdownFn = Callable[[], Awaitable[None]]

@dataclass
class AppDeps:
    index: Any
    retriever: Any
    generate_rag_answer: RagGenFn
    query_model: QueryModelFn
    call_local_llm: CallLocalFn
    build_index: BuildIndexFn
    is_openrouter_blocked: IsBlockedFn
    on_core_shutdown: Optional[ShutdownFn] = None

_STATE: Optional[AppDeps] = None
_user_last: Dict[int, float] = {}

class _AsyncSessionHolder:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._session: Optional[ClientSession] = None

    async def get(self) -> ClientSession:
        async with self._lock:
            if self._session is None or self._session.closed:
                self._session = ClientSession(connector=TCPConnector(limit=cfg.HTTP_CONN_LIMIT))
            return self._session

    async def close(self) -> None:
        async with self._lock:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None

_SESSION_HOLDER = _AsyncSessionHolder()

def _require_state() -> AppDeps:
    if _STATE is None:
        raise RuntimeError("Discord module is not initialized. Call run_bot(...) or start_bot_async(...).")
    return _STATE

def _check_cooldown(user_id: int) -> bool:
    now = time.monotonic()
    last = _user_last.get(user_id, 0.0)
    if now - last < cfg.USER_COOLDOWN:
        return False
    _user_last[user_id] = now
    return True

def _sanitize(text: str) -> str:
    return (
        text.replace("@", "@\u200b")
        .replace("```", " ")
        .replace("</sys>", " ")
        .replace("<sys>", " ")
    )

def _split_for_discord(text: str, limit: int = 2000) -> list[str]:
    parts: list[str] = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        cut = text.rfind("\n", 0, limit)
        if cut == -1:
            cut = limit
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return parts

async def _send_long(ctx: commands.Context, text: str, limit: int = 1900) -> None:
    for chunk in _split_for_discord(text, limit):
        await ctx.send(chunk)

_bot: Optional[commands.Bot] = None
_shutdown_lock = asyncio.Lock()
_shutdown_flag = False

def _build_bot() -> commands.Bot:
    intents = discord.Intents.all()
    return commands.Bot(command_prefix="!", intents=intents)

def _check_admin_only():
    def predicate(ctx: commands.Context) -> bool:
        if ctx.author.id not in cfg.ADMIN_IDS:
            raise commands.CheckFailure("not_admin")
        return True
    return commands.check(predicate)

def _check_channel_allowed():
    def predicate(ctx: commands.Context) -> bool:
        if ctx.channel.id not in cfg.ALLOWED_CHANNELS:
            raise commands.CheckFailure("not_allowed_channel")
        return True
    return commands.check(predicate)

async def _answer_rag(ctx: commands.Context, question_raw: str, *, force_local: bool | None = None) -> None:
    deps = _require_state()
    q = _sanitize((question_raw or "").strip().replace("\n", " "))
    if not q or len(q) > cfg.MAX_QUESTION_LEN or not cfg.ALLOWED_CHARS.match(q):
        await ctx.send("❌ Invalid query format.")
        return
    if not _check_cooldown(ctx.author.id):
        await ctx.send("⏳ Please wait before sending another query.")
        return
    is_blocked = deps.is_openrouter_blocked()
    use_remote = (force_local is None and not is_blocked) or (force_local is False)
    await ctx.send("🧠 Thinking locally…" if not use_remote else "🔍 Thinking…")
    try:
        async with cfg.REQUEST_SEMAPHORE:
            sys_prompt = Path(cfg.PROMPT_STRICT).read_text(encoding="utf-8")
            answer = await deps.generate_rag_answer(q, sys_prompt, use_remote=use_remote)
    except discord.HTTPException as http_exc:
        await _send_long(ctx, f"⚠️ Answer too long: {http_exc}")
        return
    except FileNotFoundError:
        await ctx.send("⚠️ System prompt file is missing or unreadable.")
        return
    except Exception as exc:
        disc_log.exception("[_answer_rag] unexpected error: %r", exc)
        await ctx.send(f"❌ Error: {exc}")
        return
    await _send_long(ctx, answer or "❌ No answer.")
    if force_local is None and is_blocked:
        await ctx.send("⚠️ OpenRouter unavailable, local model used.")

def _register_commands(bot: commands.Bot) -> None:
    @bot.command(name="strict", help="Answer using strict factual QA prompt and RAG.")
    @_check_channel_allowed()
    async def cmd_strict(ctx: commands.Context, *, q: str) -> None:
        await _answer_rag(ctx, q, force_local=None)

    @bot.command(name="think", help="Answer using reasoning/verbose QA prompt and RAG.")
    @_check_channel_allowed()
    async def cmd_think(ctx: commands.Context, *, q: str) -> None:
        await _answer_rag(ctx, q, force_local=None)

    @bot.command(name="local", help="Answer only with local LLM; cloud is skipped.")
    @_check_channel_allowed()
    async def cmd_local(ctx: commands.Context, *, q: str) -> None:
        await _answer_rag(ctx, q, force_local=True)

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
    @_check_admin_only()
    async def cmd_reload_index(ctx: commands.Context) -> None:
        deps = _require_state()
        try:
            purge_filter_cache()
            new_index = await deps.build_index()
            deps.index = new_index
            deps.retriever = new_index.as_retriever(similarity_top_k=cfg.TOP_K)
            await ctx.send("✅ Index reloaded.")
        except Exception as exc:
            disc_log.exception("[reload_index] error: %r", exc)
            await ctx.send(f"❌ Error: {exc}")

    @bot.command(name="stop", help="Stop the bot (admin only).")
    @_check_admin_only()
    async def cmd_stop(ctx: commands.Context) -> None:
        await ctx.send("🛑 Shutting down bot...")
        await _shutdown_once(bot)

    @bot.event
    async def on_ready() -> None:
        disc_log.info("Bot started as %s (ID: %s)", bot.user, getattr(bot.user, "id", "?"))
        try:
            ch = bot.get_channel(next(iter(cfg.ALLOWED_CHANNELS)))
            if ch:
                await ch.send("✅ Bot started and ready.")
        except Exception as exc:
            disc_log.error("[on_ready] startup notify failed: %s", exc)

    @bot.event
    async def on_command_error(ctx: commands.Context, error: Exception) -> None:
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

async def _shutdown_once(bot: commands.Bot) -> None:
    global _shutdown_flag
    async with _shutdown_lock:
        if _shutdown_flag:
            return
        _shutdown_flag = True
    try:
        ch = bot.get_channel(next(iter(cfg.ALLOWED_CHANNELS)))  # type: ignore[arg-type]
        if ch:
            await ch.send("🛑 Bot stopped.")
    except Exception as exc:
        disc_log.error("[shutdown] notify failed: %s", exc)
    try:
        await _SESSION_HOLDER.close()
    finally:
        deps = _require_state()
        if deps.on_core_shutdown:
            try:
                await deps.on_core_shutdown()
            except Exception as exc:
                disc_log.exception("[shutdown] on_core_shutdown error: %s", exc)
        await bot.close()

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
    global _STATE, _bot, _shutdown_flag
    _shutdown_flag = False
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
    global _STATE, _bot, _shutdown_flag
    _shutdown_flag = False
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
        await _shutdown_once(_bot)
