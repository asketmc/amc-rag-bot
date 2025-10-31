# src/asketmc_bot/main.py
#!/usr/bin/env python3
"""Asketmc RAG Bot — async entry point.

Responsibilities:
- Load .env and configuration.
- Configure logging.
- Initialize index, reranker, and LLM client.
- Start Discord bot asynchronously.
- Gracefully shut down resources on SIGINT/SIGTERM or KeyboardInterrupt.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex

# Local imports
import config as cfg
import discord_bot as discord_module
from index_builder import build_index
from lemma import LEMMA_POOL, extract_lemmas
from llm_client import LLMClient, LLMConfig
from rag_filter import build_context, get_filtered_nodes
from rerank import init_reranker, rerank, shutdown_reranker


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(debug: bool = False) -> None:
    """Configure structured logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ── Config & Settings ─────────────────────────────────────────────────────────
def load_settings() -> dict:
    """Load .env and config values with safe defaults."""
    current = Path(__file__).resolve()
    if current.parent.name == "asketmc_bot":
        project_root = current.parents[2]
    else:
        project_root = current.parent

    candidates = [
        project_root / ".env",
        project_root / ".env.local",
        project_root / ".env-example",
    ]
    env_loaded = False
    for env_path in candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            env_loaded = True
            break
    if not env_loaded:
        print(f"[WARN] No .env found in {project_root}", file=sys.stderr)

    required = ["DISCORD_TOKEN", "OPENROUTER_API_KEY"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        sys.exit(f"Missing required env vars: {', '.join(missing)}")

    return {
        "discord_token": os.getenv("DISCORD_TOKEN"),
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY"),
        "api_url": getattr(cfg, "API_URL", "https://openrouter.ai/api/v1/chat/completions"),
        "or_model": getattr(cfg, "OR_MODEL", "openrouter/auto"),
        "or_max_tokens": int(getattr(cfg, "OR_MAX_TOKENS", 512)),
        "ollama_url": getattr(cfg, "OLLAMA_URL", "http://localhost:11434/api/generate"),
        "local_model": getattr(cfg, "LOCAL_MODEL", "qwen2.5:7b-instruct-q4_K_M"),
        "top_k": int(getattr(cfg, "TOP_K", 16)),
        "ctx_len_remote": int(getattr(cfg, "CTX_LEN_REMOTE", 20_000)),
        "ctx_len_local": int(getattr(cfg, "CTX_LEN_LOCAL", 12_000)),
        "http_conn_limit": int(getattr(cfg, "HTTP_CONN_LIMIT", 5)),
        "or_retries": int(getattr(cfg, "OR_RETRIES", 3)),
        "http_timeout_total": int(getattr(cfg, "HTTP_TIMEOUT_TOTAL", 240)),
        "breaker_base_block_sec": int(getattr(cfg, "OPENROUTER_BLOCK_SEC", 120)),
        "breaker_max_block_sec": int(getattr(cfg, "OPENROUTER_BLOCK_MAX_SEC", 900)),
    }


def make_llm_config(s: dict) -> LLMConfig:
    """Convert settings dict into LLMConfig."""
    return LLMConfig(
        api_url=s["api_url"],
        or_model=s["or_model"],
        or_max_tokens=s["or_max_tokens"],
        openrouter_api_key=s["openrouter_api_key"],
        ollama_url=s["ollama_url"],
        local_model=s["local_model"],
        http_conn_limit=s["http_conn_limit"],
        or_retries=s["or_retries"],
        http_timeout_total=s["http_timeout_total"],
        breaker_base_block_sec=s["breaker_base_block_sec"],
        breaker_max_block_sec=s["breaker_max_block_sec"],
    )


# ── Core RAG logic ────────────────────────────────────────────────────────────
async def generate_rag_answer(
    retriever,
    query: str,
    sys_prompt: str,
    llm_client: LLMClient,
    settings: dict,
    **kwargs,
) -> str:
    """Build a RAG prompt and generate an answer via the LLM."""
    use_remote = kwargs.get("use_remote", None)

    qlem = extract_lemmas(query)
    raw_nodes = await retriever.aretrieve(query)
    reranked_nodes = await rerank(query, raw_nodes)
    nodes = await get_filtered_nodes(reranked_nodes or raw_nodes, qlem)

    if not nodes:
        return "⚠️ Not enough data."

    char_limit = settings["ctx_len_remote"]
    ctx_txt = build_context(nodes, qlem, char_limit)

    if use_remote is False:
        prompt_text = (
            f"{sys_prompt.strip()}\n\nCONTEXT:\n{ctx_txt.strip()}\n\nQUESTION: {query.strip()}\nANSWER:"
        )
        return await llm_client.call_local_llm(prompt_text)

    text, _used_fallback = await llm_client.query_model(
        sys_prompt=sys_prompt,
        ctx_txt=ctx_txt,
        q=query,
    )
    return text


# ── Application lifecycle ─────────────────────────────────────────────────────
async def main() -> None:
    """Main async entry point."""
    settings = load_settings()
    setup_logging(getattr(cfg, "DEBUG", False))
    log = logging.getLogger("asketmc.main")

    log.info("Building document index...")
    index: VectorStoreIndex = await build_index()
    retriever = index.as_retriever(similarity_top_k=settings["top_k"])

    await init_reranker()
    llm = LLMClient(make_llm_config(settings), logger=logging.getLogger("asketmc.llm"))

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    llm.attach_loop(loop)

    async def shutdown() -> None:
        """Graceful async shutdown for core services."""
        log.info("Shutting down core systems...")
        try:
            await shutdown_reranker()
        finally:
            try:
                await llm.close()
            finally:
                LEMMA_POOL.shutdown(wait=True)
                log.info("Shutdown complete.")
                stop_event.set()

    async def query_model_text(prompt: str) -> str:
        """Simple text query wrapper for the bot."""
        text, _ = await llm.query_model(messages=[{"role": "user", "content": prompt}])
        return text

    def is_openrouter_blocked_fn() -> bool:
        """Synchronous breaker check for bot throttling decisions."""
        return llm.is_remote_blocked_sync()

    # Start Discord bot
    bot_task = asyncio.create_task(
        discord_module.start_bot_async(
            token=settings["discord_token"],
            index=index,
            retriever=retriever,
            generate_rag_answer=lambda q, p, **kw: generate_rag_answer(
                retriever, q, p, llm, settings, **kw
            ),
            query_model=query_model_text,
            call_local_llm=llm.call_local_llm,
            build_index=build_index,
            is_openrouter_blocked=is_openrouter_blocked_fn,
            on_core_shutdown=shutdown,
        )
    )

    # Signal handling
    for sig_name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, sig_name):
            try:
                loop.add_signal_handler(getattr(signal, sig_name), lambda: asyncio.create_task(shutdown()))
            except NotImplementedError:
                pass

    await stop_event.wait()

    # Cancel bot gracefully
    bot_task.cancel()
    try:
        await bot_task
    except asyncio.CancelledError:
        pass


# ── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.getLogger("asketmc.main").warning("Interrupted by user.")
