#!/usr/bin/env python3.10
"""
config.py — Configuration for the Asketmc RAG Discord bot.

Goals:
- Clear separation of concerns: path layout, model/RAG settings, runtime limits.
- All paths resolve to absolute Paths and live under project root by default.
- No .env reads here (handled in main.py); values are code-defined for stability.

Environment overrides (optional, read by your own code where applicable):
- ASKETMC_VAR_DIR     → directory for logs/caches (default: <PROJECT_ROOT>/var)
- ASKETMC_DATA_DIR    → directory for data (default: <PROJECT_ROOT>/data)
- ASKETMC_PROMPTS_DIR → directory for prompt files (default: <PKG_ROOT>/data)
"""

from __future__ import annotations

from pathlib import Path
from typing import Pattern, Set
import asyncio
import logging
import os
import re

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

# Package root (.../src/asketmc_bot) and project root (.../LM)
PKG_ROOT: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = PKG_ROOT.parents[1]

# Mutable artifacts (logs, caches)
VAR_ROOT: Path = Path(os.getenv("ASKETMC_VAR_DIR", PROJECT_ROOT / "var")).resolve()
LOG_DIR: Path = VAR_ROOT / "logs"
CACHE_PATH: Path = VAR_ROOT / "rag_cache"
HASH_FILE: Path = CACHE_PATH / "docs_hash.json"

# Data & prompts
DATA_ROOT: Path = Path(os.getenv("ASKETMC_DATA_DIR", PROJECT_ROOT / "data")).resolve()
DOCS_PATH: Path = DATA_ROOT / "parsed"

PROMPTS_DIR: Path = Path(os.getenv("ASKETMC_PROMPTS_DIR", PKG_ROOT / "data")).resolve()
PROMPT_STRICT: Path = PROMPTS_DIR / "system_prompt_strict.txt"
PROMPT_REASON: Path = PROMPTS_DIR / "system_prompt_reason.txt"
PROMPT_REPHRASE: Path = PROMPTS_DIR / "rephrase.txt"

# Ensure directories exist (idempotent)
for _p in (VAR_ROOT, LOG_DIR, CACHE_PATH, DATA_ROOT, PROMPTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Rerank settings
# ──────────────────────────────────────────────────────────────────────────────

RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
RERANK_INPUT_K: int = 18
RERANK_OUTPUT_K: int = 9
BATCH_SIZE: int = 8
MAX_LEN: int = 512
QUERY_MAX_CHARS: int = 2048
EXECUTOR_WORKERS: int = 2
RERANKER_DEVICE: str = "cpu"  # or "cuda"

# ──────────────────────────────────────────────────────────────────────────────
# Models and parameters
# ──────────────────────────────────────────────────────────────────────────────

TOP_K: int = 24
MIN_SCORE: float = 0.35

# Remote / local LLM defaults (OpenRouter / Ollama)
OR_MODEL: str = "xiaomi/mimo-v2-flash:free"
LOCAL_MODEL: str = "llama3.1:8b"
OR_MAX_TOKENS: int = 2048
API_URL: str = "https://openrouter.ai/api/v1/chat/completions"
OLLAMA_URL: str = "http://127.0.0.1:11434/api/generate"

# Context building / chunking
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 128
LEMMA_MATCH_RATIO: float = 0.2
SCORE_RELATIVE_THRESHOLD: float = 0.15

# Context limits (align with main.py expectations)
CTX_LEN_REMOTE: int = 20_000  # when using remote model
CTX_LEN_LOCAL: int = 12_000   # when using local model

DEBUG: bool = True

# ──────────────────────────────────────────────────────────────────────────────
# Stop-words / POS filters (RU-focused with some general noise)
# ──────────────────────────────────────────────────────────────────────────────

GOOD_POS: Set[str] = {
    "NOUN", "ADJF", "ADJS", "VERB", "INFN", "PRTF", "GRND", "NUMR",
}

STOP_WORDS: Set[str] = {
    # pronouns
    "я", "ты", "он", "она", "оно", "мы", "вы", "они",
    "мой", "твой", "его", "её", "наш", "ваш", "их", "свой",
    "меня", "тебя", "него", "неё", "нас", "вас", "них",
    "мне", "тебе", "ему", "ей", "нам", "вам", "им",
    "этот", "эта", "это", "эти", "тот", "та", "то", "те",
    "такой", "такая", "такое", "такие",
    "кто", "что", "ничто", "никто", "некто", "нечто",
    "кто-то", "что-то", "кто-нибудь", "что-нибудь",
    "кто-либо", "что-либо",
    # prepositions
    "в", "во", "на", "за", "к", "ко", "с", "со", "от", "перед",
    "при", "об", "обо", "по", "до", "из", "иза", "без", "для",
    "над", "под", "между", "около", "через", "про", "среди",
    "из-за", "из-под", "внутри", "вне", "после", "согласно",
    # conjunctions
    "и", "а", "но", "да", "или", "либо", "тоже", "также",
    "что", "чтобы", "если", "когда", "пока", "хотя", "потому",
    "поскольку", "так как", "раз", "как", "будто",
    # particles
    "ли", "бы", "же", "ведь", "разве", "уж", "то", "де",
    "даже", "вон", "вот", "лишь", "только", "именно", "как-раз",
    "чуть-ли", "едва-ли",
    # modal/parenthetical
    "можно", "нельзя", "нужно", "надо", "следует", "должно",
    "может", "наверное", "возможно", "пожалуй", "кажется",
    "видимо", "якобы", "типа",
    # copulas/aux
    "быть", "есть", "нет", "являться", "становиться", "стать",
    "бывать", "оказаться", "мочь", "смочь",
    # dummy verbs
    "делать", "сделать", "делаться", "сказать", "говорить",
    "думать", "хотеть", "хотеться", "получаться",
    # interrogatives
    "где", "куда", "откуда", "зачем", "почему", "когда", "сколько", "как", "каков",
    # abstract/general nouns
    "дело", "ситуация", "случай", "момент", "период", "процесс",
    "способ", "метод", "вариант", "часть", "раз", "два", "три", "несколько", "много",
    # interjections/phrases
    "да", "нет", "ага", "угу", "ох", "ах", "ой", "эй", "эх",
    "ладно", "ок", "ну", "блин", "чёрт", "тьфу",
    "спасибо", "пожалуйста", "извините", "простите",
    "здравствуй", "привет", "пока",
    # chat noise
    "сорри", "имхо", "лол", "кек", "хех",
}

# ──────────────────────────────────────────────────────────────────────────────
# Runtime constants (used in main.py and RAG stages)
# ──────────────────────────────────────────────────────────────────────────────

log = logging.getLogger("asketmc.config")

def get_conf(name: str, default, typ=None):
    """
    Fetch a configuration variable from this module's globals with optional type casting.
    Allows overriding via monkey-patching in tests; does NOT read .env directly.
    """
    try:
        val = globals().get(name, default)
        if typ is not None and not isinstance(val, typ):
            val = typ(val)
        return val
    except Exception as e:
        log.warning("[config.get_conf] Error retrieving %s: %s", name, e)
        return default

# Retries, context limits, housekeeping
OR_RETRIES: int = 3
HTTP_CONN_LIMIT: int = 5              # max concurrent HTTP sessions
HTTP_TIMEOUT_TOTAL: int = 240         # total request timeout (seconds)
OPENROUTER_BLOCK_SEC: int = 120       # circuit-breaker base block (seconds)
OPENROUTER_BLOCK_MAX_SEC: int = 900   # circuit-breaker max block (seconds)

USER_LAST_CLEAN: int = 3600           # Cleanup interval for user_last (seconds)
EMBED_LOG_EVERY: int = 1000           # Log every N embeddings
LEMMA_CACHE_SIZE: int = 200_000       # Lemmatization cache size

# Discord and rate limiting
ALLOWED_CHANNELS: Set[int] = {1384890201544982568}  # replace with your channel ID(s)
MAX_QUESTION_LEN: int = 500
USER_COOLDOWN: int = 10  # seconds
REQUEST_SEMAPHORE: asyncio.Semaphore = asyncio.Semaphore(3)
ALLOWED_CHARS: Pattern[str] = re.compile(r"^[ а-яА-ЯёЁa-zA-Z0-9,.?!()\-—]+$")

# Administrator permissions
ADMIN_IDS: Set[int] = {267614224631463937}  # Discord user IDs allowed to run admin commands

__all__ = [
    # Paths
    "PKG_ROOT", "PROJECT_ROOT", "VAR_ROOT", "DATA_ROOT", "PROMPTS_DIR",
    "DOCS_PATH", "CACHE_PATH", "HASH_FILE", "LOG_DIR",
    "PROMPT_STRICT", "PROMPT_REASON", "PROMPT_REPHRASE",
    # Rerank
    "RERANKER_MODEL_NAME", "RERANK_INPUT_K", "RERANK_OUTPUT_K", "BATCH_SIZE",
    "MAX_LEN", "QUERY_MAX_CHARS", "EXECUTOR_WORKERS", "RERANKER_DEVICE",
    # Models / RAG params
    "TOP_K", "MIN_SCORE", "OR_MODEL", "LOCAL_MODEL", "OR_MAX_TOKENS",
    "API_URL", "OLLAMA_URL",
    "CHUNK_SIZE", "CHUNK_OVERLAP", "LEMMA_MATCH_RATIO", "SCORE_RELATIVE_THRESHOLD",
    "CTX_LEN_REMOTE", "CTX_LEN_LOCAL", "DEBUG",
    "GOOD_POS", "STOP_WORDS",
    # Runtime constants
    "get_conf", "OR_RETRIES", "HTTP_CONN_LIMIT", "HTTP_TIMEOUT_TOTAL",
    "OPENROUTER_BLOCK_SEC", "OPENROUTER_BLOCK_MAX_SEC",
    "USER_LAST_CLEAN", "EMBED_LOG_EVERY", "LEMMA_CACHE_SIZE",
    # Discord / limits / admin
    "ALLOWED_CHANNELS", "MAX_QUESTION_LEN", "USER_COOLDOWN",
    "REQUEST_SEMAPHORE", "ALLOWED_CHARS", "ADMIN_IDS",
]
