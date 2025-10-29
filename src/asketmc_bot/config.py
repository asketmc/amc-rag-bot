# config.py — configuration for the RAG bot Asketmc
from pathlib import Path
import re
import asyncio
import os
import logging

# ─── Paths ───────────────────────────────────────────────
# Project root (…/LLM). This file lives in src/asketmc_bot/config.py:
# …/LLM/src/asketmc_bot/config.py → parents[1] = …/LLM/src, parents[2] = …/LLM
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Mutable artifacts (logs, caches) live outside the package in /var by default.
VAR_ROOT = Path(os.getenv("ASKETMC_VAR_DIR", PROJECT_ROOT / "var"))

# Input data (parsed KB, prompts) live in /data by default.
DATA_ROOT = Path(os.getenv("ASKETMC_DATA_DIR", PROJECT_ROOT / "data"))

# Optional: separate dir for prompt files; by default reuse DATA_ROOT.
PROMPTS_DIR = Path(os.getenv("ASKETMC_PROMPTS_DIR", DATA_ROOT))

# Knowledge base documents (parsed text/markdown).
DOCS_PATH = DATA_ROOT / "parsed"

# RAG cache and hash file for index invalidation.
CACHE_PATH = VAR_ROOT / "rag_cache"
HASH_FILE = CACHE_PATH / "docs_hash.json"

# Centralized logs directory (used by RotatingFileHandler).
LOG_DIR = VAR_ROOT / "logs"

# Prompt files (place them in /data by default; override via ASKETMC_PROMPTS_DIR).
PROMPT_STRICT = PROMPTS_DIR / "system_prompt_strict.txt"
PROMPT_REASON = PROMPTS_DIR / "system_prompt_reason.txt"
PROMPT_REPHRASE = PROMPTS_DIR / "rephrase.txt"

# ─── Rerank settings ─────────────────────────────────────
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_INPUT_K = 18
RERANK_OUTPUT_K = 9
BATCH_SIZE = 8
MAX_LEN = 512
QUERY_MAX_CHARS = 2048
EXECUTOR_WORKERS = 2
RERANKER_DEVICE = "cpu"  # or "cuda"

# ─── Models and parameters ───────────────────────────────
TOP_K = 24
MIN_SCORE = 0.35
OR_MODEL = "deepseek/deepseek-chat:free"
LOCAL_MODEL = "llama3.1:8b"
OR_MAX_TOKENS = 2048
API_URL = "https://openrouter.ai/api/v1/chat/completions"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OPENROUTER_BLOCK_SEC = 900
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
LEMMA_MATCH_RATIO = 0.2
SCORE_RELATIVE_THRESHOLD = 0.15
DEBUG = True

GOOD_POS = {
    "NOUN", "ADJF", "ADJS", "VERB", "INFN", "PRTF", "GRND", "NUMR",
}

STOP_WORDS = {
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
    "из-за", "из-под", "внутри", "вне", "после", "перед", "согласно",
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
    "видимо", "якобы", "будто", "типа",
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

# ─── Runtime constants (used in main.py and rag_multistage) ───────────────
log = logging.getLogger("asketmc.config")

def get_conf(name: str, default, typ=None):
    """
    Allows overriding config.py variables via environment or uses default.
    Type can be specified for safe casting.
    """
    try:
        val = globals().get(name, default)
        if typ and not isinstance(val, typ):
            val = typ(val)
        return val
    except Exception as e:
        log.warning("[config.get_conf] Error retrieving %s: %s", name, e)
        return default

# ─── Runtime constants (hardcoded, not loaded from .env) ───────────────
OR_RETRIES = 1           # Retry count for OpenRouter
CTX_LEN_REMOTE = 12_000  # Max context length when using OpenRouter
CTX_LEN_LOCAL = 20_000   # Max context length for local model
USER_LAST_CLEAN = 3600   # Cleanup interval for user_last (in seconds)
EMBED_LOG_EVERY = 1000   # Log every N embeddings
LEMMA_CACHE_SIZE = 200_000  # Lemmatization cache size

# ─── Discord ────────────────────────────────────────────
ALLOWED_CHANNELS = {1384890201544982568}  # Replace with your channel ID
MAX_QUESTION_LEN = 500
USER_COOLDOWN = 10  # sec
REQUEST_SEMAPHORE = asyncio.Semaphore(3)
ALLOWED_CHARS = re.compile(r"^[ а-яА-ЯёЁa-zA-Z0-9,.?!()\-—]+$")
HTTP_CONN_LIMIT = 5  # 10 under load

# ─── Administrator permissions ──────────────────────────
ADMIN_IDS = {267614224631463937}  # Put your admin Discord ID here. Discord user IDs allowed to run !reload_index
