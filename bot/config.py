# config.py — configuration for the RAG bot Asketmc
from pathlib import Path
import re
import asyncio

# ─── Paths ───────────────────────────────────────────────
DOCS_PATH   = Path("C:/LLM/parsed")
CACHE_PATH  = Path("C:/LLM/rag_cache")
HASH_FILE   = CACHE_PATH / "docs_hash.json"
LOG_DIR     = Path("C:/LLM/logs")
PROMPT_STRICT  = "C:/LLM/system_prompt_strict.txt"
PROMPT_REASON  = "C:/LLM/system_prompt_reason.txt"
PROMPT_REPHRASE = "C:/LLM/rephrase.txt"

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
TOP_K        = 24
MIN_SCORE    = 0.35
OR_MODEL     = "deepseek/deepseek-chat:free"
LOCAL_MODEL  = "llama3.1:8b"
OR_MAX_TOKENS = 2048
API_URL      = "https://openrouter.ai/api/v1/chat/completions"
OLLAMA_URL   = "http://127.0.0.1:11434/api/generate"
OPENROUTER_BLOCK_SEC = 900
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
LEMMA_MATCH_RATIO = 0.2
SCORE_RELATIVE_THRESHOLD = 0.15
DEBUG = True

GOOD_POS = {
    "NOUN",   # nouns
    "ADJF",   # full adjectives
    "ADJS",   # short adjectives
    "VERB",   # verbs
    "INFN",   # infinitives
    "PRTF",   # participles
    "GRND",   # gerunds
    "NUMR",   # numerals
}

STOP_WORDS = {
    # ──────────────── pronouns (personal, demonstrative, indefinite, negative)
    "я", "ты", "он", "она", "оно", "мы", "вы", "они",
    "мой", "твой", "его", "её", "наш", "ваш", "их", "свой",
    "меня", "тебя", "него", "неё", "нас", "вас", "них",
    "мне", "тебе", "ему", "ей", "нам", "вам", "им",
    "этот", "эта", "это", "эти", "тот", "та", "то", "те",
    "такой", "такая", "такое", "такие",
    "кто", "что", "ничто", "никто", "некто", "нечто",
    "кто-то", "что-то", "кто-нибудь", "что-нибудь",
    "кто-либо", "что-либо",

    # ──────────────── prepositions
    "в", "во", "на", "за", "к", "ко", "с", "со", "от", "перед",
    "при", "об", "обо", "по", "до", "из", "иза", "без", "для",
    "над", "под", "между", "около", "через", "про", "среди",
    "из-за", "из-под", "внутри", "вне", "после", "перед", "согласно",

    # ──────────────── conjunctions
    "и", "а", "но", "да", "или", "либо", "тоже", "также",
    "что", "чтобы", "если", "когда", "пока", "хотя", "потому",
    "поскольку", "так как", "раз", "как", "будто",

    # ──────────────── particles
    "ли", "бы", "же", "ведь", "разве", "уж", "то", "де",
    "даже", "вон", "вот", "лишь", "только", "именно", "как-раз",
    "чуть-ли", "едва-ли",

    # ──────────────── parenthetical/modal words
    "можно", "нельзя", "нужно", "надо", "следует", "должно",
    "может", "наверное", "возможно", "пожалуй", "кажется",
    "видимо", "якобы", "будто", "типа",

    # ──────────────── copulas and auxiliaries
    "быть", "есть", "нет",
    "являться", "являться", "явился",
    "становиться", "стать", "стать",
    "бывать", "оказаться", "оказаться",
    "мочь", "смочь",

    # ──────────────── dummy verbs
    "делать", "сделать", "делаться", "сказать", "говорить",
    "думать", "хотеть", "хотеться", "получаться",

    # ──────────────── interrogative/refining
    "где", "куда", "откуда", "зачем", "почему",
    "когда", "сколько", "как", "каков",

    # ──────────────── abstract/general nouns
    "дело", "ситуация", "случай", "момент", "период", "процесс",
    "способ", "метод", "вариант", "часть",
    "раз", "два", "три", "несколько", "много",

    # ──────────────── interjections / phrases
    "да", "нет", "ага", "угу", "ох", "ах", "ой", "эй", "эх",
    "ладно", "ок", "ну", "блин", "чёрт", "тьфу",
    "спасибо", "пожалуйста", "извините", "простите",
    "здравствуй", "привет", "пока",

    # ──────────────── technical chat noise
    "сорри", "имхо", "лол", "кек", "хех",
}

# ─── Runtime constants (used in main.py and rag_multistage) ───────────────
import logging

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
OR_RETRIES        = 1          # Retry count for OpenRouter
CTX_LEN_REMOTE    = 12_000     # Max context length when using OpenRouter
CTX_LEN_LOCAL     = 20_000     # Max context length for local model
USER_LAST_CLEAN   = 3600       # Cleanup interval for user_last (in seconds)
EMBED_LOG_EVERY   = 1000       # Log every N embeddings
LEMMA_CACHE_SIZE  = 200_000    # Lemmatization cache size

# ─── Discord ────────────────────────────────────────────
ALLOWED_CHANNELS  = {1384890201544982568}  # Replace with your channel ID
MAX_QUESTION_LEN  = 500
USER_COOLDOWN     = 10  # sec
REQUEST_SEMAPHORE = asyncio.Semaphore(3)
ALLOWED_CHARS = re.compile(r"^[ а-яА-ЯёЁa-zA-Z0-9,.?!()\-—]+$")
HTTP_CONN_LIMIT = 5  # 10 under load

# ─── Administrator permissions ──────────────────────────
ADMIN_IDS = {267614224631463937}  # Put your admin Discord ID here. Discord user IDs allowed to run !reload_index
