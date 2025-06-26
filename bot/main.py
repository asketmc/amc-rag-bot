#!/usr/bin/env python3.10
"""
Main entry point for the Asketmc RAG Discord bot.

Handles configuration, startup diagnostics, and Discord bot lifecycle.
"""

from __future__ import annotations

# Standard library
import asyncio
import functools
import hashlib
import json
import logging
import os
import re
import signal
import sys
import threading
import time
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Final,
    FrozenSet,
    List,
    Optional,
    Set,
)

# Third-party libraries
import aiohttp
import discord
import spacy
import stanza
import torch
from aiohttp import ClientSession, TCPConnector
from discord.ext import commands
from dotenv import load_dotenv
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Local project modules
import config as cfg
from config import DEBUG, GOOD_POS, PROMPT_REASON, PROMPT_STRICT, STOP_WORDS
from rerank import init_reranker, rerank, shutdown_reranker

# ──────────────
# STARTUP LOGGING & ENVIRONMENT CHECKS
# ──────────────

# Print out basic startup diagnostics to the console (useful for debugging and CI/CD logs)
print(f"[STARTUP] Python: {sys.version}")                               # Display current Python version
print(f"[STARTUP] Script: {__file__}")                                 # Show the running script name
print(f"[STARTUP] Working dir: {os.getcwd()}")                         # Show the current working directory
print(f"[STARTUP] torch version: {torch.__version__}, cuda available: {torch.cuda.is_available()}")  # PyTorch/CUDA info

# Ensure the log directory exists before logging starts
try:
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)                     # Create LOG_DIR if missing
    print(f"[STARTUP] LOG_DIR created/found: {cfg.LOG_DIR}")           # Success log
except Exception as e:
    print(f"[STARTUP] Error creating LOG_DIR: {e}", file=sys.stderr)   # Error log to stderr
    sys.exit(1)                                                        # Stop execution if logging can't work

# Configure root logging: INFO level, nice format, output to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Startup logger for bootstrap phase
log = logging.getLogger("asketmc.startup")

# Initial logging: confirm imports and start of config load
log.info("[STARTUP] Module import complete.")
log.info("[STARTUP] Starting .env and config loading.")

# ────────────────────────────────
# LOADING .env
# ────────────────────────────────

# Define paths where .env files may be located (current and parent directory)
env_paths = [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / ".env"
]
env_loaded = False  # Track if any .env file is loaded

# Iterate over possible .env file locations
for dp in env_paths:
    log.info("[ENV] Checking for .env file at path: %s", dp)  # Log current check
    if dp.exists():  # If .env file exists at this path
        load_dotenv(dp)  # Load environment variables from this file
        log.info("[ENV] .env file loaded: %s", dp)  # Log success
        env_loaded = True  # Mark as loaded
        break  # Stop after first found

# If no .env file was found and loaded, log a warning
if not env_loaded:
    log.warning("[ENV] .env file not found in any of the paths: %r", [str(p) for p in env_paths])
else:
    log.info("[ENV] Environment variables loaded successfully")

# Helper function to get configuration value from config or fallback/default
def get_conf(name: str, default, typ=None):
    if hasattr(cfg, name):  # If value exists in config.py
        val = getattr(cfg, name)
        log.info("[CONSTANTS] %s found in config.py: %r", name, val)
    else:
        val = default  # Use provided default value
        log.warning("[CONSTANTS] %s not found in config.py, using default: %r", name, val)
    # If a type is required and not already satisfied, attempt casting
    if typ is not None and not isinstance(val, typ):
        try:
            val = typ(val)  # Try to cast value to specified type
            log.info("[CONSTANTS] %s cast to type %s", name, typ)
        except Exception as e:
            log.error("[CONSTANTS] %s: failed to cast to type %s: %s", name, typ, e)
    return val  # Return resolved value

# Read critical secrets or fail fast if missing (these must be set in env)
DISCORD_TOKEN: str = os.getenv("DISCORD_TOKEN") or sys.exit("DISCORD_TOKEN missing")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY") or sys.exit("OPENROUTER_API_KEY missing")

# Read and validate numeric and path constants from config/environment
OR_RETRIES: Final[int] = get_conf("OR_RETRIES", 3, int)
CTX_LEN_REMOTE: Final[int] = get_conf("CTX_LEN_REMOTE", 20_000, int)
CTX_LEN_LOCAL: Final[int] = get_conf("CTX_LEN_LOCAL", 12_000, int)
USER_LAST_CLEAN: Final[int] = get_conf("USER_LAST_CLEAN", 3600, int)
EMBED_LOG_EVERY: Final[int] = get_conf("EMBED_LOG_EVERY", 1_000, int)
LEMMA_CACHE_SIZE: Final[int] = get_conf("LEMMA_CACHE_SIZE", 200_000, int)

# Load path for lemma index file, using config path or default
LEMMA_INDEX_FILE: Final[Path] = get_conf(
    "LEMMA_INDEX_FILE", cfg.CACHE_PATH / "lemma_index.json", Path
)

# Load thresholds and sets for internal logic
SCORE_RELATIVE_THRESHOLD: Final[float] = get_conf("SCORE_RELATIVE_THRESHOLD", 0.7, float)
LEMMA_MATCH_RATIO: Final[float] = get_conf("LEMMA_MATCH_RATIO", 0.1, float)
ADMIN_IDS: Final[Set[int]] = get_conf("ADMIN_IDS", set(), set)

# Log all loaded constants for traceability
log.info("[CONSTANTS] OR_RETRIES: %d", OR_RETRIES)
log.info("[CONSTANTS] CTX_LEN_REMOTE: %d", CTX_LEN_REMOTE)
log.info("[CONSTANTS] CTX_LEN_LOCAL: %d", CTX_LEN_LOCAL)
log.info("[CONSTANTS] USER_LAST_CLEAN: %d", USER_LAST_CLEAN)
log.info("[CONSTANTS] EMBED_LOG_EVERY: %d", EMBED_LOG_EVERY)
log.info("[CONSTANTS] LEMMA_CACHE_SIZE: %d", LEMMA_CACHE_SIZE)
log.info("[CONSTANTS] LEMMA_INDEX_FILE: %s", LEMMA_INDEX_FILE)
log.info("[CONSTANTS] SCORE_RELATIVE_THRESHOLD: %.3f", SCORE_RELATIVE_THRESHOLD)
log.info("[CONSTANTS] LEMMA_MATCH_RATIO: %.3f", LEMMA_MATCH_RATIO)
log.info("[CONSTANTS] ADMIN_IDS: %r", ADMIN_IDS)

# Utility function for creating a rotating file log handler
def _rotating_handler(name: str) -> RotatingFileHandler:
    handler = RotatingFileHandler(
        cfg.LOG_DIR / name,         # Log file path
        maxBytes=10 * 1024 * 1024,  # Maximum size of log file before rotation (10MB)
        backupCount=5,              # Number of rotated log files to keep
        encoding="utf-8",           # File encoding
        delay=True,                 # Delay file creation until first write
    )
    return handler

# Ensure that the log directory exists before logging
cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

# Create handlers for application and error logs
app_handler = _rotating_handler("app.log")
err_handler = _rotating_handler("error.log")
err_handler.setLevel(logging.ERROR)  # Only log errors and above to error.log

# Create a handler for outputting logs to console (stdout)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG if getattr(cfg, "DEBUG", False) else logging.INFO)

# Set up logging configuration for the whole application
logging.basicConfig(
    level=logging.DEBUG if getattr(cfg, "DEBUG", False) else logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        app_handler,
        err_handler,
        stream_handler,
    ]
)

# Create loggers for each major subsystem/component
log = logging.getLogger("asketmc.app")
embed_log = logging.getLogger("asketmc.embed")
disc_log = logging.getLogger("asketmc.discord")
rag_log = logging.getLogger("asketmc.rag")

# Set up debug handlers and log levels depending on config
for name, lg in [
    ("app", log), ("embed", embed_log), ("discord", disc_log), ("rag", rag_log)
]:
    if getattr(cfg, "DEBUG", False):
        # If debug mode is enabled, add a dedicated debug handler
        debug_handler = _rotating_handler(f"{name}.debug.log")
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
        ))
        lg.addHandler(debug_handler)
        lg.setLevel(logging.DEBUG)
        lg.debug("[LOGGING] DEBUG handler added for %s", name)
    else:
        lg.setLevel(logging.INFO)
    # Log the ready status and level for each logger
    lg.info("[LOGGING] Logger '%s' ready, level: %s", name, lg.level)

# Prevent logs from being propagated to parent logger (avoid duplication)
for lg in (embed_log, disc_log, rag_log, log):
    lg.propagate = False

# Log the final status of logging system setup
log.info("[LOGGING] Logs are written to %s", cfg.LOG_DIR)
log.info("[LOGGING] DEBUG mode: %r", getattr(cfg, "DEBUG", False))
log.info("[LOGGING] Startup logging initialized")

# ────────────────────────────────
# LEMMATIZER: stanza (ru) + spacy (en) + langdetect
# ────────────────────────────────

# Log start of checking for Russian Stanza model
log.info("[LEMMATIZER] Checking for Stanza language model for ru...")
try:
    # Attempt to download the Russian model for Stanza if not present
    stanza.download('ru', verbose=False)
    log.info("[LEMMATIZER] Stanza 'ru' model loaded successfully.")
except Exception as e:
    # Log and exit if downloading fails
    log.exception("[LEMMATIZER] Error loading Stanza model: %s", e)
    raise SystemExit(1)

# Log start of Stanza pipeline initialization
log.info("[LEMMATIZER] Initializing Stanza.Pipeline (ru)...")
try:
    # Create a Stanza pipeline for Russian language with tokenization, POS tagging, and lemmatization
    STANZA_NLP_RU = stanza.Pipeline(
        lang='ru',
        processors='tokenize,pos,lemma',
        use_gpu=False,
        verbose=False,
    )
    log.info("[LEMMATIZER] Stanza.Pipeline initialized successfully.")
except Exception as e:
    # Log and exit if pipeline initialization fails
    log.exception("[LEMMATIZER] Error initializing Stanza.Pipeline: %s", e)
    raise SystemExit(1)

# Log start of spaCy initialization for English
log.info("[LEMMATIZER] Initializing spaCy (en)...")
try:
    import spacy
    # Load the small English language model for spaCy
    SPACY_EN = spacy.load("en_core_web_sm")
    log.info("[LEMMATIZER] spaCy 'en_core_web_sm' loaded successfully.")
except Exception as e:
    # Log and exit if loading spaCy fails
    log.exception("[LEMMATIZER] Error loading spaCy model: %s", e)
    raise SystemExit(1)

try:
    # Create a threading lock to protect shared resources for Stanza (thread safety)
    _LEMMA_LOCK = threading.Lock()
    # Set max_workers for thread pool: min(CPU count or 4, 8)
    max_workers = min(os.cpu_count() or 4, 8)
    # Create a thread pool executor for parallel lemma extraction
    LEMMA_POOL = ThreadPoolExecutor(max_workers=max_workers)
    log.info("[LEMMATIZER] ThreadPoolExecutor started with max_workers=%d", max_workers)
except Exception as e:
    # Log and exit if thread pool creation fails
    log.exception("[LEMMATIZER] Error creating ThreadPoolExecutor: %s", e)
    raise SystemExit(1)

# Cache the function for performance on repeated calls with the same input
@functools.lru_cache(maxsize=10_000)
def _extract_lemmas(text: str) -> FrozenSet[str]:
    # Log input text length and a preview (up to 120 chars, no line breaks)
    log.debug("[_extract_lemmas] Input text (%d characters): %r", len(text), text[:120].replace("\n", " "))
    try:
        # Detect language using langdetect
        detected_lang = detect(text)
        log.debug("[_extract_lemmas] Language detected: %s", detected_lang)
    except LangDetectException as e:
        # If language detection fails, default to Russian and log warning
        detected_lang = "ru"
        log.warning("[_extract_lemmas] Language detection error: %s, language forcibly set to 'ru'", e)
    except Exception as e:
        # For unexpected errors, also default to Russian and log error
        detected_lang = "ru"
        log.error("[_extract_lemmas] Unknown language detection error: %s", e)

    # Only recognize English; anything else is Russian
    lang = "en" if detected_lang == "en" else "ru"
    log.debug("[_extract_lemmas] Language after normalization: %s", lang)

    if lang == "en":
        try:
            # Process English text with spaCy
            doc = SPACY_EN(text)
            log.debug("[_extract_lemmas] spaCy processed document: tokens=%d", len(doc))
            # Extract lowercase lemmas for alphabetic tokens that are not stop words and have length > 2
            lemmas = {
                tok.lemma_.lower()
                for tok in doc
                if tok.is_alpha and not tok.is_stop and len(tok) > 2
            }
            # Log the count and a sample of extracted lemmas
            log.debug("[_extract_lemmas] (en) Number of lemmas: %d, examples: %r", len(lemmas), list(lemmas)[:10])
            return frozenset(lemmas)
        except Exception as e:
            # Log processing error and return empty set on failure
            log.exception("[_extract_lemmas] spaCy processing error: %s", e)
            return frozenset()

    try:
        # For Russian, use Stanza pipeline (protected by thread lock)
        with _LEMMA_LOCK:
            log.debug("[_extract_lemmas] Entered lock for Stanza processing")
            doc = STANZA_NLP_RU(text)
            log.debug("[_extract_lemmas] Stanza returned %d sentences", len(doc.sentences))
    except Exception as e:
        # Log Stanza processing error and return empty set on failure
        log.exception("[_extract_lemmas] Stanza processing error: %s", e)
        return frozenset()

    try:
        # Extract lemmas from Russian text, filtering by:
        # 1. Non-empty lemma,
        # 2. Lemma length > 2,
        # 3. Lemma POS tag in GOOD_POS,
        # 4. Lemma not in STOP_WORDS
        lemmas = {
            w.lemma.lower()
            for s in doc.sentences
            for w in s.words
            if w.lemma
               and len(w.lemma) > 2
               and w.upos in GOOD_POS
               and w.lemma.lower() not in STOP_WORDS
        }
        # Log the count and a sample of extracted lemmas
        log.debug("[_extract_lemmas] (ru) Number of lemmas: %d, examples: %r", len(lemmas), list(lemmas)[:10])
        return frozenset(lemmas)
    except Exception as e:
        # Log any errors constructing the lemma set, return empty set on failure
        log.exception("[_extract_lemmas] Error constructing lemma set: %s", e)
        return frozenset()

# ────────────────────────────────
# CHUNK CACHE LOGIC
# ────────────────────────────────

# Initialize logger with an explicit, module-level name for traceability in multi-module apps
log = logging.getLogger("asketmc.lemma")

# Default file for chunk lemma cache; ensures all cache logic is localized
CHUNK_LEMMA_CACHE_FILE = Path("rag_cache/chunk_lemma_index.json")
# In-memory cache: maps chunk hash (SHA-256 hex string) to a frozenset of lemmas
CHUNK_LEMMA_CACHE: Dict[str, FrozenSet[str]] = {}

# File lemma cache (used for full-file lemmatization, not just for chunks)
FILE_LEMMAS: Dict[str, FrozenSet[str]] = {}
LEMMA_INDEX_FILE = Path("rag_cache/lemma_index.json")

def chunk_hash(text: str) -> str:
    """
    Returns a SHA-256 hash for the given text (used as chunk ID).
    Logging for traceability of both input and result.
    """
    log.debug("[chunk_hash] Input text (len=%d): %r", len(text), text[:40])
    # Calculate SHA-256 hash; ensures uniqueness for cache key
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    log.debug("[chunk_hash] Got hash='%s' for text of length %d", h, len(text))
    return h

def load_chunk_lemma_cache(
    chunk_cache_file: Path = CHUNK_LEMMA_CACHE_FILE
) -> None:
    """
    Loads chunk lemma cache from the specified file (default: CHUNK_LEMMA_CACHE_FILE).
    Populates the global CHUNK_LEMMA_CACHE. Logs operation at all stages.
    """
    global CHUNK_LEMMA_CACHE
    log.info("[load_chunk_lemma_cache] Loading chunk cache from %s", chunk_cache_file)
    if chunk_cache_file.exists():
        try:
            with chunk_cache_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # Rebuild frozensets from lists (JSON does not support sets)
            CHUNK_LEMMA_CACHE = {k: frozenset(v) for k, v in data.items()}
            log.info("[load_chunk_lemma_cache] Loaded %d chunks.", len(CHUNK_LEMMA_CACHE))
            log.debug("[load_chunk_lemma_cache] Keys: %r", sorted(CHUNK_LEMMA_CACHE.keys()))
        except Exception as e:
            log.error("[load_chunk_lemma_cache] Load error: %s", e, exc_info=True)
            CHUNK_LEMMA_CACHE = {}
    else:
        CHUNK_LEMMA_CACHE = {}
        log.info("[load_chunk_lemma_cache] File not found: %s", chunk_cache_file)

def save_chunk_lemma_cache(
    chunk_cache_file: Path = CHUNK_LEMMA_CACHE_FILE
) -> None:
    """
    Saves the current in-memory chunk lemma cache to the specified file.
    Ensures parent directories exist; handles and logs any exceptions.
    """
    log.info("[save_chunk_lemma_cache] Saving chunk cache to %s", chunk_cache_file)
    # JSON can't serialize sets, so we convert to lists
    data = {k: list(v) for k, v in CHUNK_LEMMA_CACHE.items()}
    try:
        chunk_cache_file.parent.mkdir(parents=True, exist_ok=True)
        with chunk_cache_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log.info("[save_chunk_lemma_cache] Saved %d chunks.", len(CHUNK_LEMMA_CACHE))
        log.debug("[save_chunk_lemma_cache] Keys: %r", sorted(CHUNK_LEMMA_CACHE.keys()))
    except Exception as e:
        log.error("[save_chunk_lemma_cache] Save error: %s", e, exc_info=True)

def get_lemmas_for_chunk(text: str, lemma_func) -> FrozenSet[str]:
    """
    Retrieves lemma set for a chunk of text.
    Uses cache if available; otherwise calls lemma_func and updates cache.
    All stages are logged. Errors result in empty lemma set (not None).
    """
    h = chunk_hash(text)
    log.debug("[get_lemmas_for_chunk] hash=%s, text length=%d", h, len(text))
    lemmas = CHUNK_LEMMA_CACHE.get(h)
    if lemmas is None:
        log.info("[get_lemmas_for_chunk] Not in cache, calculating for hash=%s", h)
        try:
            # lemma_func should return a frozenset of lemmas
            lemmas = lemma_func(text)
            CHUNK_LEMMA_CACHE[h] = lemmas
            log.info("[get_lemmas_for_chunk] Added %d lemmas for hash=%s", len(lemmas), h)
        except Exception as e:
            log.error("[get_lemmas_for_chunk] Lemmatization error for hash=%s: %s", h, e, exc_info=True)
            lemmas = frozenset()
            CHUNK_LEMMA_CACHE[h] = lemmas
    else:
        log.info("[get_lemmas_for_chunk] Using cache with %d lemmas for hash=%s", len(lemmas), h)
    # Post-condition: hash should always be in cache at this point
    if h not in CHUNK_LEMMA_CACHE:
        log.error("[get_lemmas_for_chunk] CRITICAL: hash %s not found in cache after addition!", h)
    else:
        log.debug("[get_lemmas_for_chunk] Validation: hash %s present, len=%d", h, len(CHUNK_LEMMA_CACHE[h]))
    return lemmas

# ────────────────────────────────
# FILE LEMMA INDEXING AND VECTOR INDEX BUILD
# ────────────────────────────────

def _read_file(fp: Path) -> str:
    """
    Read the contents of a file as UTF-8 text, ignoring errors.
    Returns file content as a string or empty string on failure.
    """
    log.debug("[_read_file] Reading file: %s", fp)
    try:
        # Safe, cross-platform read; ignore invalid characters
        data = fp.read_text("utf-8", "ignore")
        log.debug("[_read_file] Read %d characters from %s", len(data), fp)
        return data
    except Exception as e:
        log.error("[_read_file] Read error %s: %s", fp, e)
        return ""

async def _compute_and_store_lemmas(fp: Path) -> None:
    """
    Asynchronously read file, extract lemmas, and store in FILE_LEMMAS.
    Lemmatization errors result in an empty lemma set for the file.
    """
    log.info("[_compute_and_store_lemmas] Reading file: %s", fp)
    try:
        loop = asyncio.get_event_loop()
        # Offload synchronous IO to thread pool for non-blocking async
        text = await loop.run_in_executor(None, _read_file, fp)
        log.debug("[_compute_and_store_lemmas] Read %d chars from %s", len(text), fp)
    except Exception as e:
        log.error("[_compute_and_store_lemmas] Read error %s: %s", fp, e, exc_info=True)
        text = ""
    try:
        # _extract_lemmas should return frozenset for serialization consistency
        lemmas = _extract_lemmas(text)
        FILE_LEMMAS[fp.name] = lemmas
        log.info("[_compute_and_store_lemmas] %s: %d lemmas, %r", fp, len(lemmas), list(lemmas)[:10])
    except Exception as e:
        log.error("[_compute_and_store_lemmas] Lemmatization error %s: %s", fp, e, exc_info=True)
        FILE_LEMMAS[fp.name] = frozenset()

def _persist_lemmas(lemma_file: Path = None) -> None:
    """
    Persist FILE_LEMMAS dict to disk as JSON (file -> list of lemmas).
    Parent directories are created as needed. Exceptions are logged.
    """
    if lemma_file is None:
        lemma_file = LEMMA_INDEX_FILE
    try:
        log.info("[_persist_lemmas] Saving lemma cache to %s...", lemma_file)
        lemma_file.parent.mkdir(parents=True, exist_ok=True)
        # JSON requires lists, not sets
        dump = {k: list(v) for k, v in FILE_LEMMAS.items()}
        lemma_file.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("[_persist_lemmas] Lemma cache saved. Files: %d", len(FILE_LEMMAS))
    except Exception as e:
        log.error("[_persist_lemmas] Error saving lemma cache: %s", e, exc_info=True)

def _load_saved_lemmas() -> None:
    """
    Load lemma cache from LEMMA_INDEX_FILE into FILE_LEMMAS.
    If file does not exist or fails to load, FILE_LEMMAS is not modified.
    """
    if not LEMMA_INDEX_FILE.exists():
        log.info("[_load_saved_lemmas] Lemma cache not found (%s)", LEMMA_INDEX_FILE)
        return
    try:
        log.info("[_load_saved_lemmas] Reading lemma cache from %s...", LEMMA_INDEX_FILE)
        data = json.loads(LEMMA_INDEX_FILE.read_text("utf-8"))
        # Restore frozenset for consistency with in-memory model
        for fname, lst in data.items():
            FILE_LEMMAS[fname] = frozenset(lst)
        log.info("[_load_saved_lemmas] Loaded lemmas for %d files.", len(FILE_LEMMAS))
    except Exception as e:
        log.warning("[_load_saved_lemmas] Lemma cache read error: %s", e, exc_info=True)

async def update_file_lemmas_async(
    docs: List[Path], stored_hashes: Dict[str, str], new_hashes: Dict[str, str]
) -> List[Path]:
    """
    Asynchronously re-calculate and update lemmas for all files where hash differs.
    Returns a list of changed file Path objects.
    """
    # Determine changed files by comparing old and new hashes by filename
    changed = [d for d in docs if stored_hashes.get(d.name) != new_hashes.get(d.name)]
    log.info("[update_file_lemmas_async] Changed files: %r", [d.name for d in changed])
    if not changed:
        log.info("[update_file_lemmas_async] No changed files")
        return []
    tasks = []
    for d in changed:
        log.info("[update_file_lemmas_async] Starting lemma task for %s", d)
        tasks.append(asyncio.create_task(_compute_and_store_lemmas(d)))
    # Wait for all async lemma computation tasks to complete
    await asyncio.gather(*tasks)
    log.info("[update_file_lemmas_async] All lemma tasks finished, saving lemmas")
    _persist_lemmas()
    return changed

# ────────────────────────────────
# VECTOR INDEX BUILD AND EMBEDDINGS INITIALIZATION
# ────────────────────────────────

async def build_index() -> VectorStoreIndex:
    """
    Builds or loads a vector index for documents from cfg.DOCS_PATH with lemmatization.
    Includes: caches, valid metadata, change processing, saving and validation.
    """
    log.info("build_index: Scanning documents in %s...", cfg.DOCS_PATH)
    docs = list(cfg.DOCS_PATH.glob("*"))
    log.info("build_index: Found %d files: %r", len(docs), [p.name for p in docs])

    # 1. Calculate hashes for all documents
    log.info("build_index: Calculating file hashes...")
    hashes = {d.name: _doc_hash(d) for d in docs}
    log.debug("build_index: File hashes: %r", hashes)

    # 2. Load old hashes
    if cfg.HASH_FILE.exists():
        log.info("build_index: Loading saved hashes from %s", cfg.HASH_FILE)
        try:
            stored = json.loads(cfg.HASH_FILE.read_text("utf-8"))
        except Exception as e:
            log.error("build_index: Error reading hash file: %s", e, exc_info=True)
            stored = {}
    else:
        log.info("build_index: Hash file not found, building from scratch.")
        stored = {}

    # 3. Load lemma cache
    log.info("build_index: Loading lemma cache (for files)...")
    try:
        _load_saved_lemmas()
    except Exception as e:
        log.error("build_index: Error loading lemma cache: %s", e, exc_info=True)

    # 4. Update lemmas only for changed files
    log.info("build_index: Updating lemmas of changed files...")
    try:
        changed = await update_file_lemmas_async(docs, stored, hashes)
        log.info("build_index: Updated file lemmas: %r", [f.name for f in changed])
    except Exception as e:
        log.error("build_index: Error updating file lemmas: %s", e, exc_info=True)
        changed = []

    # 5. Load or build index
    if cfg.CACHE_PATH.exists() and stored == hashes:
        log.info("build_index: Index cache is up to date, loading from %s", cfg.CACHE_PATH)
        try:
            idx = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=str(cfg.CACHE_PATH))
            )
        except Exception as e:
            log.error("build_index: Error loading index: %s", e, exc_info=True)
            idx = None
    else:
        log.info("build_index: Cache not found or outdated, building new index...")
        ll_docs = []
        for fp in docs:
            log.debug("[build_index] Reading file: %s", fp)
            try:
                text = fp.read_text("utf-8", "ignore")
                ll_docs.append(Document(text=text, metadata={"file_name": fp.name}))
            except Exception as e:
                log.error("build_index: Error reading file %s: %s", fp, e, exc_info=True)
        log.info("build_index: Built %d documents, building VectorStoreIndex...", len(ll_docs))
        try:
            idx = VectorStoreIndex.from_documents(ll_docs)
            log.info("build_index: Index built.")
        except Exception as e:
            log.error("build_index: Error building index: %s", e, exc_info=True)
            idx = None

    # 6. Assign lemmas to all chunk metadata
    if idx is None:
        log.error("build_index: Index was not built, exiting.")
        return None

    log.info("build_index: Adding chunk lemmas to metadata...")
    cnt = 0
    for node in idx.docstore.docs.values():
        try:
            chunk_text = node.get_content()
            lem = get_lemmas_for_chunk(chunk_text, _extract_lemmas)
            node.metadata["lemmas"] = list(lem)
            log.debug("Lemmas for %s: %r", node.metadata.get("file_name", ""), list(lem))
            cnt += 1
        except Exception as e:
            log.error("build_index: Error adding lemmas for node: %s", e, exc_info=True)
    log.info("[build_index] Updated chunks: %d", cnt)

    # 7. Log size of chunk cache
    log.info("[build_index] Keys in CHUNK_LEMMA_CACHE: %d", len(CHUNK_LEMMA_CACHE))
    log.debug("[build_index] All CHUNK_LEMMA_CACHE keys: %r", sorted(CHUNK_LEMMA_CACHE.keys()))

    # 8. Save index to disk
    log.info("build_index: Saving index to %s", cfg.CACHE_PATH)
    try:
        idx.storage_context.persist(str(cfg.CACHE_PATH))
    except Exception as e:
        log.error("build_index: Error saving index: %s", e, exc_info=True)

    # 9. Save actual hashes
    log.info("build_index: Saving hashes to %s", cfg.HASH_FILE)
    try:
        cfg.HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
        cfg.HASH_FILE.write_text(
            json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        log.error("build_index: Error saving hash file: %s", e, exc_info=True)

    log.info("build_index: Done.")

    # 10. Save and validate chunk cache
    save_chunk_lemma_cache()
    if not CHUNK_LEMMA_CACHE_FILE.exists():
        log.error("[build_index] CRITICAL: Cache file %s did not appear after saving!",
                  CHUNK_LEMMA_CACHE_FILE)
    else:
        try:
            with open(CHUNK_LEMMA_CACHE_FILE, "r", encoding="utf-8") as f:
                persisted = json.load(f)
            log.info("[build_index] Post-save check: %d keys actually saved.",
                     len(persisted))
            log.debug("[build_index] Check keys in file: %r", sorted(persisted.keys()))
        except Exception as e:
            log.error("[build_index] Error validating cache file: %s", e, exc_info=True)

    return idx

# Embedding and SentenceSplitter initialization (English logs)

log.info("[EMBEDDINGS] torch.__version__ = %s", torch.__version__)
log.info("[EMBEDDINGS] torch.cuda.is_available() = %r", torch.cuda.is_available())
log.info("[EMBEDDINGS] torch.cuda.device_count() = %d", torch.cuda.device_count())
if torch.cuda.is_available():
    log.info("[EMBEDDINGS] torch.cuda.get_device_name(0) = %s", torch.cuda.get_device_name(0))
else:
    log.info("[EMBEDDINGS] CUDA not found, using CPU.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info("[EMBEDDINGS] Initializing model BAAI/bge-m3 on device: %s", DEVICE)

class LoggingBGE(HuggingFaceEmbedding):
    _counter: int = 0
    __slots__ = ()

    def _get_text_embedding(self, text: str):
        log.debug("[LoggingBGE] EMB #%d, device=%s, text preview=%r",
                  type(self)._counter + 1, self._target_device, text[:80].replace("\n", " "))
        t0 = time.time()
        vec = super()._get_text_embedding(text)
        t1 = time.time()
        type(self)._counter += 1
        log.debug("[LoggingBGE] EMB #%d, len(text)=%d, vec[:8]=%r, %.3fs",
                  type(self)._counter, len(text), vec[:8], t1 - t0)
        if type(self)._counter % EMBED_LOG_EVERY == 0:
            embed_log.info(
                "EMB %d | dev=%s | %s | %s… | %.3fs",
                type(self)._counter,
                self._target_device,
                text[:120].replace("\n", " "),
                str(vec[:8]),
                t1 - t0,
            )
        return vec

try:
    Settings.embed_model = LoggingBGE("BAAI/bge-m3", normalize=True, device=DEVICE)
    log.info("[EMBEDDINGS] LoggingBGE successfully initialized")
except Exception as e:
    log.error("[EMBEDDINGS] LoggingBGE initialization error: %s", e, exc_info=True)
    raise

try:
    Settings.node_parser = SentenceSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
        include_metadata=False,
        paragraph_separator="\n\n",
    )
    log.info("[EMBEDDINGS] SentenceSplitter successfully initialized: chunk_size=%d, chunk_overlap=%d",
             cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
except Exception as e:
    log.error("[EMBEDDINGS] SentenceSplitter initialization error: %s", e, exc_info=True)
    raise

# ────────────────────────────────
# VECTOR INDEX
# ────────────────────────────────

def _doc_hash(fp: Path) -> str:
    """
    Compute a SHA-256 hash for a file's contents.

    Args:
        fp (Path): Path to the file.

    Returns:
        str: Hex digest of the file's SHA-256 hash.
    """
    log.debug("Hashing file: %s", fp)
    h = hashlib.sha256()
    # Read file in 1 MB chunks to handle large files efficiently
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    hex_digest = h.hexdigest()
    log.debug("Hash for %s: %s", fp, hex_digest)
    return hex_digest

async def build_index() -> VectorStoreIndex:
    """
    Build or load a vector index for documents in cfg.DOCS_PATH.
    Handles hash validation, lemma cache, metadata updates, and persistence.

    Returns:
        VectorStoreIndex: The built or loaded index.
    """
    # 1. Gather document paths
    log.info("build_index: Scanning documents in %s...", cfg.DOCS_PATH)
    docs = list(cfg.DOCS_PATH.glob("*"))
    log.info("build_index: Found %d files: %r", len(docs), [p.name for p in docs])

    # 2. Calculate file hashes
    log.info("build_index: Calculating file hashes...")
    hashes = {d.name: _doc_hash(d) for d in docs}
    log.debug("build_index: File hashes: %r", hashes)

    # 3. Load stored hashes, if present
    if cfg.HASH_FILE.exists():
        log.info("build_index: Loading stored hashes from %s", cfg.HASH_FILE)
        try:
            stored = json.loads(cfg.HASH_FILE.read_text("utf-8"))
        except Exception as e:
            log.error("build_index: Error reading hash file: %s", e, exc_info=True)
            stored = {}
    else:
        log.info("build_index: Hash file not found, building from scratch.")
        stored = {}

    # 4. Load saved lemmas (file-level)
    log.info("build_index: Loading lemma cache (for files)...")
    try:
        _load_saved_lemmas()
    except Exception as e:
        log.error("build_index: Error loading lemma cache: %s", e, exc_info=True)

    # 5. Update lemmas for changed files
    log.info("build_index: Updating lemmas of changed files...")
    try:
        changed = await update_file_lemmas_async(docs, stored, hashes)
        log.info("build_index: Updated file lemmas: %r", [f.name for f in changed])
    except Exception as e:
        log.error("build_index: Error updating file lemmas: %s", e, exc_info=True)
        changed = []

    # 6. Load or build the index
    idx = None
    if cfg.CACHE_PATH.exists() and stored == hashes:
        log.info("build_index: Index cache is up to date, loading from %s", cfg.CACHE_PATH)
        try:
            idx = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=str(cfg.CACHE_PATH))
            )
        except Exception as e:
            log.error("build_index: Error loading index: %s", e, exc_info=True)
            idx = None

    if idx is None:
        log.info("build_index: Cache not found or outdated, building new index...")
        ll_docs = []
        for fp in docs:
            log.debug("build_index: Reading file: %s", fp)
            try:
                text = fp.read_text("utf-8", "ignore")
                ll_docs.append(Document(text=text, metadata={"file_name": fp.name}))
            except Exception as e:
                log.error("build_index: Error reading file %s: %s", fp, e, exc_info=True)
        log.info("build_index: Built %d documents, building VectorStoreIndex...", len(ll_docs))
        try:
            idx = VectorStoreIndex.from_documents(ll_docs)
            log.info("build_index: Index built.")
        except Exception as e:
            log.error("build_index: Error building index: %s", e, exc_info=True)
            raise

    # 7. Assign lemma metadata to each node (chunk)
    log.info("build_index: Adding lemmas to document (chunk) metadata...")
    missing_in_cache = 0
    missing_file_name = 0
    total_nodes = len(idx.docstore.docs)
    for i, node in enumerate(idx.docstore.docs.values()):
        fname = node.metadata.get("file_name", None)
        if not fname:
            log.warning(
                "build_index: Skipping node without file_name! node.metadata=%r",
                node.metadata,
            )
            missing_file_name += 1
            continue
        if fname in FILE_LEMMAS:
            lem = FILE_LEMMAS[fname]
            log.debug(
                "build_index: Lemmas for file %r taken from cache (%d items)",
                fname, len(lem)
            )
        else:
            log.warning(
                "build_index: Lemmas not in cache for file %r, recalculating.",
                fname
            )
            text = node.get_content()
            lem = _extract_lemmas(text)
            missing_in_cache += 1
        node.metadata["lemmas"] = list(lem)
        log.debug("build_index: Lemmas for %s: %r", fname, list(lem))

    log.info(
        "build_index: Skipped nodes without file_name: %d of %d",
        missing_file_name, total_nodes,
    )
    log.info(
        "build_index: Files missing cache, lemmas recalculated: %d",
        missing_in_cache,
    )

    # 8. Persist the index to disk
    log.info("build_index: Saving index to %s", cfg.CACHE_PATH)
    try:
        idx.storage_context.persist(str(cfg.CACHE_PATH))
    except Exception as e:
        log.error("build_index: Error saving index: %s", e, exc_info=True)

    # 9. Persist hashes to disk
    log.info("build_index: Saving hashes to %s", cfg.HASH_FILE)
    try:
        cfg.HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
        cfg.HASH_FILE.write_text(
            json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        log.error("build_index: Error saving hash file: %s", e, exc_info=True)

    log.info(
        "build_index: Completed. Index contains %d chunks.",
        total_nodes - missing_file_name,
    )
    return idx

# ────────────────────────────────
# FILTER NODE
# ────────────────────────────────

# Module-level cache for filtered nodes: key is str, value is list of NodeWithScore
FILTER_CACHE: Dict[str, List[NodeWithScore]] = {}

def _cache_key(qlem: FrozenSet[str]) -> str:
    """
    Compute a cache key for a query lemma set.

    Args:
        qlem (FrozenSet[str]): Lemmas representing the query.

    Returns:
        str: SHA-256 hash hex digest as cache key.
    """
    key = hashlib.sha256(" ".join(sorted(qlem)).encode("utf-8")).hexdigest()
    rag_log.debug("[_cache_key] qlem=%r, key=%s", list(qlem), key)
    return key

async def _filter_nodes(
    raw_nodes: List[NodeWithScore], qlem: FrozenSet[str]
) -> List[NodeWithScore]:
    """
    Filter and rank nodes based on score and lemma intersection.

    Args:
        raw_nodes (List[NodeWithScore]): Input nodes with scores.
        qlem (FrozenSet[str]): Query lemmas.

    Returns:
        List[NodeWithScore]: Top filtered nodes.
    """
    rag_log.info("[_filter_nodes] Filtering started. raw_nodes: %d, qlem=%r", len(raw_nodes), list(qlem))
    if not raw_nodes:
        rag_log.warning("[_filter_nodes] No raw_nodes provided, returning empty list.")
        return []

    max_score = max(n.score for n in raw_nodes)
    rag_log.debug("[_filter_nodes] max_score=%.4f", max_score)

    alpha = getattr(cfg, "FILTER_ALPHA", 0.5)
    strict: List[Tuple[NodeWithScore, float]] = []
    for idx, n in enumerate(raw_nodes):
        lemmas = frozenset(n.node.metadata.get("lemmas", []))
        inter = len(qlem & lemmas)
        rel_score = n.score / (max_score or 1)
        ratio = inter / (len(qlem) or 1)
        weight = n.score + alpha * ratio
        rag_log.debug(
            "[_filter_nodes] idx=%d file=%r score=%.4f rel_score=%.3f inter=%d ratio=%.3f weight=%.3f",
            idx, n.node.metadata.get("file_name", "n/a"), n.score, rel_score, inter, ratio, weight
        )
        if (
            rel_score >= getattr(cfg, "SCORE_RELATIVE_THRESHOLD", 0.7)
            or ratio >= getattr(cfg, "LEMMA_MATCH_RATIO", 0.1)
        ):
            strict.append((n, weight))

    rag_log.debug("[_filter_nodes] After strict filtering: strict=%d", len(strict))
    if strict:
        strict.sort(key=lambda x: -x[1])
        for i, (n, weight) in enumerate(strict[:cfg.TOP_K]):
            rag_log.debug(
                "[_filter_nodes] TOP #%d: %r score=%.4f weight=%.3f",
                i, n.node.metadata.get("file_name", "n/a"), n.score, weight
            )
        rag_log.info("[_filter_nodes] strict=%d, returning top results.", len(strict))
        return [n for n, _ in strict[:cfg.TOP_K]]

    # Fallback: top-K by weight with at least one intersecting lemma
    fallback: List[Tuple[NodeWithScore, float, int, float]] = []
    for idx, n in enumerate(raw_nodes):
        lemmas = frozenset(n.node.metadata.get("lemmas", []))
        inter = len(qlem & lemmas)
        ratio = inter / (len(qlem) or 1)
        weight = n.score + alpha * ratio
        if inter > 0:
            fallback.append((n, weight, inter, n.score))
        rag_log.debug(
            "[_filter_nodes-fallback] idx=%d file=%r score=%.4f inter=%d weight=%.3f",
            idx, n.node.metadata.get("file_name", "n/a"), n.score, inter, weight
        )

    fallback.sort(key=lambda x: -x[1])
    rag_log.info("[_filter_nodes-fallback] Fallback items: %d", len(fallback))
    for i, (n, weight, inter, score) in enumerate(fallback[:cfg.TOP_K]):
        rag_log.debug(
            "[_filter_nodes-fallback] TOP #%d: %r score=%.4f inter=%d weight=%.3f",
            i, n.node.metadata.get("file_name", "n/a"), score, inter, weight
        )
    result = [n for n, _, _, _ in fallback[:cfg.TOP_K]]
    rag_log.info("[_filter_nodes-fallback] Returning top-%d fallback.", len(result))
    return result

async def get_filtered_nodes(
    raw_nodes: List[NodeWithScore], qlem: FrozenSet[str]
) -> List[NodeWithScore]:
    """
    Retrieve filtered nodes from cache or perform filtering if not cached.

    Args:
        raw_nodes (List[NodeWithScore]): Nodes to filter.
        qlem (FrozenSet[str]): Lemmas for query.

    Returns:
        List[NodeWithScore]: Filtered nodes.
    """
    key = _cache_key(qlem)
    rag_log.debug("[get_filtered_nodes] Checking cache for key=%s", key)
    if key in FILTER_CACHE:
        rag_log.info("[get_filtered_nodes] Using cache for key %s", key)
        rag_log.debug("[get_filtered_nodes] Cached node count=%d", len(FILTER_CACHE[key]))
        return FILTER_CACHE[key]
    rag_log.info("[get_filtered_nodes] No cache, filtering.")
    nodes = await _filter_nodes(raw_nodes, qlem)
    FILTER_CACHE[key] = nodes
    rag_log.info(
        "[get_filtered_nodes] Stored filter result in cache for key %s (nodes=%d)", key, len(nodes)
    )
    return nodes

def _content_from_node(n: NodeWithScore) -> str:
    """
    Safely extract content from a node.

    Args:
        n (NodeWithScore): Node with content.

    Returns:
        str: Extracted text (may be empty if not available).
    """
    txt = getattr(n.node, "get_content", lambda: "")()
    rag_log.debug(
        "[_content_from_node] %r, len=%d", n.node.metadata.get("file_name", "n/a"), len(txt)
    )
    return txt

def build_context(
    nodes: List[NodeWithScore], qlem: FrozenSet[str], char_limit: int
) -> str:
    """
    Build a response context by concatenating unique node contents, up to a character limit.

    Args:
        nodes (List[NodeWithScore]): Nodes to use as context.
        qlem (FrozenSet[str]): Query lemmas for weighting.
        char_limit (int): Maximum length of the context.

    Returns:
        str: Composed multi-part context string.
    """
    rag_log.info(
        "[build_context] Building context from %d nodes, char_limit=%d", len(nodes), char_limit
    )
    beta = 0.1
    scored: List[Tuple[NodeWithScore, float]] = []
    for n in nodes:
        lemmas = frozenset(n.node.metadata.get("lemmas", []))
        inter = len(qlem & lemmas)
        weight = n.score + beta * inter
        scored.append((n, weight))
        rag_log.debug(
            "[build_context] node=%r score=%.4f inter=%d weight=%.4f",
            n.node.metadata.get("file_name", "n/a"), n.score, inter, weight
        )
    scored.sort(key=lambda x: x[1], reverse=True)
    rag_log.debug("[build_context] Sorted by weight.")

    parts: List[str] = []
    seen_hashes: Set[int] = set()
    total = 0
    for idx, (n, weight) in enumerate(scored):
        txt = _content_from_node(n).strip()
        if not txt:
            rag_log.debug("[build_context] SKIP empty content: %r", n.node.metadata.get("file_name", "n/a"))
            continue
        h = hash(txt)
        if h in seen_hashes:
            rag_log.debug("[build_context] SKIP duplicate: %r", n.node.metadata.get("file_name", "n/a"))
            continue
        if total + len(txt) > char_limit:
            rag_log.info("[build_context] Char limit reached (%d), stopping.", total)
            break
        seen_hashes.add(h)
        parts.append(txt)
        total += len(txt) + 4  # 4 for separator
        rag_log.debug(
            "[build_context] Added node=%r (len=%d, total=%d)", n.node.metadata.get("file_name", "n/a"), len(txt), total
        )
    rag_log.info("[build_context] Final context: parts=%d, total size=%d", len(parts), total)
    return "\n---\n".join(parts)

# ────────────────────────────────
# DISCORD-UTILITY
# ────────────────────────────────

def split_message(text: str, limit: int = 2000) -> List[str]:
    """
    Split a long message into chunks suitable for Discord.

    Args:
        text (str): The input message string.
        limit (int, optional): Maximum chunk length. Defaults to 2000.

    Returns:
        List[str]: List of message chunks.
    """
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
    """
    Send a long message in multiple chunks if necessary.

    Args:
        ctx (commands.Context): Discord context for sending messages.
        text (str): The message content.
        limit (int, optional): Max chunk length. Defaults to 1900.
    """
    for chunk in split_message(text, limit):
        await ctx.send(chunk)

# ────────────────────────────────
# DISCORD BOT: GLOBALS & SESSION
# ────────────────────────────────

# Setup bot and global state
bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())
user_last: Dict[int, float] = {}
index: Optional["VectorStoreIndex"] = None
retriever: Any = None
openrouter_until = 0.0

class AsyncSessionHolder:
    """
    Thread-safe async holder for a reusable aiohttp ClientSession.
    """
    def __init__(self):
        self._lock = asyncio.Lock()
        self._session: Optional[ClientSession] = None
        log.debug("[AsyncSessionHolder] Initialized.")

    async def get(self) -> ClientSession:
        """
        Get or create an aiohttp session.
        Returns:
            ClientSession: active aiohttp session.
        """
        log.debug("[AsyncSessionHolder] get() called.")
        async with self._lock:
            log.debug("[AsyncSessionHolder] Acquired lock for session.")
            if self._session is None or self._session.closed:
                log.info("[AsyncSessionHolder] Creating new aiohttp session.")
                self._session = ClientSession(
                    connector=TCPConnector(limit=cfg.HTTP_CONN_LIMIT)
                )
            else:
                log.debug("[AsyncSessionHolder] Using open session.")
            return self._session

    async def close(self) -> None:
        """
        Close the session, if open.
        """
        log.debug("[AsyncSessionHolder] close() called.")
        async with self._lock:
            log.debug("[AsyncSessionHolder] Acquired lock for closing session.")
            if self._session and not self._session.closed:
                log.info("[AsyncSessionHolder] Closing aiohttp session.")
                await self._session.close()
                self._session = None
            else:
                log.debug("[AsyncSessionHolder] Session already closed or not initialized.")

session_holder = AsyncSessionHolder()

# ────────────────────────────────
# DISCORD BOT: SANITIZATION
# ────────────────────────────────

INJECTION_RE = re.compile(
    r"(?is)(?:^|\s)(?:assistant|system)\s*:|```|</?sys>|###\s*(?:assistant|system)"
)

def sanitize(text: str) -> str:
    """
    Remove dangerous Discord/system injection content and escape '@'.

    Args:
        text (str): Message to sanitize.

    Returns:
        str: Sanitized message.
    """
    log.debug("[sanitize] Input: %r", text)
    text = text.replace("@", "@\u200b")
    result = INJECTION_RE.sub(" ", text)
    disc_log.debug("[sanitize] Original: %r, sanitized: %r", text, result)
    return result

# ────────────────────────────────
# DISCORD BOT: LLM API INTEGRATION
# ────────────────────────────────

async def _call_openrouter(messages: List[dict]) -> Optional[str]:
    """
    Make a call to the OpenRouter API with retry/backoff.

    Args:
        messages (List[dict]): API payload.

    Returns:
        Optional[str]: Response content or None on failure.
    """
    s = await session_holder.get()
    log.info("[_call_openrouter] Starting OpenRouter request (max %d attempts)", OR_RETRIES)
    for attempt in range(1, OR_RETRIES + 1):
        try:
            log.debug("[_call_openrouter] Attempt #%d, messages=%r", attempt, messages)
            async with s.post(
                cfg.API_URL,
                json={
                    "model": cfg.OR_MODEL,
                    "messages": messages,
                    "max_tokens": cfg.OR_MAX_TOKENS,
                },
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            ) as r:
                log.info("[_call_openrouter] HTTP response: %d", r.status)
                if r.status in {429, 500, 502, 503}:
                    log.warning("[_call_openrouter] Server error: %d", r.status)
                    raise RuntimeError(f"HTTP {r.status}")
                data = await r.json()
                log.debug("[_call_openrouter] Response data: %r", data)
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            log.warning("[_call_openrouter] Attempt %d failed: %s", attempt, e)
            if attempt < OR_RETRIES:
                log.info("[_call_openrouter] Retrying in %d sec", 2 ** attempt)
                await asyncio.sleep(2 ** attempt)
    log.error("[_call_openrouter] Failed to get response from OpenRouter.")
    return None

async def call_local_llm(prompt_text: str) -> str:
    """
    Send prompt to local LLM (Ollama) and return response.

    Args:
        prompt_text (str): The prompt.

    Returns:
        str: Model response or error message.
    """
    log.debug("[call_local_llm] Call with prompt: %r", prompt_text)
    s = await session_holder.get()
    try:
        log.debug("[call_local_llm] Sending POST to Ollama (%s)...", cfg.OLLAMA_URL)
        async with s.post(
            cfg.OLLAMA_URL,
            json={"model": cfg.LOCAL_MODEL, "prompt": prompt_text, "stream": False},
            timeout=aiohttp.ClientTimeout(total=180)
        ) as r:
            log.info("[call_local_llm] HTTP response: %d", r.status)
            data = await r.json()
            log.info("[call_local_llm] Local LLM response: %r", data)
            return data.get("response", "❌ No response.")
    except asyncio.TimeoutError:
        log.error("[call_local_llm] Ollama timeout.")
        return "⚠️ Local LLM did not respond (timeout)."
    except Exception as e:
        log.error("[call_local_llm] Local LLM error: %s", e, exc_info=True)
        return f"⚠️ Local LLM error: {e}"

async def query_model(
    messages: List[dict], sys_prompt: str, ctx_txt: str, q: str, timeout_sec: int = 240
) -> Tuple[str, bool]:
    """
    Query the main model with OpenRouter fallback. Returns answer and fallback flag.

    Args:
        messages (List[dict]): OpenRouter-style messages.
        sys_prompt (str): System prompt.
        ctx_txt (str): Context text.
        q (str): User question.
        timeout_sec (int, optional): Timeout for local LLM. Defaults to 240.

    Returns:
        Tuple[str, bool]: (Answer text, fallback_used)
    """
    global openrouter_until
    now = time.time()
    fallback = False
    log.info("[query_model] Request: openrouter_until=%.2f, now=%.2f", openrouter_until, now)
    if now >= openrouter_until:
        log.info("[query_model] OpenRouter allowed, making request.")
        txt = await _call_openrouter(messages)
        if txt is not None:
            log.info("[query_model] Got response from OpenRouter.")
            return txt, False
        fallback = True
        openrouter_until = now + cfg.OPENROUTER_BLOCK_SEC
        log.warning("[query_model] Switching to local model, blocking OpenRouter until %.2f", openrouter_until)
    else:
        fallback = True
        log.info("[query_model] Fallback in use, OpenRouter blocked.")

    prompt_text = (
        sys_prompt.strip()
        + "\n\nCONTEXT:\n"
        + ctx_txt.strip()
        + "\n\nQUESTION: "
        + q.strip()
        + "\nANSWER:"
    )

    log.debug("[query_model] prompt_text=%r", prompt_text)
    s = await session_holder.get()
    try:
        log.debug("[query_model] Sending POST to Ollama (%s)...", cfg.OLLAMA_URL)
        async with s.post(
            cfg.OLLAMA_URL,
            json={"model": cfg.LOCAL_MODEL, "prompt": prompt_text, "stream": False},
            timeout=aiohttp.ClientTimeout(total=timeout_sec)
        ) as r:
            log.info("[query_model] HTTP response: %d", r.status)
            data = await r.json()
            log.info("[query_model] Local model response: %r", data)
            return data.get("response", ""), True
    except asyncio.TimeoutError:
        log.error("[query_model] Timeout: LLM did not respond in %d seconds", timeout_sec)
        return "⚠️ LLM did not respond (timeout). Try again later.", True
    except Exception as e:
        log.error("[query_model] LLM request error: %s", e, exc_info=True)
        return f"⚠️ LLM request error: {e}", True

# ────────────────────────────────
# DISCORD BOT: RAG-PIPELINE & COMMAND HANDLER
# ────────────────────────────────

def _cleanup_user_last() -> None:
    """
    Remove old entries from user_last based on timeout threshold.
    """
    log.debug("[_cleanup_user_last] Starting user_last cleanup.")
    cutoff = time.monotonic() - USER_LAST_CLEAN
    orig_len = len(user_last)
    for uid, ts in list(user_last.items()):
        if ts < cutoff:
            user_last.pop(uid, None)
            log.debug("[_cleanup_user_last] Removed user_id=%d (ts=%.2f < cutoff=%.2f)", uid, ts, cutoff)
    disc_log.debug("[_cleanup_user_last] user_last cleanup: was %d, now %d", orig_len, len(user_last))

async def generate_rag_answer(q: str, sys_prompt: str, use_remote: bool) -> Tuple[str, Any, Any]:
    """
    Main RAG pipeline: extract lemmas, retrieve, rerank, filter and build prompt.

    Args:
        q (str): User question.
        sys_prompt (str): System prompt.
        use_remote (bool): Use remote model if available.

    Returns:
        Tuple[str, Any, Any]: (Prompt for model, filtered nodes, built context text)
    """
    log.info("[generate_rag_answer] Received query: %r (use_remote=%r)", q, use_remote)
    qlem = _extract_lemmas(q)
    log.debug("[generate_rag_answer] Query lemmas: %r", list(qlem))
    log.debug("[generate_rag_answer] Getting raw_nodes...")
    raw_nodes = await retriever.aretrieve(q)
    log.debug("[generate_rag_answer] Retrieved raw_nodes: %d", len(raw_nodes))

    reranked_nodes = await rerank(q, raw_nodes)
    log.debug("[generate_rag_answer] After rerank: %d", len(reranked_nodes) if reranked_nodes else 0)

    if reranked_nodes:
        for i, n in enumerate(reranked_nodes[:10]):
            log.debug(
                "[generate_rag_answer] Reranked node #%d: file=%r score=%.6f lemmas(len)=%d, lemmas(sample)=%r",
                i,
                n.node.metadata.get("file_name", "n/a"),
                n.score,
                len(n.node.metadata.get("lemmas", [])),
                list(n.node.metadata.get("lemmas", []))[:10]
            )
    else:
        log.warning("[generate_rag_answer] reranked_nodes empty or None.")

    nodes = await get_filtered_nodes(reranked_nodes or raw_nodes, qlem)
    log.info("[generate_rag_answer] Filtered nodes: %d", len(nodes) if nodes else 0)
    if not nodes:
        log.warning("[generate_rag_answer] Not enough data to answer.")
        return "⚠️ Not enough data.", None, None
    char_limit = CTX_LEN_REMOTE if use_remote else CTX_LEN_LOCAL
    log.debug("[generate_rag_answer] char_limit=%d", char_limit)
    ctx_txt = build_context(nodes, qlem, char_limit)
    log.debug("[generate_rag_answer] Context built, length=%d", len(ctx_txt))
    prompt = (
        sys_prompt.strip()
        + "\n\nCONTEXT:\n"
        + ctx_txt.strip()
        + "\n\nQUESTION: "
        + q.strip()
        + "\nANSWER:"
    )
    log.debug("[generate_rag_answer] prompt=%r", prompt)
    return prompt, nodes, ctx_txt

async def ask_rag(ctx: commands.Context, q_raw: str, sys_prompt: str) -> None:
    """
    Main Discord entrypoint for RAG QA. Checks cooldown, formats, and sends answer.

    Args:
        ctx (commands.Context): Discord context.
        q_raw (str): Raw user question.
        sys_prompt (str): System/system prompt or None.
    """
    disc_log.info("[ask_rag] Called in chat #%d (%s) by %s(%d)", ctx.channel.id, ctx.channel, ctx.author.name, ctx.author.id)
    if ctx.channel.id not in cfg.ALLOWED_CHANNELS:
        disc_log.warning("[ask_rag] Channel %d not allowed, ignoring.", ctx.channel.id)
        return

    q = sanitize(q_raw.strip().replace("\n", " "))
    disc_log.info("[ask_rag] Query: %r", q)
    if len(q) > cfg.MAX_QUESTION_LEN or not cfg.ALLOWED_CHARS.match(q):
        disc_log.warning("[ask_rag] Invalid query: len=%d, match=%r", len(q), cfg.ALLOWED_CHARS.match(q))
        await ctx.send("❌ Invalid query format.")
        return

    now = time.monotonic()
    if now - user_last.get(ctx.author.id, 0) < cfg.USER_COOLDOWN:
        disc_log.info("[ask_rag] User %d in cooldown.", ctx.author.id)
        await ctx.send("⏳ Please wait a bit.")
        return
    user_last[ctx.author.id] = now
    _cleanup_user_last()

    disc_log.info(
        "cmd=%s user=%s(%d) ch=%d q=%r", ctx.command, ctx.author.name, ctx.author.id, ctx.channel.id, q
    )
    disc_log.debug("ask_rag: user_last=%r FILTER_CACHE=%d", user_last, len(FILTER_CACHE))
    await ctx.send("🔍 Thinking…")
    async with cfg.REQUEST_SEMAPHORE:
        disc_log.info("[ask_rag] Running RAG pipeline for request.")

        use_remote = time.time() < openrouter_until
        disc_log.debug("[ask_rag] use_remote=%r (openrouter_until=%.2f, now=%.2f)", use_remote, openrouter_until, time.time())
        sys_prompt_ = Path(PROMPT_STRICT).read_text("utf-8") if sys_prompt is None else sys_prompt
        prompt, nodes, ctx_txt = await generate_rag_answer(q, sys_prompt_, use_remote)
        if not nodes or prompt == "⚠️ Not enough data.":
            disc_log.info("[ask_rag] No suitable documents for answer.")
            await ctx.send("⚠️ Not enough data.")
            return

        disc_log.debug("[ask_rag] prompt for model: %r", prompt)
        answer, fb = await query_model(
            [
                {"role": "system", "content": sys_prompt_},
                {"role": "user", "content": prompt},
            ],
            sys_prompt=sys_prompt_,
            ctx_txt=ctx_txt,
            q=q,
        )

        rag_log.debug(
            "FALLBACK=%s\nQ: %s\nSYS_PROMPT:\n%s\nCTX:\n%s\nPROMPT:\n%s\nANSWER:\n%s",
            fb, q, sys_prompt_, ctx_txt, prompt, answer,
        )

        if fb:
            disc_log.warning("[ask_rag] OpenRouter unavailable, local model used.")
            await send_long(ctx, "⚠️ OpenRouter unavailable, local model used.")

        try:
            disc_log.info("[ask_rag] Sending answer, length %d", len(answer or ""))
            await send_long(ctx, answer or "❌ No answer.")
        except discord.HTTPException as e:
            disc_log.error("[ask_rag] Discord HTTPException: %s", e)
            await send_long(ctx, f"⚠️ Answer too long: {e}")

# ────────────────────────────────
# Alias for modular compatibility
# ────────────────────────────────

extract_lemmas = _extract_lemmas  # Alias to expose internal function as public API

# ─────────────────────────────────────────────────────────────
# rag_multistage — advanced RAG pipeline with multi-step querying via LLM-based decomposition
# ─────────────────────────────────────────────────────────────

async def decompose_question_llm(q0: str, sys_prompt: str) -> List[str]:
    """
    Decompose a user query into 2–3 simpler sub-questions using the LLM.

    Args:
        q0 (str): Original user question.
        sys_prompt (str): System prompt used for LLM context.

    Returns:
        List[str]: A list of sub-questions. Falls back to [q0] if parsing fails.
    """
    log.info("[decompose_question_llm] Original query: %r", q0)

    # Prompt for decomposition task (instruction + example)
    prompt = (
        "Разбей пользовательский вопрос на 2–3 более простых подвопроса, "
        "которые можно задать поисковой системе по документам.\n"
        "Пиши только список подвопросов без пояснений.\n"
        f"\nПользовательский вопрос: {q0.strip()}\nПодвопросы:"
    )

    # Compose OpenRouter-style message format
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    # Call LLM (remote or local fallback handled inside)
    answer, fb = await query_model(
        messages,
        sys_prompt=sys_prompt,
        ctx_txt="",
        q=q0,
    )

    log.debug("[decompose_question_llm] Raw model output: %r", answer)

    # Parse answer into sub-questions (1 line = 1 question)
    subq = [line.strip("-•* ") for line in answer.splitlines() if line.strip()]
    if not subq:
        log.warning("[decompose_question_llm] No subquestions parsed. Returning original.")
        return [q0]

    log.info("[decompose_question_llm] Parsed subquestions: %r", subq)
    return subq


async def multi_query_rag(q0: str, sys_prompt: str) -> str:
    """
    Execute a multi-step RAG pipeline with sub-question decomposition,
    retrieval, reranking, filtering, and context assembly.

    Args:
        q0 (str): The original user query.
        sys_prompt (str): System prompt for the LLM.

    Returns:
        str: Final generated answer based strictly on retrieved content,
        or a fallback message if no relevant data is found.
    """
    log.info("[multi_query_rag] Input question: %r", q0)

    # Decompose the input question into subquestions using LLM
    subquestions = await decompose_question_llm(q0, sys_prompt)
    log.info("[multi_query_rag] Subquestions from LLM: %r", subquestions)

    all_nodes = []

    for qn in subquestions:
        log.debug("[multi_query_rag] Processing subquestion: %r", qn)

        # Extract lemmas from the subquestion
        qlem = extract_lemmas(qn)
        log.debug("[multi_query_rag] Extracted lemmas: %r", list(qlem)[:10])

        # Retrieve candidate nodes for the subquestion
        raw_nodes = await retriever.aretrieve(qn)
        log.info("[multi_query_rag] Retrieved raw_nodes: %d", len(raw_nodes))

        # Apply reranking using CrossEncoder
        reranked = await rerank(qn, raw_nodes)
        log.debug("[multi_query_rag] Reranked nodes: %d", len(reranked))

        # Apply strict filtering based on lemmas
        filtered = await get_filtered_nodes(reranked or raw_nodes, qlem)
        log.info("[multi_query_rag] Filtered nodes: %d", len(filtered))

        all_nodes.extend(filtered)

    if not all_nodes:
        log.warning("[multi_query_rag] No relevant nodes found across all subquestions.")
        return "No relevant nodes found across all subquestions."

    # Rerank final node set against the original full query
    final_reranked = await rerank(q0, all_nodes)
    log.info("[multi_query_rag] Final reranked nodes: %d", len(final_reranked))

    # Take top-16 for context construction
    top_nodes = final_reranked[:16] if final_reranked else all_nodes[:16]
    log.debug("[multi_query_rag] Top nodes selected: %d", len(top_nodes))

    # Extract lemmas from the original full query
    qlem0 = extract_lemmas(q0)
    log.debug("[multi_query_rag] Final query lemmas: %r", list(qlem0)[:10])

    # Build final context string from selected nodes
    context = build_context(top_nodes, qlem0, CTX_LEN_LOCAL)
    log.info("[multi_query_rag] Final context length: %d", len(context))

    # Compose final prompt to LLM
    prompt = (
        "Ты — помощник на RPG-сервере Asketmc (Minecraft 1.12.2) в Discord.\n"
        "Отвечай только коротко на русском языке.\n"
        "Используй только факты из загруженных документов: рецепты, профессии, предметы, механики, эффекты.\n"
        "Не используй знания из реального мира. Не придумывай, не догадывайся.\n"
        "Если нет ответа в тексте, напиши: 'Недостаточно данных.'\n"
        f"\nCONTEXT:\n{context}\n\nQUESTION: {q0}\nANSWER:"
    )
    log.debug("[multi_query_rag] Final prompt: %r", prompt[:500])

    # Send query to model (OpenRouter or fallback to local)
    answer, fb = await query_model(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        sys_prompt=sys_prompt,
        ctx_txt=context,
        q=q0,
    )

    log.info("[multi_query_rag] Got answer (fallback=%r), length=%d", fb, len(answer or ""))
    return answer.strip()


# ────────────────────────────────
# BOT COMMANDS — Refactored Production Grade
# ────────────────────────────────

# ── Localization / Constants ─────────────────────────────
MSG_ACCESS_DENIED = "❌ Access denied."
MSG_NOT_ALLOWED_CHANNEL = "❌ This command is not allowed in this channel."
MSG_INVALID_QUERY = "❌ Invalid query format."
MSG_TOO_FAST = "⏳ Please wait before sending another query."
MSG_THINKING_LOCAL = "🧠 Thinking locally…"
MSG_NOT_ENOUGH_DATA = "⚠️ Not enough data."
MSG_NO_ANSWER = "❌ No answer."
MSG_ANSWER_TOO_LONG = "⚠️ Answer too long: {}"
MSG_INDEX_RELOADED = "✅ Index reloaded."
MSG_SHUTDOWN = "🛑 Shutting down bot..."
MSG_MULTI_ANALYSIS = "🔎 Running multi-step analysis…"
MSG_ERROR = "❌ Error: {}"

# ── Decorators ──────────────────────────────────────────

def admin_only(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    @wraps(func)
    async def wrapper(ctx: commands.Context, *args, **kwargs):
        if ctx.author.id not in ADMIN_IDS:
            disc_log.warning(f"[{func.__name__}] Access denied: not admin")
            await ctx.send(MSG_ACCESS_DENIED)
            return
        return await func(ctx, *args, **kwargs)
    return wrapper

def channel_allowed_only(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    @wraps(func)
    async def wrapper(ctx: commands.Context, *args, **kwargs):
        if ctx.channel.id not in cfg.ALLOWED_CHANNELS:
            await ctx.send(MSG_NOT_ALLOWED_CHANNEL)
            return
        return await func(ctx, *args, **kwargs)
    return wrapper


# ── Bot Commands ────────────────────────────────────────

@bot.command(name="strict")
async def cmd_strict(ctx: commands.Context, *, q: str) -> None:
    """
    Run strict RAG query using the strict prompt.
    
    Args:
        ctx: Discord context.
        q: User query.
    """
    disc_log.info(
        "[cmd:strict] user=%s(%d) channel=%s(%d) q=%r",
        ctx.author.name, ctx.author.id, ctx.channel.name, ctx.channel.id, q,
    )
    await ask_rag(ctx, q, Path(PROMPT_STRICT).read_text("utf-8"))

@bot.command(name="local", help="RAG answer only with local model, cloud is skipped")
@channel_allowed_only
async def cmd_local_llm(ctx: commands.Context, *, q: str) -> None:
    """
    Run RAG query using only the local LLM model, skipping the cloud.
    
    Args:
        ctx: Discord context.
        q: User query.
    """
    disc_log.info(
        "[cmd:local] user=%s(%d) channel=%s(%d) q=%r",
        ctx.author.name, ctx.author.id, ctx.channel.name, ctx.channel.id, q,
    )

    q = sanitize(q.strip().replace("\n", " "))
    if len(q) > cfg.MAX_QUESTION_LEN or not cfg.ALLOWED_CHARS.match(q):
        await ctx.send(MSG_INVALID_QUERY)
        return

    now = time.monotonic()
    if now - user_last.get(ctx.author.id, 0) < cfg.USER_COOLDOWN:
        await ctx.send(MSG_TOO_FAST)
        return
    user_last[ctx.author.id] = now
    _cleanup_user_last()

    await ctx.send(MSG_THINKING_LOCAL)
    try:
        async with cfg.REQUEST_SEMAPHORE:
            sys_prompt = Path(PROMPT_STRICT).read_text("utf-8")
            prompt, nodes, ctx_txt = await generate_rag_answer(q, sys_prompt, use_remote=False)
            if not nodes or prompt == MSG_NOT_ENOUGH_DATA:
                await ctx.send(MSG_NOT_ENOUGH_DATA)
                return

            disc_log.debug("[cmd:local] Prompt for local LLM: %r", prompt)
            answer = await call_local_llm(prompt)
            await send_long(ctx, answer or MSG_NO_ANSWER)
    except discord.HTTPException as e:
        await send_long(ctx, MSG_ANSWER_TOO_LONG.format(e))
    except Exception as exc:
        disc_log.exception("[cmd:local] Unexpected error: %r", exc)
        await ctx.send(MSG_ERROR.format(exc))

@bot.command(name="think")
async def cmd_think(ctx: commands.Context, *, q: str) -> None:
    """
    Run RAG query using the 'reason' prompt.
    
    Args:
        ctx: Discord context.
        q: User query.
    """
    disc_log.info(
        "[cmd:think] user=%s(%d) channel=%s(%d) q=%r",
        ctx.author.name, ctx.author.id, ctx.channel.name, ctx.channel.id, q,
    )
    await ask_rag(ctx, q, Path(PROMPT_REASON).read_text("utf-8"))

@bot.command(name="status")
async def cmd_status(ctx: commands.Context) -> None:
    """
    Report bot/document index/cache/OpenRouter status.
    
    Args:
        ctx: Discord context.
    """
    disc_log.info(
        "[cmd:status] user=%s(%d) channel=%s(%d)",
        ctx.author.name, ctx.author.id, ctx.channel.name, ctx.channel.id,
    )
    await ctx.send(
        f"🧠 Documents: {len(index.docstore.docs)}\n"
        f"💾 Cache: {'yes' if cfg.CACHE_PATH.exists() else 'no'}\n"
        f"🌐 OpenRouter: {'blocked' if time.time() < openrouter_until else 'ok'}"
    )

@bot.command(name="reload_index")
@admin_only
async def cmd_reload_index(ctx: commands.Context) -> None:
    """
    Reload vector index and retriever. Only for admins.
    
    Args:
        ctx: Discord context.
    """
    disc_log.info(
        "[cmd:reload_index] user=%s(%d) channel=%s(%d)",
        ctx.author.name, ctx.author.id, ctx.channel.name, ctx.channel.id,
    )
    FILTER_CACHE.clear()
    global index, retriever
    index = await build_index()
    retriever = index.as_retriever(similarity_top_k=cfg.TOP_K)
    await ctx.send(MSG_INDEX_RELOADED)

@bot.command(name="stop")
@admin_only
async def cmd_stop(ctx: commands.Context) -> None:
    """
    Stop the Discord bot (admin only).
    
    Args:
        ctx: Discord context.
    """
    disc_log.info(
        "[cmd:stop] user=%s(%d) channel=%s(%d)",
        ctx.author.name, ctx.author.id, ctx.channel.name, ctx.channel.id,
    )
    await ctx.send(MSG_SHUTDOWN)
    disc_log.info("[cmd:stop] Shutdown initiated.")
    await shutdown()

@bot.command(name="multy")
@channel_allowed_only
async def cmd_multiquery(ctx: commands.Context, *, q: str) -> None:
    """
    Run multi-step RAG pipeline on a single query.
    
    Args:
        ctx: Discord context.
        q: User query.
    """
    await ctx.send(MSG_MULTI_ANALYSIS)
    sys_prompt = Path(cfg.PROMPT_STRICT).read_text("utf-8")
    try:
        answer = await multi_query_rag(q.strip(), sys_prompt)
        await send_long(ctx, answer or MSG_NO_ANSWER)
    except Exception as exc:
        disc_log.exception("[cmd_multiquery] Error: %r", exc)
        await ctx.send(MSG_ERROR.format(exc))

# ────────────────────────────────
# BOT LIFECYCLE
# ────────────────────────────────

@bot.event
async def on_ready() -> None:
    """
    Discord bot startup event handler.
    Sends a notification message to the allowed channel and logs bot details.
    """
    log.info("Bot started as %s (ID: %d)", bot.user, bot.user.id)
    print(f"🟢 Discord bot started as {bot.user} (ID: {bot.user.id})")
    try:
        ch = bot.get_channel(next(iter(cfg.ALLOWED_CHANNELS)))
        if ch:
            await ch.send("✅ Bot started and ready.")
        else:
            log.warning("[on_ready] Allowed channel not found.")
    except Exception as e:
        log.error("[on_ready] Failed to send startup notification: %s", e)

async def shutdown() -> None:
    """
    Gracefully shutdown the Discord bot.
    Sends a shutdown notification, closes sessions, pools and the bot.
    """
    try:
        ch = bot.get_channel(next(iter(cfg.ALLOWED_CHANNELS)))
        if ch:
            await ch.send("🛑 Bot stopped.")
        else:
            log.warning("[shutdown] Allowed channel not found.")
    except Exception as e:
        log.error("[shutdown] Failed to send shutdown notification: %s", e)
    await bot.close()
    await session_holder.close()
    await shutdown_reranker()
    LEMMA_POOL.shutdown(wait=True)

# Cross-platform shutdown signal handling
if sys.platform != "win32":
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))

# ────────────────────────────────
# ENTRY POINT
# ────────────────────────────────

if __name__ == "__main__":
    try:
        print("🔄 Building document index...")
        index = asyncio.run(build_index())
        retriever = index.as_retriever(similarity_top_k=cfg.TOP_K)
        asyncio.run(init_reranker())
        print("✅ Index built. Starting Discord bot...")
        log.info("✅ Index built, starting bot.")
        bot.run(DISCORD_TOKEN, log_handler=None)
    except Exception as e:
        log.critical("Fatal error in main entry: %s", e)
        raise
