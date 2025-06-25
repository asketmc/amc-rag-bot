#!/usr/bin/env python3.10
# main.py — Asketmc RAG-бот v2.8+ (stanza-only, без pymorphy2, best practices, async-safe ClientSession)

from __future__ import annotations

import sys
import os
import re
import json
import time
import signal
import threading
import functools
import hashlib
import logging
import asyncio
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import (
    Any,
    Dict,
    Final,
    FrozenSet,
    List,
    Optional,
    Set,
)
from concurrent.futures import ThreadPoolExecutor

import torch
import aiohttp
import stanza
from dotenv import load_dotenv
import discord
from aiohttp import ClientSession, TCPConnector
from discord.ext import commands
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import spacy
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# ────────────────────────────────
# PROJECT IMPORTS
# ────────────────────────────────
import config as cfg
from config import PROMPT_REASON, PROMPT_STRICT, GOOD_POS, STOP_WORDS
from rerank import init_reranker, rerank, shutdown_reranker

# ────────────────────────────────
# STARTUP LOGGING & ENV CHECKS
# ────────────────────────────────
print(f"[STARTUP] Python: {sys.version}")
print(f"[STARTUP] Script: {__file__}")
print(f"[STARTUP] Working dir: {os.getcwd()}")
print(f"[STARTUP] torch version: {torch.__version__}, cuda available: {torch.cuda.is_available()}")

try:
    cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[STARTUP] LOG_DIR создан/найден: {cfg.LOG_DIR}")
except Exception as e:
    print(f"[STARTUP] Ошибка создания LOG_DIR: {e}", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("asketmc.startup")

log.info("[STARTUP] Импорт модулей завершён.")
log.info("[STARTUP] Начинается загрузка .env и конфигов.")

# ────────────────────────────────
# ЗАГРУЗКА .env
# ────────────────────────────────
env_paths = [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / ".env"
]
env_loaded = False
for dp in env_paths:
    log.info("[ENV] Проверяю наличие .env файла по пути: %s", dp)
    if dp.exists():
        load_dotenv(dp)
        log.info("[ENV] Загружен .env файл: %s", dp)
        env_loaded = True
        break

if not env_loaded:
    log.warning("[ENV] .env файл не найден ни в одном из путей: %r", [str(p) for p in env_paths])
else:
    log.info("[ENV] Переменные окружения успешно загружены")

def get_conf(name: str, default, typ=None):
    if hasattr(cfg, name):
        val = getattr(cfg, name)
        log.info("[CONSTANTS] %s найден в config.py: %r", name, val)
    else:
        val = default
        log.warning("[CONSTANTS] %s не найден в config.py, используется дефолт: %r", name, val)
    if typ is not None and not isinstance(val, typ):
        try:
            val = typ(val)
            log.info("[CONSTANTS] %s приведён к типу %s", name, typ)
        except Exception as e:
            log.error("[CONSTANTS] %s: не удалось привести к типу %s: %s", name, typ, e)
    return val

DISCORD_TOKEN: str = os.getenv("DISCORD_TOKEN") or sys.exit("DISCORD_TOKEN missing")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY") or sys.exit("OPENROUTER_API_KEY missing")

OR_RETRIES: Final[int] = get_conf("OR_RETRIES", 3, int)
CTX_LEN_REMOTE: Final[int] = get_conf("CTX_LEN_REMOTE", 20_000, int)
CTX_LEN_LOCAL: Final[int] = get_conf("CTX_LEN_LOCAL", 12_000, int)
USER_LAST_CLEAN: Final[int] = get_conf("USER_LAST_CLEAN", 3600, int)
EMBED_LOG_EVERY: Final[int] = get_conf("EMBED_LOG_EVERY", 1_000, int)
LEMMA_CACHE_SIZE: Final[int] = get_conf("LEMMA_CACHE_SIZE", 200_000, int)

LEMMA_INDEX_FILE: Final[Path] = get_conf(
    "LEMMA_INDEX_FILE", cfg.CACHE_PATH / "lemma_index.json", Path
)

SCORE_RELATIVE_THRESHOLD: Final[float] = get_conf("SCORE_RELATIVE_THRESHOLD", 0.7, float)
LEMMA_MATCH_RATIO: Final[float] = get_conf("LEMMA_MATCH_RATIO", 0.1, float)
ADMIN_IDS: Final[Set[int]] = get_conf("ADMIN_IDS", set(), set)

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

def _rotating_handler(name: str) -> RotatingFileHandler:
    handler = RotatingFileHandler(
        cfg.LOG_DIR / name,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )
    return handler

cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)

app_handler = _rotating_handler("app.log")
err_handler = _rotating_handler("error.log")
err_handler.setLevel(logging.ERROR)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG if getattr(cfg, "DEBUG", False) else logging.INFO)

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

log = logging.getLogger("asketmc.app")
embed_log = logging.getLogger("asketmc.embed")
disc_log = logging.getLogger("asketmc.discord")
rag_log = logging.getLogger("asketmc.rag")
for name, lg in [
    ("app", log), ("embed", embed_log), ("discord", disc_log), ("rag", rag_log)
]:
    if getattr(cfg, "DEBUG", False):
        debug_handler = _rotating_handler(f"{name}.debug.log")
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
        ))
        lg.addHandler(debug_handler)
        lg.setLevel(logging.DEBUG)
        lg.debug("[LOGGING] DEBUG handler добавлен для %s", name)
    else:
        lg.setLevel(logging.INFO)
    lg.info("[LOGGING] Logger '%s' готов, уровень: %s", name, lg.level)

for lg in (embed_log, disc_log, rag_log, log):
    lg.propagate = False

log.info("[LOGGING] Логи пишутся в %s", cfg.LOG_DIR)
log.info("[LOGGING] DEBUG режим: %r", getattr(cfg, "DEBUG", False))
log.info("[LOGGING] Стартовое логирование инициализировано")

# ────────────────────────────────
# LEMMATIZER: stanza (ru) + spacy (en) + langdetect
# ────────────────────────────────
log.info("[LEMMATIZER] Проверка наличия языковой модели Stanza для ru...")
try:
    stanza.download('ru', verbose=False)
    log.info("[LEMMATIZER] Stanza 'ru' модель успешно загружена.")
except Exception as e:
    log.exception("[LEMMATIZER] Ошибка при загрузке модели Stanza: %s", e)
    raise SystemExit(1)

log.info("[LEMMATIZER] Инициализация Stanza.Pipeline (ru)...")
try:
    STANZA_NLP_RU = stanza.Pipeline(
        lang='ru',
        processors='tokenize,pos,lemma',
        use_gpu=False,
        verbose=False,
    )
    log.info("[LEMMATIZER] Stanza.Pipeline успешно инициализирован.")
except Exception as e:
    log.exception("[LEMMATIZER] Ошибка инициализации Stanza.Pipeline: %s", e)
    raise SystemExit(1)

log.info("[LEMMATIZER] Инициализация spaCy (en)...")
try:
    import spacy
    SPACY_EN = spacy.load("en_core_web_sm")
    log.info("[LEMMATIZER] spaCy 'en_core_web_sm' успешно загружен.")
except Exception as e:
    log.exception("[LEMMATIZER] Ошибка загрузки spaCy модели: %s", e)
    raise SystemExit(1)

try:
    _LEMMA_LOCK = threading.Lock()
    max_workers = min(os.cpu_count() or 4, 8)
    LEMMA_POOL = ThreadPoolExecutor(max_workers=max_workers)
    log.info("[LEMMATIZER] ThreadPoolExecutor запущен с max_workers=%d", max_workers)
except Exception as e:
    log.exception("[LEMMATIZER] Ошибка создания ThreadPoolExecutor: %s", e)
    raise SystemExit(1)

@functools.lru_cache(maxsize=10_000)
def _extract_lemmas(text: str) -> FrozenSet[str]:
    log.debug("[_extract_lemmas] Входной текст (%d символов): %r", len(text), text[:120].replace("\n", " "))
    try:
        detected_lang = detect(text)
        log.debug("[_extract_lemmas] Язык определён: %s", detected_lang)
    except LangDetectException as e:
        detected_lang = "ru"
        log.warning("[_extract_lemmas] Ошибка определения языка: %s, язык принудительно установлен в 'ru'", e)
    except Exception as e:
        detected_lang = "ru"
        log.error("[_extract_lemmas] Неизвестная ошибка определения языка: %s", e)

    lang = "en" if detected_lang == "en" else "ru"
    log.debug("[_extract_lemmas] Язык после нормализации: %s", lang)

    if lang == "en":
        try:
            doc = SPACY_EN(text)
            log.debug("[_extract_lemmas] spaCy обработал документ: токенов=%d", len(doc))
            lemmas = {
                tok.lemma_.lower()
                for tok in doc
                if tok.is_alpha and not tok.is_stop and len(tok) > 2
            }
            log.debug("[_extract_lemmas] (en) Кол-во лемм: %d, примеры: %r", len(lemmas), list(lemmas)[:10])
            return frozenset(lemmas)
        except Exception as e:
            log.exception("[_extract_lemmas] Ошибка обработки spaCy: %s", e)
            return frozenset()

    try:
        with _LEMMA_LOCK:
            log.debug("[_extract_lemmas] Вошёл в lock для обработки Stanza")
            doc = STANZA_NLP_RU(text)
            log.debug("[_extract_lemmas] Stanza вернул %d предложений", len(doc.sentences))
    except Exception as e:
        log.exception("[_extract_lemmas] Ошибка обработки Stanza: %s", e)
        return frozenset()

    try:
        lemmas = {
            w.lemma.lower()
            for s in doc.sentences
            for w in s.words
            if w.lemma
               and len(w.lemma) > 2
               and w.upos in GOOD_POS
               and w.lemma.lower() not in STOP_WORDS
        }
        log.debug("[_extract_lemmas] (ru) Кол-во лемм: %d, примеры: %r", len(lemmas), list(lemmas)[:10])
        return frozenset(lemmas)
    except Exception as e:
        log.exception("[_extract_lemmas] Ошибка при построении множества лемм: %s", e)
        return frozenset()

# ────────────────────────────────
# CHUNK CACHE (c детальным построчным логированием)
# ────────────────────────────────

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, FrozenSet, List

log = logging.getLogger("asketmc.lemma")

CHUNK_LEMMA_CACHE_FILE = Path("rag_cache/chunk_lemma_index.json")
CHUNK_LEMMA_CACHE: Dict[str, FrozenSet[str]] = {}

FILE_LEMMAS: Dict[str, FrozenSet[str]] = {}
LEMMA_INDEX_FILE = Path("rag_cache/lemma_index.json")

def chunk_hash(text: str) -> str:
    log.debug("[chunk_hash] Входной текст (len=%d): %r", len(text), text[:40])
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    log.debug("[chunk_hash] Получен hash='%s' для текста длиной %d", h, len(text))
    return h


def load_chunk_lemma_cache(
    chunk_cache_file: Path = CHUNK_LEMMA_CACHE_FILE
) -> None:
    global CHUNK_LEMMA_CACHE
    log.info("[load_chunk_lemma_cache] Загрузка кэша чанков из %s", chunk_cache_file)
    if chunk_cache_file.exists():
        try:
            with chunk_cache_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            CHUNK_LEMMA_CACHE = {k: frozenset(v) for k, v in data.items()}
            log.info("[load_chunk_lemma_cache] Загружено %d чанков.", len(CHUNK_LEMMA_CACHE))
            log.debug("[load_chunk_lemma_cache] Ключи: %r", sorted(CHUNK_LEMMA_CACHE.keys()))
        except Exception as e:
            log.error("[load_chunk_lemma_cache] Ошибка загрузки: %s", e, exc_info=True)
            CHUNK_LEMMA_CACHE = {}
    else:
        CHUNK_LEMMA_CACHE = {}
        log.info("[load_chunk_lemma_cache] Нет файла: %s", chunk_cache_file)


def save_chunk_lemma_cache(
    chunk_cache_file: Path = CHUNK_LEMMA_CACHE_FILE
) -> None:
    log.info("[save_chunk_lemma_cache] Сохранение кэша чанков в %s", chunk_cache_file)
    data = {k: list(v) for k, v in CHUNK_LEMMA_CACHE.items()}
    try:
        chunk_cache_file.parent.mkdir(parents=True, exist_ok=True)
        with chunk_cache_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log.info("[save_chunk_lemma_cache] Сохранено %d чанков.", len(CHUNK_LEMMA_CACHE))
        log.debug("[save_chunk_lemma_cache] Ключи: %r", sorted(CHUNK_LEMMA_CACHE.keys()))
    except Exception as e:
        log.error("[save_chunk_lemma_cache] Ошибка сохранения: %s", e, exc_info=True)


def get_lemmas_for_chunk(text: str, lemma_func) -> FrozenSet[str]:
    h = chunk_hash(text)
    log.debug("[get_lemmas_for_chunk] hash=%s, длина текста=%d", h, len(text))
    lemmas = CHUNK_LEMMA_CACHE.get(h)
    if lemmas is None:
        log.info("[get_lemmas_for_chunk] Нет в кэше, считаю для hash=%s", h)
        try:
            lemmas = lemma_func(text)
            CHUNK_LEMMA_CACHE[h] = lemmas
            log.info(
                "[get_lemmas_for_chunk] Добавлено %d лемм для hash=%s", len(lemmas), h
            )
        except Exception as e:
            log.error(
                "[get_lemmas_for_chunk] Ошибка лемматизации для hash=%s: %s",
                h, e, exc_info=True
            )
            lemmas = frozenset()
            CHUNK_LEMMA_CACHE[h] = lemmas
    else:
        log.info(
            "[get_lemmas_for_chunk] Использую из кэша %d лемм для hash=%s",
            len(lemmas), h
        )
    # Валидация после каждого обращения:
    if h not in CHUNK_LEMMA_CACHE:
        log.error(
            "[get_lemmas_for_chunk] CRITICAL: hash %s не найден в кэше после добавления!", h
        )
    else:
        log.debug(
            "[get_lemmas_for_chunk] Проверка: hash %s присутствует, len=%d",
            h, len(CHUNK_LEMMA_CACHE[h])
        )
    return lemmas


def _read_file(fp: Path) -> str:
    log.debug("[_read_file] Чтение файла: %s", fp)
    try:
        data = fp.read_text("utf-8", "ignore")
        log.debug("[_read_file] Прочитано %d символов из %s", len(data), fp)
        return data
    except Exception as e:
        log.error("[_read_file] Ошибка чтения файла %s: %s", fp, e)
        return ""


async def _compute_and_store_lemmas(fp: Path) -> None:
    log.info("[_compute_and_store_lemmas] Чтение файла: %s", fp)
    try:
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, _read_file, fp)
        log.debug(
            "[_compute_and_store_lemmas] Прочитано %d символов из %s", len(text), fp
        )
    except Exception as e:
        log.error(
            "[_compute_and_store_lemmas] Ошибка чтения файла %s: %s", fp, e, exc_info=True
        )
        text = ""
    try:
        lemmas = _extract_lemmas(text)
        FILE_LEMMAS[fp.name] = lemmas
        log.info(
            "[_compute_and_store_lemmas] %s: %d лемм, %r",
            fp, len(lemmas), list(lemmas)[:10]
        )
    except Exception as e:
        log.error(
            "[_compute_and_store_lemmas] Ошибка лемматизации %s: %s", fp, e, exc_info=True
        )
        FILE_LEMMAS[fp.name] = frozenset()


def _persist_lemmas(lemma_file: Path = None) -> None:
    if lemma_file is None:
        lemma_file = LEMMA_INDEX_FILE
    try:
        log.info("[_persist_lemmas] Сохраняю кэш лемм в %s...", lemma_file)
        lemma_file.parent.mkdir(parents=True, exist_ok=True)
        dump = {k: list(v) for k, v in FILE_LEMMAS.items()}
        lemma_file.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info(
            "[_persist_lemmas] Кэш лемм успешно сохранён. Файлов: %d",
            len(FILE_LEMMAS)
        )
    except Exception as e:
        log.error("[_persist_lemmas] Ошибка при сохранении кэша: %s", e, exc_info=True)


def _load_saved_lemmas() -> None:
    if not LEMMA_INDEX_FILE.exists():
        log.info("[_load_saved_lemmas] Кэш лемм не найден (%s)", LEMMA_INDEX_FILE)
        return
    try:
        log.info("[_load_saved_lemmas] Чтение кэша лемм из %s...", LEMMA_INDEX_FILE)
        data = json.loads(LEMMA_INDEX_FILE.read_text("utf-8"))
        for fname, lst in data.items():
            FILE_LEMMAS[fname] = frozenset(lst)
        log.info(
            "[_load_saved_lemmas] Загружено лемм для %d файлов.", len(FILE_LEMMAS)
        )
    except Exception as e:
        log.warning("[_load_saved_lemmas] Ошибка чтения кэша лемм: %s", e, exc_info=True)


async def update_file_lemmas_async(
    docs: List[Path], stored_hashes: Dict[str, str], new_hashes: Dict[str, str]
) -> List[Path]:
    changed = [d for d in docs if stored_hashes.get(d.name) != new_hashes.get(d.name)]
    log.info(
        "[update_file_lemmas_async] Список изменённых файлов: %r",
        [d.name for d in changed]
    )
    if not changed:
        log.info("[update_file_lemmas_async] Нет изменённых файлов")
        return []
    tasks = []
    for d in changed:
        log.info("[update_file_lemmas_async] Стартую задачу лемматизации для %s", d)
        tasks.append(asyncio.create_task(_compute_and_store_lemmas(d)))
    await asyncio.gather(*tasks)
    log.info(
        "[update_file_lemmas_async] Все задачи лемматизации завершены, сохраняю леммы"
    )
    _persist_lemmas()
    return changed

async def build_index() -> VectorStoreIndex:
    """
    Строит/загружает векторный индекс для документов из cfg.DOCS_PATH с леммированием.
    Включает: кеши, валидные метаданные, обработку изменений, сохранение и валидацию.
    """
    log.info("build_index: Сканирую документы в %s...", cfg.DOCS_PATH)
    docs = list(cfg.DOCS_PATH.glob("*"))
    log.info("build_index: Найдено %d файлов: %r", len(docs), [p.name for p in docs])

    # 1. Вычисление хешей всех документов
    log.info("build_index: Вычисляю хеши файлов...")
    hashes = {d.name: _doc_hash(d) for d in docs}
    log.debug("build_index: Хеши файлов: %r", hashes)

    # 2. Загрузка старых хешей
    if cfg.HASH_FILE.exists():
        log.info("build_index: Загружаю сохранённые хеши из %s", cfg.HASH_FILE)
        try:
            stored = json.loads(cfg.HASH_FILE.read_text("utf-8"))
        except Exception as e:
            log.error("build_index: Ошибка чтения файла хешей: %s", e, exc_info=True)
            stored = {}
    else:
        log.info("build_index: Файл хешей не найден, строю с нуля.")
        stored = {}

    # 3. Загрузка кэша лемм
    log.info("build_index: Загружаю кэш лемм (для файлов)...")
    try:
        _load_saved_lemmas()
    except Exception as e:
        log.error("build_index: Ошибка загрузки кэша лемм: %s", e, exc_info=True)

    # 4. Обновление лемм только для изменённых файлов
    log.info("build_index: Обновляю леммы изменённых файлов...")
    try:
        changed = await update_file_lemmas_async(docs, stored, hashes)
        log.info("build_index: Обновлено лемм файлов: %r", [f.name for f in changed])
    except Exception as e:
        log.error("build_index: Ошибка обновления лемм файлов: %s", e, exc_info=True)
        changed = []

    # 5. Загрузка или построение индекса
    if cfg.CACHE_PATH.exists() and stored == hashes:
        log.info("build_index: Кэш индекса актуален, загружаю из %s", cfg.CACHE_PATH)
        try:
            idx = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=str(cfg.CACHE_PATH))
            )
        except Exception as e:
            log.error("build_index: Ошибка загрузки индекса: %s", e, exc_info=True)
            idx = None
    else:
        log.info("build_index: Кэш не найден или устарел, строю новый индекс...")
        ll_docs = []
        for fp in docs:
            log.debug("[build_index] Читаю файл: %s", fp)
            try:
                text = fp.read_text("utf-8", "ignore")
                ll_docs.append(Document(text=text, metadata={"file_name": fp.name}))
            except Exception as e:
                log.error("build_index: Ошибка чтения файла %s: %s", fp, e, exc_info=True)
        log.info("build_index: Построено %d документов, строю VectorStoreIndex...", len(ll_docs))
        try:
            idx = VectorStoreIndex.from_documents(ll_docs)
            log.info("build_index: Индекс построен.")
        except Exception as e:
            log.error("build_index: Ошибка построения индекса: %s", e, exc_info=True)
            idx = None

    # 6. Присваиваем леммы в метаданные всех чанков
    if idx is None:
        log.error("build_index: Индекс не построен, выход.")
        return None

    log.info("build_index: Добавляю леммы чанков в метаданные...")
    cnt = 0
    for node in idx.docstore.docs.values():
        try:
            chunk_text = node.get_content()
            lem = get_lemmas_for_chunk(chunk_text, _extract_lemmas)
            node.metadata["lemmas"] = list(lem)
            log.debug("Леммы для %s: %r", node.metadata.get("file_name", ""), list(lem))
            cnt += 1
        except Exception as e:
            log.error("build_index: Ошибка при добавлении лемм для node: %s", e, exc_info=True)
    log.info("[build_index] Обновлено чанков: %d", cnt)

    # 7. Логируем размер чанкового кэша
    log.info("[build_index] Ключей в CHUNK_LEMMA_CACHE: %d", len(CHUNK_LEMMA_CACHE))
    log.debug("[build_index] Все ключи CHUNK_LEMMA_CACHE: %r", sorted(CHUNK_LEMMA_CACHE.keys()))

    # 8. Сохраняем индекс на диск
    log.info("build_index: Сохраняю индекс в %s", cfg.CACHE_PATH)
    try:
        idx.storage_context.persist(str(cfg.CACHE_PATH))
    except Exception as e:
        log.error("build_index: Ошибка при сохранении индекса: %s", e, exc_info=True)

    # 9. Сохраняем актуальные хеши
    log.info("build_index: Сохраняю хеши в %s", cfg.HASH_FILE)
    try:
        cfg.HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
        cfg.HASH_FILE.write_text(
            json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        log.error("build_index: Ошибка при сохранении файла хешей: %s", e, exc_info=True)

    log.info("build_index: Завершено.")

    # 10. Сохраняем и валидируем кэш чанков
    save_chunk_lemma_cache()
    if not CHUNK_LEMMA_CACHE_FILE.exists():
        log.error("[build_index] CRITICAL: Кэш файл %s не появился после сохранения!",
                  CHUNK_LEMMA_CACHE_FILE)
    else:
        try:
            with open(CHUNK_LEMMA_CACHE_FILE, "r", encoding="utf-8") as f:
                persisted = json.load(f)
            log.info("[build_index] Проверка после сохранения: %d ключей реально сохранено.",
                     len(persisted))
            log.debug("[build_index] Проверочные ключи в файле: %r", sorted(persisted.keys()))
        except Exception as e:
            log.error("[build_index] Ошибка при валидации файла кэша: %s", e, exc_info=True)

    return idx


# Логирование и инициализация эмбеддингов и SentenceSplitter
log.info("[EMBEDDINGS] torch.__version__ = %s", torch.__version__)
log.info("[EMBEDDINGS] torch.cuda.is_available() = %r", torch.cuda.is_available())
log.info("[EMBEDDINGS] torch.cuda.device_count() = %d", torch.cuda.device_count())
if torch.cuda.is_available():
    log.info("[EMBEDDINGS] torch.cuda.get_device_name(0) = %s", torch.cuda.get_device_name(0))
else:
    log.info("[EMBEDDINGS] CUDA не найден, используется CPU.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info("[EMBEDDINGS] Инициализация модели BAAI/bge-m3 на устройстве: %s", DEVICE)

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
    log.info("[EMBEDDINGS] LoggingBGE успешно инициализирован")
except Exception as e:
    log.error("[EMBEDDINGS] Ошибка инициализации LoggingBGE: %s", e, exc_info=True)
    raise

try:
    Settings.node_parser = SentenceSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
        include_metadata=False,
        paragraph_separator="\n\n",
    )
    log.info("[EMBEDDINGS] SentenceSplitter успешно инициализирован: chunk_size=%d, chunk_overlap=%d",
             cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
except Exception as e:
    log.error("[EMBEDDINGS] Ошибка инициализации SentenceSplitter: %s", e, exc_info=True)
    raise

# ────────────────────────────────
# VECTOR INDEX
# ────────────────────────────────

def _doc_hash(fp: Path) -> str:
    log.debug(f"Hashing file: {fp}")
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    hex_digest = h.hexdigest()
    log.debug(f"Hash for {fp}: {hex_digest}")
    return hex_digest

async def build_index() -> VectorStoreIndex:
    log.info("build_index: Сканирую документы в %s...", cfg.DOCS_PATH)
    docs = list(cfg.DOCS_PATH.glob("*"))
    log.info("build_index: Найдено %d файлов: %r", len(docs), [p.name for p in docs])

    log.info("build_index: Вычисляю хеши файлов...")
    hashes = {d.name: _doc_hash(d) for d in docs}
    log.debug("build_index: Хеши файлов: %r", hashes)

    if cfg.HASH_FILE.exists():
        log.info("build_index: Загружаю сохранённые хеши из %s", cfg.HASH_FILE)
        try:
            stored = json.loads(cfg.HASH_FILE.read_text("utf-8"))
        except Exception as e:
            log.error("build_index: Ошибка чтения файла хешей: %s", e, exc_info=True)
            stored = {}
    else:
        log.info("build_index: Файл хешей не найден, строю с нуля.")
        stored = {}

    log.info("build_index: Загружаю кэш лемм (для файлов)...")
    try:
        _load_saved_lemmas()
    except Exception as e:
        log.error("build_index: Ошибка загрузки кэша лемм: %s", e, exc_info=True)

    log.info("build_index: Обновляю леммы изменённых файлов...")
    try:
        changed = await update_file_lemmas_async(docs, stored, hashes)
        log.info("build_index: Обновлено лемм файлов: %r", [f.name for f in changed])
    except Exception as e:
        log.error("build_index: Ошибка обновления лемм файлов: %s", e, exc_info=True)
        changed = []

    idx = None
    if cfg.CACHE_PATH.exists() and stored == hashes:
        log.info("build_index: Кэш индекса актуален, загружаю из %s", cfg.CACHE_PATH)
        try:
            idx = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=str(cfg.CACHE_PATH))
            )
        except Exception as e:
            log.error("build_index: Ошибка загрузки индекса: %s", e, exc_info=True)
            idx = None
    if idx is None:
        log.info("build_index: Кэш не найден или устарел, строю новый индекс...")
        ll_docs = []
        for fp in docs:
            log.debug(f"Читаю файл: {fp}")
            try:
                text = fp.read_text("utf-8", "ignore")
                ll_docs.append(Document(text=text, metadata={"file_name": fp.name}))
            except Exception as e:
                log.error("build_index: Ошибка чтения файла %s: %s", fp, e, exc_info=True)
        log.info("build_index: Построено %d документов, строю VectorStoreIndex...", len(ll_docs))
        try:
            idx = VectorStoreIndex.from_documents(ll_docs)
            log.info("build_index: Индекс построен.")
        except Exception as e:
            log.error("build_index: Ошибка построения индекса: %s", e, exc_info=True)
            raise

    log.info("build_index: Добавляю леммы в метаданные документов (чанков)...")
    missing_in_cache = 0
    missing_file_name = 0
    total_nodes = len(idx.docstore.docs)
    for i, node in enumerate(idx.docstore.docs.values()):
        fname = node.metadata.get("file_name", None)
        if not fname:
            log.warning("build_index: Пропускаю node без file_name! node.metadata=%r", node.metadata)
            missing_file_name += 1
            continue
        if fname in FILE_LEMMAS:
            lem = FILE_LEMMAS[fname]
            log.debug("build_index: Для файла %r леммы взяты из кэша (%d шт.)", fname, len(lem))
        else:
            log.warning("build_index: Для файла %r лемм нет в кэше, считаем заново.", fname)
            text = node.get_content()
            lem = _extract_lemmas(text)
            missing_in_cache += 1
        node.metadata["lemmas"] = list(lem)
        log.debug("build_index: Леммы для %s: %r", fname, list(lem))

    log.info("build_index: Пропущено узлов без file_name: %d из %d", missing_file_name, total_nodes)
    log.info("build_index: Файлов, для которых не было кэша и леммы были рассчитаны заново: %d", missing_in_cache)

    log.info("build_index: Сохраняю индекс в %s", cfg.CACHE_PATH)
    try:
        idx.storage_context.persist(str(cfg.CACHE_PATH))
    except Exception as e:
        log.error("build_index: Ошибка при сохранении индекса: %s", e, exc_info=True)

    log.info("build_index: Сохраняю хеши в %s", cfg.HASH_FILE)
    try:
        cfg.HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
        cfg.HASH_FILE.write_text(
            json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        log.error("build_index: Ошибка при сохранении файла хешей: %s", e, exc_info=True)

    log.info("build_index: Завершено. Индекс содержит %d чанков.", total_nodes - missing_file_name)
    return idx


# ────────────────────────────────
# FILTER NODE
# ────────────────────────────────

FILTER_CACHE: Dict[str, List[NodeWithScore]] = {}

def _cache_key(qlem: FrozenSet[str]) -> str:
    key = hashlib.sha256(" ".join(sorted(qlem)).encode("utf-8")).hexdigest()
    rag_log.debug("[_cache_key] qlem=%r, key=%s", list(qlem), key)
    return key

async def _filter_nodes(
    raw_nodes: List[NodeWithScore], qlem: FrozenSet[str]
) -> List[NodeWithScore]:
    rag_log.info("[filter_nodes] Старт фильтрации. Число raw_nodes: %d, qlem=%r", len(raw_nodes), list(qlem))
    if not raw_nodes:
        rag_log.warning("[filter_nodes] No raw_nodes provided, возвращаю пустой список.")
        return []

    max_score = max(n.score for n in raw_nodes)
    rag_log.debug("[filter_nodes] max_score=%.4f", max_score)

    alpha = getattr(cfg, "FILTER_ALPHA", 0.5)
    strict = []
    for idx, n in enumerate(raw_nodes):
        lemmas = frozenset(n.node.metadata.get("lemmas", []))
        inter = len(qlem & lemmas)
        rel_score = n.score / (max_score or 1)
        ratio = inter / (len(qlem) or 1)
        weight = n.score + alpha * ratio
        rag_log.debug(
            "[filter_nodes] idx=%d file=%r score=%.4f rel_score=%.3f inter=%d ratio=%.3f weight=%.3f",
            idx, n.node.metadata.get("file_name", "n/a"), n.score, rel_score, inter, ratio, weight
        )
        if rel_score >= getattr(cfg, "SCORE_RELATIVE_THRESHOLD", 0.7) or ratio >= getattr(cfg, "LEMMA_MATCH_RATIO", 0.1):
            strict.append((n, weight))

    rag_log.debug("[filter_nodes] После оптимизированной фильтрации strict=%d", len(strict))
    if strict:
        strict.sort(key=lambda x: -x[1])
        for i, (n, weight) in enumerate(strict[:cfg.TOP_K]):
            rag_log.debug("[filter_nodes] TOP #%d: %r score=%.4f weight=%.3f", i, n.node.metadata.get("file_name", "n/a"), n.score, weight)
        rag_log.info("[filter_nodes] strict=%d, возвращаю top-результаты.", len(strict))
        return [n for n, _ in strict[:cfg.TOP_K]]

    # Fallback — просто top-K по весу, только где есть хотя бы 1 inter
    fallback = []
    for idx, n in enumerate(raw_nodes):
        lemmas = frozenset(n.node.metadata.get("lemmas", []))
        inter = len(qlem & lemmas)
        ratio = inter / (len(qlem) or 1)
        weight = n.score + alpha * ratio
        if inter > 0:
            fallback.append((n, weight, inter, n.score))
        rag_log.debug("[filter_nodes-fallback] idx=%d file=%r score=%.4f inter=%d weight=%.3f", idx, n.node.metadata.get("file_name", "n/a"), n.score, inter, weight)

    fallback.sort(key=lambda x: -x[1])
    rag_log.info("[filter_nodes-fallback] После fallback осталось: %d", len(fallback))
    for i, (n, weight, inter, score) in enumerate(fallback[:cfg.TOP_K]):
        rag_log.debug("[filter_nodes-fallback] TOP #%d: %r score=%.4f inter=%d weight=%.3f", i, n.node.metadata.get("file_name", "n/a"), score, inter, weight)
    result = [n for n, _, _, _ in fallback[:cfg.TOP_K]]
    rag_log.info("[filter_nodes-fallback] Возвращаю топ-%d fallback.", len(result))
    return result


async def get_filtered_nodes(
    raw_nodes: List[NodeWithScore], qlem: FrozenSet[str]
) -> List[NodeWithScore]:
    key = _cache_key(qlem)
    rag_log.debug("[get_filtered_nodes] Проверка кэша для key=%s", key)
    if key in FILTER_CACHE:
        rag_log.info("[get_filtered_nodes] Использую кэш по ключу %s", key)
        rag_log.debug("[get_filtered_nodes] Кол-во кэшированных nodes=%d", len(FILTER_CACHE[key]))
        return FILTER_CACHE[key]
    rag_log.info("[get_filtered_nodes] Нет кэша, фильтрую заново.")
    nodes = await _filter_nodes(raw_nodes, qlem)
    FILTER_CACHE[key] = nodes
    rag_log.info("[get_filtered_nodes] Сохранил результат фильтрации в кэш по ключу %s (кол-во nodes=%d)", key, len(nodes))
    return nodes

def _content_from_node(n: NodeWithScore) -> str:
    txt = getattr(n.node, "get_content", lambda: "")()
    rag_log.debug("[_content_from_node] %r, len=%d", n.node.metadata.get("file_name", "n/a"), len(txt))
    return txt

def build_context(
    nodes: List[NodeWithScore], qlem: FrozenSet[str], char_limit: int
) -> str:
    rag_log.info("[build_context] Формирую контекст из %d nodes, лимит символов=%d", len(nodes), char_limit)
    beta = 0.1
    scored: List[tuple[NodeWithScore, float]] = []
    for n in nodes:
        lemmas = frozenset(n.node.metadata.get("lemmas", []))
        inter = len(qlem & lemmas)
        weight = n.score + beta * inter
        scored.append((n, weight))
        rag_log.debug("[build_context] node=%r score=%.4f inter=%d weight=%.4f",
                      n.node.metadata.get("file_name", "n/a"), n.score, inter, weight)
    scored.sort(key=lambda x: x[1], reverse=True)
    rag_log.debug("[build_context] Сортировка по weight завершена.")

    parts: List[str] = []
    seen_hashes: set[int] = set()
    total = 0
    for idx, (n, weight) in enumerate(scored):
        txt = _content_from_node(n).strip()
        if not txt:
            rag_log.debug("[build_context] SKIP пустой контент: %r", n.node.metadata.get("file_name", "n/a"))
            continue
        h = hash(txt)
        if h in seen_hashes:
            rag_log.debug("[build_context] SKIP повтор: %r", n.node.metadata.get("file_name", "n/a"))
            continue
        if total + len(txt) > char_limit:
            rag_log.info("[build_context] Достигнут лимит символов (%d), остановка.", total)
            break
        seen_hashes.add(h)
        parts.append(txt)
        total += len(txt) + 4
        rag_log.debug("[build_context] Добавлен node=%r (len=%d, total=%d)", n.node.metadata.get("file_name", "n/a"), len(txt), total)
    rag_log.info("[build_context] Итоговый контекст: частей=%d, общий размер=%d", len(parts), total)
    return "\n---\n".join(parts)

# ────────────────────────────────
# DISCORD-UTILITY
# ────────────────────────────────
def split_message(text: str, limit: int = 2000) -> List[str]:
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
    for chunk in split_message(text, limit):
        await ctx.send(chunk)


# ────────────────────────────────
# DISCORD BOT
# ────────────────────────────────

bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())
user_last: Dict[int, float] = {}
index: Optional[VectorStoreIndex] = None
retriever: Any = None
openrouter_until = 0.0

class AsyncSessionHolder:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._session: Optional[ClientSession] = None
        log.debug("[AsyncSessionHolder] Инициализация завершена.")

    async def get(self) -> ClientSession:
        log.debug("[AsyncSessionHolder] get() called.")
        async with self._lock:
            log.debug("[AsyncSessionHolder] Получен lock для сессии.")
            if self._session is None or self._session.closed:
                log.info("[AsyncSessionHolder] Открываю новую сессию aiohttp")
                self._session = ClientSession(
                    connector=TCPConnector(limit=cfg.HTTP_CONN_LIMIT)
                )
            else:
                log.debug("[AsyncSessionHolder] Использую открытую сессию.")
            return self._session

    async def close(self):
        log.debug("[AsyncSessionHolder] close() called.")
        async with self._lock:
            log.debug("[AsyncSessionHolder] Получен lock для закрытия сессии.")
            if self._session and not self._session.closed:
                log.info("[AsyncSessionHolder] Закрываю сессию aiohttp")
                await self._session.close()
                self._session = None
            else:
                log.debug("[AsyncSessionHolder] Сессия уже закрыта или не инициализирована.")

session_holder = AsyncSessionHolder()

INJECTION_RE = re.compile(
    r"(?is)(?:^|\s)(?:assistant|system)\s*:|```|</?sys>|###\s*(?:assistant|system)"
)

def sanitize(text: str) -> str:
    log.debug("[sanitize] Вход: %r", text)
    text = text.replace("@", "@\u200b")
    result = INJECTION_RE.sub(" ", text)
    disc_log.debug("[sanitize] Исходный текст: %r, результат: %r", text, result)
    return result

async def _call_openrouter(messages: List[dict]) -> Optional[str]:
    s = await session_holder.get()
    log.info("[_call_openrouter] Стартую запрос к OpenRouter (%d попыток макс)", OR_RETRIES)
    for attempt in range(1, OR_RETRIES + 1):
        try:
            log.debug("[_call_openrouter] Попытка #%d, messages=%r", attempt, messages)
            async with s.post(
                cfg.API_URL,
                json={
                    "model": cfg.OR_MODEL,
                    "messages": messages,
                    "max_tokens": cfg.OR_MAX_TOKENS,
                },
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            ) as r:
                log.info("[_call_openrouter] HTTP-ответ: %d", r.status)
                if r.status in {429, 500, 502, 503}:
                    log.warning("[_call_openrouter] Серверная ошибка: %d", r.status)
                    raise RuntimeError(f"HTTP {r.status}")
                data = await r.json()
                log.debug("[_call_openrouter] Ответ данных: %r", data)
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            log.warning("[_call_openrouter] Попытка %d неудачна: %s", attempt, e)
            if attempt < OR_RETRIES:
                log.info("[_call_openrouter] Повтор через %d сек", 2**attempt)
                await asyncio.sleep(2**attempt)
    log.error("[_call_openrouter] Не удалось получить ответ от OpenRouter")
    return None

import asyncio
import aiohttp

async def call_local_llm(prompt_text: str) -> str:
    log.debug("[call_local_llm] Вызов с prompt: %r", prompt_text)
    s = await session_holder.get()
    try:
        log.debug("[call_local_llm] Отправляю POST в Ollama (%s)...", cfg.OLLAMA_URL)
        async with s.post(
            cfg.OLLAMA_URL,
            json={"model": cfg.LOCAL_MODEL, "prompt": prompt_text, "stream": False},
            timeout=aiohttp.ClientTimeout(total=180)
        ) as r:
            log.info("[call_local_llm] HTTP-ответ: %d", r.status)
            data = await r.json()
            log.info("[call_local_llm] Ответ от локальной LLM: %r", data)
            return data.get("response", "❌ Нет ответа.")
    except asyncio.TimeoutError:
        log.error("[call_local_llm] Timeout от Ollama")
        return "⚠️ Локальная LLM не ответила (timeout)."
    except Exception as e:
        log.error("[call_local_llm] Ошибка локальной LLM: %s", e, exc_info=True)
        return f"⚠️ Ошибка обращения к локальной LLM: {e}"

async def query_model(
    messages: List[dict], sys_prompt: str, ctx_txt: str, q: str, timeout_sec: int = 240
) -> tuple[str, bool]:
    global openrouter_until
    now = time.time()
    fallback = False
    log.info("[query_model] Отправляю запрос: openrouter_until=%.2f, now=%.2f", openrouter_until, now)
    if now >= openrouter_until:
        log.info("[query_model] Разрешён запрос в OpenRouter")
        txt = await _call_openrouter(messages)
        if txt is not None:
            log.info("[query_model] Ответ получен от OpenRouter")
            return txt, False
        fallback = True
        openrouter_until = now + cfg.OPENROUTER_BLOCK_SEC
        log.warning("[query_model] Переключаюсь на локальную модель, блок OpenRouter до %.2f", openrouter_until)
    else:
        fallback = True
        log.info("[query_model] Использую fallback, OpenRouter заблокирован")

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
        log.debug("[query_model] Отправляю POST в Ollama (%s)...", cfg.OLLAMA_URL)
        async with s.post(
            cfg.OLLAMA_URL,
            json={"model": cfg.LOCAL_MODEL, "prompt": prompt_text, "stream": False},
            timeout=aiohttp.ClientTimeout(total=timeout_sec)
        ) as r:
            log.info("[query_model] HTTP-ответ: %d", r.status)
            data = await r.json()
            log.info("[query_model] Ответ от локальной модели: %r", data)
            return data.get("response", ""), True
    except asyncio.TimeoutError:
        log.error("[query_model] TimeoutError: LLM не отвечает за %d секунд", timeout_sec)
        return "⚠️ LLM не отвечает (timeout). Попробуйте позже.", True
    except Exception as e:
        log.error("[query_model] Ошибка при запросе к LLM: %s", e, exc_info=True)
        return f"⚠️ Ошибка при обращении к LLM: {e}", True

def _cleanup_user_last() -> None:
    log.debug("[_cleanup_user_last] Старт очистки user_last.")
    cutoff = time.monotonic() - USER_LAST_CLEAN
    orig_len = len(user_last)
    for uid, ts in list(user_last.items()):
        if ts < cutoff:
            user_last.pop(uid, None)
            log.debug("[_cleanup_user_last] Удалён user_id=%d (ts=%.2f < cutoff=%.2f)", uid, ts, cutoff)
    disc_log.debug("[_cleanup_user_last] Очистка user_last: было %d, стало %d", orig_len, len(user_last))

async def generate_rag_answer(q: str, sys_prompt: str, use_remote: bool):
    log.info("[generate_rag_answer] Получен запрос: %r (use_remote=%r)", q, use_remote)
    qlem = _extract_lemmas(q)
    log.debug("[generate_rag_answer] Леммы вопроса: %r", list(qlem))
    log.debug("[generate_rag_answer] Получаю raw_nodes...")
    raw_nodes = await retriever.aretrieve(q)
    log.debug("[generate_rag_answer] Получено raw_nodes: %d", len(raw_nodes))

    reranked_nodes = await rerank(q, raw_nodes)
    log.debug("[generate_rag_answer] После rerank: %d", len(reranked_nodes) if reranked_nodes else 0)

    # Подробный лог по содержимому reranked_nodes (top 10)
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
        log.warning("[generate_rag_answer] reranked_nodes пуст или None")

    nodes = await get_filtered_nodes(reranked_nodes or raw_nodes, qlem)
    log.info("[generate_rag_answer] Отфильтровано nodes: %d", len(nodes) if nodes else 0)
    if not nodes:
        log.warning("[generate_rag_answer] Недостаточно данных для ответа.")
        return "⚠️ Недостаточно данных.", None, None
    char_limit = CTX_LEN_REMOTE if use_remote else CTX_LEN_LOCAL
    log.debug("[generate_rag_answer] char_limit=%d", char_limit)
    ctx_txt = build_context(nodes, qlem, char_limit)
    log.debug("[generate_rag_answer] Контекст сформирован, длина=%d", len(ctx_txt))
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
    disc_log.info("[ask_rag] Вызвано в чате #%d (%s) от %s(%d)", ctx.channel.id, ctx.channel, ctx.author.name, ctx.author.id)
    if ctx.channel.id not in cfg.ALLOWED_CHANNELS:
        disc_log.warning("[ask_rag] Канал %d не разрешён, игнор.", ctx.channel.id)
        return

    q = sanitize(q_raw.strip().replace("\n", " "))
    disc_log.info("[ask_rag] Запрос: %r", q)
    if len(q) > cfg.MAX_QUESTION_LEN or not cfg.ALLOWED_CHARS.match(q):
        disc_log.warning("[ask_rag] Неверный формат запроса: len=%d, match=%r", len(q), cfg.ALLOWED_CHARS.match(q))
        await ctx.send("❌ Неверный формат запроса.")
        return

    now = time.monotonic()
    if now - user_last.get(ctx.author.id, 0) < cfg.USER_COOLDOWN:
        disc_log.info("[ask_rag] Пользователь %d в кулдауне.", ctx.author.id)
        await ctx.send("⏳ Подождите немного.")
        return
    user_last[ctx.author.id] = now
    _cleanup_user_last()

    disc_log.info("cmd=%s user=%s(%d) ch=%d q=%r", ctx.command, ctx.author.name, ctx.author.id, ctx.channel.id, q)
    disc_log.debug("ask_rag: user_last=%r FILTER_CACHE=%d", user_last, len(FILTER_CACHE))
    await ctx.send("🔍 Думаю…")
    async with cfg.REQUEST_SEMAPHORE:
        disc_log.info("[ask_rag] Запуск RAG-пайплайна для запроса.")

        use_remote = time.time() < openrouter_until
        disc_log.debug("[ask_rag] use_remote=%r (openrouter_until=%.2f, now=%.2f)", use_remote, openrouter_until, time.time())
        sys_prompt_ = Path(PROMPT_STRICT).read_text("utf-8") if sys_prompt is None else sys_prompt
        prompt, nodes, ctx_txt = await generate_rag_answer(q, sys_prompt_, use_remote)
        if not nodes or prompt == "⚠️ Недостаточно данных.":
            disc_log.info("[ask_rag] Нет подходящих документов для ответа.")
            await ctx.send("⚠️ Недостаточно данных.")
            return

        disc_log.debug("[ask_rag] prompt для модели: %r", prompt)
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
            disc_log.warning("[ask_rag] OpenRouter недоступен, используется локальная модель.")
            await send_long(ctx, "⚠️ OpenRouter недоступен, используется локальная модель.")

        try:
            disc_log.info("[ask_rag] Отправка ответа длиной %d", len(answer or ""))
            await send_long(ctx, answer or "❌ Нет ответа.")
        except discord.HTTPException as e:
            disc_log.error("[ask_rag] Discord HTTPException: %s", e)
            await send_long(ctx, f"⚠️ Ответ слишком длинный: {e}")

# ────────────────────────────────

# BOT COMMANDS
# ────────────────────────────────
@bot.command(name="справка")
async def cmd_spravka(ctx: commands.Context, *, q: str):
    disc_log.info("[cmd:справка] user=%s(%d) channel=%s(%d) q=%r",
                  ctx.author.name, ctx.author.id, ctx.channel.name, ctx.channel.id, q)
    await ask_rag(ctx, q, Path(PROMPT_STRICT).read_text("utf-8"))

@bot.command(name="локально", help="RAG-ответ только локально, минуя облако")
async def cmd_local_llm(ctx: commands.Context, *, q: str):
    disc_log.info("[cmd:локально] user=%s(%d) channel=%s(%d) q=%r",
                  ctx.author.name, ctx.author.id, ctx.channel.name, ctx.channel.id, q)
    if ctx.channel.id not in cfg.ALLOWED_CHANNELS:
        return

    q = sanitize(q.strip().replace("\n", " "))
    if len(q) > cfg.MAX_QUESTION_LEN or not cfg.ALLOWED_CHARS.match(q):
        await ctx.send("❌ Неверный формат запроса.")
        return

    now = time.monotonic()
    if now - user_last.get(ctx.author.id, 0) < cfg.USER_COOLDOWN:
        await ctx.send("⏳ Подождите немного.")
        return
    user_last[ctx.author.id] = now
    _cleanup_user_last()

    await ctx.send("🧠 Думаю локально…")
    async with cfg.REQUEST_SEMAPHORE:
        sys_prompt = Path(PROMPT_STRICT).read_text("utf-8")
        prompt, nodes, ctx_txt = await generate_rag_answer(q, sys_prompt, use_remote=False)
        if not nodes or prompt == "⚠️ Недостаточно данных.":
            await ctx.send("⚠️ Недостаточно данных.")
            return

        disc_log.debug("[cmd:локально] prompt для локальной LLM: %r", prompt)
        answer = await call_local_llm(prompt)
        try:
            await send_long(ctx, answer or "❌ Нет ответа.")
        except discord.HTTPException as e:
            await send_long(ctx, f"⚠️ Ответ слишком длинный: {e}")

@bot.command(name="подумай")
async def cmd_podumay(ctx: commands.Context, *, q: str):
    disc_log.info("[cmd:подумай] user=%s(%d) channel=%s(%d) q=%r",
                  ctx.author.name, ctx.author.id, ctx.channel.name, ctx.channel.id, q)
    await ask_rag(ctx, q, Path(PROMPT_REASON).read_text("utf-8"))

@bot.command(name="статус")
async def cmd_status(ctx: commands.Context):
    disc_log.info("[cmd:статус] user=%s(%d) channel=%s(%d)",
                  ctx.author.name, ctx.author.id, ctx.channel.name, ctx.channel.id)
    await ctx.send(
        f"🧠 Документов: {len(index.docstore.docs)}\n"
        f"💾 Кеш: {'да' if cfg.CACHE_PATH.exists() else 'нет'}\n"
        f"🌐 OpenRouter: {'блок' if time.time() < openrouter_until else 'ok'}"
    )

@bot.command(name="reload_index")
async def cmd_reload_index(ctx: commands.Context):
    disc_log.info("[cmd:reload_index] user=%s(%d) channel=%s(%d)",
                  ctx.author.name, ctx.author.id, ctx.channel.name, ctx.channel.id)
    if ctx.author.id not in ADMIN_IDS:
        disc_log.warning("[cmd:reload_index] Отказ: нет доступа")
        await ctx.send("❌ Нет доступа.")
        return
    FILTER_CACHE.clear()
    global index, retriever
    index = await build_index()
    retriever = index.as_retriever(similarity_top_k=cfg.TOP_K)
    await ctx.send("✅ Индекс перезагружен.")

@bot.command(name="stop")
async def cmd_stop(ctx: commands.Context):
    disc_log.info("[cmd:stop] user=%s(%d) channel=%s(%d)",
                  ctx.author.name, ctx.author.id, ctx.channel.name, ctx.channel.id)
    if ctx.author.id not in ADMIN_IDS:
        disc_log.warning("[cmd:stop] Отказ: нет доступа")
        await ctx.send("❌ Нет доступа.")
        return
    await ctx.send("🛑 Останавливаю бота...")
    disc_log.info("[cmd:stop] Инициировано завершение работы бота...")
    await shutdown()

@bot.command(name="многозапрос")
async def cmd_multirag(ctx: commands.Context, *, q: str):
    if ctx.channel.id not in cfg.ALLOWED_CHANNELS:
        return
    await ctx.send("🔎 Выполняю многошаговый анализ…")
    sys_prompt = Path(cfg.PROMPT_STRICT).read_text("utf-8")
    try:
        answer = await multi_query_rag(q.strip(), sys_prompt)
    except Exception as e:
        await ctx.send(f"❌ Ошибка: {e}")
        raise
    await send_long(ctx, answer or "❌ Нет ответа.")


# ────────────────────────────────
# BOT LIFECYCLE
# ────────────────────────────────
@bot.event
async def on_ready():
    log.info("Бот запущен как %s (ID: %d)", bot.user, bot.user.id)
    print(f"🟢 Discord-бот запущен как {bot.user} (ID: {bot.user.id})")
    try:
        ch = bot.get_channel(next(iter(cfg.ALLOWED_CHANNELS)))
        if ch:
            await ch.send("✅ Бот запущен и готов к работе.")
    except Exception as e:
        log.error("[on_ready] Не удалось отправить уведомление о запуске: %s", e)

async def shutdown() -> None:
    try:
        ch = bot.get_channel(next(iter(cfg.ALLOWED_CHANNELS)))
        if ch:
            await ch.send("🛑 Бот остановлен.")
    except Exception as e:
        log.error("[shutdown] Не удалось отправить уведомление об остановке: %s", e)
    await bot.close()
    await session_holder.close()
    await shutdown_reranker()
    LEMMA_POOL.shutdown(wait=True)

if sys.platform != "win32":
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown()))


# ────────────────────────────────
# ENTRY POINT
# ────────────────────────────────
if __name__ == "__main__":
    print("🔄 Построение индекса документов...")
    index = asyncio.run(build_index())
    retriever = index.as_retriever(similarity_top_k=cfg.TOP_K)
    asyncio.run(init_reranker())
    print("✅ Индекс построен. Запускаем Discord-бота...")
    log.info("✅ Индекс успешно построен, бот запускается.")
    bot.run(DISCORD_TOKEN, log_handler=None)
