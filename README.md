![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![LLM](https://img.shields.io/badge/LLM-Ollama%20%7C%20Llama3-orange)
![Discord](https://img.shields.io/badge/Discord-Bot-informational)
![NLP](https://img.shields.io/badge/NLP-Stanza%20%7C%20spaCy-purple)

# Asketmc RAG Discord Bot

Discord bot for question answering over a local RU/EN knowledge base using Retrieval-Augmented Generation (RAG).

---

## Purpose and Behavior

* **Retrieval**: `llama_index.core.VectorStoreIndex` + `SentenceSplitter` (chunk size/overlap from `config.py`).
* **Embeddings**: `HuggingFaceEmbedding("BAAI/bge-m3", normalize=True, device=("cuda" if available else "cpu"))`.
* **Lemmatization**: Russian via `stanza`, English via `spaCy en_core_web_sm`; language detection via `langdetect`.
* **Index persistence**: stored in `cfg.CACHE_PATH`; per-file SHA-256 hashes (`cfg.HASH_FILE`); index rebuilt only when hashes change.
* **Reranking**: `sentence_transformers.CrossEncoder` model from config (default `BAAI/bge-reranker-v2-m3`); device from `RERANKER_DEVICE`.
* **Filtering/context**: lemma intersection and relative score threshold (`cfg.SCORE_RELATIVE_THRESHOLD`, `cfg.LEMMA_MATCH_RATIO`, `cfg.TOP_K`). Context assembled up to a character limit.
* **LLM routing**: primary via OpenRouter (configurable URL/model/limits) with retries and circuit breaker; fallback to local Ollama (`cfg.LOCAL_MODEL`).
* **Discord runtime**: commands, channel/admin allow-lists, per-user cooldown, concurrency semaphore, input validation/sanitization, long message splitting.
* **Shutdown**: handles `SIGINT`/`SIGTERM`; closes HTTP sessions, reranker, and lemma thread pool.

---

## Behavior

* Queries exceeding `cfg.MAX_QUESTION_LEN` or failing `cfg.ALLOWED_CHARS` regex return `❌ Invalid query format.`
* Per-user cooldown: `cfg.USER_COOLDOWN` seconds.
* Parallel requests limited by `cfg.REQUEST_SEMAPHORE` (default: 3).
* If no relevant context after retrieval/reranking/filtering: returns `⚠️ Not enough data.`
* On remote path block: routes to local model and notifies with `⚠️ OpenRouter unavailable, local model used.`

---

## Architecture

1. **Indexing** (`index_builder.py`)

   * Configures `Settings.embed_model` and `Settings.node_parser`.
   * Reads documents from `cfg.DOCS_PATH`.
   * Computes SHA-256 hashes; loads cached index if unchanged.
   * Updates lemma cache (`lemma.FILE_LEMMAS`), assigns lemmas to each node.
   * Persists index, hashes, and lemma caches.

2. **Lemmatization** (`lemma.py`)

   * Initializes `stanza` (RU) and `spaCy` (EN); language detection via `langdetect`.
   * Uses a `ThreadPoolExecutor` for concurrent processing.
   * Persists lemma caches (`rag_cache/*.json`).

3. **Reranker** (`rerank.py`)

   * Loads `CrossEncoder` (CPU/GPU depending on config).
   * `rerank()` validates query (regex/length), performs scoring, and returns top `cfg.RERANK_OUTPUT_K` nodes.
   * `shutdown_reranker()` releases resources and clears CUDA cache if used.

4. **Filtering and Context** (`rag_filter.py`)

   * Filters nodes based on lemma overlap and score thresholds.
   * Builds context string up to configured length.
   * Caches results (LRU-based).

5. **LLM Client** (`llm_client.py`)

   * Manages shared `aiohttp` session with connection limits and timeouts.
   * Handles OpenRouter retries, 401/429/5xx codes, exponential backoff, and circuit breaker.
   * Fallback to local Ollama (`/api/generate`, `stream=False`).
   * Exposes breaker state for sync checks.

6. **Discord Bot** (`discord_bot.py`)

   * Commands: `!strict`, `!think`, `!local`, `!status`, `!reload_index` (admin), `!stop` (admin).
   * Guards: `cfg.ALLOWED_CHANNELS`, `cfg.ADMIN_IDS`, regex validation, cooldown, semaphore.
   * Sanitizes mentions/code fences; splits long replies.

7. **Entry Point** (`main.py`)

   * Loads `.env` (`DISCORD_TOKEN`, `OPENROUTER_API_KEY` required).
   * Sets up logging, builds index, initializes reranker and LLM client.
   * Starts Discord bot.
   * Handles shutdown signals and closes all components.

8. **Optional** (`rag_langchain.py`)

   * Defines a LangGraph RAG pipeline (not used in main runtime).

---

## Discord Commands

* `!strict <question>` — RAG with factual system prompt (`cfg.PROMPT_STRICT`).
* `!think <question>` — alternate prompt (`cfg.PROMPT_REASON`).
* `!local <question>` — use local model only.
* `!status` — show document count, cache state, and breaker status.
* `!reload_index` — rebuild index (admin only).
* `!stop` — orderly shutdown (admin only).

---

## Configuration

**Required (.env):**

* `DISCORD_TOKEN`
* `OPENROUTER_API_KEY`

**Key parameters (`config.py`):**

* Paths: `VAR_ROOT`, `CACHE_PATH`, `HASH_FILE`, `DATA_ROOT`, `DOCS_PATH`, `PROMPTS_DIR`, `PROMPT_STRICT`, `PROMPT_REASON`.
* Models/routing: `API_URL`, `OR_MODEL`, `OR_MAX_TOKENS`, `OR_RETRIES`, `OLLAMA_URL`, `LOCAL_MODEL`, `HTTP_CONN_LIMIT`, `HTTP_TIMEOUT_TOTAL`, `OPENROUTER_BLOCK_SEC`, `OPENROUTER_BLOCK_MAX_SEC`.
* RAG: `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `LEMMA_MATCH_RATIO`, `SCORE_RELATIVE_THRESHOLD`, `CTX_LEN_REMOTE`, `CTX_LEN_LOCAL`.
* Rerank: `RERANKER_MODEL_NAME`, `RERANK_INPUT_K`, `RERANK_OUTPUT_K`, `BATCH_SIZE`, `MAX_LEN`, `QUERY_MAX_CHARS`, `EXECUTOR_WORKERS`, `RERANKER_DEVICE`.
* Discord: `ALLOWED_CHANNELS`, `MAX_QUESTION_LEN`, `USER_COOLDOWN`, `REQUEST_SEMAPHORE`, `ALLOWED_CHARS`, `ADMIN_IDS`.

**Optional environment overrides:**

* `ASKETMC_VAR_DIR`, `ASKETMC_DATA_DIR`, `ASKETMC_PROMPTS_DIR`

---

## Setup and Run

**Requirements:**

* Python 3.10+
* Optional CUDA GPU for reranker.
* Installed models: stanza (RU), spaCy (EN), `BAAI/bge-m3` (auto-downloaded).

**Steps:**

```bash
python3.10 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python -m spacy download en_core_web_sm
python - <<'PY'
import stanza; stanza.download('ru')
PY
```

Create `.env`:

```ini
DISCORD_TOKEN=...
OPENROUTER_API_KEY=...
```

Run:

```bash
cd src/asketmc_bot
python main.py
```

> Local model requires a running Ollama instance with `cfg.LOCAL_MODEL` pulled.

---

## Repository Layout

```
LLM/
├─ .env                       # canonical env (DISCORD_TOKEN, OPENROUTER_API_KEY, …)
├─ .env-example               # non-secret sample
├─ .gitignore
├─ pyproject.toml             # src-layout packaging (askemc-bot)
├─ README.md
├─ requirements.txt
├─ requirements-dev.txt
├─ requirements-backup.txt
├─ data/                      # input data (immutable / versioned)
│  ├─ parsed/                 # KB sources (md/txt)
│  └─ parsers/                # offline preprocessors
├─ src/
│  └─ asketmc_bot/            # Python package (import: asketmc_bot.*)
│     ├─ __init__.py
│     ├─ config.py            # paths, tunables (uses VAR_ROOT/DATA_ROOT)
│     ├─ main.py              # async entrypoint, DI, lifecycle, .env loader
│     ├─ discord_bot.py       # commands, RBAC, cooldowns
│     ├─ index_builder.py     # vector index build/cache/load
│     ├─ rag_filter.py        # hybrid keyword + semantic filtering
│     ├─ rerank.py            # CrossEncoder init & scoring
│     ├─ lemma.py             # RU/EN lemmatization (thread pool)
│     ├─ rag_langchain.py     # optional LC pipeline (experimental)
│     ├─ data/                # (optional) package-local resources
│     └─ rag_cache/           # (temporary; prefer var/rag_cache in prod)
├─ tests/                     # test suite (pytest)
│  ├─ test_entrypoint.py      # smoke: docstring + entry guard
│  └─ test_query_model_unit.py# unit: query_model + circuit breaker
└─ var/                       # mutable runtime artifacts (not versioned)
   ├─ logs/                   # rotating logs (app/error/rag/…)
   └─ rag_cache/              # indices/cache (runtime)
```

---

## Security and Permissions

* Uses `discord.Intents.all()` but enforces `cfg.ALLOWED_CHANNELS` and `cfg.ADMIN_IDS`.
* Secrets loaded from `.env`.
* Input sanitized (mentions/code fences removed, regex-validated).
* Logs/caches/data stored under project-local directories.

---

## License

MIT

Contact: [asketmc.team+ragbot@gmail.com](mailto:asketmc.team+ragbot@gmail.com)
