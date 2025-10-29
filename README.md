# 🧠 Asketmc RAG Discord Bot — Local LLM + Hybrid Retrieval

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![LLM](https://img.shields.io/badge/LLM-Ollama%20%7C%20Llama3-orange)
![Discord](https://img.shields.io/badge/Discord-Bot-informational)
![NLP](https://img.shields.io/badge/NLP-Stanza%20%7C%20spaCy-purple)
![Status](https://img.shields.io/badge/status-internal%20use-yellow)
![Updated](https://img.shields.io/badge/last%20update-October%202025-blueviolet)

> Hybrid-RAG Discord assistant for file-based knowledge bases (RU/EN). Vector retrieval + keyword fallback, CrossEncoder rerank, secure async bot runtime. Designed for reliability, observability, and low-latency local inference.

---

## ✅ Scope & Value Proposition

This repository demonstrates a production-style **Retrieval-Augmented Generation (RAG)** workload with:

* **Local inference** via **Ollama + Llama3**
* **Hybrid retrieval** (`llama-index` + keyword fallback)
* **Dual-stage reranking** with `CrossEncoder`
* **Russian/English lemmatization** (Stanza, spaCy)
* **Hardened Discord bot** (RBAC, cooldowns, sanitization)

Primary use cases: internal documentation assistants, QA/dev tooling, knowledge-heavy game worlds.

---

## 🔍 Feature Highlights

* **Hybrid Retrieval** — vector search (BAAI/bge) + keyword fallback with score gating
* **Rerank Pipeline** — `BAAI/bge-reranker-v2-m3` (CPU/GPU switchable)
* **Lemmatized KB** — RU+EN, SHA256 caches, sentence chunking, per-file indices
* **Discord Runtime** — RBAC, command parsing, rate/cooldown limiting, message sanitization
* **Resilience** — circuit breaker + automatic fallback to local LLM when OpenRouter is unavailable
* **Telemetry** — rotating log channels per module (`app`, `error`, `rerank`, `embed`, `rag`)
* **Config as Code** — `.env` + `config.py` + externalized prompts

---

## 🧠 Multistep RAG (`!multy`)

`!multy` decomposes the primary question `Q0 → [Q1..Qn]`, runs per-question retrieve+rerank, merges candidates, and executes a final rerank against `Q0`. Tunables (`K`, `R`, `F`) are set in `config.py`.

Benefits: high recall, per-facet relevance, coherent final grounding, reduced hallucinations.

---

## 🛠️ Technology Stack

| Layer         | Tooling / Library                     |
| ------------- | ------------------------------------- |
| LLM           | `ollama` + `llama3` (local inference) |
| Retrieval     | `llama-index` + `VectorStoreIndex`    |
| Embeddings    | `BAAI/bge-m3`                         |
| Rerank        | `CrossEncoder` (`bge-reranker-v2-m3`) |
| Lemmatization | `stanza`, `spaCy`, `langdetect`       |
| Bot API       | `discord.py`, `aiohttp`               |
| Infra         | `asyncio`, rotating logs, `.env`      |

---

## 🔄 Model Routing Strategy

* **Primary:** OpenRouter (**DeepSeek-v3**)
* **Fallback:** Local **Llama3‑8B** via Ollama (validated on GeForce **GTX 1060 6GB**)

**Supported / Tested**

* DeepSeek‑v3 (OpenRouter)
* DeepSeek‑v1 8B
* Phi‑3 Mini
* Llama3‑8B (local)

---

## 📁 Repository Layout

```text
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

## 💬 Command Surface (Discord)

* `!strict <q>` — RAG with rerank (factual QA)
* `!local <q>` — force local LLM only
* `!think <q>` — alternate prompt mode
* `!multy <q>` — decomposed retrieval pipeline
* `!reload_index` — rebuild index (admin)
* `!status` — status/diagnostics
* `!stop` — controlled shutdown (admin)

---

## 🚀 Setup & Run

> **Prereqs**: Python **3.10+**, (optional) CUDA-capable GPU for reranker/LLM acceleration.

From repo root:

```bash
cd LLM/bot
python3.10 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# language models for NLP
python -m spacy download en_core_web_sm
python -m stanza.download ru
```

Create `.env` (see `.env-example`):

```ini
DISCORD_TOKEN=your_discord_bot_token
OPENROUTER_API_KEY=your_openrouter_api_key
# optional overrides
# API_URL=https://openrouter.ai/api/v1/chat/completions
# OR_MODEL=openrouter/auto
# OR_MAX_TOKENS=512
# OLLAMA_URL=http://localhost:11434/api/generate
# LOCAL_MODEL=llama3:8b
# HTTP_CONN_LIMIT=5
# OR_RETRIES=3
# OPENROUTER_BLOCK_SEC=900
# CTX_LEN_REMOTE=20000
# CTX_LEN_LOCAL=12000
# TOP_K=16
# HTTP_TIMEOUT_TOTAL=240
```

Add your knowledge base under `LLM/parsed/` (Markdown/Text).

Run the bot:

```bash
python main.py
```

---

## 🧪 QA, Telemetry & Troubleshooting

* **Traceability**: retrieve → rerank → context assembly → LLM call (timestamps + durations)
* **Debug surfaces**: similarity scores, cache hits, selected context chunks
* **Input hygiene**: character filtering, max length guards, unique-token thresholds
* **Controls**: cooldowns, admin whitelist, regex filters
* **Fallbacks**: vector ↔ keyword ↔ local LLM; OpenRouter circuit breaker

**Common checks**

* Ensure `.env` is loaded from `LLM/bot` or parent dir
* Validate GPU visibility if using CUDA (`torch.cuda.is_available()` at startup logs)
* Clear/rebuild `rag_cache/` when changing corpora

---

## 🔒 Security & Compliance

* Secrets via `.env`; never commit tokens or logs containing credentials
* Minimal Discord bot permissions (principle of least privilege)
* Network hardening: timeouts, retries, circuit breaker for upstream provider
* Log redaction paths reserved for sensitive payloads

---

## ⚙️ Operations (Config & Observability)

* Configuration driven by `config.py` + environment variables
* Rotating logs under `LLM/logs/` with separate channels (`app`, `error`, `rerank`, `embed`, `rag`)
* CrossEncoder runs on CPU by default; can switch to GPU if available
* Context length limits for remote/local models are tunable to meet latency/SLOs

---

## 📈 Performance & Resource Profile (reference rig)

**Hardware**: GeForce **GTX 1060 6GB**, **Intel Core i7‑7700K** @ **4.5 GHz**

**VRAM (Ollama + reranker)**

* Active RAG runs: **~2.5 GB → ~5.3 GB** (increase due to sequential model loads)

**Host RAM (per‑process working set)**

* `python.exe`: **~3.9 GB**
* `ollama`: **~3.4 GB**

**CPU utilization**

* Idle: **~0%**
* Strict local query (active): **~50%**

> Figures vary with model choice, prompt/context size, and corpus volume.

---

## ⚠️ Known Limitations & Trade‑offs

* GTX 1060 6GB constrains local model sizes and throughput
* Lemmatization (RU/EN) may miss rare morphological forms
* Answer quality depends on KB structure and source quality in `parsed/`

---

## 🧭 Roadmap (short)

* Optional metrics export (latency p50/p95 for retrieve/rerank/LLM)
* Lightweight eval harness for regression testing on a seed query set
* Threat model note for token handling & Discord scopes

---

## 📌 Notes

* Focus on clarity of the pipeline and failure modes; designed for constrained GPUs and intermittent upstream availability.

---

## 🛡 License

**MIT**

**Contact**: [asketmc.team+ragbot@gmail.com](mailto:asketmc.team+ragbot@gmail.com)
