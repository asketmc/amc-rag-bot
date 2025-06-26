# 🧠 Asketmc RAG Discord Bot — Local LLM + Hybrid Retrieval

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![LLM](https://img.shields.io/badge/LLM-Ollama%20%7C%20Llama3-orange)
![Discord](https://img.shields.io/badge/Discord-Bot-informational)
![NLP](https://img.shields.io/badge/NLP-Stanza%20%7C%20spaCy-purple)
![Status](https://img.shields.io/badge/status-internal%20use-yellow)
![Updated](https://img.shields.io/badge/last%20update-June%202025-blueviolet)

> Lightweight hybrid-RAG Discord bot (vector + keyword fallback) for Russian-language file-based knowledge bases.  
> Built in 2 days from scratch (first-time Python) as a demonstration of LLM orchestration, retrieval QA logic, and modular design.

---

## ✅ Purpose & Highlights

This project was built **in just 2 days**, with no prior Python experience.  
It demonstrates how to build a fully functional **Retrieval-Augmented Generation (RAG)** assistant using:

- **Local LLM inference** (Ollama + Llama3)
- **Hybrid retrieval** via `llama-index` + keyword fallback
- **Full rerank pipeline** with `CrossEncoder`
- **Russian-language lemmatization** (Stanza, spaCy)
- **Secure, async-safe Discord bot** architecture

Originally developed for an RPG Minecraft server, this RAG bot is also designed to support internal tech documentation — e.g., for DevOps/QA teams.  
It supports structured configs, multilingual corpora, and real-world adaptation (e.g., Salesforce internal docs at Fleetcor).

---

## 🔍 Key Features

- **Hybrid Retrieval**: vector search (BAAI/bge) + keyword fallback with score filtering  
- **Async-safe rerank**: `BAAI/bge-reranker-v2-m3` with thread pool + CPU/GPU switch  
- **Lemmatized KB**: Russian+English, SHA256 cache, sentence chunking, per-file indices  
- **Discord Bot**: role-based access, command parsing, cooldowns, message sanitization  
- **Fault Tolerance**: automatic fallback to local LLM on OpenRouter errors  
- **Full Logging**: rotating logs per module (`chat`, `embedding`, `errors`, etc.)  
- **Configurable**: `.env` + `config.py` + isolated prompt files (`system_prompt.txt`, etc.)  
- **Flexible Backend**: tested with multiple LLMs, easily switchable via config  

---

## 🧠 Advanced Multistep RAG (`!multy`)

The `!multy` command implements a **multi-question, dual-rerank pipeline** for complex queries — ideal for lore-heavy worlds or layered documentation.

---

### ⚙️ How it works

**Step 1:** Decompose original query `Q0` → [`Q1`, `Q2`, `Q3`] via LLM

**Step 2:** For each `Qn`:
- Retrieve top‑**K** chunks (_default: 18, `config.RERANK_INPUT_K`_)
- Rerank with `CrossEncoder` (`BAAI/bge-reranker-v2-m3`)
- Select top‑**R** chunks (_default: 9, `config.RERANK_OUTPUT_K`_) via semantic + lemma-based filtering

**Step 3:** Merge all filtered chunks → `[C1 ... Cn]`

**Step 4:** Final rerank: `Q0` vs `[C1 ... Cn]`
- Select top‑**F** chunks (_default: 16, `config.TOP_K`_) for context

**Step 5:** Query LLM with:
- `SYSTEM` prompt  
- Final `CONTEXT`  
- Original `QUESTION = Q0`

---

### ✅ Benefits

- ✅ **High recall** — decomposition covers multiple facets of the original query  
- ✅ **Per‑question relevance** — each `Qn` is filtered independently  
- ✅ **Global coherence** — final rerank aligns all content to the original `Q0`  
- ✅ **Low hallucination** — responses are grounded in retrieved facts only  

Parameters like `K`, `R`, `F` are fully configurable in `config.py`.

> This structure is ideal for **multi-hop RAG**, world modeling, and decomposed fact retrieval.

---

## 🛠️ Stack

| Layer         | Tool / Library                        |
|---------------|---------------------------------------|
| LLM           | `ollama` + `llama3` (local inference) |
| Retrieval     | `llama-index` + VectorStoreIndex      |
| Embeddings    | `BAAI/bge-m3`                         |
| Rerank        | `CrossEncoder` (`bge-reranker-v2-m3`) |
| Lemmatization | `stanza`, `spaCy`, `langdetect`       |
| Bot API       | `discord.py`, `aiohttp`               |
| Infra         | `asyncio`, `rotating logs`, `.env` isolation |

---

### 🔄 Model Routing

By default, the bot uses **OpenRouter (DeepSeek-v3)** for high-quality completions.  
If OpenRouter is unavailable (e.g., quota exceeded or downtime), it falls back to **local inference (Llama3-8B)** via Ollama on RTX 1060 6GB.

#### ✅ Tested / Supported Models:
- DeepSeek-v3 (OpenRouter)
- DeepSeek-v1 8B
- Phi-3 Mini
- Llama3-8B (local)

**Final choices**:
- **DeepSeek-v3** — for high-quality reasoning
- **Llama3-8B** — for reliable results under resource constraints

---

## 📁 File Layout
```text
/LLM
├── bot/ # Main bot application (entry point, core logic)
│ ├── main.py # Entry point: init, load index, launch Discord bot
│ ├── config.py # Global settings and constants
│ ├── rerank.py # CrossEncoder reranking logic
│ ├── requirements.txt # Python dependencies (active version)
│ ├── requirements-backup.txt # Backup dependency list (pip freeze)
│ ├── rag_cache/ # Vector index + lemma cache
│ └── pycache/ # Python bytecode cache
│
├── parsed/ # Input text/markdown documents for knowledge base
├── logs/ # Rotating log files (runtime/debug output)
├── parsers/ # Optional data parsers / preprocessors (WIP)
│
├── system_prompt_reason.txt # System prompt for reasoning / verbose QA mode
├── system_prompt_strict.txt # System prompt for strict factual QA
├── rephrase.txt # Prompt for question rewriting (optional)
│
├── .env # Environment variables (API keys etc.)
├── .env-example # Example env file for configuration
├── .gitignore # Git exclusion rules
└── README.md # Project documentation
---
```
## 🧪 QA & Observability

- Full trace logs (per block, per retrieval step)
- Debug mode shows similarity scores, cache hits, context chunks
- Input sanitization: char filter, length guard, unique word threshold
- Commands protected by cooldowns, admin whitelist, regex filters
- Clean fallback between vector / keyword / local LLM

---

## 💬 Example Log — Step-by-Step Breakdown

```text
📥 [Retrieval Stage]
[INFO ] RAG: Received query → "Does this character bio contain lore violations?"
[INFO ] Lemma: Loaded index with 45 documents, 0 new files detected
[INFO ] Vector Search: Top 24 chunks matched (BAAI/bge-m3)

🧠 [Rerank Stage]
[DEBUG] Rerank: Using model BAAI/bge-reranker-v2-m3 (CPU), query_len=456, input_k=18
[DEBUG] Filtering 18 candidate pairs (sample: "...Night Spirit. Her followers...")
[INFO ] Rerank completed in 37.45 sec
[DEBUG] Top scores: [0.728, 0.013, 0.0027, ...]

🌐 [LLM Query Stage]
[INFO ] Fallback system: OpenRouter access allowed
[INFO ] OpenRouter call started (model: DeepSeek-v3)
[INFO ] OpenRouter HTTP 200 OK
[INFO ] OpenRouter response successfully received
```

📢 Discord Commands
!strict <question> — standard RAG reply with rerank

!local <question> — local LLM-only answer

!think <question> — alternate prompt mode

!reload_index — admin-only index rebuild

!status — debug/info panel

!stop — admin-only shutdown

!multy <question> — multistep decomposed retrieval pipeline (for complex queries)

🛠️ Setup
Clone the repo
```text
git clone https://github.com/youruser/amc-rag-bot.git
cd amc-rag-bot
```
Install dependencies
```text
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m stanza.download ru
```
Add API keys
```text
DISCORD_TOKEN=...
OPENROUTER_API_KEY=...
```
Place your knowledge base into /parsed/

Run the bot
```text
python main.py
```
⚡ Why This Matters
This project shows how an LLM QA Engineer or RAG Architect can:

Build a working hybrid retrieval system

Handle edge cases, multilingual input, and fallback routing

Deploy a real-time QA bot using only Python and open APIs

Deliver results under real constraints (2 days, no prior Python background)

Use this repo as a reference, PoC baseline, or technical interview sample.

🛡 License
MIT

📬 Contact: asketmc.team+ragbot@gmail.com
