"""
tests/test_query_model_unit.py

Purpose:
--------
This file contains *unit tests* for the function `query_model()` defined in main.py.

We test its decision-making logic — how it chooses between:
  • calling the remote cloud model (OpenRouter)
  • falling back to the local model (Ollama)
  • activating a circuit breaker when the remote call fails

The tests are isolated from any real dependencies.
All external modules (torch, llama_index, rerank, etc.) are replaced with lightweight *stubs*.
This guarantees fast, deterministic, and reproducible test runs.

Terminology for manual QA audience:
-----------------------------------
• Unit test → Tests one small "unit" of logic (a function or class) in isolation.
• Mock/stub → Fake object that replaces a real external dependency.
• Circuit breaker → A safety switch that stops sending requests to a broken external service for a period of time.
"""
# ── ensure project root (C:/LLM/bot) is importable ────────────────────────────
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ──────────────────────────────────────────────────────────────────────────────
# 🧩 STANDARD IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import sys
import types
import importlib
import os
import tempfile
from pathlib import Path
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# 🧠 STUBS: Replace heavy or external dependencies BEFORE importing main.py
# ──────────────────────────────────────────────────────────────────────────────
# Each "stub" is a fake lightweight module registered into sys.modules,
# so that when main.py tries to import them, Python finds these fake ones instead.
# This avoids importing large frameworks like PyTorch, llama_index, etc.


# --- Stub for torch (removes CUDA/VRAM checks) -------------------------------
torch = types.ModuleType("torch")
torch.__version__ = "0"
torch.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules["torch"] = torch


# --- Stub for rerank (skips sentence-transformers import) --------------------
rerank_mod = types.ModuleType("rerank")

async def init_reranker():
    """Fake async init function for reranker."""
    return None

async def rerank(q, nodes):
    """Fake async rerank — returns nodes unchanged."""
    return nodes

async def shutdown_reranker():
    """Fake async shutdown."""
    return None

# Assign our fake functions to the stub module
rerank_mod.init_reranker = init_reranker
rerank_mod.rerank = rerank
rerank_mod.shutdown_reranker = shutdown_reranker
sys.modules["rerank"] = rerank_mod


# --- Stub for llama_index (core + schema) ------------------------------------
# We fake just enough structure so that main.py and rag_filter.py imports succeed.

ll_pkg = types.ModuleType("llama_index")
ll_pkg.__path__ = []  # mark as package

ll_core = types.ModuleType("llama_index.core")
ll_core.__path__ = []  # mark as package

class VectorStoreIndex:
    """Minimal fake of the real VectorStoreIndex class."""
    def as_retriever(self, similarity_top_k=5):
        """Return a fake retriever with a dummy async method."""
        class _R:
            async def aretrieve(self, q):
                return []
        return _R()

ll_core.VectorStoreIndex = VectorStoreIndex

# Fake schema module used in rag_filter
ll_schema = types.ModuleType("llama_index.core.schema")

class TextNode:
    """Fake node class representing one text chunk."""
    def __init__(self, text=""):
        self.text = text

class NodeWithScore:
    """Fake wrapper adding a numeric similarity score to a node."""
    def __init__(self, node=None, score=0.0):
        self.node = node if node is not None else TextNode("")
        self.score = float(score)

ll_schema.TextNode = TextNode
ll_schema.NodeWithScore = NodeWithScore

# Register all of them into sys.modules
sys.modules["llama_index"] = ll_pkg
sys.modules["llama_index.core"] = ll_core
sys.modules["llama_index.core.schema"] = ll_schema


# --- Stub for lemma (fake lemmatization pool) --------------------------------
lemma_mod = types.ModuleType("lemma")
lemma_mod.extract_lemmas = lambda q: []  # always returns empty list

class _Pool:
    """Fake thread pool with a no-op shutdown method."""
    def shutdown(self, wait=True):
        pass

lemma_mod.LEMMA_POOL = _Pool()
sys.modules["lemma"] = lemma_mod


# --- Stub for index_builder ---------------------------------------------------
idx_builder = types.ModuleType("index_builder")

async def build_index():
    """Fake async index builder that returns a dummy retriever."""
    class _Idx:
        def as_retriever(self, similarity_top_k=5):
            class _R:
                async def aretrieve(self, q):
                    return []
            return _R()
    return _Idx()

idx_builder.build_index = build_index
sys.modules["index_builder"] = idx_builder


# --- Minimal fake config (so main.py logging setup doesn't crash) ------------
cfg = types.ModuleType("config")
tmp = Path(tempfile.mkdtemp(prefix="ragbot-unit-"))
(cfg.__dict__).update(
    dict(
        LOG_DIR=tmp / "logs",
        DEBUG=False,
        API_URL="https://example.test/api",
        OR_MODEL="openrouter/auto",
        OR_MAX_TOKENS=128,
        OLLAMA_URL="http://127.0.0.1:11434/api/generate",
        LOCAL_MODEL="llama3:8b",
        HTTP_CONN_LIMIT=2,
        OR_RETRIES=1,
        OPENROUTER_BLOCK_SEC=1,
        CTX_LEN_REMOTE=1000,
        CTX_LEN_LOCAL=1000,
        TOP_K=4,
        HTTP_TIMEOUT_TOTAL=5,
    )
)
sys.modules["config"] = cfg


# --- Fake environment variables (required by load_settings()) ----------------
os.environ.setdefault("DISCORD_TOKEN", "dummydiscord")
os.environ.setdefault("OPENROUTER_API_KEY", "dummykey")


# ──────────────────────────────────────────────────────────────────────────────
# 📦 IMPORT SYSTEM UNDER TEST (main.py)
# ──────────────────────────────────────────────────────────────────────────────
# "Reload" ensures that the module is freshly executed in this clean stubbed context.
from asketmc_bot import main
main = importlib.reload(main)


# ──────────────────────────────────────────────────────────────────────────────
# ✅ ACTUAL TEST CASES
# ──────────────────────────────────────────────────────────────────────────────

# Mark this file for pytest-asyncio (enables async test functions)
pytestmark = pytest.mark.asyncio


@pytest.mark.skip(reason="query_model moved to llm_client module - see test_llm_client.py")
async def test_validation_error():
    """
    Test 1 — Input validation for `query_model()`.

    Scenario:
        User calls `query_model()` without providing either:
            * messages (prebuilt array of messages for LLM)
            * OR all three components: sys_prompt, ctx_txt, q

    Meaning of parameters:
        sys_prompt — system-level instruction for the model
        ctx_txt — contextual retrieved text
        q — user's question

    Expected result:
        Function raises ValueError, because at least one of these inputs must be provided.

    Why:
        This ensures the function doesn't proceed with an incomplete input, which could
        otherwise cause runtime errors or nonsense queries to the model.
    """
    with pytest.raises(ValueError):
        await main.query_model(messages=None, sys_prompt=None, ctx_txt=None, q=None)


@pytest.mark.skip(reason="query_model moved to llm_client module - see test_llm_client.py")
async def test_remote_success(monkeypatch):
    """
    Test 2 — Successful remote (OpenRouter) call.

    Scenario:
        Remote API responds successfully, so the local fallback should NOT be used.

    Steps:
        1. Monkeypatch `_call_openrouter()` to always return "REMOTE_OK".
        2. Call `query_model()` with valid sys_prompt, ctx_txt, q.
        3. Verify that result is from the remote model, and no fallback was used.

    Expected:
        txt == "REMOTE_OK"
        used_fb == False
    """

    async def _ok(msgs):
        return "REMOTE_OK"

    # Replace real HTTP function with our stub.
    monkeypatch.setattr(main, "_call_openrouter", _ok)

    txt, used_fb = await main.query_model(sys_prompt="S", ctx_txt="C", q="Q")

    assert txt == "REMOTE_OK", "Remote model response not propagated"
    assert used_fb is False, "Fallback should not activate when remote succeeds"


@pytest.mark.skip(reason="query_model moved to llm_client module - see test_llm_client.py")
async def test_remote_fail_triggers_fallback_and_breaker(monkeypatch):
    """
    Test 3 — Remote failure triggers fallback and circuit breaker.

    Scenario:
        The remote API fails (returns None). The function should:
          1. Use the local model instead.
          2. Activate the circuit breaker to block further remote calls temporarily.

    Steps:
        1. Stub `_call_openrouter()` → always returns None (simulate API failure).
        2. Stub `call_local_llm()` → returns "LOCAL_OK" (simulate local success).
        3. Call `query_model()`.
        4. Verify fallback used, breaker active.

        Then:
        5. Call again while breaker still active.
        6. Verify that remote is NOT called at all (breaker skipped it).

    Expected:
        - First call: fallback used, text contains "LOCAL_OK".
        - Circuit breaker flag is True (OpenRouter temporarily blocked).
        - Second call: remote not called, still "LOCAL_OK".
    """

    # (1) Simulate remote failure
    async def _fail(msgs):
        return None
    monkeypatch.setattr(main, "_call_openrouter", _fail)

    # (2) Simulate local model success
    async def _local(prompt_text, timeout_sec=None):
        return "LOCAL_OK"
    monkeypatch.setattr(main, "call_local_llm", _local)

    # First call: triggers breaker + fallback
    txt, used_fb = await main.query_model(sys_prompt="S", ctx_txt="C", q="Q")
    assert used_fb is True, "Fallback should be True when remote fails"
    assert "LOCAL_OK" in txt, "Local model output missing"
    assert main.is_openrouter_blocked() is True, "Circuit breaker not activated"

    # Second call: breaker active → remote must not be called
    called = {"n": 0}

    async def _count(msgs):
        called["n"] += 1
        return "REMOTE_SHOULD_NOT_BE_CALLED"

    # Replace the remote function with counter to detect unwanted calls
    monkeypatch.setattr(main, "_call_openrouter", _count)

    txt2, used_fb2 = await main.query_model(sys_prompt="S", ctx_txt="C", q="Q")
    assert used_fb2 is True, "Still expect fallback while breaker active"
    assert "LOCAL_OK" in txt2
    assert called["n"] == 0, "Breaker failed — remote was called unexpectedly"
