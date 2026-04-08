"""
tests/bootstrap.py

Single entry-point for test environment setup and shared test doubles.

Responsibilities
----------------
- sys.path configuration so ``asketmc_bot`` is importable without installation.
- Heavy-dependency stubs (chromadb, spacy, torch, llama_index, …) so unit
  tests run without a full ML stack.
- ``MockNode`` / ``MockNodeWithScore`` test doubles for llama_index node types.

``configure()`` is called once from ``conftest.pytest_configure``; everything
else in this module is safe to import at any time.
"""
from __future__ import annotations

import os
import sys
import traceback
import types
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _dbg_enabled() -> bool:
    return os.getenv("ASKETMC_TEST_DEBUG", "0") in {"1", "true", "True", "yes", "YES"}


def _dbg(*parts: object) -> None:
    if _dbg_enabled():
        print("[bootstrap]", *parts, file=sys.stderr)


def _module_origin(name: str) -> str:
    mod = sys.modules.get(name)
    if mod is None:
        return "<not in sys.modules>"
    f = getattr(mod, "__file__", None)
    p = getattr(mod, "__path__", None)
    return f"file={f!r} path={list(p) if p is not None else None!r} type={type(mod)!r}"


# ---------------------------------------------------------------------------
# sys.path
# ---------------------------------------------------------------------------

def ensure_src_on_path() -> None:
    """Add ``src/`` to sys.path so asketmc_bot is importable without editable install."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    _dbg("root=", str(root), "src=", str(src))


# ---------------------------------------------------------------------------
# Stub installers
# ---------------------------------------------------------------------------

def _stubs_enabled() -> bool:
    v = os.getenv("ASKETMC_TEST_STUBS", "1")
    enabled = v not in {"0", "false", "False", "no", "NO"}
    _dbg("ASKETMC_TEST_STUBS=", v, "-> enabled=", enabled)
    return enabled


def _install_failfast_stub(module_name: str, *, attrs: dict[str, Any] | None = None) -> None:
    """Register a stub that raises on use; prevents silent fallbacks in CI."""
    if module_name in sys.modules:
        _dbg(module_name, "already present:", _module_origin(module_name))
        return

    m = types.ModuleType(module_name)
    m.__dict__["__version__"] = "0"

    def _fail(*_a: Any, **_kw: Any) -> None:
        raise RuntimeError(
            f"Stubbed dependency '{module_name}' was used in tests. "
            "Install real deps or set ASKETMC_TEST_STUBS=0."
        )

    m.__dict__.setdefault("load", _fail)
    m.__dict__.setdefault("Client", _fail)
    m.__dict__.setdefault("OpenAI", _fail)

    if attrs:
        m.__dict__.update(attrs)

    sys.modules[module_name] = m
    _dbg("stubbed", module_name)


def _install_rerank_stub() -> None:
    """Stub asketmc_bot.rerank to avoid heavy model initialisation in unit tests."""
    if "asketmc_bot.rerank" in sys.modules:
        _dbg("asketmc_bot.rerank already present:", _module_origin("asketmc_bot.rerank"))
        return

    rerank_mod = types.ModuleType("asketmc_bot.rerank")

    async def init_reranker() -> None:
        return None

    async def rerank(_q, nodes):  # noqa: ANN001
        return nodes

    async def shutdown_reranker() -> None:
        return None

    rerank_mod.init_reranker = init_reranker
    rerank_mod.rerank = rerank
    rerank_mod.shutdown_reranker = shutdown_reranker

    sys.modules["asketmc_bot.rerank"] = rerank_mod
    if "rerank" not in sys.modules:
        sys.modules["rerank"] = rerank_mod

    _dbg("stubbed asketmc_bot.rerank")


def _install_llama_index_stub() -> None:
    """Stub llama_index when the real package is absent; falls back gracefully."""
    _dbg("attempt import llama_index")
    try:
        import llama_index as _llama_index  # noqa: F401
        _dbg("import llama_index: OK")
        try:
            import llama_index.core as _core  # noqa: F401
            _dbg("import llama_index.core: OK", _module_origin("llama_index.core"))
        except Exception:
            _dbg("import llama_index.core: FAILED\n" + traceback.format_exc())
        return
    except Exception:
        _dbg("import llama_index: FAILED\n" + traceback.format_exc())

    _dbg("falling back to stub llama_index.* modules")

    if "llama_index" not in sys.modules:
        ll_pkg = types.ModuleType("llama_index")
        ll_pkg.__path__ = []
        sys.modules["llama_index"] = ll_pkg

    if "llama_index.core" not in sys.modules:
        ll_core = types.ModuleType("llama_index.core")
        ll_core.__path__ = []
        sys.modules["llama_index.core"] = ll_core
    else:
        ll_core = sys.modules["llama_index.core"]

    if "llama_index.core.schema" not in sys.modules:
        ll_schema = types.ModuleType("llama_index.core.schema")

        class TextNode:
            def __init__(self, text: str = "") -> None:
                self.text = text

            def get_content(self) -> str:
                return self.text

        class NodeWithScore:
            def __init__(self, node: TextNode | None = None, score: float = 0.0) -> None:
                self.node = node if node is not None else TextNode("")
                self.score = float(score)

        ll_schema.TextNode = TextNode
        ll_schema.NodeWithScore = NodeWithScore
        sys.modules["llama_index.core.schema"] = ll_schema

    if not hasattr(ll_core, "schema"):
        ll_core.schema = sys.modules["llama_index.core.schema"]

    _dbg("llama_index stub install done")


def install_stubs() -> None:
    """Install all heavy-dependency stubs. No-op when ASKETMC_TEST_STUBS=0."""
    if not _stubs_enabled():
        return

    _dbg("install stubs: begin")
    _install_failfast_stub("chromadb")
    _install_failfast_stub("spacy")
    _install_failfast_stub("transformers")
    _install_failfast_stub("openai")
    _install_failfast_stub("ollama")
    _install_failfast_stub("onnxruntime")

    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.__version__ = "0"
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        sys.modules["torch"] = torch
        _dbg("stubbed torch")
    else:
        _dbg("torch already present:", _module_origin("torch"))

    if "sentence_transformers" not in sys.modules:
        st = types.ModuleType("sentence_transformers")
        sys.modules["sentence_transformers"] = st
        _dbg("stubbed sentence_transformers")
    else:
        st = sys.modules["sentence_transformers"]
        _dbg("sentence_transformers already present:", _module_origin("sentence_transformers"))

    # Ensure CrossEncoder is available for reranker tests
    if not hasattr(st, "CrossEncoder"):
        from typing import Iterable as _Iterable

        class _CrossEncoder:
            """Deterministic stub for sentence_transformers.CrossEncoder."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def predict(self, pairs: _Iterable[Any], **kwargs: Any) -> list[float]:
                try:
                    n = len(pairs)  # type: ignore[arg-type]
                except Exception:
                    n = sum(1 for _ in pairs)
                return [float(n - i) for i in range(n)]

        st.CrossEncoder = _CrossEncoder
        _dbg("added CrossEncoder stub to sentence_transformers")

    _install_rerank_stub()
    _install_llama_index_stub()
    _dbg("install stubs: done")


# ---------------------------------------------------------------------------
# Public entry-point called from conftest.pytest_configure
# ---------------------------------------------------------------------------

def configure() -> None:
    """Bootstrap the test environment: sys.path + stubs. Call once per session."""
    ensure_src_on_path()
    install_stubs()


# ---------------------------------------------------------------------------
# Shared test doubles for llama_index node types
# ---------------------------------------------------------------------------

class MockNode:
    def __init__(self, text: str = "", metadata: dict | None = None) -> None:
        self.text = text
        self.metadata = metadata or {}
        self.id = f"node_{id(self)}"

    def get_content(self, *_args, **_kwargs) -> str:  # noqa: ANN001
        return self.text


class MockNodeWithScore:
    def __init__(self, text: str = "", score: float = 0.5, lemmas: list | None = None) -> None:
        self.node = MockNode(text, {"lemmas": lemmas or []})
        self.score = score
