from __future__ import annotations

import os
import sys
import types
import traceback
from pathlib import Path


def _dbg_enabled() -> bool:
    return os.getenv("ASKETMC_TEST_DEBUG", "0") in {"1", "true", "True", "yes", "YES"}


def _dbg(*parts: object) -> None:
    if _dbg_enabled():
        print("[conftest]", *parts, file=sys.stderr)


def _ensure_sys_path_src() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    _dbg("root=", str(root))
    _dbg("src=", str(src))
    _dbg("sys.path[0:5]=", sys.path[:5])


def _stubs_enabled() -> bool:
    v = os.getenv("ASKETMC_TEST_STUBS", "1")
    enabled = v not in {"0", "false", "False", "no", "NO"}
    _dbg("ASKETMC_TEST_STUBS=", v, "-> enabled=", enabled)
    return enabled


def _module_origin(name: str) -> str:
    mod = sys.modules.get(name)
    if mod is None:
        return "<not in sys.modules>"
    f = getattr(mod, "__file__", None)
    p = getattr(mod, "__path__", None)
    return f"file={f!r} path={list(p) if p is not None else None!r} type={type(mod)!r}"


def _install_rerank_stub() -> None:
    """
    Prevent heavy reranker initialization during unit tests.
    Must stub the package-qualified path because production imports use asketmc_bot.*.
    """
    if "asketmc_bot.rerank" in sys.modules:
        _dbg("asketmc_bot.rerank already present:", _module_origin("asketmc_bot.rerank"))
        return

    rerank_mod = types.ModuleType("asketmc_bot.rerank")

    async def init_reranker() -> None:
        return None

    async def rerank(q, nodes):
        return nodes

    async def shutdown_reranker() -> None:
        return None

    rerank_mod.init_reranker = init_reranker
    rerank_mod.rerank = rerank
    rerank_mod.shutdown_reranker = shutdown_reranker

    sys.modules["asketmc_bot.rerank"] = rerank_mod
    # Optional compatibility if any code imports plain "rerank"
    if "rerank" not in sys.modules:
        sys.modules["rerank"] = rerank_mod

    _dbg("stubbed asketmc_bot.rerank")


def _install_stub_modules() -> None:
    _dbg("install stubs: begin")

    # torch stub (avoid CUDA checks / heavy import)
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.__version__ = "0"
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        sys.modules["torch"] = torch
        _dbg("stubbed torch")
    else:
        _dbg("torch already present:", _module_origin("torch"))

    # sentence_transformers stub (avoid pulling transformers/torch)
    if "sentence_transformers" not in sys.modules:
        st = types.ModuleType("sentence_transformers")
        sys.modules["sentence_transformers"] = st
        _dbg("stubbed sentence_transformers")
    else:
        _dbg("sentence_transformers already present:", _module_origin("sentence_transformers"))

    # Rerank stub: avoids heavy model init during tests
    _install_rerank_stub()

    # llama_index minimal surface used by some modules
    # Only stub if llama_index is not already importable / installed.
    _dbg("attempt import llama_index with current sys.modules state")
    _dbg(
        "pre-import origins:",
        "llama_index=",
        _module_origin("llama_index"),
        "llama_index.core=",
        _module_origin("llama_index.core"),
        "sentence_transformers=",
        _module_origin("sentence_transformers"),
    )

    try:
        import llama_index  # noqa: F401

        _dbg("import llama_index: OK")
        try:
            import llama_index.core as c  # noqa: F401

            _dbg("import llama_index.core: OK", _module_origin("llama_index.core"))
            _dbg("has Settings=", hasattr(c, "Settings"))
        except Exception:
            _dbg("import llama_index.core: FAILED")
            _dbg("traceback:\n" + traceback.format_exc())
        return
    except Exception:
        _dbg("import llama_index: FAILED")
        _dbg("traceback:\n" + traceback.format_exc())

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

    # Avoid clobbering real schema if it becomes available later.
    if not hasattr(ll_core, "schema"):
        ll_core.schema = sys.modules["llama_index.core.schema"]

    _dbg(
        "stub install done:",
        "llama_index=",
        _module_origin("llama_index"),
        "llama_index.core=",
        _module_origin("llama_index.core"),
        "llama_index.core.schema=",
        _module_origin("llama_index.core.schema"),
    )


def pytest_configure(config) -> None:
    _ensure_sys_path_src()
    if _stubs_enabled():
        _install_stub_modules()
