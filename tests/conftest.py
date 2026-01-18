from __future__ import annotations

import os
import sys
import types
from pathlib import Path


def _ensure_sys_path_src() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _stubs_enabled() -> bool:
    # Default: enabled (keeps current behavior), allow disabling for integration runs.
    return os.getenv("ASKETMC_TEST_STUBS", "1") not in {"0", "false", "False", "no", "NO"}


def _install_stub_modules() -> None:
    # torch stub (avoid CUDA checks / heavy import)
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.__version__ = "0"
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        sys.modules["torch"] = torch

    # sentence_transformers stub (avoid pulling transformers/torch)
    if "sentence_transformers" not in sys.modules:
        st = types.ModuleType("sentence_transformers")
        sys.modules["sentence_transformers"] = st

    # llama_index minimal surface used by rerank/rag_filter
    # Only stub if llama_index is not already importable / installed.
    try:
        import llama_index  # noqa: F401
        return
    except Exception:
        pass

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


def pytest_configure(config) -> None:
    # Apply before test collection imports modules under test.
    _ensure_sys_path_src()

    if _stubs_enabled():
        _install_stub_modules()
