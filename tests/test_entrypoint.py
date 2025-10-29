from pathlib import Path
import re

def test_module_has_docstring_and_entry_guard():
    src = Path("main.py").read_text(encoding="utf-8")
    assert re.search(r'^\s*"""[^"]+', src, re.M), "missing module docstring"
    assert 'if __name__ == "__main__":' in src, "missing entrypoint guard"
# tests/test_entrypoint.py
from pathlib import Path
import re

def test_module_has_docstring_and_entry_guard():
    root = Path(__file__).resolve().parents[1]  # <-- LLM/bot
    src = (root / "main.py").read_text(encoding="utf-8")
    assert re.search(r'^\s*"""[^"]+', src, re.M)
    assert 'if __name__ == "__main__":' in src
