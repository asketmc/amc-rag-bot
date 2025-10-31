from pathlib import Path
import re

def test_module_has_docstring_and_entry_guard():
    root = Path(__file__).resolve().parents[1]
    src = (root / "src" / "asketmc_bot" / "main.py").read_text(encoding="utf-8")
    assert re.search(r'^\s*"""[^"]+', src, re.M), "missing module docstring"
    assert 'if __name__ == "__main__":' in src, "missing entrypoint guard"
