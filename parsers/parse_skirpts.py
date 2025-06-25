"""
Парсер Skript-скриптов → «C:/LLM/parsed/skript.txt»
Требуется Python 3.10+
"""

import re
from pathlib import Path

# ─────────── НАСТРОЙКИ ──────────────────────────────────────────────────
SKRIPT_DIR  = Path(r"C:/Nextgen/plugins/Skript/scripts")   # папка .sk
OUTPUT_FILE = Path(r"C:/LLM/parsed/skript.txt")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
# -----------------------------------------------------------------------

CMD_RE       = re.compile(r'^\s*command\s+/([\w\d_]+)', re.I)
DESC_RE      = re.compile(r'^\s*description\s*:\s*(.+)', re.I)
COMMENT_RE   = re.compile(r'^\s*#\s*(.+)')
EVENT_RE     = re.compile(r'^\s*on\s+(.+?):', re.I)
VAR_RE       = re.compile(r'\{[^{}]+\}')
EFFECT_RE    = re.compile(r'^\s*(add|remove|set|give|execute|broadcast|make|spawn)\b', re.I)

commands = []           # list[dict]: {"name":"/x","desc":"", "file":Path}
events, variables, effects = set(), set(), set()

for file in SKRIPT_DIR.glob("**/*.sk"):
    lines = file.read_text(encoding="utf-8", errors="ignore").splitlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # --- команды -----------------------------------------------------
        m = CMD_RE.match(line)
        if m:
            cmd_name = "/" + m.group(1)
            base_indent = len(line) - len(line.lstrip())   # для выхода из блока
            desc = None

            # Ищем комментарий сверху (до ближайшей пустой строки / начала файла)
            j = i - 1
            while j >= 0 and lines[j].strip():
                cmt = COMMENT_RE.match(lines[j])
                if cmt:
                    desc = cmt.group(1).strip()
                    break
                j -= 1

            # Ищем description: внутри блока
            k = i + 1
            while k < len(lines):
                ln = lines[k]
                if len(ln) - len(ln.lstrip()) <= base_indent:
                    break       # вышли из блока
                d = DESC_RE.match(ln)
                if d:
                    desc = d.group(1).strip()
                    break
                k += 1

            if not desc:
                desc = "Нет описания"

            commands.append({"name": cmd_name, "desc": desc, "file": file.name})

        # --- ивенты ------------------------------------------------------

        # --- переменные --------------------------------------------------

        # --- эффекты -----------------------------------------------------

        i += 1

# ─────────── ЗАПИСЬ RAG-ФАЙЛА ────────────────────────────────────────────
with OUTPUT_FILE.open("w", encoding="utf-8") as out:
    out.write("Команды Skript\n")
    out.write("==============\n\n")
    for cmd in sorted(commands, key=lambda x: x["name"]):
        out.write(f"{cmd['name']}: {cmd['desc']}  (из {cmd['file']})\n")
    out.write("\n---\nИвенты:\n")
    for e in sorted(events):   out.write(f" - {e}\n")
    out.write("\nПеременные:\n")
    for v in sorted(variables): out.write(f" - {v}\n")
    out.write("\nЭффекты:\n")
    for fx in sorted(effects): out.write(f" - {fx}\n")
