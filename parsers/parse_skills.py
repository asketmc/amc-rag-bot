"""
Парсер SkillAPI-данных (Python 3.10)

Создаёт «C:/LLM/parsed/skills.txt» с краткой сводкой по классам:
• уровень открытия навыка берётся из строки “Required level” в icon-lore;
• кулдаун — из attributes.cooldown-base;
• описание — очищенные строки icon-lore без цветовых кодов, плейсхолдеров
  и служебных упоминаний (“Required level”, “Cooldown”).
"""

import os
import re
import yaml

# ──────────────────────────────── НАСТРОЙКИ ────────────────────────────────
CLASS_DIR  = r"C:/Nextgen/plugins/SkillAPI/dynamic/class"
SKILL_DIR  = r"C:/Nextgen/plugins/SkillAPI/dynamic/skill"   # поправьте путь при необходимости
OUTPUT_FILE = r"C:/LLM/parsed/skills.txt"
# ───────────────────────────────────────────────────────────────────────────


# ──────────────────────────── ВСПОМОГАТЕЛЬНЫЕ ─────────────────────────────
COLOR_RE        = re.compile(r"&[0-9a-fk-or]", re.I)   # §-коды Bukkit &x
PLACEHOLDER_RE  = re.compile(r"\{[^}]+\}")             # {attr:cooldown}, {type} …
DIGIT_RE        = re.compile(r"\d+")


def load_yaml(path: str):
    """Читает YAML-файл и возвращает dict или {}."""
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}
    except Exception:
        return {}


def clean_line(text: str) -> str:
    """Удаляет цветовые коды и плейсхолдеры, обрезает пробелы."""
    text = COLOR_RE.sub("", text)
    text = PLACEHOLDER_RE.sub("", text)
    return text.strip()
# ───────────────────────────────────────────────────────────────────────────


# ───────────────────── ШАГ 1: ИНДЕКС НАВЫКОВ ───────────────────────────────
skill_data: dict[str, dict] = {}

for fname in os.listdir(SKILL_DIR):
    if not fname.endswith(".yml"):
        continue
    for key, section in load_yaml(os.path.join(SKILL_DIR, fname)).items():
        if not isinstance(section, dict):
            continue

        name = section.get("name", key)

        # 1. Уровень открытия
        unlock_level: int | None = None
        for line in section.get("icon-lore", []):
            if "Required level" in line:
                m = DIGIT_RE.search(line)
                if m:
                    unlock_level = int(m.group())
                    break
        if unlock_level is None:
            # fallback: skill-req-lvl или 1
            unlock_level = int(section.get("skill-req-lvl", 1))

        # 2. Кулдаун
        try:
            cooldown_raw = section.get("attributes", {}).get("cooldown-base", "0")
            cooldown = float(cooldown_raw)
        except Exception:
            cooldown = 0.0

        # 3. Описание
        description: list[str] = []
        for line in section.get("icon-lore", []):
            if "Required level" in line.lower():
                continue
            if "cooldown" in line.lower():
                continue
            cleaned = clean_line(line)
            if cleaned:
                description.append(cleaned)

        skill_data[name] = {
            "level": unlock_level,
            "cooldown": cooldown,
            "desc": description,
        }
# ───────────────────────────────────────────────────────────────────────────


# ──────────────── ШАГ 2: ГЕНЕРАЦИЯ ОТЧЁТА ПО КЛАССАМ ──────────────────────
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for fname in os.listdir(CLASS_DIR):
        if not fname.endswith(".yml"):
            continue

        for key, section in load_yaml(os.path.join(CLASS_DIR, fname)).items():
            if not isinstance(section, dict):
                continue

            class_name = section.get("name", key)
            group      = section.get("group", "Нет")
            attrs      = section.get("attributes", {})
            health     = attrs.get("health-base", "—")
            mana       = attrs.get("mana-base",  "—")
            skills     = section.get("skills", [])

            out.write(f"Класс: {class_name}\n")
            out.write(f"Группа: {group}\n")
            out.write(f"Здоровье: {health}, Мана: {mana}\n")
            out.write("Навыки:\n")

            for skill in skills:
                info = skill_data.get(skill, {})
                lvl  = info.get("level", "—")
                cd   = info.get("cooldown", "—")
                desc = info.get("desc", [])

                out.write(f"  - {skill} (ур. {lvl}, CD: {cooldown} сек)\n")
                for line in desc:
                    out.write(f"      {line}\n")

            out.write("---\n")
            out.write("---\n")
