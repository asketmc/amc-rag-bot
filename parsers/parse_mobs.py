"""
Парсер данных монстров Grimoire of Gaia (Python 3.10)

Создаёт «C:/LLM/parsed/mobs.txt» с краткой сводкой:
• минимальное и максимальное здоровье (generic.maxHealth)
• минимальный и максимальный урон (generic.attackDamage)
"""

import os
import json
from pathlib import Path

# ──────────────────────────────── НАСТРОЙКИ ────────────────────────────────
MOBS_DIR = r"C:/Nextgen/config/MobsPropertiesRandomness/json"    # путь к JSON-файлам
OUTPUT_FILE = r"C:/LLM/parsed/mobs.txt"
# ───────────────────────────────────────────────────────────────────────────


# ──────────────────────────── ВСПОМОГАТЕЛЬНЫЕ ─────────────────────────────
def parse_mob(path: Path) -> dict:
    """Возвращает словарь с инфой о монстре."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    mob_id = data.get("mob_id", path.stem)

    result = {
        "mob_id": mob_id,
        "min_hp": "—",
        "max_hp": "—",
        "min_dmg": "—",
        "max_dmg": "—",
    }

    for attr in data.get("attributes", []):
        aid = attr.get("id", "")
        mod = attr.get("modifier", {})
        if aid == "generic.maxHealth":
            result["min_hp"] = mod.get("min", "—")
            result["max_hp"] = mod.get("max", "—")
        elif aid == "generic.attackDamage":
            result["min_dmg"] = mod.get("min", "—")
            result["max_dmg"] = mod.get("max", "—")

    return result
# ───────────────────────────────────────────────────────────────────────────


# ──────────────── ШАГ 1: ЧТЕНИЕ ВСЕХ МОБОВ ────────────────────────────────
mob_data: dict[str, dict] = {}

for fname in os.listdir(MOBS_DIR):
    if not fname.endswith(".json"):
        continue

    full_path = Path(MOBS_DIR) / fname
    try:
        mob = parse_mob(full_path)
        mob_data[mob["mob_id"]] = mob
    except Exception as e:
        print(f"[WARN] Ошибка при разборе {fname}: {e}")
# ───────────────────────────────────────────────────────────────────────────


# ──────────────── ШАГ 2: ГЕНЕРАЦИЯ СВОДКИ ─────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for mob_id, mob in sorted(mob_data.items()):
        out.write(f"Моб: {mob_id}\n")
        out.write(f"Здоровье: {mob['min_hp']}–{mob['max_hp']}\n")
        out.write(f"Урон: {mob['min_dmg']}–{mob['max_dmg']}\n")
        out.write("---\n")
