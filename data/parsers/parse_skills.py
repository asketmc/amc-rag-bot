#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parse_skills.py — SkillAPI parser (Python 3.10+)

Purpose:
  Parse SkillAPI YAML (skills & classes) and generate a deterministic text report.

Key points:
  - PEP 8 / PEP 257 compliant.
  - CLI + .env configuration (no hardcoded paths).
  - Structured JSON logging (channels: app, error, parse).
  - Windows-safe logging: separate log files, optional NO-ROTATION mode.
  - Deterministic output (sorted file lists and keys).
  - Async-ready: concurrent file parsing via asyncio.to_thread.
  - Guards: timeouts, file limits, lore length cap, atomic write.

Usage example:
  python parse_skills.py \
    --class-dir "C:/Nextgen/plugins/SkillAPI/dynamic/class" \
    --skill-dir "C:/Nextgen/plugins/SkillAPI/dynamic/skill" \
    --output "C:/LLM/parsed/skills.txt"

Environment variables (.env supported):
  CLASS_DIR, SKILL_DIR, OUTPUT_FILE
  LOG_DIR (default C:/LLM/logs/parser), LOG_LEVEL, LOG_MAX_BYTES, LOG_BACKUPS
  LOG_PREFIX (default "parser_"), LOG_DISABLE_ROTATION ("1" to disable)
  IO_TIMEOUT_SEC, MAX_FILES, MAX_ICON_LORE_LEN, CIRCUIT_BREAKER_ERR_RATIO
  CONCURRENCY
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Optional .env loader
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # optional

import yaml  # PyYAML


# ────────────────────────────────── Config ──────────────────────────────────


@dataclass(frozen=True)
class Config:
    """Runtime configuration loaded from env and CLI."""

    class_dir: Path
    skill_dir: Path
    output_file: Path

    # Logging
    log_dir: Path
    log_level: str
    log_max_bytes: int
    log_backups: int
    log_prefix: str
    disable_rotation: bool

    # Guards
    io_timeout_sec: float
    max_files: int
    max_icon_lore_len: int
    circuit_breaker_err_ratio: float

    # Concurrency
    concurrency: int

    @staticmethod
    def from_env_and_args(args: argparse.Namespace) -> "Config":
        """Construct Config using .env (if present) and CLI overrides."""
        if load_dotenv is not None:
            load_dotenv()

        def env_path(key: str, default: str) -> Path:
            return Path(os.getenv(key, default))

        def env_int(key: str, default: int) -> int:
            try:
                return int(os.getenv(key, str(default)))
            except Exception:
                return default

        def env_float(key: str, default: float) -> float:
            try:
                return float(os.getenv(key, str(default)))
            except Exception:
                return default

        class_dir = Path(args.class_dir or os.getenv("CLASS_DIR", r"C:/Nextgen/plugins/SkillAPI/dynamic/class"))
        skill_dir = Path(args.skill_dir or os.getenv("SKILL_DIR", r"C:/Nextgen/plugins/SkillAPI/dynamic/skill"))
        output_file = Path(args.output or os.getenv("OUTPUT_FILE", r"C:/LLM/parsed/skills.txt"))

        # Default to a separate folder to avoid conflicts with the running bot
        log_dir = env_path("LOG_DIR", os.getenv("LOG_DIR", r"C:/LLM/logs/parser"))
        log_level = (args.log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
        log_max_bytes = env_int("LOG_MAX_BYTES", 1_000_000)
        log_backups = env_int("LOG_BACKUPS", 1)
        log_prefix = os.getenv("LOG_PREFIX", "parser_")
        disable_rotation = os.getenv("LOG_DISABLE_ROTATION", "0") == "1"

        io_timeout_sec = env_float("IO_TIMEOUT_SEC", 10.0)
        max_files = env_int("MAX_FILES", 10_000)
        max_icon_lore_len = env_int("MAX_ICON_LORE_LEN", 4096)
        circuit_breaker_err_ratio = env_float("CIRCUIT_BREAKER_ERR_RATIO", 0.5)

        concurrency = env_int("CONCURRENCY", max(1, (args.concurrency or 8)))

        return Config(
            class_dir=class_dir,
            skill_dir=skill_dir,
            output_file=output_file,
            log_dir=log_dir,
            log_level=log_level,
            log_max_bytes=log_max_bytes,
            log_backups=log_backups,
            log_prefix=log_prefix,
            disable_rotation=disable_rotation,
            io_timeout_sec=io_timeout_sec,
            max_files=max_files,
            max_icon_lore_len=max_icon_lore_len,
            circuit_breaker_err_ratio=circuit_breaker_err_ratio,
            concurrency=concurrency,
        )


# ───────────────────────────── Structured logging ───────────────────────────


class JsonFormatter(logging.Formatter):
    """Lightweight structured JSON formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": int(record.created * 1000),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            payload["extra"] = getattr(record, "extra")
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def setup_logging(cfg: Config) -> tuple[logging.Logger, logging.Logger, logging.Logger]:
    """Create per-channel loggers with isolated files (Windows-safe)."""
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    def make_handler(filename: str) -> logging.Handler:
        path = cfg.log_dir / filename
        if cfg.disable_rotation:
            return logging.FileHandler(str(path), encoding="utf-8", delay=True)
        return RotatingFileHandler(
            filename=str(path),
            maxBytes=cfg.log_max_bytes,
            backupCount=cfg.log_backups,
            encoding="utf-8",
            delay=True,
        )

    def make_logger(name: str, filename: str, level: int) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False

        handler = make_handler(filename)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

        if name == "app":
            stream = logging.StreamHandler(sys.stderr)
            stream.setFormatter(JsonFormatter())
            stream.setLevel(level)
            logger.addHandler(stream)
        return logger

    level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    app_log = make_logger("app", f"{cfg.log_prefix}app.log", level)
    err_log = make_logger("error", f"{cfg.log_prefix}error.log", logging.WARNING)
    parse_log = make_logger("parse", f"{cfg.log_prefix}parse.log", level)
    return app_log, err_log, parse_log


# ──────────────────────────────── Domain logic ──────────────────────────────

COLOR_RE = re.compile(r"&[0-9a-fk-or]", re.I)     # Bukkit-style color codes
PLACEHOLDER_RE = re.compile(r"\{[^}]+\}")          # {attr:cooldown}, {type}, etc.
DIGIT_RE = re.compile(r"\d+")
REQUIRED_RE = re.compile(r"required\s*level", re.I)
COOLDOWN_RE = re.compile(r"cooldown", re.I)


def load_yaml_file(path: Path) -> Dict:
    """Load a YAML file; return {} on any failure."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}
    except Exception:
        return {}


def clean_line(text: str) -> str:
    """Remove color codes & placeholders and trim spaces."""
    text = COLOR_RE.sub("", text)
    text = PLACEHOLDER_RE.sub("", text)
    return text.strip()


def parse_skill_section(key: str, section: Dict, max_icon_lore_len: int) -> Optional[Tuple[str, Dict]]:
    """Extract normalized skill info from one SkillAPI section."""
    if not isinstance(section, dict):
        return None

    name = str(section.get("name", key))
    icon_lore: List[str] = list(map(str, section.get("icon-lore", []) or []))

    # Guard: cap total lore length to avoid pathological inputs
    if sum(len(line) for line in icon_lore) > max_icon_lore_len:
        icon_lore = icon_lore[:64]

    # 1) Unlock level (from icon-lore "Required level" first, then fallback)
    unlock_level: Optional[int] = None
    for line in icon_lore:
        if REQUIRED_RE.search(line):
            m = DIGIT_RE.search(line)
            if m:
                try:
                    unlock_level = int(m.group())
                except Exception:
                    unlock_level = None
            if unlock_level is not None:
                break
    if unlock_level is None:
        try:
            unlock_level = int(section.get("skill-req-lvl", 1))
        except Exception:
            unlock_level = 1

    # 2) Cooldown
    cooldown_val = 0.0
    try:
        cooldown_raw = (section.get("attributes", {}) or {}).get("cooldown-base", "0")
        cooldown_val = float(cooldown_raw)
    except Exception:
        cooldown_val = 0.0

    # 3) Description (filtered)
    description: List[str] = []
    for line in icon_lore:
        if REQUIRED_RE.search(line) or COOLDOWN_RE.search(line):
            continue
        cleaned = clean_line(line)
        if cleaned:
            description.append(cleaned)

    return name, {"level": unlock_level, "cooldown": cooldown_val, "desc": description}


def parse_skill_file(path: Path, max_icon_lore_len: int) -> Dict[str, Dict]:
    """Parse a single skill YAML into name->info mapping."""
    data = load_yaml_file(path)
    result: Dict[str, Dict] = {}
    if not isinstance(data, dict):
        return result
    for key in sorted(data.keys()):
        section = data[key]
        parsed = parse_skill_section(str(key), section, max_icon_lore_len)
        if parsed is None:
            continue
        name, info = parsed
        result[name] = info
    return result


def parse_all_skills(
    skill_dir: Path,
    max_files: int,
    max_icon_lore_len: int,
    concurrency: int,
    timeout: float,
    parse_log: logging.Logger,
) -> Dict[str, Dict]:
    """Parse all *.yml in skill_dir concurrently (deterministic merge)."""
    files = sorted([p for p in skill_dir.iterdir() if p.suffix.lower() == ".yml"])
    if len(files) > max_files:
        files = files[:max_files]

    sem = asyncio.Semaphore(concurrency)
    results: Dict[str, Dict] = {}
    errors: List[Tuple[Path, str]] = []

    async def worker(file_path: Path) -> None:
        async with sem:
            start = time.perf_counter()
            try:
                chunk = await asyncio.wait_for(
                    asyncio.to_thread(parse_skill_file, file_path, max_icon_lore_len),
                    timeout=timeout,
                )
                for k in sorted(chunk.keys()):
                    results[k] = chunk[k]
                parse_log.info(
                    "parsed-skill-file",
                    extra={"extra": {"file": str(file_path), "ms": int((time.perf_counter() - start) * 1000)}},
                )
            except Exception as exc:
                errors.append((file_path, repr(exc)))
                parse_log.warning("parse-failed", extra={"extra": {"file": str(file_path), "error": repr(exc)}})

    async def run_all() -> None:
        await asyncio.gather(*(worker(p) for p in files))

    asyncio.run(run_all())

    if errors:
        parse_log.warning("skill-parse-errors", extra={"extra": {"count": len(errors)}})

    return results


def class_entry_to_report_lines(
    class_name: str,
    group: str,
    health: str,
    mana: str,
    skills: Iterable[str],
    skill_data: Dict[str, Dict],
) -> List[str]:
    """Render one class entry into stable list of lines."""
    lines: List[str] = []
    lines.append(f"Класс: {class_name}")
    lines.append(f"Группа: {group}")
    lines.append(f"Здоровье: {health}, Мана: {mana}")
    lines.append("Навыки:")

    for skill in sorted(skills):
        info = skill_data.get(skill, {})
        lvl = info.get("level", "—")
        cd = info.get("cooldown", "—")
        desc = info.get("desc", [])

        # Note: fix original bug — use 'cd' variable in output
        lines.append(f"  - {skill} (ур. {lvl}, CD: {cd} сек)")
        for line in desc:
            lines.append(f"      {line}")

    lines.append("---")
    lines.append("---")
    return lines


def parse_class_file(path: Path, skill_data: Dict[str, Dict]) -> List[str]:
    """Parse one class YAML file and return rendered lines."""
    data = load_yaml_file(path)
    if not isinstance(data, dict):
        return []
    all_lines: List[str] = []
    for key in sorted(data.keys()):
        section = data[key]
        if not isinstance(section, dict):
            continue
        class_name = str(section.get("name", key))
        group = str(section.get("group", "Нет"))
        attrs = section.get("attributes", {}) or {}
        health = str(attrs.get("health-base", "—"))
        mana = str(attrs.get("mana-base", "—"))
        skills = list(map(str, section.get("skills", []) or []))
        all_lines.extend(
            class_entry_to_report_lines(class_name, group, health, mana, skills, skill_data)
        )
    return all_lines


def render_report(
    class_dir: Path,
    skill_data: Dict[str, Dict],
    max_files: int,
    concurrency: int,
    timeout: float,
    parse_log: logging.Logger,
) -> List[str]:
    """Render full report from class files concurrently and deterministically."""
    files = sorted([p for p in class_dir.iterdir() if p.suffix.lower() == ".yml"])
    if len(files) > max_files:
        files = files[:max_files]

    sem = asyncio.Semaphore(concurrency)
    all_chunks: List[Tuple[str, List[str]]] = []
    errors: List[Tuple[Path, str]] = []

    async def worker(file_path: Path) -> None:
        async with sem:
            start = time.perf_counter()
            try:
                lines = await asyncio.wait_for(
                    asyncio.to_thread(parse_class_file, file_path, skill_data),
                    timeout=timeout,
                )
                all_chunks.append((str(file_path), lines))
                parse_log.info(
                    "parsed-class-file",
                    extra={"extra": {"file": str(file_path), "ms": int((time.perf_counter() - start) * 1000)}},
                )
            except Exception as exc:
                errors.append((file_path, repr(exc)))
                parse_log.warning("class-parse-failed", extra={"extra": {"file": str(file_path), "error": repr(exc)}})

    async def run_all() -> None:
        await asyncio.gather(*(worker(p) for p in files))

    asyncio.run(run_all())

    all_chunks.sort(key=lambda x: x[0])  # deterministic concat
    report_lines: List[str] = []
    for _, chunk in all_chunks:
        report_lines.extend(chunk)

    if errors:
        parse_log.warning("class-parse-errors", extra={"extra": {"count": len(errors)}})

    return report_lines


# ───────────────────────────────── Utilities ────────────────────────────────


def circuit_breaker_should_trip(total: int, err_count: int, ratio: float) -> bool:
    """Return True if error ratio exceeds threshold."""
    if total == 0:
        return False
    return (err_count / total) > ratio


def write_report(output_file: Path, lines: List[str]) -> None:
    """Write report atomically to output_file with LF newlines."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_file.with_suffix(output_file.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line)
            f.write("\n")
    tmp.replace(output_file)


def validate_paths(cfg: Config) -> Tuple[bool, str]:
    """Basic sanity checks for input directories."""
    if not cfg.skill_dir.exists():
        return False, f"Skill dir not found: {cfg.skill_dir}"
    if not cfg.class_dir.exists():
        return False, f"Class dir not found: {cfg.class_dir}"
    return True, ""


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI definition."""
    p = argparse.ArgumentParser(description="SkillAPI YAML parser -> text report.")
    p.add_argument("--class-dir", type=str, default=None, help="Path to SkillAPI class YAML directory.")
    p.add_argument("--skill-dir", type=str, default=None, help="Path to SkillAPI skill YAML directory.")
    p.add_argument("--output", type=str, default=None, help="Output file path.")
    p.add_argument("--concurrency", type=int, default=None, help="Concurrent workers (default 8).")
    p.add_argument("--log-level", type=str, default=None, help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return p


# ────────────────────────────────── Main ────────────────────────────────────


def main() -> int:
    """Entry point."""
    args = build_arg_parser().parse_args()
    cfg = Config.from_env_and_args(args)

    app_log, err_log, parse_log = setup_logging(cfg)
    app_log.info(
        "startup",
        extra={
            "extra": {
                "cfg": {
                    "class_dir": str(cfg.class_dir),
                    "skill_dir": str(cfg.skill_dir),
                    "output_file": str(cfg.output_file),
                    "concurrency": cfg.concurrency,
                    "io_timeout_sec": cfg.io_timeout_sec,
                    "max_files": cfg.max_files,
                    "circuit_breaker_err_ratio": cfg.circuit_breaker_err_ratio,
                    "log_dir": str(cfg.log_dir),
                    "log_prefix": cfg.log_prefix,
                    "disable_rotation": cfg.disable_rotation,
                }
            }
        },
    )

    ok, reason = validate_paths(cfg)
    if not ok:
        err_log.error("invalid-paths", extra={"extra": {"reason": reason}})
        print(f"ERROR: {reason}", file=sys.stderr)
        return 2

    # Phase 1: parse skills
    start = time.perf_counter()
    skill_data = parse_all_skills(
        cfg.skill_dir,
        cfg.max_files,
        cfg.max_icon_lore_len,
        cfg.concurrency,
        cfg.io_timeout_sec,
        parse_log,
    )
    t1_ms = int((time.perf_counter() - start) * 1000)
    app_log.info("skills-parsed", extra={"extra": {"skills": len(skill_data), "ms": t1_ms}})

    # Phase 2: render report
    start = time.perf_counter()
    report_lines = render_report(
        cfg.class_dir,
        skill_data,
        cfg.max_files,
        cfg.concurrency,
        cfg.io_timeout_sec,
        parse_log,
    )
    t2_ms = int((time.perf_counter() - start) * 1000)
    app_log.info("classes-rendered", extra={"extra": {"lines": len(report_lines), "ms": t2_ms}})

    # Simple circuit breaker: if we had skills but produced no lines
    total_inputs = len(skill_data)
    err_ratio_approx = 0.0 if report_lines else (1.0 if total_inputs > 0 else 0.0)
    if circuit_breaker_should_trip(
        max(1, total_inputs), int(err_ratio_approx * max(1, total_inputs)), cfg.circuit_breaker_err_ratio
    ):
        msg = "Circuit breaker tripped: no renderable classes"
        err_log.error("circuit-breaker", extra={"extra": {"reason": msg}})
        print(f"ERROR: {msg}", file=sys.stderr)
        return 3

    # Write output
    try:
        write_report(cfg.output_file, report_lines)
        app_log.info(
            "report-written",
            extra={"extra": {"path": str(cfg.output_file), "lines": len(report_lines)}},
        )
    except Exception as exc:
        err_log.error("write-failed", extra={"extra": {"path": str(cfg.output_file), "error": repr(exc)}})
        print("\n".join(report_lines))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
