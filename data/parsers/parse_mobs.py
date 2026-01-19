#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parse_mobs.py — Grimoire of Gaia mobs parser (Python 3.10)

Purpose:
  Parse Grimoire of Gaia mob JSON files and generate a deterministic text report:
    • min/max health (generic.maxHealth)
    • min/max damage (generic.attackDamage)

Key points:
  - PEP 8 / PEP 257 compliant.
  - CLI + .env configuration (no hardcoded paths).
  - Structured JSON logging (channels: app, error, parse).
  - Windows-safe logging: separate log files, optional NO-ROTATION mode.
  - Deterministic output (sorted file lists and keys).
  - Async-ready: concurrent file parsing via asyncio.to_thread.
  - Guards: timeouts, file limits, atomic write.

Usage example:
  python parse_mobs.py \
    --mobs-dir "C:/Nextgen/config/MobsPropertiesRandomness/json" \
    --output "C:/LLM/parsed/mobs.txt"

Environment variables (.env supported):
  MOBS_DIR, OUTPUT_FILE
  LOG_DIR (default C:/LLM/logs/parser), LOG_LEVEL, LOG_MAX_BYTES, LOG_BACKUPS
  LOG_PREFIX (default "parser_"), LOG_DISABLE_ROTATION ("1" to disable)
  IO_TIMEOUT_SEC, MAX_FILES, CONCURRENCY
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional .env loader
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # optional


# ────────────────────────────────── Config ──────────────────────────────────


@dataclass(frozen=True)
class Config:
    """Runtime configuration loaded from env and CLI."""

    mobs_dir: Path
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

        mobs_dir = Path(args.mobs_dir or os.getenv("MOBS_DIR", r"C:/Nextgen/config/MobsPropertiesRandomness/json"))
        output_file = Path(args.output or os.getenv("OUTPUT_FILE", r"C:/LLM/parsed/mobs.txt"))

        # Default to a separate folder to avoid conflicts with any running bot
        log_dir = env_path("LOG_DIR", os.getenv("LOG_DIR", r"C:/LLM/logs/parser"))
        log_level = (args.log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
        log_max_bytes = env_int("LOG_MAX_BYTES", 1_000_000)
        log_backups = env_int("LOG_BACKUPS", 1)
        log_prefix = os.getenv("LOG_PREFIX", "parser_")
        disable_rotation = os.getenv("LOG_DISABLE_ROTATION", "0") == "1"

        io_timeout_sec = env_float("IO_TIMEOUT_SEC", 10.0)
        max_files = env_int("MAX_FILES", 20_000)
        concurrency = env_int("CONCURRENCY", max(1, (args.concurrency or 8)))

        return Config(
            mobs_dir=mobs_dir,
            output_file=output_file,
            log_dir=log_dir,
            log_level=log_level,
            log_max_bytes=log_max_bytes,
            log_backups=log_backups,
            log_prefix=log_prefix,
            disable_rotation=disable_rotation,
            io_timeout_sec=io_timeout_sec,
            max_files=max_files,
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


def load_json_file(path: Path) -> Dict:
    """Load a JSON file; return {} on any failure."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data or {}
    except Exception:
        return {}


def parse_mob_record(data: Dict, fallback_id: str) -> Optional[Dict[str, object]]:
    """Parse a single mob JSON object to normalized dict."""
    if not isinstance(data, dict):
        return None

    mob_id = str(data.get("mob_id", fallback_id))

    # Defaults to em dash if missing
    result: Dict[str, object] = {
        "mob_id": mob_id,
        "min_hp": "—",
        "max_hp": "—",
        "min_dmg": "—",
        "max_dmg": "—",
    }

    attributes = data.get("attributes", [])
    if isinstance(attributes, list):
        for attr in attributes:
            if not isinstance(attr, dict):
                continue
            aid = str(attr.get("id", ""))
            mod = attr.get("modifier", {})
            if not isinstance(mod, dict):
                mod = {}

            if aid == "generic.maxHealth":
                result["min_hp"] = mod.get("min", "—")
                result["max_hp"] = mod.get("max", "—")
            elif aid == "generic.attackDamage":
                result["min_dmg"] = mod.get("min", "—")
                result["max_dmg"] = mod.get("max", "—")

    return result


def parse_mob_file(path: Path) -> Optional[Dict[str, object]]:
    """Parse one mob JSON file into a normalized dict."""
    data = load_json_file(path)
    record = parse_mob_record(data, fallback_id=path.stem)
    return record


def parse_all_mobs(
    mobs_dir: Path,
    max_files: int,
    concurrency: int,
    timeout: float,
    parse_log: logging.Logger,
) -> Dict[str, Dict[str, object]]:
    """Parse all *.json in mobs_dir concurrently (deterministic merge)."""
    files = sorted([p for p in mobs_dir.iterdir() if p.suffix.lower() == ".json"])
    if len(files) > max_files:
        files = files[:max_files]

    sem = asyncio.Semaphore(concurrency)
    results: Dict[str, Dict[str, object]] = {}
    errors: List[Tuple[Path, str]] = []

    async def worker(file_path: Path) -> None:
        async with sem:
            start = time.perf_counter()
            try:
                record = await asyncio.wait_for(asyncio.to_thread(parse_mob_file, file_path), timeout=timeout)
                if record and isinstance(record, dict):
                    # merge deterministically by mob_id
                    mob_id = str(record.get("mob_id", file_path.stem))
                    results[mob_id] = record  # last write wins but files are sorted
                parse_log.info(
                    "parsed-mob-file",
                    extra={"extra": {"file": str(file_path), "ms": int((time.perf_counter() - start) * 1000)}},
                )
            except Exception as exc:
                errors.append((file_path, repr(exc)))
                parse_log.warning("mob-parse-failed", extra={"extra": {"file": str(file_path), "error": repr(exc)}})

    async def run_all() -> None:
        await asyncio.gather(*(worker(p) for p in files))

    asyncio.run(run_all())

    if errors:
        parse_log.warning("mob-parse-errors", extra={"extra": {"count": len(errors)}})

    return results


def render_report_lines(mobs: Dict[str, Dict[str, object]]) -> List[str]:
    """Render deterministic text lines for all mobs."""
    lines: List[str] = []
    for mob_id in sorted(mobs.keys()):
        mob = mobs[mob_id]
        min_hp = mob.get("min_hp", "—")
        max_hp = mob.get("max_hp", "—")
        min_dmg = mob.get("min_dmg", "—")
        max_dmg = mob.get("max_dmg", "—")

        lines.append(f"Моб: {mob_id}")
        lines.append(f"Здоровье: {min_hp}–{max_hp}")
        lines.append(f"Урон: {min_dmg}–{max_dmg}")
        lines.append("---")
    return lines


# ───────────────────────────────── Utilities ────────────────────────────────


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
    """Basic sanity checks for input directory."""
    if not cfg.mobs_dir.exists():
        return False, f"Mobs dir not found: {cfg.mobs_dir}"
    return True, ""


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI definition."""
    p = argparse.ArgumentParser(description="Grimoire of Gaia mob JSON parser -> text report.")
    p.add_argument("--mobs-dir", type=str, default=None, help="Path to mobs JSON directory.")
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
                    "mobs_dir": str(cfg.mobs_dir),
                    "output_file": str(cfg.output_file),
                    "concurrency": cfg.concurrency,
                    "io_timeout_sec": cfg.io_timeout_sec,
                    "max_files": cfg.max_files,
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

    # Phase 1: parse mobs
    start = time.perf_counter()
    mobs = parse_all_mobs(
        cfg.mobs_dir,
        cfg.max_files,
        cfg.concurrency,
        cfg.io_timeout_sec,
        parse_log,
    )
    t1_ms = int((time.perf_counter() - start) * 1000)
    app_log.info("mobs-parsed", extra={"extra": {"count": len(mobs), "ms": t1_ms}})

    # Phase 2: render report
    lines = render_report_lines(mobs)

    # Write output
    try:
        write_report(cfg.output_file, lines)
        app_log.info(
            "report-written",
            extra={"extra": {"path": str(cfg.output_file), "lines": len(lines)}},
        )
    except Exception as exc:
        err_log.error("write-failed", extra={"extra": {"path": str(cfg.output_file), "error": repr(exc)}})
        print("\n".join(lines))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
