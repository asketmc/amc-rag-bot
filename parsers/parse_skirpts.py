# amc-rag-bot/parsers/parse_skirpts.py
"""
Skript (.sk) parser -> aggregates commands, events, variables, and effects
into a single RAG-friendly text file.

- Python: 3.10+
- Encoding: UTF-8
- Style: PEP 8 / PEP 257

The parser scans a Skript source tree, extracts structured signals, and writes
a deterministic summary suitable for downstream retrieval pipelines.

Usage (defaults are Windows-style paths; override via CLI):
    python parse_skirpts.py \
        --src "C:/Nextgen/plugins/Skript/scripts" \
        --out "C:/LLM/parsed/skript.txt"
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Set, Tuple

# -----------------------------------------------------------------------------
# Configuration (can be overridden via CLI)
# -----------------------------------------------------------------------------

DEFAULT_SKRIPT_DIR = Path(r"C:/Nextgen/plugins/Skript/scripts")
DEFAULT_OUTPUT_FILE = Path(r"C:/LLM/parsed/skript.txt")

# -----------------------------------------------------------------------------
# Regexes
# -----------------------------------------------------------------------------

CMD_RE = re.compile(r"^\s*command\s+/([\w\d_]+)", re.IGNORECASE)
DESC_RE = re.compile(r"^\s*description\s*:\s*(.+)", re.IGNORECASE)
COMMENT_RE = re.compile(r"^\s*#\s*(.+)")
EVENT_RE = re.compile(r"^\s*on\s+(.+?):\s*$", re.IGNORECASE)
VAR_RE = re.compile(r"\{[^{}]+\}")
EFFECT_RE = re.compile(
    r"^\s*(add|remove|set|give|execute|broadcast|make|spawn)\b",
    re.IGNORECASE,
)

# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Command:
    """Represents a Skript command definition."""

    name: str
    description: str
    source_file: str


@dataclass(frozen=True)
class ParseResult:
    """Aggregated parse results for the entire tree."""

    commands: List[Command]
    events: Set[str]
    variables: Set[str]
    effects: Set[str]


# -----------------------------------------------------------------------------
# Parsing utilities
# -----------------------------------------------------------------------------

def _read_lines(path: Path) -> List[str]:
    """Read a file as UTF-8 (ignoring errors) and return a list of lines."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:  # pragma: no cover
        logging.warning("Failed to read %s: %s", path, exc)
        return []
    return text.splitlines()


def _leading_indent(s: str) -> int:
    """Return the number of leading whitespace characters (spaces/tabs)."""
    return len(s) - len(s.lstrip())


def _find_block_description(
    lines: List[str],
    start_index: int,
    base_indent: int,
) -> Optional[str]:
    """Search for an inline `description:` field inside the current indented block."""
    k = start_index + 1
    while k < len(lines):
        ln = lines[k]
        if _leading_indent(ln) <= base_indent:
            break  # left the block
        desc_m = DESC_RE.match(ln)
        if desc_m:
            return desc_m.group(1).strip()
        k += 1
    return None


def _find_above_comment_description(lines: List[str], start_index: int) -> Optional[str]:
    """Search upwards for a comment that can serve as a description."""
    j = start_index - 1
    while j >= 0 and lines[j].strip():
        cmt = COMMENT_RE.match(lines[j])
        if cmt:
            return cmt.group(1).strip()
        j -= 1
    return None


def _scan_commands(lines: List[str], source_file: str) -> List[Command]:
    """Extract command blocks with descriptions."""
    results: List[Command] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        cmd_m = CMD_RE.match(line)
        if cmd_m:
            cmd_name = "/" + cmd_m.group(1)
            base_indent = _leading_indent(line)
            description = (
                _find_above_comment_description(lines, i)
                or _find_block_description(lines, i, base_indent)
                or "No description"
            )
            results.append(Command(name=cmd_name, description=description, source_file=source_file))
        i += 1
    return results


def _scan_events(lines: Iterable[str]) -> Set[str]:
    """Extract event names from `on <event>:` lines."""
    events: Set[str] = set()
    for line in lines:
        m = EVENT_RE.match(line)
        if m:
            event = m.group(1).strip()
            if event:
                events.add(event)
    return events


def _scan_variables(lines: Iterable[str]) -> Set[str]:
    """Extract variable placeholders `{...}`."""
    vars_found: Set[str] = set()
    for line in lines:
        for var in VAR_RE.findall(line):
            # Normalize whitespace inside braces to improve deduplication.
            normalized = re.sub(r"\s+", " ", var)
            vars_found.add(normalized)
    return vars_found


def _scan_effects(lines: Iterable[str]) -> Set[str]:
    """Extract effect verbs at the beginning of a line."""
    effects: Set[str] = set()
    for line in lines:
        m = EFFECT_RE.match(line)
        if m:
            effects.add(m.group(1).lower())
    return effects


def parse_file(path: Path) -> Tuple[List[Command], Set[str], Set[str], Set[str]]:
    """Parse one `.sk` file and return (commands, events, variables, effects)."""
    lines = _read_lines(path)
    commands = _scan_commands(lines, source_file=path.name)
    events = _scan_events(lines)
    variables = _scan_variables(lines)
    effects = _scan_effects(lines)
    return commands, events, variables, effects


def parse_tree(src_dir: Path) -> ParseResult:
    """Parse all `.sk` files under `src_dir` (recursive)."""
    commands: List[Command] = []
    events: Set[str] = set()
    variables: Set[str] = set()
    effects: Set[str] = set()

    for file_path in src_dir.glob("**/*.sk"):
        file_cmds, file_events, file_vars, file_effects = parse_file(file_path)
        commands.extend(file_cmds)
        events.update(file_events)
        variables.update(file_vars)
        effects.update(file_effects)

    # Deterministic ordering for output reproducibility.
    commands_sorted = sorted(commands, key=lambda c: (c.name.lower(), c.source_file.lower()))
    return ParseResult(
        commands=commands_sorted,
        events=events,
        variables=variables,
        effects=effects,
    )


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def write_summary(out_path: Path, result: ParseResult) -> None:
    """Write a text summary that is stable and RAG-friendly."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        # Commands
        out.write("Skript Commands\n")
        out.write("===============\n\n")
        for cmd in result.commands:
            out.write(f"{cmd.name}: {cmd.description}  (from {cmd.source_file})\n")

        # Events
        out.write("\n---\nEvents:\n")
        for e in sorted(result.events, key=lambda s: s.lower()):
            out.write(f" - {e}\n")

        # Variables
        out.write("\nVariables:\n")
        for v in sorted(result.variables, key=lambda s: s.lower()):
            out.write(f" - {v}\n")

        # Effects
        out.write("\nEffects:\n")
        for fx in sorted(result.effects, key=lambda s: s.lower()):
            out.write(f" - {fx}\n")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    """Create an argument parser for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Parse Skript (.sk) sources and produce a consolidated text summary."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_SKRIPT_DIR,
        help="Root directory containing .sk files (recursive).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output text file path.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity.",
    )
    return parser


def main() -> int:
    """Entry point for CLI execution."""
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.src.exists():
        logging.error("Source directory does not exist: %s", args.src)
        return 2

    logging.info("Scanning Skript tree: %s", args.src)
    result = parse_tree(args.src)
    logging.info(
        "Parsed: %d commands, %d events, %d variables, %d effects",
        len(result.commands),
        len(result.events),
        len(result.variables),
        len(result.effects),
    )

    write_summary(args.out, result)
    logging.info("Summary written: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
