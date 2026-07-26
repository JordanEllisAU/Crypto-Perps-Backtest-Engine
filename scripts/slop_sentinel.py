#!/usr/bin/env python3
"""slop_sentinel.py - diff-based ratchet that fails on NEW slop.

Scans the current branch diff against a base ref (default origin/main) and fails
if any changed production/gate file introduces:
  - bare ``except:`` or ``except ...: pass`` (Python)
  - new ``print()`` calls in production code (src/)
  - a production module growing beyond 500 source lines

Existing slop on the base branch is ignored; only newly added slop fails the gate.

Usage:
  python scripts/slop_sentinel.py [--base-ref origin/main] [--strict]

Environment:
  SLOP_BASE_REF   override base ref (used by CI)
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PRINT_DIRS = {"src"}
SIZE_DIRS = {"src"}
MAX_SOURCE_LINES = 500

PRINT_RE = re.compile(r"^\s*print\s*\(")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        **kw,
    )


def fetch_base(base_ref: str) -> None:
    if base_ref.startswith("origin/"):
        result = run(["git", "fetch", "origin", base_ref.split("/", 1)[1]])
        if result.returncode != 0:
            print(f"warning: could not fetch {base_ref}: {result.stderr.strip()}", file=sys.stderr)


def _git_status_files() -> list[str]:
    result = run(["git", "status", "--porcelain", "--untracked-files=all"])
    if result.returncode != 0:
        return []
    files: list[str] = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        status = line[:2]
        name = line[3:]
        if status.startswith("??") or status[0] in "AMCR":
            files.append(name.split(" -> ")[-1])
    return files


def changed_files(base_ref: str) -> list[Path]:
    lines: list[str] = []

    triple = f"{base_ref}...HEAD"
    result = run(["git", "diff", "--name-only", "--diff-filter=ACM", triple])
    if result.returncode == 0:
        lines.extend(result.stdout.splitlines())

    result = run(["git", "diff", "--name-only", "--diff-filter=ACM", base_ref])
    if result.returncode == 0:
        lines.extend(result.stdout.splitlines())
    lines.extend(_git_status_files())

    if not lines:
        print(f"error: could not diff against {base_ref}", file=sys.stderr)
        sys.exit(1)

    seen: set[str] = set()
    paths: list[Path] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        p = ROOT / line
        if p.exists():
            paths.append(p)
    return sorted(paths)


def base_text(path: Path, base_ref: str) -> str | None:
    rel = path.relative_to(ROOT).as_posix()
    result = run(["git", "show", f"{base_ref}:{rel}"])
    if result.returncode != 0:
        return None
    return result.stdout


def is_in_dirs(rel: Path, dirs: set[str]) -> bool:
    parts = rel.as_posix().split("/")
    return parts[0] in dirs if parts else False


def source_line_count(text: str) -> int:
    return sum(
        1 for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def find_print_lines(text: str) -> set[int]:
    return {
        text[:m.start()].count("\n") + 1
        for m in PRINT_RE.finditer(text)
    }


def find_bare_except_lines(text: str) -> set[int]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                lines.add(node.lineno)
            elif len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                lines.add(node.lineno)
    return lines


def line_at(text: str, lineno: int) -> str:
    lines = text.splitlines()
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""


def analyse(path: Path, base_ref: str) -> list[dict]:
    rel = path.relative_to(ROOT)
    current = path.read_text(encoding="utf-8", errors="ignore")
    base = base_text(path, base_ref) or ""
    offenses: list[dict] = []

    if path.suffix == ".py":
        in_print_dirs = is_in_dirs(rel, PRINT_DIRS)
        if in_print_dirs:
            current_prints = find_print_lines(current)
            base_prints = find_print_lines(base)
            for lineno in sorted(current_prints - base_prints):
                offenses.append({
                    "rule": "NEW_PRINT",
                    "file": rel.as_posix(),
                    "line": lineno,
                    "snippet": line_at(current, lineno),
                })

        current_bare = find_bare_except_lines(current)
        base_bare = find_bare_except_lines(base)
        for lineno in sorted(current_bare - base_bare):
            offenses.append({
                "rule": "BARE_EXCEPT",
                "file": rel.as_posix(),
                "line": lineno,
                "snippet": line_at(current, lineno),
            })

        if is_in_dirs(rel, SIZE_DIRS):
            cur_size = source_line_count(current)
            base_size = source_line_count(base) if base else 0
            if cur_size > MAX_SOURCE_LINES and base_size <= MAX_SOURCE_LINES:
                offenses.append({
                    "rule": "MODULE_SIZE",
                    "file": rel.as_posix(),
                    "line": 1,
                    "snippet": f"{cur_size} source lines (cap {MAX_SOURCE_LINES}); base had {base_size}",
                })

    return offenses


def main() -> int:
    parser = argparse.ArgumentParser(description="Diff-based slop sentinel")
    parser.add_argument("--base-ref", default=os.environ.get("SLOP_BASE_REF", "origin/main"))
    parser.add_argument("--strict", action="store_true", help="Fail on existing slop too")
    args = parser.parse_args()

    fetch_base(args.base_ref)
    paths = changed_files(args.base_ref)

    all_offenses: list[dict] = []
    for path in paths:
        all_offenses.extend(analyse(path, args.base_ref))
        if args.strict:
            rel = path.relative_to(ROOT)
            current = path.read_text(encoding="utf-8", errors="ignore")
            if path.suffix == ".py":
                if is_in_dirs(rel, PRINT_DIRS):
                    for lineno in find_print_lines(current):
                        all_offenses.append({
                            "rule": "EXISTING_PRINT",
                            "file": rel.as_posix(),
                            "line": lineno,
                            "snippet": line_at(current, lineno),
                        })
                for lineno in find_bare_except_lines(current):
                    all_offenses.append({
                        "rule": "EXISTING_BARE_EXCEPT",
                        "file": rel.as_posix(),
                        "line": lineno,
                        "snippet": line_at(current, lineno),
                    })

    if all_offenses:
        seen: set[tuple[str, str, int]] = set()
        unique: list[dict] = []
        for o in all_offenses:
            key = (o["rule"], o["file"], o["line"])
            if key not in seen:
                seen.add(key)
                unique.append(o)

        print(f"Slop sentinel FAILED: {len(unique)} new offense(s)", file=sys.stderr)
        for o in unique:
            print(f"  [{o['rule']}] {o['file']}:{o['line']} {o['snippet']}", file=sys.stderr)
        return 1

    print(f"Slop sentinel passed: {len(paths)} changed file(s), no new slop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
