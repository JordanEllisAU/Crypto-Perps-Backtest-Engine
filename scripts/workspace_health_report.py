#!/usr/bin/env python3
"""workspace_health_report.py - Report workspace hygiene and process debt.

Run: python scripts/workspace_health_report.py [--leader-repo ../DeceptionLeaderBot]
"""
from __future__ import annotations

import argparse
import ast
import logging
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PROD_DIRS = {"src"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "dist", "build", "__pycache__", ".pytest_cache"}
WORKSPACE_FILES = {".github/PULL_REQUEST_TEMPLATE.md", "CONTRIBUTING.md", "AGENTS.md"}


def _run(cmd: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)


def _branch_age_report() -> list[dict]:
    _run(["git", "fetch", "--prune"], cwd=ROOT)
    default = (
        _run(["git", "symbolic-ref", "refs/remotes/origin/HEAD"], cwd=ROOT)
        .stdout.strip()
        .replace("refs/remotes/origin/", "")
    )
    refs = (
        _run(
            ["git", "for-each-ref", "refs/remotes/origin/", "--format=%(refname:short)\t%(committerdate:iso8601)\t%(committername)"],
            cwd=ROOT,
        )
        .stdout.strip()
        .splitlines()
    )
    rows: list[dict] = []
    now = datetime.now(timezone.utc)
    for line in refs:
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue
        ref, date_str, author = parts
        branch = ref.split("/", 1)[1] if "/" in ref else ref
        if branch in (default, "HEAD"):
            continue
        try:
            dt = datetime.fromisoformat(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_days = (now - dt).total_seconds() / 86400
        except ValueError:
            age_days = -1.0
        merged = (
            _run(["git", "merge-base", "--is-ancestor", ref, f"origin/{default}"], cwd=ROOT)
            .returncode
            == 0
        )
        rows.append({
            "branch": branch,
            "age_days": round(age_days, 1),
            "author": author,
            "merged": merged,
            "action": "delete" if merged else ("review" if age_days > 7 else "keep"),
        })
    return sorted(rows, key=lambda r: r["age_days"], reverse=True)


def _oversized_modules() -> list[dict]:
    oversized: list[dict] = []
    for p in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        rel = p.relative_to(ROOT)
        in_prod = rel.parts[0] in PROD_DIRS if rel.parts else False
        text = p.read_text(encoding="utf-8", errors="ignore")
        line_count = sum(
            1 for line in text.splitlines() if line.strip() and not line.strip().startswith("#")
        )
        if in_prod and line_count > 500:
            oversized.append({"file": str(rel), "lines": line_count, "cap": 500})
    return sorted(oversized, key=lambda r: r["lines"], reverse=True)


def _missing_workspace_files() -> list[str]:
    return [f for f in WORKSPACE_FILES if not (ROOT / f).is_file()]


def _silent_except_count() -> list[dict]:
    findings: list[dict] = []
    for p in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        rel = p.relative_to(ROOT)
        in_prod = rel.parts[0] in PROD_DIRS if rel.parts else False
        if not in_prod:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            log.warning("parse_failed: %s - %s", rel, exc)
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                body = node.body
                if node.type is None or (len(body) == 1 and isinstance(body[0], ast.Pass)):
                    findings.append({
                        "file": str(rel),
                        "line": node.lineno,
                        "type": "bare" if node.type is None else "empty_pass",
                    })
    return findings


def _print_in_prod_modules() -> list[dict]:
    findings: list[dict] = []
    for p in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        rel = p.relative_to(ROOT)
        in_prod = rel.parts[0] in PROD_DIRS if rel.parts else False
        if not in_prod:
            continue
        rel_posix = rel.as_posix()
        text = p.read_text(encoding="utf-8", errors="ignore")
        for m in re.finditer(r"^\s*print\s*\(", text, re.MULTILINE):
            line = text[:m.start()].count("\n") + 1
            findings.append({"file": rel_posix, "line": line})
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Workspace health report")
    parser.add_argument("--leader-repo", type=Path, help="Path to DeceptionLeaderBot repo for cross-repo alignment checks")
    args = parser.parse_args()

    branches = _branch_age_report()
    oversized = _oversized_modules()
    missing = _missing_workspace_files()
    silent_excepts = _silent_except_count()
    prints = _print_in_prod_modules()

    log.info("workspace_health_summary branches=%s oversized=%s missing=%s silent_excepts=%s prints=%s",
             len(branches), len(oversized), len(missing), len(silent_excepts), len(prints))

    cross_repo: list[dict] = []
    if args.leader_repo and args.leader_repo.is_dir():
        leader = args.leader_repo.resolve()
        for name in ("DeceptionSignal", "DeceptionModule"):
            here = _run(["git", "grep", "-l", name], cwd=ROOT).stdout.strip().splitlines()
            there = _run(["git", "grep", "-l", name], cwd=leader).stdout.strip().splitlines()
            cross_repo.append({"symbol": name, "local_files": here, "remote_files": there})
        log.info("cross_repo_alignment: %s", cross_repo)

    log.info("Workspace Health Report")
    log.info("=" * 40)
    log.info("Remote branches: %s (oldest %.1f days)", len(branches), branches[0]["age_days"] if branches else 0)
    log.info("Oversized production modules: %s", len(oversized))
    log.info("Missing workspace files: %s", missing if missing else "none")
    log.info("Silent exception handlers: %s", len(silent_excepts))
    log.info("print() calls in production: %s", len(prints))
    if cross_repo:
        log.info("Cross-repo symbols checked: %s", len(cross_repo))
    log.info("=" * 40)
    log.info("Run `python scripts/workspace_health_report.py --leader-repo ../DeceptionLeaderBot` for cross-repo checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
