#!/usr/bin/env python3
"""quality_sweep.py — local A+ quality gate and auto-fix.

Runs ``make ci`` twice, scans for slop/placeholder/junk/bad code, applies safe
auto-fixes, and produces a module lattice/index.

Usage: python scripts/quality_sweep.py [--apply] [--aggressive] [--strict] [--ci-runs N]
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("quality_sweep")

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "runtime"
LATTICE_DIR = ROOT / "docs" / "lattices"
LINE_LIMIT = 500
FUNC_LIMIT = 50

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "_shared",
    ".devin",
    ".windsurf",
    ".cursor",
    ".claude",
    ".agents",
    ".playwright-mcp",
}
REPO_CONFIG = {
    "deception": {
        "name": "DeceptionLeaderBot",
        "prod_dirs": {"core", "exchanges", "live", "scoring", "trading"},
        "root_prod": ["main.py", "launcher.py", "web_dashboard.py"],
        "skip_dirs": SKIP_DIRS,
    },
    "crypto": {
        "name": "Crypto-Perps-Backtest-Engine",
        "prod_dirs": {"src", "engine_core"},
        "root_prod": [],
        "skip_dirs": SKIP_DIRS,
    },
}

PLACEHOLDER_RE = re.compile(r"#.*\b(TODO|FIXME|XXX|HACK)\b", re.IGNORECASE)
TYPE_IGNORE_RE = re.compile(r"#\s*type:\s*ignore\b(?![\t ]+[A-Za-z0-9])")
BARE_EXCEPT_RE = re.compile(r"^(\s*)except\s*:(.*)$")


@dataclass
class Issue:
    rule: str
    line: int
    message: str
    severity: str  # error, warn, info
    fixable: bool = False


@dataclass
class ModuleInfo:
    path: Path
    rel: str
    lane: str
    sloc: int
    func_count: int
    max_func_sloc: int
    issues: list[Issue] = field(default_factory=list)
    trust_tier: str = "A+"


def detect_repo() -> tuple[str, dict[str, set[str] | list[str]]]:
    agents = ROOT / "AGENTS.md"
    if agents.exists():
        text = agents.read_text(encoding="utf-8", errors="ignore")
        if "DeceptionLeaderBot" in text:
            return "deception", REPO_CONFIG["deception"]
        if "Crypto-Perps-Backtest-Engine" in text or "backtest" in text.lower():
            return "crypto", REPO_CONFIG["crypto"]
    if (ROOT / "src" / "engine.py").exists() or (ROOT / "engine_core").is_dir():
        return "crypto", REPO_CONFIG["crypto"]
    if (ROOT / "exchanges").is_dir():
        return "deception", REPO_CONFIG["deception"]
    return "deception", REPO_CONFIG["deception"]


def sloc(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith("#"))


def function_sloc(lines: list[str], start: int, end: int) -> int:
    return sum(1 for line in lines[start - 1 : end] if line.strip() and not line.strip().startswith("#"))


def py_files(config: dict[str, set[str] | list[str]]) -> list[Path]:
    try:
        result = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            files = [ROOT / p for p in result.stdout.splitlines() if p]
        else:
            files = list(ROOT.rglob("*.py"))
    except FileNotFoundError:
        files = list(ROOT.rglob("*.py"))

    skip_dirs = config.get("skip_dirs", set())  # type: ignore[union-attr]
    out: list[Path] = []
    for p in files:
        rel = p.relative_to(ROOT)
        if any(part in skip_dirs for part in rel.parts):
            continue
        if not p.is_file():
            continue
        out.append(p)
    return sorted(out)


def is_production(p: Path, config: dict[str, set[str] | list[str]]) -> bool:
    rel = p.relative_to(ROOT)
    return (rel.parts and rel.parts[0] in config.get("prod_dirs", set())) or rel.name in config.get("root_prod", [])


def lane_for(p: Path) -> str:
    rel = p.relative_to(ROOT)
    if rel.parts:
        return rel.parts[0]
    return "root"


def make_ci(runs: int) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for i in range(1, runs + 1):
        logger.info("Running make ci (pass %d/%d)", i, runs)
        result = subprocess.run(["make", "ci"], cwd=ROOT, capture_output=True, text=True, check=False)
        results.append({"pass": i, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode != 0:
            logger.error("make ci pass %d failed", i)
            break
    return results


def _used_names(tree: ast.AST) -> set[str]:
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                used.add(root.id)
    return used


def _imported_aliases(tree: ast.AST) -> list[tuple[str, str, ast.AST]]:
    """Return list of (alias_name, kind, import_node) for top-level imports."""
    aliases: list[tuple[str, str, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name.split(".")[0]
                aliases.append((name, "import", node))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    aliases.append(("*", "wildcard", node))
                else:
                    name = alias.asname if alias.asname else alias.name
                    aliases.append((name, "from", node))
    return aliases


def _is_silent_except(handler: ast.ExceptHandler) -> bool:
    body = [n for n in handler.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
    if not body:
        return True
    for node in body:
        if isinstance(node, ast.Raise):
            return False
        if isinstance(node, ast.Expr):
            call = node.value
            if isinstance(call, ast.Call):
                func = call.func
                if isinstance(func, ast.Attribute) and func.attr in ("debug", "info", "warning", "error", "exception", "critical"):
                    return False
                if isinstance(func, ast.Name) and func.id in ("print", "logging"):
                    return False
        if isinstance(node, (ast.Pass, ast.Break, ast.Continue)):
            continue
        if isinstance(node, ast.Return) and node.value is None:
            continue
        return False
    return True


def _is_placeholder_body(body: list[ast.AST]) -> bool:
    nodes = [n for n in body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
    if not nodes:
        return True
    if len(nodes) == 1:
        n = nodes[0]
        if isinstance(n, ast.Pass):
            return True
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and n.value.value is Ellipsis:
            return True
    return False


def _has_reflection(text: str) -> bool:
    return any(name in text for name in ("getattr(", "setattr(", "hasattr(", "eval(", "exec(", "globals(", "locals("))


def detect_issues(path: Path, text: str, tree: ast.AST, config: dict[str, set[str] | list[str]]) -> list[Issue]:
    issues: list[Issue] = []
    prod = is_production(path, config)
    lines = text.splitlines()

    for m in PLACEHOLDER_RE.finditer(text):
        line = text[: m.start()].count("\n") + 1
        issues.append(Issue("TODO_COMMENT", line, f"{m.group(1)} comment", "info"))

    for m in TYPE_IGNORE_RE.finditer(text):
        line = text[: m.start()].count("\n") + 1
        issues.append(Issue("TYPE_IGNORE", line, "unqualified `type: ignore`", "info"))

    used = _used_names(tree)
    reflection_file = _has_reflection(text)
    has_all = "__all__" in text

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            func_sloc = function_sloc(lines, node.lineno, node.end_lineno or node.lineno)
            if func_sloc > FUNC_LIMIT:
                issues.append(Issue("LONG_FUNCTION", node.lineno, f"function ~{func_sloc} source lines", "warn"))
            if _is_placeholder_body(node.body):
                issues.append(Issue("PLACEHOLDER", node.lineno, "empty/placeholder function body", "warn"))

            defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]
            for d in defaults:
                if isinstance(d, ast.List | ast.Dict | ast.Set):
                    issues.append(Issue("MUTABLE_DEFAULT", d.lineno, "mutable default argument", "warn", fixable=False))

            for child in ast.walk(node):
                if isinstance(child, ast.Global):
                    issues.append(Issue("GLOBAL_IN_FUNC", child.lineno, "`global` in function", "warn"))
                    break

        elif isinstance(node, ast.ClassDef):
            if _is_placeholder_body(node.body):
                issues.append(Issue("PLACEHOLDER", node.lineno, "empty/placeholder class body", "warn"))

        elif isinstance(node, ast.ExceptHandler):
            if node.type is None:
                issues.append(Issue("BARE_EXCEPT", node.lineno, "bare `except:`", "error", fixable=True))
            elif _is_silent_except(node):
                issues.append(Issue("SILENT_EXCEPT", node.lineno, "silent exception swallow", "warn"))

        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id == "print":
                    rel_posix = path.relative_to(ROOT).as_posix()
                    allowed_print = rel_posix.startswith("core/launcher") or rel_posix == "launcher.py" or rel_posix.startswith("scripts/")
                    print_sev = "warn" if allowed_print else ("error" if prod else "warn")
                    issues.append(Issue("PRINT", node.lineno, "`print()` call", print_sev))
                elif func.id in ("eval", "exec", "compile"):
                    issues.append(Issue("DANGEROUS_BUILTIN", node.lineno, f"`{func.id}()` call", "warn"))
                elif func.id in ("getattr", "setattr", "hasattr"):
                    issues.append(Issue("REFLECTION", node.lineno, f"`{func.id}()` call", "warn"))
                elif func.id == "breakpoint":
                    issues.append(Issue("DEBUGGER", node.lineno, "`breakpoint()` call", "warn"))

        elif isinstance(node, ast.Raise):
            exc = node.exc
            if isinstance(exc, ast.Call | ast.Name):
                name = ""
                if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                    name = exc.func.id
                elif isinstance(exc, ast.Name):
                    name = exc.id
                if name == "NotImplementedError":
                    issues.append(Issue("NOT_IMPLEMENTED", node.lineno, "NotImplementedError raised", "warn"))

        elif isinstance(node, ast.Name):
            if node.id == "Any":
                issues.append(Issue("ANY_TYPE", node.lineno, "`Any` type annotation", "warn"))

    if not has_all and not reflection_file:
        for name, kind, node in _imported_aliases(tree):
            if kind == "wildcard":
                mod = node.module or "."
                issues.append(Issue("WILDCARD_IMPORT", node.lineno, f"wildcard import from {mod}", "warn"))
                continue
            if name not in used:
                issues.append(Issue("UNUSED_IMPORT", 1, f"unused import `{name}`", "info"))

    return issues


def _apply_bare_except_fix(text: str, lines: set[int]) -> str:
    new_lines: list[str] = []
    for idx, line in enumerate(text.splitlines(keepends=True), start=1):
        if idx not in lines:
            new_lines.append(line)
            continue
        stripped_line = line.rstrip("\r\n")
        eol = line[len(stripped_line):]
        match = BARE_EXCEPT_RE.match(stripped_line)
        if match:
            indent, rest = match.groups()
            new_lines.append(f"{indent}except Exception:{rest}{eol}")
        else:
            new_lines.append(line)
    return "".join(new_lines)


def _apply_whitespace_fix(text: str) -> str:
    lines = text.splitlines(keepends=True)
    cleaned = []
    for line in lines:
        if line.endswith("\n"):
            body = line[:-1].rstrip()
            cleaned.append(body + "\n")
        elif line.endswith("\r\n"):
            body = line[:-2].rstrip()
            cleaned.append(body + "\r\n")
        else:
            cleaned.append(line.rstrip())
    text = "".join(cleaned)
    if not text.endswith("\n"):
        text += "\n"
    return text


def apply_safe_fixes(path: Path, issues: list[Issue], apply: bool, aggressive: bool, config: dict[str, set[str] | list[str]]) -> tuple[str, list[str]]:
    if not apply or not issues:
        return path.read_text(encoding="utf-8", errors="ignore"), []

    text = path.read_text(encoding="utf-8", errors="ignore")
    applied: list[str] = []
    prod = is_production(path, config)

    rules = {i.rule for i in issues}
    if "BARE_EXCEPT" in rules and (aggressive or not prod):
        bare_lines = {i.line for i in issues if i.rule == "BARE_EXCEPT"}
        text = _apply_bare_except_fix(text, bare_lines)
        applied.append("BARE_EXCEPT")

    text = _apply_whitespace_fix(text)
    applied.append("WHITESPACE")
    return text, applied


def scan_files(files: list[Path], config: dict[str, set[str] | list[str]]) -> list[ModuleInfo]:
    modules: list[ModuleInfo] = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            logger.warning("%s: syntax error skipped (%s)", p.relative_to(ROOT), exc)
            continue
        rel = p.relative_to(ROOT).as_posix()
        info = ModuleInfo(path=p, rel=rel, lane=lane_for(p), sloc=sloc(text), func_count=sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)), max_func_sloc=0)
        lines = text.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                fs = function_sloc(lines, node.lineno, node.end_lineno or node.lineno)
                info.max_func_sloc = max(info.max_func_sloc, fs)
        info.issues = detect_issues(p, text, tree, config)
        modules.append(info)
    return modules


def trust_tier(info: ModuleInfo) -> str:
    errors = [i for i in info.issues if i.severity == "error"]
    warns = [i for i in info.issues if i.severity == "warn"]
    if errors or info.sloc > LINE_LIMIT or info.max_func_sloc > FUNC_LIMIT:
        return "C"
    if warns:
        return "B"
    return "A+"


def build_lattice(modules: list[ModuleInfo]) -> str:
    out: list[str] = ["# Module Quality Lattice", "| module | lane | sloc | func_count | max_func_sloc | line_limit | trust | issues |", "|---|---|---:|---:|---:|:---:|:---:|---|"]
    for info in sorted(modules, key=lambda m: (m.lane, m.rel)):
        line_ok = "OK" if info.sloc <= LINE_LIMIT else f"OVER {LINE_LIMIT}"
        func_ok = "OK" if info.max_func_sloc <= FUNC_LIMIT else f"OVER {FUNC_LIMIT}"
        issue_names = ", ".join(sorted({i.rule for i in info.issues})) or "-"
        tier = trust_tier(info)
        info.trust_tier = tier
        out.append(f"| `{info.rel}` | {info.lane} | {info.sloc} | {info.func_count} | {info.max_func_sloc} | {line_ok} / {func_ok} | {tier} | {issue_names} |")
    return "\n".join(out)


def write_report(modules: list[ModuleInfo], ci_results: list[dict[str, object]]) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    LATTICE_DIR.mkdir(parents=True, exist_ok=True)

    lattice_path = LATTICE_DIR / "QUALITY_LATTICE.md"
    lattice_path.write_text(build_lattice(modules), encoding="utf-8")

    report = {
        "ci_results": [{"pass": r["pass"], "returncode": r["returncode"]} for r in ci_results],
        "modules": [
            {
                "rel": m.rel,
                "lane": m.lane,
                "sloc": m.sloc,
                "func_count": m.func_count,
                "max_func_sloc": m.max_func_sloc,
                "trust_tier": m.trust_tier,
                "issues": [
                    {"rule": i.rule, "line": i.line, "message": i.message, "severity": i.severity}
                    for i in m.issues
                ],
            }
            for m in modules
        ],
    }
    report_path = RUNTIME_DIR / "quality_sweep_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("Lattice written to %s", lattice_path.relative_to(ROOT))
    logger.info("JSON report written to %s", report_path.relative_to(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Local A+ quality sweep")
    parser.add_argument("--apply", action="store_true", help="Apply safe auto-fixes")
    parser.add_argument("--aggressive", action="store_true", help="Apply fixes in production modules too")
    parser.add_argument("--skip-ci", action="store_true", help="Skip make ci baseline")
    parser.add_argument("--ci-runs", type=int, default=2, help="Times to run make ci")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    parser.add_argument("--repo", choices=list(REPO_CONFIG), help="Repo profile")
    args = parser.parse_args()

    if args.repo:
        repo_name = args.repo
        config = REPO_CONFIG[args.repo]
    else:
        repo_name, config = detect_repo()
    logger.info("Quality sweep for %s", repo_name)

    ci_results: list[dict[str, object]] = []
    if not args.skip_ci:
        ci_results = make_ci(args.ci_runs)
        if any(r["returncode"] != 0 for r in ci_results):
            logger.error("make ci failed; fix before continuing")
            return 1

    files = py_files(config)
    logger.info("Scanning %d Python files", len(files))
    modules = scan_files(files, config)

    if args.apply:
        fixed_count = 0
        for info in modules:
            if not info.issues:
                continue
            new_text, applied = apply_safe_fixes(info.path, info.issues, True, args.aggressive, config)
            if applied:
                info.path.write_text(new_text, encoding="utf-8")
                fixed_count += 1
                logger.info("Fixed %s: %s", info.rel, ", ".join(applied))
        if fixed_count:
            logger.info("Re-scanning after auto-fix")
            modules = scan_files(files, config)

    write_report(modules, ci_results)

    errors = sum(1 for m in modules for i in m.issues if i.severity == "error")
    warns = sum(1 for m in modules for i in m.issues if i.severity == "warn")
    logger.info("Found %d errors and %d warnings across %d modules", errors, warns, len(modules))

    if errors:
        logger.error("Quality sweep FAILED: %d production errors", errors)
        return 1
    if args.strict and warns:
        logger.error("Quality sweep FAILED: %d warnings in strict mode", warns)
        return 1
    logger.info("Quality sweep PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
