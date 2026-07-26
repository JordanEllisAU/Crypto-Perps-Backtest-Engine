"""Enforce the repo's own logging directive on gate/validation scripts.

`.claude/rules/anti-circular-debug.md` and `AGENTS.md` forbid bare `print()` in
validation, build, and reporting scripts. Nothing enforced that, so 138 calls
accumulated across `scripts/`. This repo has no lint/hygiene gate of its own,
so the check lives in the test suite — the only gate that actually runs here.

Scripts emit through `scripts/_gate_log.py` instead: `get_logger()` for
diagnostics, `emit_report()` for column-aligned report bodies, and
`emit_payload()` for machine-readable JSON.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

# _gate_log.py is the sanctioned writer: it owns the sys.stdout writes that
# every other script routes through.
EXEMPT = {"_gate_log.py"}


def _script_files() -> list[Path]:
    return sorted(p for p in SCRIPTS_DIR.glob("*.py") if p.name not in EXEMPT)


def _bare_prints(path: Path) -> list[int]:
    """Line numbers of `print(...)` calls, found via AST rather than regex.

    A regex over source text would miss `print` inside a nested scope and
    would falsely match the word in a docstring or comment.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]


def test_scripts_directory_is_not_empty():
    """Guard the guard: an empty glob would make the check below vacuous."""
    assert _script_files(), f"no scripts found under {SCRIPTS_DIR}"


@pytest.mark.parametrize("script", _script_files(), ids=lambda p: p.name)
def test_no_bare_print_in_gate_scripts(script: Path):
    offenders = _bare_prints(script)
    assert not offenders, (
        f"{script.relative_to(SCRIPTS_DIR.parent)} uses bare print() at lines {offenders}. "
        "Use scripts/_gate_log.py: get_logger() for diagnostics, emit_report() for "
        "report bodies, emit_payload() for machine-readable JSON."
    )


def test_gate_log_module_exposes_the_three_emitters():
    import sys

    sys.path.insert(0, str(SCRIPTS_DIR))
    import _gate_log

    assert callable(_gate_log.get_logger)
    assert callable(_gate_log.emit_report)
    assert callable(_gate_log.emit_payload)


def test_emit_report_writes_verbatim_without_a_log_prefix(capsys):
    """Report alignment is the reason these do not go through the logger."""
    import sys

    sys.path.insert(0, str(SCRIPTS_DIR))
    import _gate_log

    row = "  live_tp0.8_full        long  win= 53.0%  net=  -20.4bps"
    _gate_log.emit_report(row)
    out = capsys.readouterr().out
    assert out == row + "\n", f"emit_report altered the line: {out!r}"


def test_emit_report_supports_partial_lines(capsys):
    import sys

    sys.path.insert(0, str(SCRIPTS_DIR))
    import _gate_log

    _gate_log.emit_report("Checking BTCUSDT...", end=" ")
    _gate_log.emit_report("OK")
    assert capsys.readouterr().out == "Checking BTCUSDT... OK\n"


def test_emit_payload_is_parseable_json(capsys):
    import json
    import sys

    sys.path.insert(0, str(SCRIPTS_DIR))
    import _gate_log

    _gate_log.emit_payload({"parity_check": "PASS", "diff_bps": 0.0})
    assert json.loads(capsys.readouterr().out) == {"parity_check": "PASS", "diff_bps": 0.0}


def test_logger_writes_to_stderr_not_stdout(capsys):
    """Diagnostics must not contaminate a report or payload on stdout."""
    import sys

    sys.path.insert(0, str(SCRIPTS_DIR))
    import _gate_log

    _gate_log.get_logger("unit_test").warning("diagnostic line")
    captured = capsys.readouterr()
    assert "diagnostic line" not in captured.out
