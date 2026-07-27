"""Reporting diagnostics must not disable themselves in silence.

Several metric fallbacks and sanity checks in ``src/reporting.py`` are wrapped
in broad handlers so a missing or malformed artifact never fails a run. That is
the right call, but swallowing the error silently makes a check that *never
ran* indistinguishable from a check that *passed* — and the default it leaves
behind (slippage 0, win_rate 0.0, "no degeneracy") reads as a real result.

These tests pin the contract: the run still succeeds, and the degradation is
reported as a warning.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from unittest.mock import patch

from src.reporting import ReportGenerator


def _generator(tmp_path: Path) -> ReportGenerator:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    gen = ReportGenerator.__new__(ReportGenerator)
    gen.artifacts_dir = artifacts
    return gen


def _unreadable_csv(only: str):
    """Make reads of one specific artifact raise, leaving other reads intact.

    Writing malformed bytes is not enough — pandas parses almost any garbage
    into *some* frame, so the handler would never be exercised. Patching every
    read is too blunt: the routines under test read other artifacts first and
    would die before reaching the branch. So the failure is scoped by filename.
    """
    real = pd.read_csv

    def _side_effect(path, *args, **kwargs):
        if only in str(path):
            raise OSError(f"simulated unreadable artifact: {only}")
        return real(path, *args, **kwargs)

    return patch("pandas.read_csv", side_effect=_side_effect)


class TestHandlersAreNotSilent:
    """Each degraded path must emit a warning rather than pass silently."""

    def test_source_has_no_silent_exception_pass_in_reporting(self):
        """Guard against the pattern being reintroduced."""
        import ast

        import src.reporting as reporting_mod

        tree = ast.parse(Path(reporting_mod.__file__).read_text(encoding="utf-8"))
        silent = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                body = [
                    s
                    for s in node.body
                    if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
                ]
                if len(body) == 1 and isinstance(body[0], ast.Pass):
                    silent.append(node.lineno)
        assert silent == [], (
            f"silent `except ...: pass` reintroduced in src/reporting.py at lines {silent}"
        )

    def test_warnings_is_importable_at_module_scope(self):
        """The handlers call warnings.warn outside the blocks that import it."""
        import src.reporting as reporting_mod

        assert hasattr(reporting_mod, "warnings")
        assert reporting_mod.warnings is warnings


def _warnings_from(fn) -> str:
    """Run ``fn`` and return its warning text.

    The call is allowed to raise: these handlers sit inside larger routines
    that may fail later for unrelated reasons on a synthetic run. What is
    under test is whether the degraded artifact was *reported*.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fn()
        except Exception:
            pass
    return " | ".join(str(w.message) for w in caught)


# The handlers inside `_calculate_metrics` (slippage stats, win_rate fallback,
# slippage-degeneracy check) are not covered behaviourally here: that method
# takes 20+ required arguments including a live `portfolio_state`, so any
# synthetic driver would pin the fixture rather than the behaviour. They are
# covered structurally by `test_source_has_no_silent_exception_pass_in_reporting`
# above, which is what actually prevents the pattern from coming back.


class TestPositionsHistoryVisibility:
    def test_unreadable_positions_csv_warns(self, tmp_path: Path):
        gen = _generator(tmp_path)
        (gen.artifacts_dir / "fills.csv").write_text(
            "ts,leg,side,qty,price,position_id,symbol,fee_usd,slippage_cost_usd\n"
            "2021-01-01T00:00:00Z,ENTRY,BUY,1.0,100.0,p1,BTCUSDT,0.1,0.0\n"
        )
        (gen.artifacts_dir / "positions.csv").write_text("ts,position_id\n1,p1\n")

        with _unreadable_csv("positions.csv"):
            messages = _warnings_from(lambda: gen._write_trades_artifact([]))

        assert "positions.csv" in messages, (
            f"unreadable positions.csv silently degraded the rebuild; got: {messages!r}"
        )
