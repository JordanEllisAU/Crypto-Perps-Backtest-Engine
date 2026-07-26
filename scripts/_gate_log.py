"""Shared logger for the repo's gate / validation / reporting scripts.

`.claude/rules/anti-circular-debug.md` forbids bare `print()` in validation,
build, and reporting scripts. This module is the single logger those scripts
share, so the rule lives in one place instead of being re-implemented (or
ignored) in each script.

Three different things used to travel over `print()` here, and they are not
interchangeable:

* **Diagnostics** — progress lines, status, pass/fail. These go through
  :func:`get_logger`, which writes to stderr so a caller can redirect the
  report body independently of the running commentary.
* **Reports** — the column-aligned tables and banners these scripts exist to
  produce. Routing them through the logger would prefix every row with a
  timestamp and level and destroy the alignment, so :func:`emit_report` writes
  them to stdout unchanged.
* **Payloads** — machine-readable JSON that a caller parses. Those bytes are
  the script's output contract, so :func:`emit_payload` writes them verbatim
  with no decoration.

The repo has no third-party logging dependency (see requirements.txt), so this
is built on the standard library.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

__all__ = ["get_logger", "emit_report", "emit_payload"]

_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger("engine_gate")
    root.addHandler(handler)
    root.setLevel(os.environ.get("GATE_LOG_LEVEL", "INFO").upper())
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return the shared logger for a gate script."""
    _configure()
    return logging.getLogger(f"engine_gate.{name}")


def emit_report(line: str = "", end: str = "\n") -> None:
    """Write one line of a formatted human-readable report to stdout.

    Report bodies are column-aligned tables and banners: they are the script's
    deliverable, not diagnostics, and a log prefix on every row would break the
    alignment. ``end`` supports the incremental "Checking X... OK" style that
    some validators use.
    """
    sys.stdout.write(f"{line}{end}")
    sys.stdout.flush()


def emit_payload(payload: Any) -> None:
    """Write a machine-readable JSON document to stdout verbatim."""
    json.dump(payload, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    sys.stdout.flush()
