# AGENTS.md — Agent Directives

## Anti-circular debugging and hygiene directives

These directives are mandatory for all agents working in this repository. They replace previous ad-hoc orchestration tooling and prevent circular debugging.

- Stop on a blocker; do not re-run the same failing command more than twice without a new hypothesis.
- Run `make ci` before every commit; it compiles all Python files, runs the diff-based slop sentinel, and runs `pytest`.
- No bare `print()` / `console.log()` / `printf()` in gate scripts; use structured logging.
- Generated files must be deterministic (no timestamps, strip trailing whitespace).
- Do not modify tests to pass; fix the code or skip with a reason.
- Use feature branches and PRs; never `git reset --hard` on the default branch.
- No session/chat artifacts in commits (`.claude/*.json`, `.vscode/`, `.playwright-mcp/`, scratch files).
- Document root cause, not just the fix.

## Research and data integrity

- Start every research task with the backtest research spine: `.devin/skills/backtest-research-spine/SKILL.md`.
- Search the memory MCP (`.devin/skills/memory/SKILL.md`) for prior findings on the same signal, invariant, or parameter before starting new research.
- Run `scripts/validate_data_integrity.py` and `make ci` before wiring a new signal or parameter into the engine.
- Use MCP tools (`context7` for library docs, `deepwiki` for repo architecture, `parallel` for live web research, `memory` for prior findings) and record a source corpus in every research artifact.
- Cross-reference DeceptionLeaderBot research findings in `DeceptionLeaderBot/docs/research/` before changing shared primitives.

## Quality sweep

- Run the local A+ sweep before every PR: `python scripts/quality_sweep.py --ci-runs 2`.
- See `.devin/skills/code-quality/quality-sweep/SKILL.md` for the full sweep protocol, rule table, and auto-fix guidance.
- Do not auto-fix `src/` modules that touch accounting invariants, margin math, fee models, or fill simulation without a regression test.

## Workspace process

- Batch related changes into one PR; see `.github/PULL_REQUEST_TEMPLATE.md` and `CONTRIBUTING.md`.
- Delete merged remote branches promptly; use `python scripts/workspace_health_report.py` to find candidates.
- Reusable trading primitives that overlap with `DeceptionLeaderBot` should be proposed as a shared library, not copy-pasted.

## Parallel & batch tool-calling

- Batch independent reads, searches, dry-run checks, and MCP lookups in parallel. Keep ordered gates (`make ci`, `validate_data_integrity.py`, baseline runs) sequential.
- Use `asyncio.gather` or `ThreadPoolExecutor` with bounded concurrency for independent I/O in scripts.
- Never parallelize destructive or data-modifying operations.

Full rule files:
- `.cursor/rules/anti-circular-debug.mdc`
- `.claude/rules/anti-circular-debug.md`

## Change log

### Reporting diagnostics no longer self-disable (2026-07-26)

- **Symptom:** `metrics.json` could report slippage stats of 0, `win_rate` of
  0.0, and `slippage_degeneracy_warning: false` on runs where those values had
  never actually been computed. The `avg_r` and `avg_trade_duration_bars`
  sanity checks could likewise never fire.
- **Root cause:** seven handlers in `src/reporting.py` were written as
  `except Exception: pass`. Not failing a run on a missing or malformed
  artifact is correct, but swallowing the error silently left each default
  indistinguishable from a real result — and for the degeneracy and sanity
  checks it meant a *validation signal that reported "clean" without ever
  having run*. `warnings` was also only imported inside nested blocks, so the
  handlers had no module-scope name to report through.
- **Fix:** `import warnings` moved to module scope; all seven handlers now
  emit `warnings.warn` describing the degradation. No control flow changed —
  every fallback still returns the same default, it is just no longer silent.
  This matches the idiom already used at the two handlers in `_calculate_metrics`
  that did report (`Error reading rebuilt trades.csv`, `Failed to calculate avg_r`).
- **Coverage:** `tests/test_reporting_diagnostics_visible.py` asserts the
  `except ...: pass` pattern is absent from `src/reporting.py` (AST guard,
  prevents reintroduction) and that an unreadable `positions.csv` warns rather
  than silently degrading the trade rebuild. The handlers inside
  `_calculate_metrics` are covered structurally only: that method takes 20+
  required arguments including a live `portfolio_state`, so a synthetic driver
  would pin the fixture rather than the behaviour.
- **Verification:**
  - `python -m pytest -q` -> 59 passed, 6 skipped (was 56 passed, 6 skipped).
  - `python scripts/run_example_oracle.py` -> PASS.

### Gate scripts moved off bare `print()` (2026-07-26)

- **Symptom:** `scripts/*.py` contained **138 bare `print()` calls**, directly
  contradicting the directive above.
- **Root cause:** nothing enforced the rule. This repo has no lint or hygiene
  gate — the GitHub Actions workflows were removed in #6 — so the only gate
  that runs is `pytest`, and it had no opinion on script output.
- **Fix:** `scripts/_gate_log.py` is the shared logger the directive asks for.
  It separates three things that were all `print()` before, and are not
  interchangeable:
  - `get_logger()` — diagnostics and pass/fail, on **stderr**, so a caller can
    redirect the report body independently of the running commentary;
  - `emit_report()` — column-aligned report bodies, written to stdout verbatim
    (a log prefix on every row would destroy the alignment; `end=` supports the
    incremental `Checking X... OK` style);
  - `emit_payload()` — machine-readable JSON, whose bytes are a caller's
    contract and must carry no decoration.
  Built on the standard library, since this repo has no third-party logging
  dependency. All 138 calls converted; `print` count under `scripts/` is 0.
- **Enforcement:** `tests/test_gate_script_hygiene.py` fails the suite if a bare
  `print()` reappears under `scripts/`, detected via AST rather than regex so
  nested scopes are caught and docstrings are not false-positives. It also pins
  that `emit_report` writes verbatim, `emit_payload` round-trips as JSON, and
  the logger stays off stdout. Verified by reintroducing a `print()` and
  confirming the suite fails with the offending file and line.
- **Verification:**
  - `python -m pytest -q` -> 76 passed, 6 skipped (was 56 passed, 6 skipped).
  - `python scripts/run_example_oracle.py` -> exit 0, report formatting intact.
  - `python -m compileall -q scripts/` -> clean; every script still imports.

### Known debt (reported, not addressed)

- `src/engine.py` is **5017 lines** and `src/reporting.py` is 2262; there is no
  module-size gate in this repo to hold that line. Splitting them is a real
  refactor of live execution code and is not something to do incidentally.
- PR #3 is stale: both of its fixes (the sequencer tie-breaker test and the
  per-trade parity entry-cost attribution) are already present on `main` via
  #4/#5. Verified by running the named test and reading `scripts/parity_replay.py`.
