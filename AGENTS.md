# AGENTS.md — Agent Directives

## Anti-circular debugging and hygiene directives

These directives are mandatory for all agents working in this repository. They replace previous ad-hoc orchestration tooling and prevent circular debugging.

- Stop on a blocker; do not re-run the same failing command more than twice without a new hypothesis.
- Run the repo's local lint/test/build gates before every commit (e.g., `make ci`, `npm test`, `pytest`, `npm run build`).
- No bare `print()` / `console.log()` / `printf()` in gate scripts; use structured logging.
- Generated files must be deterministic (no timestamps, strip trailing whitespace).
- Do not modify tests to pass; fix the code or skip with a reason.
- Use feature branches and PRs; never `git reset --hard` on the default branch.
- No session/chat artifacts in commits (`.claude/*.json`, `.vscode/`, `.playwright-mcp/`, scratch files).
- Document root cause, not just the fix.

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

### Known debt (reported, not addressed)

- `scripts/*.py` contain **138 bare `print()` calls**, contradicting the
  directive above. Converting them changes the human-readable output these
  validation tools produce, so it needs an explicit decision on the target
  format rather than a mechanical rewrite.
- `src/engine.py` is **5017 lines** and `src/reporting.py` is 2262; there is no
  module-size gate in this repo to hold that line.
- PR #3 is stale: both of its fixes (the sequencer tie-breaker test and the
  per-trade parity entry-cost attribution) are already present on `main` via
  #4/#5. Verified by running the named test and reading `scripts/parity_replay.py`.
