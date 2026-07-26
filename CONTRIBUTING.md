# Contributing to Crypto-Perps-Backtest-Engine

This backtesting engine simulates futures execution and accounting. Every change must preserve the `Cash + Unrealized PnL = Total Equity` invariant and avoid lookahead.

## Branch and PR cadence

- Use one branch per coherent theme. Batch related fixes into one PR.
- Name branches `devin/<timestamp>-<short-desc>`.
- Keep PRs focused; prefer multiple small, testable PRs over one giant refactor.
- Delete merged remote branches promptly.

## Pre-commit verification

```bash
python -m compileall src
python -m pytest -q
```

## Code conventions

- No bare `except:` or silent `except Exception: pass`.
- Use type hints on new public functions.
- Avoid lookahead: signals generated at bar `t` execute at `t+1`.
- Keep risk, execution, and accounting math deterministic and testable.

## Cross-repo shared code

The engine is designed to replay and validate `DeceptionLeaderBot` signals (`DeceptionSignal` / `DeceptionModule`). Reusable trading primitives that overlap with the live bot should be proposed as a shared library instead of duplicated. The signal CSV schema is the canonical contract between the two repos.

## Documentation

- Update `docs/` if a fill model, risk guardrail, or module behavior changes.
- Explain root cause, not just the diff, in the PR body.
