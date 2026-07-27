---
name: quality-sweep
description: |
  Run the local A+ quality sweep — `make ci` twice, deep slop/placeholder/junk/bad-code
  detection, safe auto-fixes, module lattice/index, and line-limit / atomic-structure
  enforcement. Use before any PR or when the user asks for "sweep", "A+ code",
  "clean slop", "line limit", or "atomic modules".
---

# Quality Sweep Skill

**Scope:** Crypto-Perps-Backtest-Engine (and reusable conventions from DeceptionLeaderBot).

**Finish-first directive:** The sweep ends when `make ci` passes twice and
`scripts/quality_sweep.py` reports **0 errors**. Warnings are acceptable unless
`--strict` is used for an A+ pass.

**Purpose:** Catch slop, placeholders, junk, and bad-code patterns the standard
gates do not yet cover, then auto-fix the safe ones and publish a module
lattice/index.

## When to invoke

- User asks for a sweep, cleanup, or A+ quality pass.
- Before opening a PR.
- After deleting CI or changing the local gate setup.
- When a module is suspected to be bloated or a placeholder.

## Sweep procedure

1. **Baseline gate — run `make ci` twice.**
   ```bash
   make ci && make ci
   ```

2. **Run the deep quality sweep.**
   ```bash
   python scripts/quality_sweep.py --ci-runs 2
   ```
   Writes `docs/lattices/QUALITY_LATTICE_<timestamp>.md` and
   `runtime/quality_sweep_report_<timestamp>.json`.

3. **Review the lattice.**
   - `error` = production blocker
   - `warn` = A+ debt
   - `info` = hygiene note
   - `trust_tier` = `A+` / `B` / `C`

4. **Apply safe auto-fixes after reviewing the diff.**
   ```bash
   python scripts/quality_sweep.py --apply --ci-runs 2
   ```

5. **Optional A+ strict pass.**
   ```bash
   python scripts/quality_sweep.py --strict --ci-runs 2
   ```

## Detected rules

See the DeceptionLeaderBot `quality-sweep` skill for the full rule table; the
same rule set applies here, scoped to `src/` and `engine_core/` production
code.

## Money-path / accounting safety

- Do **not** auto-fix `src/` modules that change accounting invariants, margin
  math, fee models, or fill simulation without reviewing the diff and adding a
  regression test.
- Never change leverage, position sizing, margin mode, SL/TP, or fee
  percentages during a sweep.

## Batched execution

- The two `make ci` passes are sequential gates.
- `QUALITY_LATTICE` and report reads can be parallel; lint checks over non-overlapping file sets may run concurrently.

## Verification chain

- `python scripts/quality_sweep.py --ci-runs 2`
- `python scripts/quality_sweep.py --strict`
- `make ci`
