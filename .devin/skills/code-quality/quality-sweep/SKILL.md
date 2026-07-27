---
name: quality-sweep
description: |
  Run the local A+ quality sweep — `make ci` twice, deep slop/placeholder/junk/bad-code
  detection, safe auto-fixes, module lattice/index, and line-limit / atomic-structure
  enforcement. Use before any PR or when the user asks for "sweep", "A+ code",
  "clean slop", "line limit", or "atomic modules".
---

# Quality Sweep Skill

**Scope:** DeceptionLeaderBot and Crypto-Perps-Backtest-Engine.

**Finish-first directive:** The sweep ends when `make ci` passes twice and
`scripts/quality_sweep.py` reports **0 errors**. Warnings are acceptable unless
`--strict` is used for an A+ pass.

**Purpose:** Catch slop, placeholders, junk, and bad-code patterns the standard
gates do not yet cover, then auto-fix the safe ones and publish a module
lattice/index.

## When to invoke

- User asks for a sweep, cleanup, or A+ quality pass.
- Before opening a PR, especially after a big refactor or research spike.
- After deleting CI or changing the local gate setup.
- When a module is suspected to be bloated or a placeholder.

## Sweep procedure

1. **Baseline gate — run `make ci` twice.**
   ```bash
   make ci && make ci
   ```
   Both passes must be green before deeper scanning.

2. **Run the deep quality sweep.**
   ```bash
   python scripts/quality_sweep.py --ci-runs 2
   ```
   This runs `make ci` twice internally, scans all Python files, and writes:
   - `docs/lattices/QUALITY_LATTICE.md`
   - `runtime/quality_sweep_report.json`

3. **Review the lattice.**
   - `error` issues are production blockers (e.g., `print()` outside the CLI,
     bare `except`, silent exception swallows).
   - `warn` issues are A+ debt (e.g., `Any`, long functions, mutable defaults,
     `TODO` comments).
   - `info` issues are hygiene notes (e.g., unused imports, `type: ignore`).
   - `trust_tier` is `A+` (clean), `B` (warnings), or `C` (errors / >500 lines /
     >50-line function).

4. **Apply safe auto-fixes (review first).**
   ```bash
   python scripts/quality_sweep.py --apply --ci-runs 2
   ```
   Safe fixes include: whitespace/EOF newline and bare `except:` -> `except Exception:`.
   By default, bare-except fixes are skipped in production modules (`core/`,
   `exchanges/`, `trading/`, `scoring/`, `live/`, `src/`); use `--aggressive` to
   override after reviewing. Unused imports are reported but never auto-removed.
   The script re-scans after fixing.

5. **Address remaining errors manually.**
   - Replace `print()` in production modules with `structlog` or `logging`.
   - Replace bare/silent `except` blocks with targeted catches + logging.
   - Refactor functions >50 source lines toward atomic helpers.
   - Split modules approaching 500 source lines.

6. **Optional A+ strict pass.**
   ```bash
   python scripts/quality_sweep.py --strict --ci-runs 2
   ```
   Fails on warnings too. Use this when the user explicitly asks for A+.

## What the sweep detects

| Rule | Severity default | Meaning |
|------|-----------------|---------|
| `PRINT` | error in prod | `print()` outside interactive CLI |
| `BARE_EXCEPT` | error | bare `except:` |
| `SILENT_EXCEPT` | warn (error with `--strict`) | `except ...: pass` or empty swallow |
| `NOT_IMPLEMENTED` | warn | `raise NotImplementedError` |
| `PLACEHOLDER` | warn | empty function/class body (`pass` / `...`) |
| `TODO_COMMENT` | info | `TODO`/`FIXME`/`XXX`/`HACK` comment |
| `DANGEROUS_BUILTIN` | warn | `eval()` / `exec()` / `compile()` |
| `REFLECTION` | warn | `getattr()` / `setattr()` / `hasattr()` |
| `ANY_TYPE` | warn | `typing.Any` usage |
| `TYPE_IGNORE` | info | unqualified `# type: ignore` |
| `WILDCARD_IMPORT` | warn | `from x import *` |
| `DEBUGGER` | warn | `breakpoint()` / `pdb` |
| `GLOBAL_IN_FUNC` | warn | `global` inside a function |
| `MUTABLE_DEFAULT` | warn | mutable default argument |
| `UNUSED_IMPORT` | info | unused import |
| `LONG_FUNCTION` | warn | function >50 source lines |

## Money-path safety

- Do **not** auto-fix modules under `core/`, `exchanges/`, `trading/`,
  `scoring/`, `live/` (Deception) or `src/` (Crypto) without reviewing the diff.
- Never change leverage, position sizing, margin mode, SL/TP, or order params
  during a sweep. This skill targets code quality, not trading logic.

## Verification chain

- `python scripts/quality_sweep.py --ci-runs 2`
- `python scripts/quality_sweep.py --strict` (for A+ pass)
- `make ci` (after fixes)

## Output artifacts

- `docs/lattices/QUALITY_LATTICE.md` — human-readable module lattice.
- `runtime/quality_sweep_report.json` — machine-readable findings.
