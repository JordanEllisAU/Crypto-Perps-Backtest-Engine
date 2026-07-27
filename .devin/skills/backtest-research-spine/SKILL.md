---
id: backtest-research-spine
name: Backtest Research Spine
description: |
  Evidence-first research workflow for the Crypto-Perps-Backtest-Engine.
  Ensures data-integrity and accounting-invariant checks run before any
  new strategy, signal, or microstructure research is wired into the engine.
tier: standard
scope: workspace
---

# Backtest Research Spine

**Scope:** Crypto-Perps-Backtest-Engine. Use for any research task that could change signals, parameters, risk thresholds, or data sources.

**Finish-first directive:** Research ends when the idea is either (a) rejected by data or invariants, (b) validated with a persisted research artifact, or (c) promoted to a tested, `make ci` passing code change.

## When to invoke

- Adding or changing a signal module (`src/modules/` or engine integration).
- Changing risk guardrails (`src/risk/`), fill models (`src/execution/`), or parameters (`config/`).
- Researching market microstructure edges (ATR, BB width, OI, CVD, TFI, funding, basis).
- Comparing the engine output to another backtester or live broker data.

## Pre-research checklist

Before starting new research, confirm the current engine is healthy:

1. `make ci` passes locally (`compileall`, `slop_sentinel`, `pytest`).
2. `python scripts/validate_data_integrity.py --data-path <path>` passes on the dataset you will use.
3. `python scripts/run_baselines.py --data-path <path> --start-date ... --end-date ...` produces stable baseline numbers.
4. Accounting invariants are checked: `equity = cash + unrealized_pnl`; position conservation; fee/funding sign.

If any of these fail, fix data/bugs before researching new edges.

## External source selector

| Question | First-pass tool | Fallback |
|----------|-----------------|----------|
| Prior findings on this signal, invariant, or parameter | `mcp_tool` server=`memory` `search_nodes` + `open_nodes` | `Crypto-Perps-Backtest-Engine/docs/` and `DeceptionLeaderBot/docs/research/` |
| `pandas` / `numpy` / `pytest` API | `mcp_tool` server=`context7` | Official docs + `web_search` |
| `freqtrade` architecture / config | `mcp_tool` server=`deepwiki` repo=`freqtrade/freqtrade` | `freqtrade.io` docs |
| `ccxt` data conventions, fees, contract sizes | `mcp_tool` server=`deepwiki` repo=`ccxt/ccxt` | CCXT docs |
| Market microstructure / crypto perp research | `mcp_tool` server=`parallel` or `web_search` | `web_get_contents` on top 5 URLs |
| Existing internal findings | `Crypto-Perps-Backtest-Engine/docs/` and `DeceptionLeaderBot/docs/research/` | N/A |

## Research procedure

1. State the hypothesis in falsifiable form (e.g., "High quote-OI coins have lower 60m forward range").
2. Search memory (`mcp_tool` server=`memory` `search_nodes`) for prior findings on the same signal, invariant, or parameter and open any relevant entities.
3. Lock the dataset: record path, date range, symbols, bar interval.
4. Run `validate_data_integrity.py` and paste the PASS/FAIL line.
5. Compute the signal on historical bars using a standalone script (not inside the live engine).
6. Run a forward-test with `metric_edge_forward_test.py` logic or equivalent:
   - Spearman correlation vs forward return / range.
   - Top-vs-bottom decile edge.
   - Signed-return predictability.
7. If the edge is stable across at least 3 independent trials, document it in `docs/research/`.
8. If the edge is to be wired into the engine, add a shadow flag first and run `make ci`.

## Research artifact format

Every new research artifact under `docs/research/` must include:

- Hypothesis
- Dataset snapshot
- Data-integrity result
- Methodology
- Results with confidence metrics
- `source_corpus` table (see `mcp-research-toolkit` skill)
- `recommendation` (adopt / reject / more data needed)

## Integration back to engine

- New signal modules go in `src/modules/` and must include oracle-mode tests.
- New parameters must be added to `config/base_params.json` with defaults that preserve existing behavior.
- Risk changes must be validated against `tests/test_accounting_invariants_toy.py` and `tests/test_risk_controls.py`.
- Any parameter that changes accounting must be behind a feature flag until it passes cross-engine parity.

## Batched execution

- `make ci`, `validate_data_integrity.py`, and baseline runs are sequential gates.
- External source queries (Context7, DeepWiki, web) for a research question can run in parallel.

## Cross-repo links

- `DeceptionLeaderBot/.devin/skills/research/mcp-research-toolkit/SKILL.md` for MCP usage.
- `DeceptionLeaderBot/.devin/skills/reasoning/reasoning-log/SKILL.md` for high-risk decision logs.
- `DeceptionLeaderBot/docs/research/MCP_TOOLING_INDEX_2026-07-27.md` for available MCP servers.
