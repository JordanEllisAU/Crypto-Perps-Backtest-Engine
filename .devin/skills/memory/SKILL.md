---
id: memory
name: Memory MCP
description: |
  Recall prior backtest baselines, invariants, and research findings and
  persist new verified results to the knowledge-graph memory MCP.
tier: standard
scope: workspace
---

# Memory MCP — Crypto-Perps-Backtest-Engine

**Scope:** Crypto-Perps-Backtest-Engine.

Use this skill before and after any research or engine change that touches:
- Signals (`src/modules/`)
- Risk guardrails (`src/risk/`)
- Fill models / execution (`src/execution/`)
- Base parameters (`config/`)
- Accounting invariants or cross-engine parity

## Memory-first workflow

1. **RECALL** — `mcp_tool` server=`memory` `tool_name=search_nodes`
   - Query for the signal name, parameter, risk rule, or invariant you are about to touch.
   - Example queries: `quote-OI edge`, `margin guard TRIM`, `cross-engine parity`, `funding window`.
2. **LOAD** — `mcp_tool` server=`memory` `tool_name=open_nodes`
   - Open the returned entities and read their observations.
3. **CONFLICT CHECK** — If memory contradicts the current code/config/docs, current repo truth wins.
   - Add a correction observation, not a silent overwrite.
4. **PERSIST** — Only after `make ci` or a standalone verification passes:
   - `create_entities` for new signals, invariants, or parity runs.
   - `add_observations` to append results to existing entities.
   - `create_relations` to link a signal to its risk guardrail, dataset, or parity report.

## Entity naming convention

| Domain | Entity name | Observations |
|--------|-------------|--------------|
| Baselines | `BTB_Baselines_<dataset>` | baseline equity, sharpe, max drawdown, run date |
| Invariants | `BTB_Invariants` | accounting rules, position conservation, fee/funding sign |
| Per signal | `BTB_Signal_<name>` | hypothesis, edge metrics, shadow flag status, dataset |
| Per risk rule | `BTB_Risk_<name>` | threshold, trigger history, rationale |
| Parity runs | `BTB_Parity_<run_id>` | cross-engine or broker parity metrics, deltas, status |

## What to persist

- Standalone signal validation results (correlation, decile edge, signed return).
- Baseline numbers and dataset snapshots.
- Accounting invariant checks and any exceptions found.
- Cross-engine / broker parity deltas and resolutions.
- Parameter changes that preserve existing behavior (with `config/base_params.json` path).

## What NOT to persist

- Raw market data or full backtest output files.
- Unverified edge hypotheses.
- Broker secrets, API keys, or live account details.

## Sample tool calls

### Search memory
```json
{
  "tool": "mcp_tool",
  "server": "memory",
  "command": "call_tool",
  "tool_name": "search_nodes",
  "tool_args": {
    "query": "quote-OI forward range edge"
  }
}
```

### Add a validated observation
```json
{
  "tool": "mcp_tool",
  "server": "memory",
  "command": "call_tool",
  "tool_name": "add_observations",
  "tool_args": {
    "observations": [
      {
        "entityName": "BTB_Signal_quote_oi_range",
        "contents": "[2026-07-27] Spearman 0.12 p<0.01 on 60m forward range across 90 days; stable in 3 trials. source: docs/research/Quote_OI_Edge_2026-07-27.md"
      }
    ]
  }
}
```

## Fallback when memory MCP is unavailable

Use Devin knowledge notes (`devin_mcp` `devin_knowledge_manage` / `get_knowledge_note` / `suggest_knowledge`) or `DeceptionLeaderBot/docs/research/` cross-references.
