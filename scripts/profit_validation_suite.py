"""
Profit validation suite.

Runs end-to-end backtests across toy market scenarios (UP, DOWN, CHOP, GAP_SHOCK),
ORACLE modes, ES caps, and cost-model toggles.  Each run is validated against:

1. `engine_core.src.reporting.validate_metrics` hard gates.
2. Independent replay of `fills.csv` via `scripts.parity_replay`.
3. Hand-computed round-trip net PnL for single-trade runs.
4. Cost invariants (fees/slippage are zero when `cost_model.enabled=False`).

Exit code is 0 only if every check passes.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine_core.config.params_loader import ParamsLoader
from engine_core.src.data.loader import DataLoader
from engine_core.src.engine import BacktestEngine
from engine_core.src.reporting import validate_metrics
from engine_core.tests.fixtures.toy_markets import generate_toy_market


SCENARIOS = {
    "UP": {"drift": +0.001, "seed": 1},
    "DOWN": {"drift": -0.001, "seed": 2},
    "CHOP": {"drift": 0.0, "seed": 3},
    "GAP_SHOCK": {"drift": 0.0005, "seed": 4},
}


def replay_fills(fills_path: Path, initial_capital: float = 100000.0) -> Dict:
    """Lightweight replay of fills.csv returning total_pnl/final_equity/fee/slippage."""
    from scripts.parity_replay import replay_pnl
    return replay_pnl(fills_path, initial_capital=initial_capital)


def hand_compute_net_pnl(fills_path: Path) -> float:
    """Recompute round-trip net PnL from fills.csv for a single position."""
    fills = pd.read_csv(fills_path)
    entry = fills[fills["leg"] == "ENTRY"].iloc[0]
    exit_ = fills[fills["leg"] == "EXIT"].iloc[0]
    qty = entry["qty"]
    side = "LONG" if entry["side"] == "BUY" else "SHORT"
    entry_px = entry["price"]
    exit_px = exit_["price"]
    fees = entry["fee_usd"] + exit_["fee_usd"]
    slippage = entry["slippage_cost_usd"] + exit_["slippage_cost_usd"]
    if side == "LONG":
        gross = (exit_px - entry_px) * qty
    else:
        gross = (entry_px - exit_px) * qty
    return gross - fees - slippage


def run_case(
    scenario: str,
    run_name: str,
    data_dir: Path,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    overrides: Dict,
) -> Tuple[bool, Dict]:
    """Run one validation case. Returns (pass, summary_row)."""
    output_dir = Path("runs/profit_validation_suite") / scenario / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dl = DataLoader(str(data_dir))
    dl.load_symbol("BTCUSDT", require_liquidity=False)

    params = ParamsLoader(overrides=overrides, strict=False)
    engine = BacktestEngine(dl, params, require_liquidity_data=False)
    engine.run(start_ts=start_ts, end_ts=end_ts, output_dir=str(output_dir))

    artifacts_dir = output_dir / "artifacts"
    metrics = json.load(open(artifacts_dir / "metrics.json"))

    # 1. validate_metrics
    val = validate_metrics(artifacts_dir, strict_canonical=False)
    if not val.passed:
        return False, {
            "scenario": scenario,
            "run": run_name,
            "passed": False,
            "reason": "validate_metrics: " + "; ".join(val.failures),
        }

    # 2. parity replay
    replay = replay_fills(artifacts_dir / "fills.csv", initial_capital=metrics["initial_equity"])
    pnl_tol = 1e-6 + abs(metrics["initial_equity"]) * 1e-7
    if abs(replay["total_pnl"] - metrics["realized_pnl_from_trades"]) > pnl_tol:
        return False, {
            "scenario": scenario,
            "run": run_name,
            "passed": False,
            "reason": f"replay pnl {replay['total_pnl']} != metrics {metrics['realized_pnl_from_trades']}",
        }
    if abs(replay["final_equity"] - metrics["final_equity"]) > pnl_tol:
        return False, {
            "scenario": scenario,
            "run": run_name,
            "passed": False,
            "reason": f"replay final_equity {replay['final_equity']} != metrics {metrics['final_equity']}",
        }

    # 3. single-trade hand calculation
    fills = pd.read_csv(artifacts_dir / "fills.csv")
    if len(fills[fills["leg"] == "ENTRY"]) == 1 and metrics.get("total_trades", 0) == 1:
        hand_pnl = hand_compute_net_pnl(artifacts_dir / "fills.csv")
        if abs(hand_pnl - metrics["realized_pnl_from_trades"]) > 1e-6:
            return False, {
                "scenario": scenario,
                "run": run_name,
                "passed": False,
                "reason": f"hand pnl {hand_pnl} != metrics {metrics['realized_pnl_from_trades']}",
            }

    # 4. cost model toggle invariant
    cost_enabled = overrides.get("cost_model", {}).get("enabled", True)
    if not cost_enabled:
        if metrics["total_fees"] != 0.0 or metrics["total_slippage_cost"] != 0.0:
            return False, {
                "scenario": scenario,
                "run": run_name,
                "passed": False,
                "reason": "cost_model disabled but fees/slippage non-zero",
            }
    else:
        if metrics["total_fees"] <= 0.0 or metrics["total_slippage_cost"] <= 0.0:
            # Edge: flat/no-trade runs have zero fees/slippage; only fail if there are trades.
            if metrics.get("total_trades", 0) > 0:
                return False, {
                    "scenario": scenario,
                    "run": run_name,
                    "passed": False,
                    "reason": "cost_model enabled but fees/slippage are not positive with trades",
                }

    row = {
        "scenario": scenario,
        "run": run_name,
        "passed": True,
        "reason": "",
        "initial_equity": metrics["initial_equity"],
        "final_equity": metrics["final_equity"],
        "total_pnl": metrics["realized_pnl_from_trades"],
        "num_trades": metrics.get("total_trades", 0),
        "fees": metrics["total_fees"],
        "slippage": metrics["total_slippage_cost"],
        "funding": metrics.get("funding_cost_total", 0.0),
    }
    return True, row


def main():
    parser = argparse.ArgumentParser(description="Profit validation suite")
    parser.add_argument("--output-dir", type=str, default="runs/profit_validation_suite")
    parser.add_argument("--num-bars", type=int, default=96)
    args = parser.parse_args()

    root_out = Path(args.output_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    start_ts = pd.Timestamp("2021-01-01 00:00:00", tz="UTC")
    end_ts = start_ts + pd.Timedelta(minutes=15 * (args.num_bars - 1))

    data_dirs: Dict[str, Path] = {}
    for scenario, info in SCENARIOS.items():
        data_dir = root_out / "data" / scenario
        data_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(info["seed"])
        df = generate_toy_market(
            scenario,
            start_ts,
            num_bars=args.num_bars,
            base_price=50000.0,
        )
        df.to_csv(data_dir / "BTCUSDT_15m.csv", index=False)
        # Empty funding file for completeness
        pd.DataFrame(
            {
                "funding_ts": pd.Series([], dtype="datetime64[ns, UTC]"),
                "funding_rate": pd.Series([], dtype=float),
            }
        ).to_csv(data_dir / "BTCUSDT_funding.csv", index=False)
        data_dirs[scenario] = data_dir

    cases: List[Tuple[str, str, Dict]] = []
    for scenario in SCENARIOS:
        cases.extend(
            [
                (
                    scenario,
                    "always_long_costs_on",
                    {
                        "general": {"oracle_mode": "always_long"},
                        "es_guardrails": {"es_cap_of_equity": 1.0},
                        "cost_model": {"enabled": True},
                    },
                ),
                (
                    scenario,
                    "always_long_costs_off",
                    {
                        "general": {"oracle_mode": "always_long"},
                        "es_guardrails": {"es_cap_of_equity": 1.0},
                        "cost_model": {"enabled": False},
                    },
                ),
                (
                    scenario,
                    "always_long_default_es",
                    {
                        "general": {"oracle_mode": "always_long"},
                        "cost_model": {"enabled": True},
                    },
                ),
            ]
        )
    # One random run per suite
    cases.append(
        (
            "CHOP",
            "random_costs_on",
            {
                "general": {"oracle_mode": "random", "oracle_random_seed": 42},
                "es_guardrails": {"es_cap_of_equity": 1.0},
                "cost_model": {"enabled": True},
            },
        )
    )

    results: List[Dict] = []
    failures: List[str] = []
    for scenario, run_name, overrides in cases:
        passed, row = run_case(
            scenario, run_name, data_dirs[scenario], start_ts, end_ts, overrides
        )
        results.append(row)
        if not passed:
            failures.append(f"{scenario}/{run_name}: {row['reason']}")

    df = pd.DataFrame(results)
    summary_path = root_out / "profit_validation_summary.csv"
    df.to_csv(summary_path, index=False)

    print(f"\nSummary written to {summary_path}\n")
    print(df.to_string(index=False))

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("\nAll profit validation checks passed.")


if __name__ == "__main__":
    main()
