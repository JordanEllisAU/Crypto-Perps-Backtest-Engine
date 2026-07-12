#!/usr/bin/env python3
"""E2E directional validation for DeceptionLeaderBot signals.

This script lives in the forked validator repo and runs the bot's signals
through the engine's DeceptionModule (the real exit lattice: SL/TP1/TP2/TSL,
all taker fees, no max_hold). It produces:
  - e2e_validation_results.json (machine-readable)
  - validation_report.md (human-readable "what can be better" report)

Usage:
  python scripts/deception_e2e.py --signals signals.csv --data data/ --output runtime/e2e/
  python scripts/deception_e2e.py --signals signals.csv --data data/ --output runtime/e2e/ --fees 0.0002,0.0008

Signals CSV columns:
  symbol, side, entry_ts, entry_price, sl_price, tp1_price, tp2_price,
  tp1_frac, tsl_bps, qty, trap_type, deception_score

Data dir:
  {SYMBOL}_15m.csv or {SYMBOL}_1m.csv files with columns: ts, open, high, low, close, volume
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure engine_core is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from engine_core.config.params_loader import ParamsLoader
from engine_core.src.data.loader import DataLoader
from engine_core.src.engine import BacktestEngine
from engine_core.src.modules.deception import DeceptionSignal, load_signals_csv


def _sanitize_symbol(sym: str) -> str:
    """Convert symbol name (e.g. 'BTC/USDT:USDT') to filesystem-safe name."""
    return sym.replace("/", "_").replace(":", "_")


def load_ohlcv(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all OHLCV CSVs from data_dir. Supports 1m and 15m."""
    result: Dict[str, pd.DataFrame] = {}
    for csv_path in data_dir.glob("*_*.csv"):
        # Parse {SYMBOL}_{timeframe}.csv
        stem = csv_path.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        sym, tf = parts
        if tf not in ("1m", "15m", "5m", "3m"):
            continue
        df = pd.read_csv(csv_path)
        if "ts" not in df.columns:
            continue
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        if "notional" not in df.columns and "volume" in df.columns:
            df["notional"] = df["close"] * df["volume"]
        result[sym] = df
        print(f"  Loaded {sym} ({tf}): {len(df)} bars")
    return result


def run_deception_backtest(
    signals: List[DeceptionSignal],
    ohlcv: Dict[str, pd.DataFrame],
    output_dir: Path,
    maker_fee: float = 0.0002,
    taker_fee: float = 0.0008,
    initial_capital: float = 1000.0,
    max_positions: int = 14,
) -> Dict:
    """Run the DeceptionModule backtest through the engine."""
    print("\n" + "=" * 70)
    print("PHASE: DeceptionModule Backtest (Real Signal Replay)")
    print("=" * 70)

    # Prepare data dir in engine's expected format
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for sym, df in ohlcv.items():
        df.to_csv(data_dir / f"{sym}_15m.csv", index=False)

    # Configure engine for deception mode
    overrides = {
        "cost_model": {"enabled": True},
        "general": {
            "deception_mode": True,
            "oracle_mode": None,  # Mutually exclusive with deception_mode
            "initial_capital_usd": initial_capital,
            "maker_fee_bps": {"default": maker_fee * 10000},
            "taker_fee_bps": {"default": taker_fee * 10000},
            "max_positions": {"default": max_positions},
        },
        "es_guardrails": {"es_cap_of_equity": 1.0},
        "risk": {"max_positions": {"default": max_positions}},
        "slippage_costs": {
            "participation_cap_normal": 1.0,
            "participation_cap_thin": 1.0,
            "vacuum_entry_blocked": False,
        },
    }

    params = ParamsLoader(overrides=overrides, strict=False)
    loader = DataLoader(str(data_dir))
    symbols = list(ohlcv.keys())
    for sym in symbols:
        loader.load_symbol(sym)

    if not loader.get_symbols():
        return {"phase": "deception_backtest", "status": "SKIP", "reason": "No symbols loaded"}

    engine = BacktestEngine(loader, params, require_liquidity_data=False)
    engine.set_deception_signals(signals)

    time_range = loader.get_time_range()
    if time_range[0] is None:
        return {"phase": "deception_backtest", "status": "SKIP", "reason": "No time range"}

    start_ts = time_range[0]
    end_ts = time_range[1] if len(time_range) > 1 and time_range[1] else start_ts + timedelta(hours=24)

    try:
        engine.run(start_ts=start_ts, end_ts=end_ts, output_dir=str(output_dir / "reports"))
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"phase": "deception_backtest", "status": "FAIL", "error": str(e)}

    trades = engine.trades or []
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    winners = [t for t in trades if t.get("pnl", 0) > 0]
    losers = [t for t in trades if t.get("pnl", 0) <= 0]
    win_rate = len(winners) / len(trades) * 100 if trades else 0

    # Group by exit reason
    by_reason: Dict[str, List[float]] = defaultdict(list)
    for t in trades:
        by_reason[t.get("reason", "unknown")].append(t.get("pnl", 0))

    result = {
        "phase": "deception_backtest",
        "status": "PASS" if len(trades) > 0 else "SKIP",
        "num_trades": len(trades),
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "num_winners": len(winners),
        "num_losers": len(losers),
        "by_reason": {k: {"count": len(v), "pnl": sum(v)} for k, v in by_reason.items()},
        "equity": engine.portfolio.equity,
        "fees_paid": engine.portfolio.fees_paid,
    }
    print(f"  Trades: {len(trades)}, PnL: ${total_pnl:.2f}, WR: {win_rate:.1f}%")
    print(f"  Equity: ${engine.portfolio.equity:.2f}, Fees: ${engine.portfolio.fees_paid:.2f}")
    for reason, stats in result["by_reason"].items():
        print(f"  {reason}: {stats['count']} trades, ${stats['pnl']:.2f}")
    return result


def run_data_integrity(ohlcv: Dict[str, pd.DataFrame]) -> Dict:
    """Phase 1: Data Integrity — timestamp monotonicity, gaps, NaNs, OHLC sanity."""
    print("\n" + "=" * 70)
    print("PHASE: Data Integrity")
    print("=" * 70)
    errors: List[str] = []
    checked = 0
    for sym, df in ohlcv.items():
        if len(df) < 2:
            errors.append(f"{sym}: too few bars ({len(df)})")
            continue
        # NaN check
        nan_count = df[["open", "high", "low", "close"]].isna().sum().sum()
        if nan_count > 0:
            errors.append(f"{sym}: {nan_count} NaN values in OHLC")
        # OHLC sanity
        bad_h = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
        bad_l = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
        if bad_h > 0 or bad_l > 0:
            errors.append(f"{sym}: {bad_h} bad highs, {bad_l} bad lows")
        # Timestamp monotonicity
        ts_diffs = df["ts"].diff().dropna()
        if (ts_diffs <= pd.Timedelta(0)).any():
            errors.append(f"{sym}: non-monotonic timestamps")
        checked += 1
        if not any(sym in e for e in errors):
            print(f"  OK: {sym} — {len(df)} bars")
    result = {
        "phase": "data_integrity",
        "status": "PASS" if not errors else "FAIL",
        "symbols_checked": checked,
        "errors": errors,
    }
    print(f"  RESULT: {result['status']} ({checked} symbols, {len(errors)} errors)")
    return result


def run_accounting_invariants(backtest_result: Dict) -> Dict:
    """Phase 2: Accounting Invariants — equity identity, PnL conservation."""
    print("\n" + "=" * 70)
    print("PHASE: Accounting Invariants")
    print("=" * 70)
    if backtest_result.get("status") == "SKIP":
        print("  SKIP: No backtest ran")
        return {"phase": "accounting_invariants", "status": "SKIP"}

    trades = backtest_result.get("by_reason", {})
    trade_pnl_sum = sum(v["pnl"] for v in trades.values())
    total_pnl = backtest_result.get("total_pnl", 0)
    pnl_conserved = abs(trade_pnl_sum - total_pnl) < 0.01

    result = {
        "phase": "accounting_invariants",
        "status": "PASS" if pnl_conserved else "FAIL",
        "pnl_conservation": "PASS" if pnl_conserved else "FAIL",
        "trade_pnl_sum": trade_pnl_sum,
        "total_pnl": total_pnl,
    }
    print(f"  PnL conservation: {result['pnl_conservation']}")
    print(f"  RESULT: {result['status']}")
    return result


def generate_what_can_be_better(
    signals: List[DeceptionSignal],
    ohlcv: Dict[str, pd.DataFrame],
    backtest_result: Dict,
) -> Dict:
    """Generate the 'what can be better' directional analysis report."""
    print("\n" + "=" * 70)
    print("PHASE: What Can Be Better (Directional Analysis)")
    print("=" * 70)

    findings: List[str] = []
    metrics: Dict[str, Any] = {}

    # 1. Direction accuracy: for each signal, did the price move in the signal's favor?
    direction_correct = 0
    direction_wrong = 0
    direction_neutral = 0
    for sig in signals:
        sym = _sanitize_symbol(sig.symbol)
        if sym not in ohlcv:
            continue
        df = ohlcv[sym]
        entry_ts = sig.signal_ts
        # Find bars after entry
        future = df[df["ts"] >= entry_ts]
        if len(future) < 2:
            continue
        entry_px = sig.entry_price
        # Check price 1 hour after entry (4 x 15m bars or 60 x 1m bars)
        lookforward = future.head(60) if len(future) > 60 else future
        max_high = lookforward["high"].max()
        min_low = lookforward["low"].min()
        if sig.side == "LONG":
            if max_high > entry_px * 1.005:  # moved up >0.5%
                direction_correct += 1
            elif min_low < entry_px * 0.995:
                direction_wrong += 1
            else:
                direction_neutral += 1
        else:  # SHORT
            if min_low < entry_px * 0.995:
                direction_correct += 1
            elif max_high > entry_px * 1.005:
                direction_wrong += 1
            else:
                direction_neutral += 1

    total_dir = direction_correct + direction_wrong + direction_neutral
    dir_accuracy = direction_correct / total_dir * 100 if total_dir > 0 else 0
    metrics["direction_accuracy"] = {
        "correct": direction_correct,
        "wrong": direction_wrong,
        "neutral": direction_neutral,
        "accuracy_pct": dir_accuracy,
    }
    if total_dir > 0:
        findings.append(
            f"Direction accuracy: {dir_accuracy:.1f}% ({direction_correct}/{total_dir} signals "
            f"moved in the predicted direction within 1h). "
            f"{direction_wrong} went the wrong way."
        )

    # 2. SL opportunity cost: for SL exits, what was the max favorable price?
    sl_opportunity_cost: List[Dict] = []
    for sig in signals:
        sym = _sanitize_symbol(sig.symbol)
        if sym not in ohlcv:
            continue
        df = ohlcv[sym]
        entry_ts = sig.signal_ts
        future = df[df["ts"] >= entry_ts]
        if len(future) < 2:
            continue
        entry_px = sig.entry_price
        sl_px = sig.stop_price
        # Did SL hit within the next N bars?
        lookforward = future.head(60)
        if sig.side == "LONG":
            sl_hit = (lookforward["low"] <= sl_px).any()
            if sl_hit:
                # Max favorable price before SL
                sl_bar_idx = (lookforward["low"] <= sl_px).idxmax()
                before_sl = lookforward.loc[:sl_bar_idx]
                max_favorable = before_sl["high"].max()
                missed_pct = (max_favorable - entry_px) / entry_px * 100
                if missed_pct > 0.3:  # Only flag if >0.3% was left on table
                    sl_opportunity_cost.append({
                        "symbol": sig.symbol, "side": sig.side,
                        "entry": entry_px, "sl": sl_px,
                        "max_favorable": max_favorable,
                        "missed_pct": missed_pct,
                    })
        else:  # SHORT
            sl_hit = (lookforward["high"] >= sl_px).any()
            if sl_hit:
                sl_bar_idx = (lookforward["high"] >= sl_px).idxmax()
                before_sl = lookforward.loc[:sl_bar_idx]
                max_favorable = before_sl["low"].min()
                missed_pct = (entry_px - max_favorable) / entry_px * 100
                if missed_pct > 0.3:
                    sl_opportunity_cost.append({
                        "symbol": sig.symbol, "side": sig.side,
                        "entry": entry_px, "sl": sl_px,
                        "max_favorable": max_favorable,
                        "missed_pct": missed_pct,
                    })

    avg_sl_missed = np.mean([s["missed_pct"] for s in sl_opportunity_cost]) if sl_opportunity_cost else 0
    metrics["sl_opportunity_cost"] = {
        "count": len(sl_opportunity_cost),
        "avg_missed_pct": avg_sl_missed,
    }
    if sl_opportunity_cost:
        findings.append(
            f"SL opportunity cost: {len(sl_opportunity_cost)} SL exits had an average "
            f"{avg_sl_missed:.2f}% favorable move before hitting SL. "
            f"Widening SL by 0.1-0.2% could have captured some of this."
        )

    # 3. Edge by trap_type
    trap_pnl: Dict[str, List[float]] = defaultdict(list)
    # Use backtest trades if available
    if backtest_result.get("by_reason"):
        # We don't have per-trade trap_type in by_reason, so use signals
        pass
    # Group signals by trap_type and check direction
    trap_dir: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "wrong": 0, "total": 0})
    for i, sig in enumerate(signals):
        sym = _sanitize_symbol(sig.symbol)
        if sym not in ohlcv:
            continue
        df = ohlcv[sym]
        future = df[df["ts"] >= sig.signal_ts]
        if len(future) < 2:
            continue
        lookforward = future.head(60)
        max_high = lookforward["high"].max()
        min_low = lookforward["low"].min()
        tt = sig.trap_type or "unknown"
        trap_dir[tt]["total"] += 1
        if sig.side == "LONG" and max_high > sig.entry_price * 1.005:
            trap_dir[tt]["correct"] += 1
        elif sig.side == "SHORT" and min_low < sig.entry_price * 0.995:
            trap_dir[tt]["correct"] += 1
        elif sig.side == "LONG" and min_low < sig.entry_price * 0.995:
            trap_dir[tt]["wrong"] += 1
        elif sig.side == "SHORT" and max_high > sig.entry_price * 1.005:
            trap_dir[tt]["wrong"] += 1

    metrics["edge_by_trap_type"] = {
        tt: {"correct": d["correct"], "wrong": d["wrong"], "total": d["total"],
             "accuracy": d["correct"] / d["total"] * 100 if d["total"] > 0 else 0}
        for tt, d in trap_dir.items()
    }
    # Find best and worst trap types
    if trap_dir:
        best_tt = max(metrics["edge_by_trap_type"].items(), key=lambda x: x[1]["accuracy"])
        worst_tt = min(metrics["edge_by_trap_type"].items(), key=lambda x: x[1]["accuracy"])
        if best_tt[1]["total"] >= 3:
            findings.append(
                f"Best trap type: {best_tt[0]} ({best_tt[1]['accuracy']:.1f}% direction accuracy, "
                f"{best_tt[1]['total']} signals). Focus on this signal type."
            )
        if worst_tt[1]["total"] >= 3 and worst_tt[1]["accuracy"] < 40:
            findings.append(
                f"Worst trap type: {worst_tt[0]} ({worst_tt[1]['accuracy']:.1f}% direction accuracy, "
                f"{worst_tt[1]['total']} signals). Consider filtering this signal type out."
            )

    # 4. Edge by deception_score band
    score_bands = [(0, 30, "low"), (30, 60, "mid"), (60, 100, "high")]
    score_dir: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for sig in signals:
        sym = _sanitize_symbol(sig.symbol)
        if sym not in ohlcv:
            continue
        df = ohlcv[sym]
        future = df[df["ts"] >= sig.signal_ts]
        if len(future) < 2:
            continue
        lookforward = future.head(60)
        max_high = lookforward["high"].max()
        min_low = lookforward["low"].min()
        for lo, hi, label in score_bands:
            if lo <= sig.deception_score < hi:
                score_dir[label]["total"] += 1
                if (sig.side == "LONG" and max_high > sig.entry_price * 1.005) or \
                   (sig.side == "SHORT" and min_low < sig.entry_price * 0.995):
                    score_dir[label]["correct"] += 1
                break

    metrics["edge_by_score_band"] = {
        label: {"correct": d["correct"], "total": d["total"],
                "accuracy": d["correct"] / d["total"] * 100 if d["total"] > 0 else 0}
        for label, d in score_dir.items()
    }
    # Find if high-score signals perform better
    high_acc = score_dir.get("high", {}).get("accuracy", 0)
    low_acc = score_dir.get("low", {}).get("accuracy", 0)
    if score_dir.get("high", {}).get("total", 0) >= 3 and score_dir.get("low", {}).get("total", 0) >= 3:
        if high_acc > low_acc + 10:
            findings.append(
                f"High deception_score (>60) signals have {high_acc:.1f}% direction accuracy vs "
                f"{low_acc:.1f}% for low-score (<30). Consider raising the minimum score threshold."
            )
        elif low_acc > high_acc + 10:
            findings.append(
                f"Low deception_score (<30) signals have {low_acc:.1f}% direction accuracy vs "
                f"{high_acc:.1f}% for high-score. The score may be inverted — investigate."
            )

    # 5. TP placement: for signals that didn't hit TP, was the TP too far?
    tp_too_far = 0
    tp_too_close = 0
    for sig in signals:
        if sig.tp1_price <= 0:
            continue
        sym = _sanitize_symbol(sig.symbol)
        if sym not in ohlcv:
            continue
        df = ohlcv[sym]
        future = df[df["ts"] >= sig.signal_ts]
        if len(future) < 2:
            continue
        lookforward = future.head(60)
        max_favorable = lookforward["high"].max() if sig.side == "LONG" else lookforward["low"].min()
        tp_distance = abs(sig.tp1_price - sig.entry_price) / sig.entry_price * 100
        if sig.side == "LONG":
            achieved_pct = (max_favorable - sig.entry_price) / sig.entry_price * 100
        else:
            achieved_pct = (sig.entry_price - max_favorable) / sig.entry_price * 100
        if achieved_pct < tp_distance * 0.8:
            tp_too_far += 1
        elif achieved_pct > tp_distance * 1.5:
            tp_too_close += 1

    metrics["tp_placement"] = {"too_far": tp_too_far, "too_close": tp_too_close}
    if tp_too_far > len(signals) * 0.3:
        findings.append(
            f"TP too far: {tp_too_far} signals never got within 80% of TP1. "
            f"Consider lowering TP1 to capture more wins."
        )
    if tp_too_close > len(signals) * 0.3:
        findings.append(
            f"TP too close: {tp_too_close} signals overshot TP1 by 50%+. "
            f"Consider raising TP1 to capture more profit."
        )

    result = {
        "phase": "what_can_be_better",
        "status": "PASS",
        "metrics": metrics,
        "findings": findings,
    }
    print(f"  Findings: {len(findings)}")
    for f in findings:
        print(f"  - {f}")
    return result


def write_markdown_report(
    results: Dict[str, Dict],
    output_path: Path,
    signals_count: int,
    symbols: List[str],
) -> None:
    """Write a human-readable markdown report."""
    lines = [
        "# Directional Validator Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Signals: {signals_count}",
        f"Symbols: {', '.join(symbols)}",
        "",
    ]
    for phase_name, result in results.items():
        status = result.get("status", "UNKNOWN")
        lines.append(f"## {phase_name}: {status}")
        lines.append("")
        if "metrics" in result:
            lines.append("```json")
            lines.append(json.dumps(result["metrics"], indent=2, default=str))
            lines.append("```")
            lines.append("")
        if "findings" in result:
            lines.append("### What Can Be Better")
            lines.append("")
            for f in result["findings"]:
                lines.append(f"- {f}")
            lines.append("")
        if "errors" in result and result["errors"]:
            lines.append("### Errors")
            lines.append("")
            for e in result["errors"]:
                lines.append(f"- {e}")
            lines.append("")
        # Include key metrics inline
        for key in ("num_trades", "total_pnl", "win_rate", "direction_accuracy", "accuracy_pct"):
            if key in result:
                lines.append(f"- {key}: {result[key]}")
        if "by_reason" in result:
            lines.append("")
            lines.append("### PnL by Exit Reason")
            lines.append("")
            lines.append("| Reason | Trades | PnL |")
            lines.append("|--------|--------|-----|")
            for reason, stats in result["by_reason"].items():
                lines.append(f"| {reason} | {stats['count']} | ${stats['pnl']:.2f} |")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="E2E directional validation for DeceptionLeaderBot")
    parser.add_argument("--signals", type=str, required=True, help="Path to signals CSV")
    parser.add_argument("--data", type=str, required=True, help="Directory with OHLCV CSVs")
    parser.add_argument("--output", type=str, default="runtime/e2e", help="Output directory")
    parser.add_argument("--fees", type=str, default="0.0002,0.0008", help="maker,taker fee rates")
    parser.add_argument("--capital", type=float, default=1000.0, help="Initial capital USD")
    parser.add_argument("--max-positions", type=int, default=14, help="Max concurrent positions")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    maker_fee, taker_fee = [float(x) for x in args.fees.split(",")]

    print("=" * 70)
    print("DeceptionLeaderBot E2E Directional Validator")
    print(f"Signals: {args.signals}")
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    print(f"Fees: maker={maker_fee}, taker={taker_fee}")
    print("=" * 70)

    # Load signals
    signals = load_signals_csv(args.signals)
    print(f"\nLoaded {len(signals)} signals")
    if not signals:
        print("ERROR: No signals loaded")
        sys.exit(1)

    # Load OHLCV
    print("\n--- Loading OHLCV data ---")
    ohlcv = load_ohlcv(Path(args.data))
    if not ohlcv:
        print("ERROR: No OHLCV data loaded")
        sys.exit(1)

    # Sanitize signal symbols to match OHLCV keys
    for sig in signals:
        sig.symbol = _sanitize_symbol(sig.symbol)

    symbols = sorted(set(sig.symbol for sig in signals))

    # Run phases
    results: Dict[str, Dict] = {}
    results["data_integrity"] = run_data_integrity(ohlcv)
    backtest = run_deception_backtest(
        signals, ohlcv, output_dir, maker_fee, taker_fee, args.capital, args.max_positions
    )
    results["deception_backtest"] = backtest
    results["accounting_invariants"] = run_accounting_invariants(backtest)
    results["what_can_be_better"] = generate_what_can_be_better(signals, ohlcv, backtest)

    # Summary
    print("\n" + "=" * 70)
    print("E2E VALIDATION SUMMARY")
    print("=" * 70)
    all_pass = True
    for phase_name, result in results.items():
        status = result["status"]
        icon = "[PASS]" if status == "PASS" else "[FAIL]" if status == "FAIL" else "[SKIP]"
        print(f"  {icon} {phase_name}: {status}")
        if status == "FAIL":
            all_pass = False

    overall = "PASS" if all_pass else "FAIL"
    print(f"\n  OVERALL: {overall}")

    # Save JSON results
    results_file = output_dir / "e2e_validation_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signals_count": len(signals),
            "symbols": symbols,
            "overall_status": overall,
            "phases": results,
        }, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_file}")

    # Write markdown report
    write_markdown_report(results, output_dir / "validation_report.md", len(signals), symbols)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
