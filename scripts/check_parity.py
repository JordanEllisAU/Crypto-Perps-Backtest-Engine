"""
Check parity between engine metrics and replay metrics.
"""
import argparse
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows console encoding safety
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        # Python < 3.7 or encoding not available, use ASCII fallback
        pass


def check_parity(metrics_path: Path, replay_path: Path, output_path: Path):
    """Compare engine metrics with replay metrics"""
    # Load engine metrics
    with open(metrics_path, 'r') as f:
        engine_metrics = json.load(f)
    
    # Load replay results
    with open(replay_path, 'r') as f:
        replay_results = json.load(f)
    
    # Compare key metrics with strict tolerances (bps-level, not %-level)
    comparisons = []
    
    # Trade count: must be identical (0 diff)
    engine_trades = engine_metrics.get('total_trades', engine_metrics.get('num_trades', 0))
    replay_trades = replay_results.get('num_trades', 0)
    trade_diff = abs(engine_trades - replay_trades)
    comparisons.append({
        'metric': 'num_trades',
        'engine': engine_trades,
        'replay': replay_trades,
        'diff': trade_diff,
        'pass': trade_diff == 0,
        'tolerance': '0 (identical)'
    })
    
    # Total PnL: < 5-10 bps equity (0.05-0.1% of equity)
    # Try multiple PnL fields from engine metrics (prefer realized_pnl_from_trades)
    engine_pnl = engine_metrics.get('realized_pnl_from_trades', 
                                     engine_metrics.get('pnl_net_usd', 
                                                       engine_metrics.get('total_pnl', 0.0)))
    replay_pnl = replay_results.get('total_pnl', 0.0)
    pnl_diff = abs(engine_pnl - replay_pnl)
    engine_equity = engine_metrics.get('final_equity', engine_metrics.get('initial_equity', 100000.0))
    pnl_diff_bps_equity = (pnl_diff / engine_equity) * 10000 if engine_equity > 0 else 0.0
    comparisons.append({
        'metric': 'total_pnl',
        'engine': engine_pnl,
        'replay': replay_pnl,
        'diff': pnl_diff,
        'diff_bps_equity': pnl_diff_bps_equity,
        'pass': pnl_diff_bps_equity < 10.0,  # < 10 bps equity
        'tolerance': '< 10 bps equity (0.1%)'
    })
    
    # Final equity: < 5-10 bps equity
    engine_equity = engine_metrics.get('final_equity', 0.0)
    replay_equity = replay_results.get('final_equity', 0.0)
    equity_diff = abs(engine_equity - replay_equity)
    equity_diff_bps = (equity_diff / engine_equity) * 10000 if engine_equity > 0 else 0.0
    comparisons.append({
        'metric': 'final_equity',
        'engine': engine_equity,
        'replay': replay_equity,
        'diff': equity_diff,
        'diff_bps': equity_diff_bps,
        'pass': equity_diff_bps < 10.0,  # < 10 bps
        'tolerance': '< 10 bps (0.1%)'
    })
    
    # Total fees: < $0.01 absolute
    engine_fees = engine_metrics.get('total_fees', 0.0)
    replay_fees = replay_results.get('total_fees', 0.0)
    fees_diff = abs(engine_fees - replay_fees)
    comparisons.append({
        'metric': 'total_fees',
        'engine': engine_fees,
        'replay': replay_fees,
        'diff': fees_diff,
        'pass': fees_diff < 0.01,  # < $0.01
        'tolerance': '< $0.01 absolute'
    })
    
    # Total slippage: < $0.01 absolute
    engine_slippage = engine_metrics.get('total_slippage_cost', 0.0)
    replay_slippage = replay_results.get('total_slippage', 0.0)
    slippage_diff = abs(engine_slippage - replay_slippage)
    comparisons.append({
        'metric': 'total_slippage',
        'engine': engine_slippage,
        'replay': replay_slippage,
        'diff': slippage_diff,
        'pass': slippage_diff < 0.01,  # < $0.01
        'tolerance': '< $0.01 absolute'
    })
    
    # Per-trade PnL diff: < 1-2 bps notional (check if trades available)
    if 'trades' in replay_results and len(replay_results['trades']) > 0:
        # Load trades from engine if available
        trades_path = Path(metrics_path).parent / 'trades.csv'
        if trades_path.exists():
            engine_trades_df = pd.read_csv(trades_path)
            replay_trades_list = replay_results['trades']
            # Match by position_id to avoid order-dependence when multiple trades
            # close at the same timestamp.
            replay_by_id = {t.get('position_id', ''): t for t in replay_trades_list}
            engine_by_id = {r.get('position_id', ''): r for _, r in engine_trades_df.iterrows()}
            common_ids = set(replay_by_id.keys()) & set(engine_by_id.keys())

            if common_ids:
                max_per_trade_diff_bps = 0.0
                for pid in common_ids:
                    engine_trade = engine_by_id[pid]
                    replay_trade = replay_by_id[pid]
                    pnl_diff = abs(float(engine_trade.get('pnl_net_usd', 0.0)) - float(replay_trade.get('pnl', 0.0)))
                    notional = abs(float(engine_trade.get('notional_entry_usd', 0.0)) + float(engine_trade.get('notional_exit_usd', 0.0)))
                    if notional > 0:
                        diff_bps = (pnl_diff / notional) * 10000
                        max_per_trade_diff_bps = max(max_per_trade_diff_bps, diff_bps)

                comparisons.append({
                    'metric': 'per_trade_pnl_max_diff',
                    'max_diff_bps_notional': max_per_trade_diff_bps,
                    'pass': max_per_trade_diff_bps < 2.0,  # < 2 bps notional
                    'tolerance': '< 2 bps notional (0.02%)'
                })
    
    # Generate report
    all_pass = all(c['pass'] for c in comparisons)
    
    report = {
        'parity_check': 'PASS' if all_pass else 'FAIL',
        'comparisons': comparisons
    }
    
    # Save report (convert numpy types to native Python types for JSON)
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    report_serializable = convert_to_native(report)
    with open(output_path, 'w') as f:
        json.dump(report_serializable, f, indent=2)
    
    # Print summary
    print("Parity Check Results:")
    print(f"Overall: {report['parity_check']}")
    print()
    for comp in comparisons:
        status = "[PASS]" if comp['pass'] else "[FAIL]"
        print(f"{status} {comp['metric']}:")
        if 'engine' in comp and 'replay' in comp:
            print(f"  Engine: {comp['engine']}")
            print(f"  Replay: {comp['replay']}")
            print(f"  Diff: {comp.get('diff', 0):.2f}")
        if 'diff_bps' in comp:
            print(f"  Diff: {comp['diff_bps']:.2f} bps")
        if 'diff_bps_equity' in comp:
            print(f"  Diff: {comp['diff_bps_equity']:.2f} bps equity")
        if 'max_diff_bps_notional' in comp:
            print(f"  Max per-trade diff: {comp['max_diff_bps_notional']:.2f} bps notional")
        if 'tolerance' in comp:
            print(f"  Tolerance: {comp['tolerance']}")
        print()
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Check parity between engine and replay')
    parser.add_argument('metrics_path', type=str, help='Path to engine metrics.json')
    parser.add_argument('replay_path', type=str, help='Path to replay results JSON')
    parser.add_argument('--output', type=str, default='parity_report.json', help='Output report path')
    
    args = parser.parse_args()
    
    report = check_parity(args.metrics_path, args.replay_path, args.output)
    
    # Exit with error code if parity fails
    if report['parity_check'] == 'FAIL':
        sys.exit(1)


if __name__ == '__main__':
    main()

