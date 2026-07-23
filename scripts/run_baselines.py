"""
Baseline Benchmark Runner
Runs simple baseline strategies (Buy & Hold, Flat, Random) through the full engine using ORACLE module.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone

# Add engine_core to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine_core.config.params_loader import ParamsLoader
from engine_core.src.engine import BacktestEngine
from engine_core.src.data.loader import DataLoader


def run_buy_and_hold(data_loader: DataLoader, start_ts: pd.Timestamp, end_ts: pd.Timestamp, output_dir: Path):
    """Run Buy & Hold strategy using ORACLE module (always_long, enters on first bar, exits on last bar)"""
    import sys
    print("    Creating params...", file=sys.stderr, flush=True)
    params = ParamsLoader(overrides={
        'general': {'oracle_mode': 'always_long'},
        'es_guardrails': {'es_cap_of_equity': 1.0},
        'risk': {'max_positions': {'default': 10}},
        'slippage_costs': {'participation_cap_normal': 0.05, 'participation_cap_thin': 0.01}
    }, strict=False)
    
    print("    Creating BacktestEngine...", file=sys.stderr, flush=True)
    engine = BacktestEngine(data_loader, params, require_liquidity_data=False)
    
    print(f"    Running engine.run({start_ts} to {end_ts})...", file=sys.stderr, flush=True)
    engine.run(start_ts=start_ts, end_ts=end_ts, output_dir=str(output_dir / "buy_and_hold"))
    print("    Engine.run() completed.", file=sys.stderr, flush=True)
    
    return engine


def run_flat(data_loader: DataLoader, start_ts: pd.Timestamp, end_ts: pd.Timestamp, output_dir: Path):
    """Run Flat strategy using ORACLE module (flat mode, no signals)"""
    params = ParamsLoader(overrides={
        'general': {'oracle_mode': 'flat'}
    }, strict=False)
    
    engine = BacktestEngine(data_loader, params, require_liquidity_data=False)
    engine.run(start_ts=start_ts, end_ts=end_ts, output_dir=str(output_dir / "flat"))
    
    return engine


def run_random(data_loader: DataLoader, start_ts: pd.Timestamp, end_ts: pd.Timestamp, output_dir: Path, seed: int = 42):
    """Run Random strategy using ORACLE module (random mode with seeded random entries/exits)"""
    params = ParamsLoader(overrides={
        'general': {'oracle_mode': 'random', 'oracle_random_seed': seed},
        'es_guardrails': {'es_cap_of_equity': 1.0},
        'risk': {'max_positions': {'default': 10}},
        'slippage_costs': {'participation_cap_normal': 0.05, 'participation_cap_thin': 0.01}
    }, strict=False)
    
    engine = BacktestEngine(data_loader, params, require_liquidity_data=False)
    engine.run(start_ts=start_ts, end_ts=end_ts, output_dir=str(output_dir / "random"))
    
    return engine


def calculate_naive_return(data_loader: DataLoader, symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> float:
    """Calculate naive Buy & Hold return: (last_close / first_close) - 1"""
    df = data_loader.get_15m_bars(symbol)
    if df is None or len(df) == 0:
        return 0.0
    
    # Filter to time range
    mask = (df['ts'] >= start_ts) & (df['ts'] <= end_ts)
    df_filtered = df[mask]
    
    if len(df_filtered) == 0:
        return 0.0
    
    first_close = df_filtered.iloc[0]['close']
    last_close = df_filtered.iloc[-1]['close']
    
    return (last_close / first_close) - 1.0


def generate_summary(baselines: dict, output_dir: Path, data_loader: DataLoader = None, start_ts: pd.Timestamp = None, end_ts: pd.Timestamp = None):
    """Generate baseline summary CSV"""
    summary_data = []
    
    for name, engine in baselines.items():
        if engine is None:
            continue
            
        # Try to load from metrics.json for accurate values
        metrics_path = output_dir / name / "artifacts" / "metrics.json"
        if metrics_path.exists():
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                initial_equity = metrics.get('initial_equity', engine.portfolio.initial_capital)
                final_equity = metrics.get('final_equity', engine.portfolio.equity)
                total_return = metrics.get('total_return', (final_equity / initial_equity) - 1.0)
                num_trades = metrics.get('total_trades', 0)
                total_pnl = metrics.get('realized_pnl_from_trades', 0.0)
                fees_paid = metrics.get('total_fees', 0.0)
                slippage_paid = metrics.get('total_slippage_cost', 0.0)
                funding_paid = metrics.get('funding_cost_total', 0.0)
        else:
            # Fallback to engine values
            initial_equity = engine.portfolio.initial_capital
            final_equity = engine.portfolio.equity
            total_return = (final_equity / initial_equity) - 1.0
            num_trades = len(engine.trades) if hasattr(engine, 'trades') and engine.trades else 0
            total_pnl = engine.portfolio.total_pnl
            fees_paid = engine.portfolio.fees_paid
            slippage_paid = engine.portfolio.slippage_paid
            funding_paid = engine.portfolio.funding_paid
        
        # For single-symbol Buy & Hold, calculate naive return for comparison
        naive_return = None
        if name == 'buy_and_hold' and data_loader and start_ts and end_ts:
            symbols = data_loader.get_symbols()
            if len(symbols) == 1:
                naive_return = calculate_naive_return(data_loader, symbols[0], start_ts, end_ts)
                return_diff_bps = abs(total_return - naive_return) * 10000  # Convert to bps
        
        row = {
            'strategy': name,
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'num_trades': num_trades,
            'total_pnl': total_pnl,
            'fees_paid': fees_paid,
            'slippage_paid': slippage_paid,
            'funding_paid': funding_paid
        }
        
        if naive_return is not None:
            row['naive_return'] = naive_return
            row['naive_return_pct'] = naive_return * 100
            row['return_diff_bps'] = return_diff_bps
        
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    summary_path = output_dir / "baselines_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"Baseline summary saved to {summary_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Run baseline benchmark strategies using ORACLE module')
    parser.add_argument('--data-path', type=str, required=True, help='Path to data directory')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='artifacts/baselines', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for random strategy')
    
    args = parser.parse_args()
    
    # Parse dates - if only date provided, set end_ts to end of day
    start_ts = pd.Timestamp(args.start_date, tz='UTC')
    end_ts = pd.Timestamp(args.end_date, tz='UTC')
    # If end_ts is at midnight (00:00:00), set to end of that day (23:59:59)
    if end_ts.hour == 0 and end_ts.minute == 0 and end_ts.second == 0:
        end_ts = end_ts.replace(hour=23, minute=59, second=59)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_loader = DataLoader(args.data_path)
    
    # DataLoader requires explicit load_symbol() calls to load data
    # Find available symbols from CSV files in data directory
    data_path = Path(args.data_path)
    available_symbols = []
    for csv_file in data_path.glob("*_15m.csv"):
        symbol = csv_file.stem.replace("_15m", "")
        available_symbols.append(symbol)
    
    if not available_symbols:
        print(f"ERROR: No symbol data found in {args.data_path}")
        print("Expected files like: BTCUSDT_15m.csv, ETHUSDT_15m.csv, etc.")
        return
    
    # Load symbols - limit to first 3 for faster testing (can be removed for full run)
    # For baselines, we typically only need BTCUSDT and ETHUSDT
    symbols_to_load = ['BTCUSDT', 'ETHUSDT'] if 'BTCUSDT' in available_symbols else available_symbols[:3]
    print(f"Loading {len(symbols_to_load)} symbols for baseline: {', '.join(symbols_to_load)}")
    loaded_count = 0
    for symbol in symbols_to_load:
        if symbol not in available_symbols:
            continue
        print(f"  Loading {symbol}...", end=' ', flush=True)
        errors = data_loader.load_symbol(symbol, require_liquidity=False)
        if errors:
            print(f"WARNING: {len(errors)} validation errors")
        else:
            print("OK")
            loaded_count += 1
    
    print(f"Loaded {loaded_count}/{len(symbols_to_load)} symbols successfully")
    
    # Check if we have any loaded symbols
    loaded_symbols = data_loader.get_symbols()
    if not loaded_symbols:
        print("ERROR: No symbols were loaded successfully!")
        return
    
    print(f"Available symbols for backtest: {', '.join(loaded_symbols)}")
    print("Running baseline strategies using ORACLE module...")
    print(f"Start: {start_ts}, End: {end_ts}")
    print(f"Output: {output_dir}")
    print()
    
    baselines = {}
    
    # Run Buy & Hold
    print("Running Buy & Hold (oracle_mode='always_long')...")
    print("  [This may take a moment...]", flush=True)
    try:
        baselines['buy_and_hold'] = run_buy_and_hold(data_loader, start_ts, end_ts, output_dir)
        print(f"  [OK] Completed")
        print(f"  Trades: {len(baselines['buy_and_hold'].trades)}")
        print(f"  Final Equity: ${baselines['buy_and_hold'].portfolio.equity:,.2f}")
        print(f"  Total Return: {((baselines['buy_and_hold'].portfolio.equity / baselines['buy_and_hold'].portfolio.initial_capital) - 1) * 100:.2f}%")
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        baselines['buy_and_hold'] = None
    
    print()
    
    # Run Flat
    print("Running Flat (oracle_mode='flat')...")
    try:
        baselines['flat'] = run_flat(data_loader, start_ts, end_ts, output_dir)
        print(f"  Trades: {len(baselines['flat'].trades)}")
        print(f"  Final Equity: ${baselines['flat'].portfolio.equity:,.2f}")
        print(f"  Total PnL: ${baselines['flat'].portfolio.total_pnl:,.2f}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        baselines['flat'] = None
    
    print()
    
    # Run Random
    print(f"Running Random (oracle_mode='random', seed={args.seed})...")
    try:
        baselines['random'] = run_random(data_loader, start_ts, end_ts, output_dir, seed=args.seed)
        print(f"  Trades: {len(baselines['random'].trades)}")
        print(f"  Final Equity: ${baselines['random'].portfolio.equity:,.2f}")
        print(f"  Total Return: {((baselines['random'].portfolio.equity / baselines['random'].portfolio.initial_capital) - 1) * 100:.2f}%")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        baselines['random'] = None
    
    print()
    
    # Generate summary
    print("Generating summary...")
    valid_baselines = {k: v for k, v in baselines.items() if v is not None}
    if valid_baselines:
        summary_df = generate_summary(valid_baselines, output_dir, data_loader, start_ts, end_ts)
        print("\nBaseline Summary:")
        print(summary_df.to_string(index=False))
        
        # Check Buy & Hold return vs naive return
        if 'buy_and_hold' in valid_baselines and 'return_diff_bps' in summary_df.columns:
            buyhold_row = summary_df[summary_df['strategy'] == 'buy_and_hold'].iloc[0]
            if pd.notna(buyhold_row.get('return_diff_bps')):
                diff_bps = buyhold_row['return_diff_bps']
                print(f"\nBuy & Hold vs Naive Return: {diff_bps:.2f} bps difference")
                if diff_bps > 50.0:
                    print(f"  WARNING: Difference exceeds 50 bps tolerance!")
                else:
                    print(f"  ✓ Within 50 bps tolerance")
    else:
        print("No baselines completed successfully")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
