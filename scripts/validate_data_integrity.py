"""
Validate Data Integrity
Checks OHLCV data for timestamps, gaps, NaNs, and sanity.
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Add engine_core to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine_core.src.data.loader import DataLoader

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _gate_log import emit_report, get_logger

log = get_logger("validate_data_integrity")



def validate_symbol(symbol: str, df: pd.DataFrame, interval_min: int = 15) -> dict:
    """
    Run validation checks on a single symbol dataframe.
    """
    issues = {
        "duplicates": 0,
        "gaps": 0,
        "nans": {},
        "ohlc_violations": 0,
        "volume_violations": 0,
        "outliers": 0,
        "price_scale_ok": True,
        "utc_aligned": True
    }
    
    if df.empty:
        return issues

    # 1. Timestamp monotonicity & duplicates
    if not df['ts'].is_monotonic_increasing:
        issues['monotonicity'] = False
        
    duplicates = df.duplicated(subset=['ts']).sum()
    issues['duplicates'] = int(duplicates)
    
    # 2. Interval gap detection
    # Expected delta
    expected_delta = pd.Timedelta(minutes=interval_min)
    deltas = df['ts'].diff().dropna()
    # Gaps > 1.5x expected
    gaps = (deltas > expected_delta * 1.5).sum()
    issues['gaps'] = int(gaps)
    
    # 3. NaN / inf / missing OHLCV
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            nan_count = df[col].isna().sum() + np.isinf(df[col]).sum()
            if nan_count > 0:
                issues['nans'][col] = int(nan_count)
                
    # 4. OHLC sanity
    # low <= min(open, close) <= max(open, close) <= high
    # We check violations
    # low > open OR low > close OR high < open OR high < close
    # Using a small epsilon for float precision?
    # Ideally exact.
    # Violations: low > min(open, close) OR high < max(open, close)
    # Also low <= high
    
    min_oc = df[['open', 'close']].min(axis=1)
    max_oc = df[['open', 'close']].max(axis=1)
    
    ohlc_bad = (
        (df['low'] > min_oc) | 
        (df['high'] < max_oc) | 
        (df['low'] > df['high'])
    )
    issues['ohlc_violations'] = int(ohlc_bad.sum())
    
    # 5. Volume sanity
    vol_bad = (df['volume'] < 0)
    issues['volume_violations'] = int(vol_bad.sum())
    
    # 6. Return outliers (log only)
    # abs(15m return) > 25%
    returns = df['close'].pct_change().dropna()
    outliers = (returns.abs() > 0.25).sum()
    issues['outliers'] = int(outliers)
    
    # 7. Price scale consistency
    median_price = df['close'].median()
    # Check if min price < 1/50th median or max price > 50x median
    if df['close'].min() < median_price / 50 or df['close'].max() > median_price * 50:
        issues['price_scale_ok'] = False
        
    # 8. Timezone & candle boundary alignment
    # Check if ts is UTC
    if df['ts'].dt.tz is None:
        issues['utc_aligned'] = False # Should be tz-aware UTC
    elif str(df['ts'].dt.tz) != 'UTC':
        # Accept +00:00 as UTC
        if str(df['ts'].dt.tz) not in ['UTC', 'datetime.timezone.utc', '+00:00']:
             issues['utc_aligned'] = False
             
    # Check alignment (00, 15, 30, 45)
    minutes = df['ts'].dt.minute
    misaligned = (~minutes.isin([0, 15, 30, 45])).sum()
    if misaligned > 0:
        issues['misaligned_bars'] = int(misaligned)
        
    return issues

def main():
    parser = argparse.ArgumentParser(description="Validate Data Integrity")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--symbols", type=str, default="ALL", help="Comma-separated list of symbols or ALL")
    parser.add_argument("--start-date", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, help="End date YYYY-MM-DD")
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    
    # Determine symbols
    if args.symbols == "ALL":
        # Scan dir for _15m.csv or .parquet
        files = list(data_path.glob("*_15m.csv")) + list(data_path.glob("*_15m.parquet"))
        symbols = sorted(list(set(f.name.split('_')[0] for f in files)))
    else:
        symbols = args.symbols.split(',')
        
    emit_report(f"Validating {len(symbols)} symbols...")
    
    # Dates
    start_ts = pd.Timestamp(args.start_date, tz='UTC') if args.start_date else None
    end_ts = pd.Timestamp(args.end_date, tz='UTC') if args.end_date else None
    
    loader = DataLoader(str(data_path), start_ts=start_ts, end_ts=end_ts)
    
    all_issues = {}
    failed_symbols = []
    
    for symbol in symbols:
        emit_report(f"Checking {symbol}...", end=" ")
        errors = loader.load_symbol(symbol, require_liquidity=False)
        if errors:
            emit_report(f"LOAD ERROR: {errors}")
            failed_symbols.append(symbol)
            continue
            
        df = loader.get_15m_bars(symbol)
        if df is None or df.empty:
            emit_report("EMPTY")
            continue
            
        issues = validate_symbol(symbol, df)
        all_issues[symbol] = issues
        
        # Determine pass/fail for this symbol based on strict criteria
        # Fail if: duplicates > 0, ohlc_violations > 0, nans > 0 (except maybe volume?)
        failed = False
        if issues['duplicates'] > 0: failed = True
        if issues['ohlc_violations'] > 0: failed = True
        if len(issues['nans']) > 0: failed = True
        if not issues.get('utc_aligned', True): failed = True
        
        if failed:
            emit_report("FAIL")
            failed_symbols.append(symbol)
        else:
            emit_report("OK")
            
    # Report generation
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    report_path = artifacts_dir / "data_integrity_report.md"
    with open(report_path, 'w') as f:
        f.write("# Data Integrity Report\n\n")
        f.write(f"Symbols Checked: {len(symbols)}\n")
        f.write(f"Symbols Failed: {len(failed_symbols)}\n\n")
        
        f.write("## Violations\n")
        for sym in failed_symbols:
            if sym in all_issues:
                f.write(f"### {sym}\n")
                f.write(f"```json\n{json.dumps(all_issues[sym], indent=2)}\n```\n")
            else:
                f.write(f"### {sym}\nLoad Error\n")
                
        f.write("\n## Summary Stats\n")
        # Aggregate stats table could go here
        
    # Flags CSV
    flags_path = artifacts_dir / "data_integrity_flags.csv"
    rows = []
    for sym, iss in all_issues.items():
        row = {'symbol': sym}
        # Flatten the 'nans' dict into columns
        nans_dict = iss.get('nans', {})
        for col in ['open', 'high', 'low', 'close', 'volume']:
            row[f'nan_count_{col}'] = nans_dict.get(col, 0)
        # Add other fields (excluding 'nans' dict)
        for key, value in iss.items():
            if key != 'nans':
                row[key] = value
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(flags_path, index=False)
        
    emit_report(f"\nReport saved to {report_path}")
    emit_report(f"Flags CSV saved to {flags_path}")
    
    # Print summary
    total_duplicates = sum(iss.get('duplicates', 0) for iss in all_issues.values())
    total_nans = sum(sum(iss.get('nans', {}).values()) for iss in all_issues.values())
    total_ohlc_violations = sum(iss.get('ohlc_violations', 0) for iss in all_issues.values())
    
    emit_report(f"\nSummary:")
    emit_report(f"  Total duplicates: {total_duplicates}")
    emit_report(f"  Total NaNs: {total_nans}")
    emit_report(f"  Total OHLC violations: {total_ohlc_violations}")
    emit_report(f"  Failed symbols: {len(failed_symbols)}")
    
    if failed_symbols:
        sys.exit(1)

if __name__ == "__main__":
    main()

