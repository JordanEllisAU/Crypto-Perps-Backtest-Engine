"""
Export normalized signals/fills from an experiment for parity replay.
"""
import argparse
import sys
import json
from pathlib import Path
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _gate_log import emit_report, get_logger

log = get_logger("export_signals")



# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def export_signals(artifacts_dir: Path, output_path: Path):
    """Export normalized signals/fills from experiment artifacts"""
    artifacts_dir = Path(artifacts_dir)
    output_path = Path(output_path)
    
    # Load fills (most detailed trade data)
    fills_path = artifacts_dir / "fills.csv"
    if not fills_path.exists():
        raise FileNotFoundError(f"Fills CSV not found: {fills_path}")
    
    fills_df = pd.read_csv(fills_path)
    
    # Normalize fills to a simple format for replay
    signals = []
    for _, fill in fills_df.iterrows():
        signals.append({
            'ts': fill['ts'],
            'symbol': fill['symbol'],
            'side': fill['side'],
            'qty': fill['qty'],
            'price': fill['price'],
            'fee_usd': fill.get('fee_usd', 0.0),
            'slippage_cost_usd': fill.get('slippage_cost_usd', 0.0),
            'leg': fill.get('leg', 'ENTRY'),
            'position_id': fill.get('position_id', ''),
            'module': fill.get('module', '')
        })
    
    signals_df = pd.DataFrame(signals)
    signals_df.to_csv(output_path, index=False)
    emit_report(f"Exported {len(signals_df)} signals to {output_path}")
    
    return signals_df


def main():
    parser = argparse.ArgumentParser(description='Export normalized signals from experiment')
    parser.add_argument('artifacts_dir', type=str, help='Path to experiment artifacts directory')
    parser.add_argument('--output', type=str, default='signals_export.csv', help='Output CSV path')
    
    args = parser.parse_args()
    
    export_signals(args.artifacts_dir, args.output)


if __name__ == '__main__':
    main()

