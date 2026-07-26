"""
Minimal Engine Example: Oracle Long Strategy

Demonstrates basic engine usage with oracle_mode=always_long.
This script loads toy UP market data and validates that PnL > 0.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add engine_core to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine_core.config.params_loader import ParamsLoader
from engine_core.src.engine import BacktestEngine
from engine_core.src.data.loader import DataLoader
from engine_core.tests.fixtures.toy_markets import create_toy_data_loader

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _gate_log import emit_report, get_logger

log = get_logger("run_example_oracle")



def main():
    """Run minimal oracle example"""
    # Create toy UP market (prices go up)
    import tempfile
    from engine_core.tests.fixtures.toy_markets import generate_toy_market
    
    tmp_dir = Path(tempfile.mkdtemp())
    start_ts = pd.Timestamp('2021-01-01 00:00:00', tz='UTC')
    np.random.seed(42)
    up_market = generate_toy_market('UP', start_ts, num_bars=100, base_price=50000.0)
    markets = {'BTCUSDT': up_market}
    data_loader = create_toy_data_loader(markets, tmp_dir)
    
    # Load oracle_long config override
    config_path = Path(__file__).parent.parent / 'config' / 'example_overrides' / 'oracle_long.json'
    params = ParamsLoader(
        base_path=Path(__file__).parent.parent / 'config' / 'base_params.json',
        overrides_path=config_path,
        strict=False
    )
    
    # Create engine
    engine = BacktestEngine(data_loader, params, require_liquidity_data=False)
    
    # Run backtest
    start_ts = pd.Timestamp('2021-01-01 00:00:00', tz='UTC')
    end_ts = pd.Timestamp('2021-01-01 23:45:00', tz='UTC')
    output_dir = Path('runs/example_oracle')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    engine.run(start_ts=start_ts, end_ts=end_ts, output_dir=str(output_dir))
    
    # Validate results
    import json

    with open(output_dir / 'artifacts' / 'metrics.json', 'r') as f:
        metrics = json.load(f)
    
    final_equity = metrics.get('final_equity', 0)
    initial_equity = metrics.get('initial_equity', 100000)
    total_pnl = metrics.get('realized_pnl_from_trades', 0)
    # Metrics use 'total_trades', not 'trade_count'
    trade_count = metrics.get('total_trades', 0)
    
    emit_report(f"Initial Equity: ${initial_equity:,.2f}")
    emit_report(f"Final Equity: ${final_equity:,.2f}")
    emit_report(f"Total PnL: ${total_pnl:,.2f}")
    emit_report(f"Trade Count: {trade_count}")
    emit_report(f"Return: {(final_equity / initial_equity - 1) * 100:.2f}%")
    
    # Validation: PnL should be positive in UP market
    assert total_pnl > 0, f"Expected PnL > 0 in UP market, got {total_pnl}"
    assert trade_count > 0, f"Expected trades > 0, got {trade_count}"
    assert final_equity > initial_equity, f"Expected final equity > initial, got {final_equity} <= {initial_equity}"
    
    emit_report("\n[OK] Example oracle run completed successfully!")
    emit_report("[OK] PnL > 0 validation passed")
    emit_report("[OK] Trade count > 0 validation passed")
    emit_report("[OK] Final equity > initial equity validation passed")


if __name__ == '__main__':
    main()

