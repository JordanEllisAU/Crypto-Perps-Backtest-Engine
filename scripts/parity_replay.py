"""
Replay PnL calculation from exported signals using pandas.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def replay_pnl(signals_path: Path, initial_capital: float = 100000.0, funding_data: dict = None) -> dict:
    """
    Replay PnL calculation from signals.
    
    Returns:
        dict with metrics: total_pnl, final_equity, num_trades, etc.
    """
    signals_df = pd.read_csv(signals_path)
    signals_df['ts'] = pd.to_datetime(signals_df['ts'])
    
    # Track positions
    positions = {}  # {position_id: {'entry_price': float, 'qty': float, 'side': str, 'entry_ts': ts}}
    cash = initial_capital
    total_pnl = 0.0
    total_fees = 0.0
    total_slippage = 0.0
    
    trades = []
    skipped_exits = []  # Track skipped EXITs for debugging
    
    # Process fills: ENTRYs first, then EXITs (to handle timestamp ordering issues)
    # This ensures all ENTRYs are processed before any EXITs
    entry_fills = signals_df[signals_df['leg'] == 'ENTRY'].sort_values('ts')
    exit_fills = signals_df[signals_df['leg'] == 'EXIT'].sort_values('ts')
    
    # Process all ENTRYs first
    for _, fill in entry_fills.iterrows():
        position_id = fill.get('position_id', '')
        side = fill['side']
        qty = fill['qty']
        price = fill['price']
        fee_usd = fill.get('fee_usd', 0.0)
        slippage_cost_usd = fill.get('slippage_cost_usd', 0.0)
        
        # Open position
        pos_record = {
            'entry_price': price,
            'qty': qty,
            'side': 'LONG' if side == 'BUY' else 'SHORT',
            'entry_ts': fill['ts'],
            'entry_fees': fee_usd,
            'entry_slippage': slippage_cost_usd,
        }
        positions[position_id] = pos_record
        
        # Track entry costs separately; cash is only adjusted on closed PnL so that
        # final equity equals initial capital + round-trip net PnL.
        total_fees += fee_usd
        total_slippage += slippage_cost_usd
    
    # Then process all EXITs
    for _, fill in exit_fills.iterrows():
        position_id = fill.get('position_id', '')
        qty = fill['qty']
        price = fill['price']
        fee_usd = fill.get('fee_usd', 0.0)
        slippage_cost_usd = fill.get('slippage_cost_usd', 0.0)
        
        # Close position
        if position_id not in positions:
            skipped_exits.append(position_id)
            continue  # Skip if position not found
        
        pos = positions[position_id]
        
        # Calculate gross PnL
        if pos['side'] == 'LONG':
            pnl = (price - pos['entry_price']) * pos['qty']
        else:  # SHORT
            pnl = (pos['entry_price'] - price) * pos['qty']
        
        # Round-trip net PnL = gross - (entry + exit fees and slippage)
        total_trade_costs = (
            pos['entry_fees'] + pos['entry_slippage'] + fee_usd + slippage_cost_usd
        )
        pnl -= total_trade_costs
        
        # Update cash and PnL
        cash += pnl
        total_pnl += pnl
        total_fees += fee_usd
        total_slippage += slippage_cost_usd
        
        # Record trade
        trades.append({
            'position_id': position_id,
            'side': pos['side'],
            'entry_price': pos['entry_price'],
            'exit_price': price,
            'qty': pos['qty'],
            'pnl': pnl,
            'fees': total_trade_costs,
            'slippage': pos['entry_slippage'] + slippage_cost_usd
        })
        
        # Remove position
        del positions[position_id]
    
    # Calculate final equity (cash + unrealized PnL from open positions)
    # For simplicity, assume open positions are closed at last price
    # Note: In a full implementation, we'd need to mark-to-market open positions
    # For parity check, we assume all positions are closed (engine closes at end_ts)
    final_equity = cash
    
    # Calculate total costs
    total_costs = total_fees + total_slippage
    
    result = {
        'initial_capital': initial_capital,
        'final_equity': final_equity,
        'total_pnl': total_pnl,
        'total_fees': total_fees,
        'total_slippage': total_slippage,
        'total_costs': total_costs,
        'num_trades': len(trades),
        'trades': trades
    }
    
    # Debug info
    if len(skipped_exits) > 0:
        result['debug_skipped_exits'] = len(skipped_exits)
        result['debug_open_positions'] = len(positions)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Replay PnL from exported signals')
    parser.add_argument('signals_path', type=str, help='Path to exported signals CSV')
    parser.add_argument('--initial-capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--output', type=str, default='parity_replay.json', help='Output JSON path')
    
    args = parser.parse_args()
    
    results = replay_pnl(args.signals_path, args.initial_capital)
    
    # Save results
    import json
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Replay results saved to {output_path}")
    print(f"Total PnL: ${results['total_pnl']:,.2f}")
    print(f"Final Equity: ${results['final_equity']:,.2f}")
    print(f"Number of Trades: {results['num_trades']}")


if __name__ == '__main__':
    main()
