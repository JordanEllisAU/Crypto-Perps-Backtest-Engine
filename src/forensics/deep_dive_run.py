"""Deep dive forensic analysis of backtest runs"""
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

LOGGER = logging.getLogger(__name__)


def analyze_run(run_dir: str, output_md_path: str) -> None:
    """
    Main analysis function that performs multi-dimensional breakdowns
    and generates a Markdown report.
    
    Args:
        run_dir: Path to the run directory (e.g., 'runs/abuse_run_kill_es_v2')
        output_md_path: Path where the Markdown report will be written
    """
    from .report_generator import generate_markdown_report
    
    run_path = Path(run_dir)
    artifacts_dir = run_path / 'artifacts'
    
    # Load artifacts
    LOGGER.info("Loading artifacts from %s", artifacts_dir)
    
    # Load metrics.json
    metrics_path = artifacts_dir / 'metrics.json'
    if not metrics_path.exists():
        # Try root level
        metrics_path = run_path / 'metrics.json'
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Load trades.csv
    trades_path = artifacts_dir / 'trades.csv'
    if not trades_path.exists():
        trades_path = run_path / 'trades.csv'
    df_trades = pd.read_csv(trades_path)
    
    # Parse timestamps
    if 'close_ts' in df_trades.columns:
        df_trades['close_ts'] = pd.to_datetime(df_trades['close_ts'], errors='coerce')
    if 'open_ts' in df_trades.columns:
        df_trades['open_ts'] = pd.to_datetime(df_trades['open_ts'], errors='coerce')
    
    # Load equity.csv
    equity_path = artifacts_dir / 'equity.csv'
    if not equity_path.exists():
        equity_path = run_path / 'equity_curve.csv'
    df_equity = pd.read_csv(equity_path)
    if 'ts' in df_equity.columns:
        df_equity['ts'] = pd.to_datetime(df_equity['ts'], errors='coerce')
    
    LOGGER.info("Loaded %s trades and %s equity records", len(df_trades), len(df_equity))
    
    # Compute all breakdowns
    LOGGER.info("Computing breakdowns...")
    analysis_results = {
        'run_dir': str(run_path),
        'run_name': run_path.name,
        'metrics': metrics,
        'time_breakdown': compute_time_breakdown(df_trades, df_equity),
        'module_breakdown': compute_module_breakdown(df_trades),
        'symbol_breakdown': compute_symbol_breakdown(df_trades),
        'direction_breakdown': compute_direction_breakdown(df_trades),
        'cost_decomposition': compute_cost_decomposition(df_trades),
    }
    
    # Generate report
    LOGGER.info("Generating report at %s", output_md_path)
    generate_markdown_report(analysis_results, Path(output_md_path))
    LOGGER.info("Report generated successfully!")


def compute_time_breakdown(df_trades: pd.DataFrame, df_equity: pd.DataFrame) -> Dict:
    """Compute year/quarter/month breakdowns"""
    if len(df_trades) == 0 or 'close_ts' not in df_trades.columns:
        return {'year': {}, 'quarter': {}, 'month': {}}
    
    # Ensure close_ts is datetime
    df_trades = df_trades.copy()
    df_trades['close_ts'] = pd.to_datetime(df_trades['close_ts'], errors='coerce')
    df_trades = df_trades.dropna(subset=['close_ts'])
    
    # Extract time components
    df_trades['year'] = df_trades['close_ts'].dt.year
    df_trades['quarter'] = df_trades['close_ts'].dt.quarter
    df_trades['month'] = df_trades['close_ts'].dt.month
    df_trades['year_quarter'] = df_trades['year'].astype(str) + '-Q' + df_trades['quarter'].astype(str)
    # Use strftime to avoid timezone issues with to_period
    df_trades['year_month'] = df_trades['close_ts'].dt.strftime('%Y-%m')
    
    results = {'year': {}, 'quarter': {}, 'month': {}}
    
    # Year breakdown
    for year in sorted(df_trades['year'].unique()):
        year_trades = df_trades[df_trades['year'] == year]
        year_metrics = compute_period_metrics(year_trades, df_equity, year, None, None)
        results['year'][str(year)] = year_metrics
    
    # Quarter breakdown
    for yq in sorted(df_trades['year_quarter'].unique()):
        yq_trades = df_trades[df_trades['year_quarter'] == yq]
        year, quarter = yq.split('-Q')
        quarter_metrics = compute_period_metrics(yq_trades, df_equity, int(year), int(quarter), None)
        results['quarter'][yq] = quarter_metrics
    
    # Month breakdown
    for ym in sorted(df_trades['year_month'].unique()):
        ym_trades = df_trades[df_trades['year_month'] == ym]
        # Parse year-month string (e.g., "2023-01")
        year, month = map(int, ym.split('-'))
        month_metrics = compute_period_metrics(ym_trades, df_equity, year, None, month)
        results['month'][ym] = month_metrics
    
    return results


def compute_period_metrics(
    period_trades: pd.DataFrame,
    df_equity: pd.DataFrame,
    year: int,
    quarter: Optional[int],
    month: Optional[int]
) -> Dict:
    """Compute metrics for a specific time period"""
    if len(period_trades) == 0:
        return {
            'net_pnl': 0.0,
            'trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
        }
    
    # Net PnL
    net_pnl = float(period_trades['pnl_net_usd'].sum())
    
    # Number of trades
    trades = len(period_trades)
    
    # Win rate
    wins = (period_trades['pnl_net_usd'] > 0).sum()
    win_rate = wins / trades if trades > 0 else 0.0
    
    # Profit factor
    gross_profit = period_trades[period_trades['pnl_net_usd'] > 0]['pnl_net_usd'].sum()
    gross_loss = abs(period_trades[period_trades['pnl_net_usd'] < 0]['pnl_net_usd'].sum())
    if gross_loss > 0:
        profit_factor = float(gross_profit / gross_loss)
    elif gross_profit > 0:
        profit_factor = 999999.0  # Large number instead of inf
    else:
        profit_factor = 0.0
    
    # Approximate max drawdown for this period
    max_dd = 0.0
    if len(df_equity) > 0 and 'ts' in df_equity.columns:
        # Filter equity to this period
        df_equity_copy = df_equity.copy()
        df_equity_copy['ts'] = pd.to_datetime(df_equity_copy['ts'], errors='coerce')
        
        # Ensure timezone consistency
        period_start = period_trades['close_ts'].min()
        period_end = period_trades['close_ts'].max()
        
        # Remove timezone if equity is naive, or make equity aware if trades are aware
        if period_start.tz is not None and df_equity_copy['ts'].dt.tz is None:
            # Trades are timezone-aware, equity is naive - make equity aware (assume UTC)
            df_equity_copy['ts'] = df_equity_copy['ts'].dt.tz_localize('UTC')
        elif period_start.tz is None and df_equity_copy['ts'].dt.tz is not None:
            # Trades are naive, equity is aware - make trades naive
            period_start = period_start.tz_localize(None) if period_start.tz is not None else period_start
            period_end = period_end.tz_localize(None) if period_end.tz is not None else period_end
            df_equity_copy['ts'] = df_equity_copy['ts'].dt.tz_localize(None)
        elif period_start.tz is not None and df_equity_copy['ts'].dt.tz is not None:
            # Both aware - ensure same timezone
            if period_start.tz != df_equity_copy['ts'].dt.tz.iloc[0]:
                df_equity_copy['ts'] = df_equity_copy['ts'].dt.tz_convert(period_start.tz)
        
        period_equity = df_equity_copy[
            (df_equity_copy['ts'] >= period_start) & 
            (df_equity_copy['ts'] <= period_end)
        ]
        
        if len(period_equity) > 0 and 'equity' in period_equity.columns:
            equity_values = period_equity['equity'].values
            if len(equity_values) > 0:
                peak = equity_values[0]
                max_dd = 0.0
                for val in equity_values:
                    if val > peak:
                        peak = val
                    dd = (peak - val) / peak if peak > 0 else 0.0
                    if dd > max_dd:
                        max_dd = dd
    
    return {
        'net_pnl': net_pnl,
        'trades': trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd,
    }


def compute_module_breakdown(df_trades: pd.DataFrame) -> Dict:
    """Compute module-level metrics"""
    if len(df_trades) == 0 or 'module' not in df_trades.columns:
        return {}
    
    results = {}
    
    for module in df_trades['module'].unique():
        if pd.isna(module):
            continue
        
        module_trades = df_trades[df_trades['module'] == module]
        module_metrics = compute_group_metrics(
            module_trades,
            include_costs=True,
            include_holding_time=True
        )
        results[str(module)] = module_metrics
    
    return results


def compute_symbol_breakdown(df_trades: pd.DataFrame) -> Dict:
    """Compute symbol and module×symbol breakdowns"""
    if len(df_trades) == 0 or 'symbol' not in df_trades.columns:
        return {'by_symbol': {}, 'by_module_symbol': {}}
    
    results = {'by_symbol': {}, 'by_module_symbol': {}}
    
    # Overall symbol breakdown
    for symbol in df_trades['symbol'].unique():
        if pd.isna(symbol):
            continue
        
        symbol_trades = df_trades[df_trades['symbol'] == symbol]
        symbol_metrics = compute_group_metrics(
            symbol_trades,
            include_costs=False,
            include_holding_time=True
        )
        results['by_symbol'][str(symbol)] = symbol_metrics
    
    # Module×Symbol breakdown
    if 'module' in df_trades.columns:
        for (module, symbol), group in df_trades.groupby(['module', 'symbol']):
            if pd.isna(module) or pd.isna(symbol):
                continue
            
            key = f"{module}-{symbol}"
            module_symbol_metrics = compute_group_metrics(
                group,
                include_costs=False,
                include_holding_time=True
            )
            results['by_module_symbol'][key] = {
                'module': str(module),
                'symbol': str(symbol),
                **module_symbol_metrics
            }
    
    return results


def compute_direction_breakdown(df_trades: pd.DataFrame) -> Dict:
    """Compute direction breakdowns (LONG vs SHORT)"""
    if len(df_trades) == 0 or 'dir' not in df_trades.columns:
        return {'overall': {}, 'by_module': {}, 'by_symbol': {}}
    
    results = {'overall': {}, 'by_module': {}, 'by_symbol': {}}
    
    # Overall direction breakdown
    for direction in df_trades['dir'].unique():
        if pd.isna(direction):
            continue
        
        dir_trades = df_trades[df_trades['dir'] == direction]
        dir_metrics = compute_group_metrics(
            dir_trades,
            include_costs=False,
            include_holding_time=False
        )
        results['overall'][str(direction)] = dir_metrics
    
    # Module×Direction breakdown
    if 'module' in df_trades.columns:
        for (module, direction), group in df_trades.groupby(['module', 'dir']):
            if pd.isna(module) or pd.isna(direction):
                continue
            
            key = f"{module}-{direction}"
            module_dir_metrics = compute_group_metrics(
                group,
                include_costs=False,
                include_holding_time=False
            )
            results['by_module'][key] = {
                'module': str(module),
                'direction': str(direction),
                **module_dir_metrics
            }
    
    # Symbol×Direction breakdown (only if >= 5 trades)
    if 'symbol' in df_trades.columns:
        for (symbol, direction), group in df_trades.groupby(['symbol', 'dir']):
            if pd.isna(symbol) or pd.isna(direction):
                continue
            
            if len(group) >= 5:  # Only include if enough trades
                key = f"{symbol}-{direction}"
                symbol_dir_metrics = compute_group_metrics(
                    group,
                    include_costs=False,
                    include_holding_time=False
                )
                results['by_symbol'][key] = {
                    'symbol': str(symbol),
                    'direction': str(direction),
                    **symbol_dir_metrics
                }
    
    return results


def compute_cost_decomposition(df_trades: pd.DataFrame) -> Dict:
    """Compute detailed cost breakdowns"""
    if len(df_trades) == 0:
        return {'by_module': {}, 'by_symbol': {}, 'overall': {}}
    
    results = {'by_module': {}, 'by_symbol': {}, 'overall': {}}
    
    # Overall cost decomposition
    results['overall'] = compute_cost_metrics(df_trades)
    
    # By module
    if 'module' in df_trades.columns:
        for module in df_trades['module'].unique():
            if pd.isna(module):
                continue
            
            module_trades = df_trades[df_trades['module'] == module]
            results['by_module'][str(module)] = compute_cost_metrics(module_trades)
    
    # By symbol
    if 'symbol' in df_trades.columns:
        for symbol in df_trades['symbol'].unique():
            if pd.isna(symbol):
                continue
            
            symbol_trades = df_trades[df_trades['symbol'] == symbol]
            results['by_symbol'][str(symbol)] = compute_cost_metrics(symbol_trades)
    
    return results


def compute_group_metrics(
    group: pd.DataFrame,
    include_costs: bool = False,
    include_holding_time: bool = False
) -> Dict:
    """Compute standard metrics for a group of trades"""
    if len(group) == 0:
        return {}
    
    trades = len(group)
    
    # Net PnL
    net_pnl = float(group['pnl_net_usd'].sum())
    
    # Win rate
    wins = (group['pnl_net_usd'] > 0).sum()
    win_rate = wins / trades if trades > 0 else 0.0
    
    # Profit factor
    gross_profit = group[group['pnl_net_usd'] > 0]['pnl_net_usd'].sum()
    gross_loss = abs(group[group['pnl_net_usd'] < 0]['pnl_net_usd'].sum())
    if gross_loss > 0:
        profit_factor = float(gross_profit / gross_loss)
    elif gross_profit > 0:
        profit_factor = 999999.0
    else:
        profit_factor = 0.0
    
    # Average gross/net PnL per trade
    avg_gross_pnl = float(group['pnl_gross_usd'].mean()) if 'pnl_gross_usd' in group.columns else 0.0
    avg_net_pnl = float(group['pnl_net_usd'].mean())
    
    metrics = {
        'trades': trades,
        'net_pnl': net_pnl,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_gross_pnl_per_trade': avg_gross_pnl,
        'avg_net_pnl_per_trade': avg_net_pnl,
    }
    
    # Add costs if requested
    if include_costs:
        gross_signal_pnl = float(group['pnl_gross_usd'].sum()) if 'pnl_gross_usd' in group.columns else 0.0
        total_costs = float(
            group['entry_costs_usd'].sum() + 
            group['exit_costs_usd'].sum() + 
            group['funding_cost_usd'].sum()
        ) if all(col in group.columns for col in ['entry_costs_usd', 'exit_costs_usd', 'funding_cost_usd']) else 0.0
        
        cost_ratio = total_costs / abs(gross_signal_pnl) if gross_signal_pnl != 0 else 0.0
        
        metrics.update({
            'gross_signal_pnl': gross_signal_pnl,
            'total_costs': total_costs,
            'cost_ratio': cost_ratio,
        })
    
    # Add holding time if requested
    if include_holding_time and 'age_bars' in group.columns:
        # Filter out zero/negative age_bars for meaningful average
        valid_ages = group[group['age_bars'] > 0]['age_bars']
        if len(valid_ages) > 0:
            avg_holding_time_bars = float(valid_ages.mean())
        else:
            avg_holding_time_bars = 0.0
        metrics['avg_holding_time_bars'] = avg_holding_time_bars
    
    return metrics


def compute_cost_metrics(group: pd.DataFrame) -> Dict:
    """Compute detailed cost metrics for a group of trades"""
    if len(group) == 0:
        return {}
    
    # Gross PnL before costs
    gross_pnl = float(group['pnl_gross_usd'].sum()) if 'pnl_gross_usd' in group.columns else 0.0
    
    # Cost components
    total_fees = 0.0
    total_slippage = 0.0
    total_funding = 0.0
    
    if all(col in group.columns for col in ['entry_costs_usd', 'exit_costs_usd', 'funding_cost_usd']):
        # Total fees: entry + exit costs (assuming these include fees)
        # Note: In reality, we'd need fills.csv to separate fees from slippage precisely
        # For now, we'll estimate: fees are typically ~0.04% of notional (4 bps)
        # Slippage is the remainder of entry_costs_usd + exit_costs_usd after fees
        
        entry_costs = group['entry_costs_usd'].sum()
        exit_costs = group['exit_costs_usd'].sum()
        total_funding = float(group['funding_cost_usd'].sum())
        
        # Estimate fees: 4 bps of notional (entry + exit)
        if 'notional_entry_usd' in group.columns and 'notional_exit_usd' in group.columns:
            total_notional = group['notional_entry_usd'].sum() + group['notional_exit_usd'].sum()
            estimated_fees = total_notional * 0.0004  # 4 bps
            total_fees = float(estimated_fees)
            # Slippage is the remainder
            total_slippage = float(entry_costs + exit_costs - estimated_fees)
        else:
            # Fallback: assume all entry+exit costs are fees
            total_fees = float(entry_costs + exit_costs)
            total_slippage = 0.0
    
    total_costs = total_fees + total_slippage + total_funding
    net_pnl = float(group['pnl_net_usd'].sum())
    
    # Cost share
    cost_share = total_costs / abs(gross_pnl) if gross_pnl != 0 else 0.0
    
    # Average cost per trade
    avg_cost_per_trade = total_costs / len(group) if len(group) > 0 else 0.0
    
    # Cost per trade distribution (p50, p90)
    if all(col in group.columns for col in ['entry_costs_usd', 'exit_costs_usd', 'funding_cost_usd']):
        cost_per_trade = (
            group['entry_costs_usd'] + 
            group['exit_costs_usd'] + 
            group['funding_cost_usd']
        )
        cost_p50 = float(cost_per_trade.quantile(0.50)) if len(cost_per_trade) > 0 else 0.0
        cost_p90 = float(cost_per_trade.quantile(0.90)) if len(cost_per_trade) > 0 else 0.0
    else:
        cost_p50 = 0.0
        cost_p90 = 0.0
    
    return {
        'gross_pnl_before_costs': gross_pnl,
        'total_fees': total_fees,
        'total_slippage': total_slippage,
        'total_funding': total_funding,
        'total_costs': total_costs,
        'net_pnl': net_pnl,
        'cost_share': cost_share,
        'avg_cost_per_trade': avg_cost_per_trade,
        'cost_per_trade_p50': cost_p50,
        'cost_per_trade_p90': cost_p90,
    }

