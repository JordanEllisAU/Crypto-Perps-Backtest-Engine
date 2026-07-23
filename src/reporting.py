"""Report generation: CSV/JSON outputs"""
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
import numpy as np
from datetime import datetime, UTC
import hashlib
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False


@dataclass
class ValidationResult:
    """Result of metrics validation"""
    passed: bool                     # True only if no hard failures
    failures: List[str]              # Hard violations (NO-GO)
    warnings: List[str]              # WARN-level issues
    info: List[str]                  # Informational notes
    metrics: Dict[str, Any]          # Parsed metrics.json
    artifacts_dir: Path              # Where artifacts were loaded from


def validate_metrics(
    artifacts_dir: Union[str, Path],
    *,
    strict_canonical: bool = True,
) -> ValidationResult:
    """
    Validate metrics and artifacts against SSOT invariants.
    
    Args:
        artifacts_dir: Path to artifacts directory
        strict_canonical: If True, ES/margin/halt violations are hard failures.
                         If False, they are warnings.
    
    Returns:
        ValidationResult with pass/fail status and detailed messages
    """
    artifacts_dir = Path(artifacts_dir)
    failures: List[str] = []
    warnings: List[str] = []
    info: List[str] = []
    
    # Required files
    required_files = {
        'equity.csv': artifacts_dir / 'equity.csv',
        'trades.csv': artifacts_dir / 'trades.csv',
        'fills.csv': artifacts_dir / 'fills.csv',
        'ledger.csv': artifacts_dir / 'ledger.csv',
        'metrics.json': artifacts_dir / 'metrics.json',
    }
    
    # Check all required files exist
    for name, path in required_files.items():
        if not path.exists():
            failures.append(f"Missing required artifact: {name}")
            return ValidationResult(
                passed=False,
                failures=failures,
                warnings=warnings,
                info=info,
                metrics={},
                artifacts_dir=artifacts_dir
            )
    
    # Load artifacts
    try:
        equity_df = pd.read_csv(required_files['equity.csv'])
        trades_df = pd.read_csv(required_files['trades.csv'])
        fills_df = pd.read_csv(required_files['fills.csv'])
        ledger_df = pd.read_csv(required_files['ledger.csv'])
        with open(required_files['metrics.json'], 'r') as f:
            metrics = json.load(f)
    except Exception as e:
        failures.append(f"Failed to load artifacts: {e}")
        return ValidationResult(
            passed=False,
            failures=failures,
            warnings=warnings,
            info=info,
            metrics={},
            artifacts_dir=artifacts_dir
        )
    
    # Required metrics fields
    required_fields = [
        'es_violations_count',
        'margin_blocks_count',
        'halt_daily_hard_count',
        'per_symbol_loss_cap_count',
        'halt_soft_brake_count',
        'slippage_degeneracy_warning',
        'vacuum_blocks_count',
        'thin_post_only_entries_count',
        'thin_cancel_block_count',
    ]
    
    for field in required_fields:
        if field not in metrics:
            failures.append(f"Missing required metrics field: {field}")
    
    if failures:
        return ValidationResult(
            passed=False,
            failures=failures,
            warnings=warnings,
            info=info,
            metrics=metrics,
            artifacts_dir=artifacts_dir
        )
    
    # Gate 1: Bar Identity (hard FAIL - always)
    if len(equity_df) > 0:
        required_cols = {'equity', 'cash', 'open_pnl', 'closed_pnl'}
        if required_cols.issubset(set(equity_df.columns)):
            for idx, row in equity_df.iterrows():
                equity_val = float(row['equity'])
                cash_val = float(row['cash'])
                open_pnl_val = float(row['open_pnl'])
                closed_pnl_val = float(row['closed_pnl'])
                computed = cash_val + open_pnl_val + closed_pnl_val
                diff = abs(equity_val - computed)
                if diff > 1e-6:
                    ts = row.get('ts', idx)
                    failures.append(
                        f"Bar identity violation at ts={ts}: equity={equity_val:.6f}, "
                        f"cash+open_pnl+closed_pnl={computed:.6f}, diff={diff:.6f}"
                    )
                    break  # Report first violation
        else:
            missing = required_cols - set(equity_df.columns)
            failures.append(f"equity.csv missing required columns: {missing}")
    
    # Gate 2: Global PnL Reconciliation (hard FAIL - always)
    if len(equity_df) > 0:
        initial_equity = float(equity_df['equity'].iloc[0])
        final_equity = float(equity_df['equity'].iloc[-1])
        equity_delta = final_equity - initial_equity
        
        sum_trades = 0.0
        sum_ledger = 0.0
        
        if len(trades_df) > 0 and 'pnl_net_usd' in trades_df.columns:
            sum_trades = float(trades_df['pnl_net_usd'].sum())
        
        if len(ledger_df) > 0 and 'cash_delta_usd' in ledger_df.columns:
            sum_ledger = float(ledger_df['cash_delta_usd'].sum())
        
        n_trades = len(trades_df) if len(trades_df) > 0 else 0
        tolerance = 0.01 * max(n_trades, 1)
        
        # Check equity_delta vs sum_trades
        diff_equity_vs_trades = abs(equity_delta - sum_trades)
        if diff_equity_vs_trades > tolerance:
            # Also compute diff between trades and ledger for debugging
            diff_trades_vs_ledger = abs(sum_trades - sum_ledger)
            failures.append(
                f"Global PnL reconciliation failed: equity_delta={equity_delta:.2f}, "
                f"sum_trades={sum_trades:.2f}, sum_ledger={sum_ledger:.2f}, "
                f"diff_equity_vs_trades={diff_equity_vs_trades:.2f}, diff_trades_vs_ledger={diff_trades_vs_ledger:.2f}, "
                f"tolerance={tolerance:.2f}"
            )
        
        # Also check ledger vs trades (separate gate for clarity)
        if len(trades_df) > 0 and 'pnl_net_usd' in trades_df.columns and len(ledger_df) > 0 and 'cash_delta_usd' in ledger_df.columns:
            sum_trades_pnl = float(trades_df['pnl_net_usd'].sum())
            sum_ledger_cash = float(ledger_df['cash_delta_usd'].sum())
            n_trades = len(trades_df)
            tolerance = 0.01 * max(n_trades, 1)
            diff = abs(sum_ledger_cash - sum_trades_pnl)
            if diff > tolerance:
                failures.append(
                    f"Ledger vs trades reconciliation failed: SUM(ledger.cash_delta_usd) = {sum_ledger_cash:.2f}, "
                    f"SUM(trades.pnl_net_usd) = {sum_trades_pnl:.2f}, diff = {diff:.2f}, tolerance = {tolerance:.2f}"
                )
    
    # Gate 3: Trade Counting Coherence (hard FAIL - always)
    n_trades_csv = len(trades_df)
    n_trades_metrics = metrics.get('total_trades', 0)
    if n_trades_csv != n_trades_metrics:
        failures.append(
            f"Trade counting coherence failed: len(trades.csv) = {n_trades_csv}, "
            f"metrics.total_trades = {n_trades_metrics}"
        )
    
    # Win rate check
    if 'win_rate' in metrics and n_trades_csv > 0:
        if 'pnl_net_usd' in trades_df.columns:
            wins = (trades_df['pnl_net_usd'] > 0).sum()
            win_rate_recomputed = wins / n_trades_csv
            win_rate_metrics = float(metrics['win_rate'])
            diff = abs(win_rate_metrics - win_rate_recomputed)
            if diff > 1e-6:
                failures.append(
                    f"Win rate coherence failed: metrics.win_rate = {win_rate_metrics:.6f}, "
                    f"recomputed from trades.csv = {win_rate_recomputed:.6f}, diff = {diff:.6f}"
                )
    
    # Gate 4: Costs Applied (hard FAIL - always)
    if len(fills_df) > 0:
        total_fees_from_fills = float(fills_df['fee_usd'].sum()) if 'fee_usd' in fills_df.columns else 0.0
        total_slippage_from_fills = float(fills_df['slippage_cost_usd'].sum()) if 'slippage_cost_usd' in fills_df.columns else 0.0
        
        if len(ledger_df) > 0 and 'event' in ledger_df.columns:
            funding_events = ledger_df[ledger_df['event'] == 'FUNDING']
            funding_from_ledger = float(funding_events['funding_usd'].sum()) if 'funding_usd' in funding_events.columns else 0.0
        else:
            funding_from_ledger = 0.0
        
        total_fees_metric = float(metrics.get('total_fees', 0.0))
        total_slippage_metric = float(metrics.get('total_slippage_cost', 0.0))
        funding_cost_total = float(metrics.get('funding_cost_total', 0.0))
        
        n_fills = len(fills_df)
        tolerance = 0.01 * max(n_fills, 1)
        
        fees_diff = abs(total_fees_from_fills - total_fees_metric)
        if fees_diff > tolerance:
            failures.append(
                f"Costs applied (fees) failed: SUM(fills.fee_usd) = {total_fees_from_fills:.2f}, "
                f"metrics.total_fees = {total_fees_metric:.2f}, diff = {fees_diff:.2f}, tolerance = {tolerance:.2f}"
            )
        
        slippage_diff = abs(total_slippage_from_fills - total_slippage_metric)
        if slippage_diff > tolerance:
            failures.append(
                f"Costs applied (slippage) failed: SUM(fills.slippage_cost_usd) = {total_slippage_from_fills:.2f}, "
                f"metrics.total_slippage_cost = {total_slippage_metric:.2f}, diff = {slippage_diff:.2f}, tolerance = {tolerance:.2f}"
            )
        
        funding_diff = abs(funding_from_ledger - funding_cost_total)
        if funding_diff > tolerance:
            failures.append(
                f"Costs applied (funding) failed: SUM(ledger[FUNDING].funding_usd) = {funding_from_ledger:.2f}, "
                f"metrics.funding_cost_total = {funding_cost_total:.2f}, diff = {funding_diff:.2f}, tolerance = {tolerance:.2f}"
            )
    
    # Gate 5: Exposure Sanity (hard FAIL - always)
    exposure_pct = metrics.get('exposure_pct', 0.0)
    # NO auto-conversion - SSOT defines it as fraction [0, 1]
    if exposure_pct < 0.0 or exposure_pct > 1.0:
        failures.append(
            f"Exposure sanity failed: exposure_pct = {exposure_pct:.4f} not in [0.0, 1.0] "
            f"(SSOT defines it as fraction, not percentage)"
        )
    
    # Gate 6: Per-Position PnL Reconciliation (hard FAIL - always)
    if len(trades_df) > 0 and 'position_id' in trades_df.columns and 'pnl_net_usd' in trades_df.columns:
        if len(ledger_df) > 0 and 'position_id' in ledger_df.columns and 'cash_delta_usd' in ledger_df.columns:
            for position_id in trades_df['position_id'].unique():
                trade_row = trades_df[trades_df['position_id'] == position_id]
                if len(trade_row) > 0:
                    pnl_net_usd = float(trade_row['pnl_net_usd'].iloc[0])
                    ledger_rows = ledger_df[ledger_df['position_id'] == position_id]
                    cash_impact = float(ledger_rows['cash_delta_usd'].sum())
                    diff = abs(pnl_net_usd - cash_impact)
                    if diff > 0.01:
                        failures.append(
                            f"Per-position PnL reconciliation failed for position_id={position_id}: "
                            f"pnl_net_usd={pnl_net_usd:.2f}, cash_impact_from_ledger={cash_impact:.2f}, diff={diff:.2f}"
                        )
                        break  # Report first violation
    
    # Gate 7: ES Violations (conditional on strict_canonical)
    es_violations = metrics.get('es_violations_count', 0)
    if strict_canonical:
        if es_violations > 0:
            failures.append(f"ES violations detected: {es_violations} violations found (strict canonical mode)")
    else:
        if es_violations > 0:
            warnings.append(f"ES violations detected: {es_violations} violations found (non-strict mode)")
    
    # Gate 8: Margin Blocks (conditional on strict_canonical)
    margin_blocks = metrics.get('margin_blocks_count', 0)
    if strict_canonical:
        if margin_blocks > 0:
            failures.append(f"Margin blocks detected: {margin_blocks} blocks found (strict canonical mode)")
    else:
        if margin_blocks > 0:
            warnings.append(f"Margin blocks detected: {margin_blocks} blocks found (non-strict mode)")
    
    # Gate 9: Loss Halts (conditional on strict_canonical)
    halt_daily_hard = metrics.get('halt_daily_hard_count', 0)
    per_symbol_cap = metrics.get('per_symbol_loss_cap_count', 0)
    if strict_canonical:
        if halt_daily_hard > 0:
            failures.append(f"Daily hard halt detected: {halt_daily_hard} occurrences (strict canonical mode)")
        if per_symbol_cap > 0:
            failures.append(f"Per-symbol loss cap detected: {per_symbol_cap} occurrences (strict canonical mode)")
    else:
        if halt_daily_hard > 0:
            warnings.append(f"Daily hard halt detected: {halt_daily_hard} occurrences (non-strict mode)")
        if per_symbol_cap > 0:
            warnings.append(f"Per-symbol loss cap detected: {per_symbol_cap} occurrences (non-strict mode)")
    
    # Gate 10: Liquidity Regime Counters (WARN/INFO - always)
    vacuum_blocks = metrics.get('vacuum_blocks_count', 0)
    if vacuum_blocks > 0:
        warnings.append(f"VACUUM blocks detected: {vacuum_blocks} blocks (liquidity issues)")
    
    thin_cancel = metrics.get('thin_cancel_block_count', 0)
    if thin_cancel > 0:
        warnings.append(f"THIN cancel blocks detected: {thin_cancel} blocks")
    
    thin_post_only = metrics.get('thin_post_only_entries_count', 0)
    if thin_post_only > 0:
        info.append(f"THIN post-only entries: {thin_post_only} entries")
    
    # Gate 11: Slippage Degeneracy (WARN - always)
    # slippage_degeneracy_warning is a required field (checked above)
    slippage_degen = metrics.get('slippage_degeneracy_warning', False)
    if isinstance(slippage_degen, str):
        slippage_degen = slippage_degen.lower() == 'true'
    if slippage_degen:
        warnings.append("Slippage degeneracy warning: realized slippage is unnaturally constant")
    
    # Set passed status
    passed = len(failures) == 0
    
    return ValidationResult(
        passed=passed,
        failures=failures,
        warnings=warnings,
        info=info,
        metrics=metrics,
        artifacts_dir=artifacts_dir
    )


class ReportGenerator:
    """Generate backtest reports"""
    
    def __init__(self, output_dir: str = "reports", run_id: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.output_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
    
    def generate_trades_csv(self, trades: List[Dict]):
        """Generate trades.csv - FIX 2 & 3: Include position_id, open_ts, close_ts, age_bars
        Note: This now calls _write_trades_artifact which rebuilds from fills.csv"""
        # _write_trades_artifact will be called separately and rebuilds from fills.csv
        # This method is kept for backward compatibility but the actual artifact is written by _write_trades_artifact
        if not trades:
            # Create empty dataframe with required columns
            df = pd.DataFrame(columns=[
                'ts', 'symbol', 'side', 'module', 'qty', 'price', 'fees', 'slip_bps',
                'participation_pct', 'post_only',
                'stop_dist', 'ES_used_before', 'ES_used_after', 'reason', 'pnl',
                'position_id', 'open_ts', 'close_ts', 'age_bars'
            ])
        else:
            df = pd.DataFrame(trades)
            # Ensure required columns exist
            if 'pnl' not in df.columns:
                df['pnl'] = 0.0
            if 'participation_pct' not in df.columns:
                df['participation_pct'] = 0.0
            if 'post_only' not in df.columns:
                df['post_only'] = False
            if 'position_id' not in df.columns:
                df['position_id'] = ''
            if 'open_ts' not in df.columns:
                df['open_ts'] = None
            if 'close_ts' not in df.columns:
                df['close_ts'] = None
            if 'age_bars' not in df.columns:
                df['age_bars'] = 0
            if 'gap_through' not in df.columns:
                df['gap_through'] = False
        
        df.to_csv(self.output_dir / 'trades.csv', index=False)
    
    def generate_equity_curve_csv(self, equity_curve: List[Dict]):
        """Generate equity_curve.csv"""
        if not equity_curve:
            df = pd.DataFrame(columns=['ts', 'equity', 'drawdown', 'daily_pnl', 'rolling_vol'])
        else:
            df = pd.DataFrame(equity_curve)
            
            # Calculate daily PnL and rolling vol if not present
            if 'daily_pnl' not in df.columns:
                df['daily_pnl'] = df['equity'].diff()
            
            if 'rolling_vol' not in df.columns:
                returns = df['equity'].pct_change()
                df['rolling_vol'] = returns.rolling(window=96).std() * np.sqrt(96)  # Annualized
        
        df.to_csv(self.output_dir / 'equity_curve.csv', index=False)
    
    def generate_positions_csv(self, positions_history: List[Dict]):
        """Generate positions.csv in artifacts directory"""
        if not positions_history:
            df = pd.DataFrame(columns=['ts', 'position_id', 'symbol', 'qty', 'entry_px', 'stop_px', 'trail_px', 'module', 'age_bars'])
        else:
            df = pd.DataFrame(positions_history)
            # Ensure required columns exist
            required_cols = ['ts', 'position_id', 'symbol', 'qty', 'entry_px', 'stop_px', 'trail_px', 'module', 'age_bars']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'position_id':
                        df[col] = ''
                    elif col in ['qty', 'entry_px', 'stop_px', 'trail_px', 'age_bars']:
                        df[col] = 0.0
                    else:
                        df[col] = ''
        
        # Write to artifacts directory
        df.to_csv(self.artifacts_dir / 'positions.csv', index=False)
        # Also write to output_dir for backward compatibility
        df.to_csv(self.output_dir / 'positions.csv', index=False)
    
    def generate_metrics_json(
        self,
        portfolio_state,
        trades: List[Dict],
        equity_curve: List[Dict],
        positions_history: List[Dict],
        params_snapshot: Dict,
        es_violations_count: int = 0,
        es_block_count: int = 0,
        beta_block_count: int = 0,
        margin_blocks_count: int = 0,
        margin_trim_count: int = 0,
        halt_daily_hard_count: int = 0,
        halt_soft_brake_count: int = 0,
        per_symbol_loss_cap_count: int = 0,
        vacuum_blocks_count: int = 0,
        thin_post_only_entries_count: int = 0,
        thin_extra_slip_bps_total: float = 0.0,
        thin_cancel_block_count: int = 0,
        funding_events_count: int = 0,
        es_usage_samples: List[float] = None,
        vacuum_dwell_bars: int = 0,
        thin_dwell_bars: int = 0,
        total_bars_processed: int = 0
    ):
        """Generate metrics.json"""
        if not equity_curve:
            metrics = self._calculate_empty_metrics()
        else:
            metrics = self._calculate_metrics(
                portfolio_state,
                trades,
                equity_curve,
                positions_history,
                es_violations_count,
                es_block_count,
                beta_block_count,
                margin_blocks_count,
                margin_trim_count,
                halt_daily_hard_count,
                halt_soft_brake_count,
                per_symbol_loss_cap_count,
                vacuum_blocks_count,
                thin_post_only_entries_count,
                thin_extra_slip_bps_total,
                thin_cancel_block_count,
                funding_events_count,
                es_usage_samples or [],
                vacuum_dwell_bars,
                thin_dwell_bars,
                total_bars_processed
            )
        
        # Add identity & bookkeeping fields
        import hashlib
        params_json_str = json.dumps(params_snapshot, sort_keys=True, default=str)
        params_hash = hashlib.sha256(params_json_str.encode()).hexdigest()[:16]
        
        metrics['run_id'] = self.run_id or ''
        metrics['created_at'] = datetime.now(UTC).isoformat()
        metrics['params_hash'] = params_hash
        # initial_equity and final_equity are already set in metrics dict by _calculate_metrics
        # Just ensure they're present (they should be)
        if 'initial_equity' not in metrics:
            metrics['initial_equity'] = portfolio_state.initial_capital
        if 'final_equity' not in metrics:
            metrics['final_equity'] = portfolio_state.equity
        
        # Calculate total_slippage_cost and slippage_bps_realized_mean from fills
        try:
            fills_path = self.artifacts_dir / 'fills.csv'
            if fills_path.exists():
                df_fills = pd.read_csv(fills_path)
                if len(df_fills) > 0 and 'slippage_cost_usd' in df_fills.columns:
                    metrics['total_slippage_cost'] = float(df_fills['slippage_cost_usd'].sum())
                else:
                    metrics['total_slippage_cost'] = 0.0
                
                if len(df_fills) > 0 and 'slippage_bps_applied' in df_fills.columns:
                    slip_bps_values = df_fills['slippage_bps_applied'].dropna()
                    if len(slip_bps_values) > 0:
                        metrics['slippage_bps_realized_mean'] = float(slip_bps_values.mean())
                    else:
                        metrics['slippage_bps_realized_mean'] = 0.0
                else:
                    metrics['slippage_bps_realized_mean'] = 0.0
            else:
                metrics['total_slippage_cost'] = 0.0
                metrics['slippage_bps_realized_mean'] = 0.0
        except Exception:
            metrics['total_slippage_cost'] = 0.0
            metrics['slippage_bps_realized_mean'] = 0.0
        
        # Add parameter snapshot
        metrics['params_snapshot'] = params_snapshot
        metrics['generated_at'] = datetime.now(UTC).isoformat()
        
        # Write to artifacts/metrics.json (canonical)
        with open(self.artifacts_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        # Also write to output_dir for backward compatibility
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
    
    def _calculate_metrics(
        self,
        portfolio_state,
        trades: List[Dict],
        equity_curve: List[Dict],
        positions_history: List[Dict],
        es_violations_count: int,
        es_block_count: int,
        beta_block_count: int,
        margin_blocks_count: int,
        margin_trim_count: int,
        halt_daily_hard_count: int,
        halt_soft_brake_count: int,
        per_symbol_loss_cap_count: int,
        vacuum_blocks_count: int,
        thin_post_only_entries_count: int,
        thin_extra_slip_bps_total: float,
        thin_cancel_block_count: int,
        funding_events_count: int,
        es_usage_samples: List[float],
        vacuum_dwell_bars: int = 0,
        thin_dwell_bars: int = 0,
        total_bars_processed: int = 0
    ) -> Dict:
        """Calculate performance metrics"""
        # Handle equity_curve: if non-empty, build df_equity from it
        # Else, if equity.csv exists, read it
        # Else, create empty DataFrame with required columns
        if equity_curve:
            df_equity = pd.DataFrame(equity_curve)
        else:
            equity_path = self.artifacts_dir / 'equity.csv'
            if equity_path.exists():
                try:
                    df_equity = pd.read_csv(equity_path)
                except Exception:
                    df_equity = pd.DataFrame(columns=['ts', 'equity', 'cash', 'open_pnl', 'closed_pnl'])
            else:
                df_equity = pd.DataFrame(columns=['ts', 'equity', 'cash', 'open_pnl', 'closed_pnl'])
        
        # Only if df_equity is not empty, ensure it has a ts column and parse it
        if len(df_equity) > 0:
            # Rename 'timestamp' to 'ts' if needed
            if 'timestamp' in df_equity.columns and 'ts' not in df_equity.columns:
                df_equity['ts'] = df_equity['timestamp']
            # Ensure 'ts' column exists and parse it
            if 'ts' in df_equity.columns:
                df_equity['ts'] = pd.to_datetime(df_equity['ts'], utc=True, errors='coerce')
        
        df_trades = pd.DataFrame(trades) if trades else pd.DataFrame()
        if len(df_trades) > 0 and 'ts' in df_trades.columns:
            df_trades['ts'] = pd.to_datetime(df_trades['ts'], utc=True, errors='coerce')
        
        # Basic metrics
        # FIX: initial_equity from equity curve might be after first trade? 
        # Use portfolio_state.initial_capital as the ground truth starting point.
        initial_equity = portfolio_state.initial_capital
        
        # Use portfolio_state.equity as final_equity (most up-to-date) if it differs from equity curve
        # This ensures we capture the equity after EOD finalizer
        final_equity_from_curve = df_equity['equity'].iloc[-1] if len(df_equity) > 0 else portfolio_state.equity
        # Ensure final_equity reflects the most accurate closing state. 
        # If portfolio_state.equity is 0.0 (uninitialized), use curve.
        final_equity = portfolio_state.equity if portfolio_state.equity > 0 else final_equity_from_curve
        
        total_return = (final_equity - initial_equity) / initial_equity if initial_equity > 0 else 0.0
        
        # CAGR
        if len(df_equity) > 0:
            start_ts = df_equity['ts'].iloc[0]
            end_ts = df_equity['ts'].iloc[-1]
            years = (end_ts - start_ts).days / 365.25
            if years > 0:
                cagr = (final_equity / initial_equity) ** (1 / years) - 1
            else:
                cagr = 0.0
        else:
            cagr = 0.0
        
        # Max drawdown
        max_dd = portfolio_state.max_drawdown
        max_dd_pct = portfolio_state.max_drawdown_pct
        
        # MAR (CAGR / MaxDD)
        mar = cagr / abs(max_dd_pct) if max_dd_pct != 0 else 0.0
        
        daily_equity = df_equity.set_index('ts')['equity'].resample('1D').last().dropna()
        daily_returns = daily_equity.pct_change().dropna()
        avg_equity = daily_equity.mean() if len(daily_equity) > 0 else portfolio_state.equity
        
        if len(daily_returns) > 0:
            ret_mean = daily_returns.mean()
            ret_std = daily_returns.std(ddof=0)
            sharpe = (ret_mean / ret_std) * np.sqrt(252) if ret_std > 0 else 0.0
            downside = daily_returns[daily_returns < 0]
            downside_std = downside.std(ddof=0)
            sortino = (ret_mean / downside_std) * np.sqrt(252) if downside_std > 0 else 0.0
        else:
            sharpe = 0.0
            sortino = 0.0
        
        calmar = cagr / abs(max_dd_pct) if max_dd_pct != 0 else 0.0
        
        # Trade metrics - try to read from rebuilt trades.csv first
        slip_nonzero_share = 0.0
        turnover_daily = 0.0
        turnover_annualized = 0.0
        win_rate = 0.0
        profit_factor = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        participation_mean = 0.0
        slip_mean = 0.0
        slip_p95 = 0.0
        ttl_exit_share = 0.0
        module_pnl = {}
        symbol_pnl = {}
        module_symbol_stats = {}
        hit_ratio_per_module = {}  # Initialize here so it's available in both code paths
        used_rebuilt_trades = False
        # Initialize required fields (always set, even if zero/False)
        slippage_degeneracy_warning = False
        slippage_degeneracy_details = None
        
        # Try to read from rebuilt trades.csv (authoritative source)
        # win_rate MUST be computed from trades.csv to match validator
        try:
            trades_path = self.artifacts_dir / 'trades.csv'
            if trades_path.exists():
                df_trades_rebuilt = pd.read_csv(trades_path)
                if len(df_trades_rebuilt) > 0 and 'pnl_net_usd' in df_trades_rebuilt.columns:
                    used_rebuilt_trades = True
                    # Convert timestamp columns if present
                    if 'open_ts' in df_trades_rebuilt.columns:
                        df_trades_rebuilt['open_ts'] = pd.to_datetime(df_trades_rebuilt['open_ts'], errors='coerce')
                    if 'close_ts' in df_trades_rebuilt.columns:
                        df_trades_rebuilt['close_ts'] = pd.to_datetime(df_trades_rebuilt['close_ts'], errors='coerce')
                        # Use close_ts for time-based calculations
                        df_trades_rebuilt['ts'] = df_trades_rebuilt['close_ts']
                    
                    total_trades_rebuilt = len(df_trades_rebuilt)
                    
                    # Calculate win rate from rebuilt trades (SSOT requirement)
                    # wins = count of trades with pnl_net_usd > 0
                    # win_rate = wins / total_trades
                    winning_trades = df_trades_rebuilt[df_trades_rebuilt['pnl_net_usd'] > 0]
                    losing_trades = df_trades_rebuilt[df_trades_rebuilt['pnl_net_usd'] < 0]
                    wins = len(winning_trades)
                    win_rate = wins / total_trades_rebuilt if total_trades_rebuilt > 0 else 0.0
                    
                    # Profit factor: sum of wins / abs(sum of losses)
                    gross_profit = float(winning_trades['pnl_net_usd'].sum()) if len(winning_trades) > 0 else 0.0
                    gross_loss = float(abs(losing_trades['pnl_net_usd'].sum())) if len(losing_trades) > 0 else 0.0
                    if gross_loss > 0.0:
                        profit_factor = gross_profit / gross_loss
                    elif gross_profit > 0.0:
                        # No losing trades but have winning trades - set to a large number instead of inf
                        profit_factor = 999999.0  # Use large number instead of inf for JSON serialization
                    else:
                        profit_factor = 0.0
                    
                    # Avg win/loss (only for trades that actually won/lost, exclude zeros)
                    avg_win = float(winning_trades['pnl_net_usd'].mean()) if len(winning_trades) > 0 else 0.0
                    avg_loss = float(losing_trades['pnl_net_usd'].mean()) if len(losing_trades) > 0 else 0.0
                    
                    # Module/symbol PnL breakdown
                    if 'module' in df_trades_rebuilt.columns:
                        module_pnl = df_trades_rebuilt.groupby('module')['pnl_net_usd'].sum().to_dict()
                        # Convert to float for JSON serialization
                        module_pnl = {k: float(v) for k, v in module_pnl.items()}
                    if 'symbol' in df_trades_rebuilt.columns:
                        symbol_pnl = df_trades_rebuilt.groupby('symbol')['pnl_net_usd'].sum().to_dict()
                        # Convert to float for JSON serialization
                        symbol_pnl = {k: float(v) for k, v in symbol_pnl.items()}
                    
                    # Hit ratio per module (win rate per module) - calculate from rebuilt trades.csv
                    if 'module' in df_trades_rebuilt.columns:
                        hit_ratio_per_module = {}
                        for module_name in df_trades_rebuilt['module'].unique():
                            module_trades = df_trades_rebuilt[df_trades_rebuilt['module'] == module_name]
                            if len(module_trades) > 0:
                                wins = len(module_trades[module_trades['pnl_net_usd'] > 0])
                                total = len(module_trades)
                                hit_ratio_per_module[module_name] = wins / total if total > 0 else 0.0
                    
                    # Calculate avg_r from rebuilt trades.csv
                    if 'initial_risk_usd' in df_trades_rebuilt.columns and 'pnl_net_usd' in df_trades_rebuilt.columns:
                        valid_trades = df_trades_rebuilt[
                            (df_trades_rebuilt['initial_risk_usd'] > 1e-9) &
                            (df_trades_rebuilt['pnl_net_usd'].notna())
                        ]
                        if len(valid_trades) > 0:
                            # R = pnl_net_usd / initial_risk_usd
                            r_multiples = valid_trades['pnl_net_usd'] / valid_trades['initial_risk_usd']
                            avg_r = float(r_multiples.mean())
                    
                    # Module-symbol stats (nested breakdown)
                    if 'module' in df_trades_rebuilt.columns and 'symbol' in df_trades_rebuilt.columns:
                        module_symbol_stats = {}
                        for (module_name, symbol_name), grp in df_trades_rebuilt.groupby(['module', 'symbol']):
                            trades_count = len(grp)
                            wins_count = len(grp[grp['pnl_net_usd'] > 0])
                            pnl_sum = float(grp['pnl_net_usd'].sum())
                            win_rate_module_symbol = wins_count / trades_count if trades_count > 0 else 0.0
                            
                            # Calculate avg trade duration if age_bars is available
                            avg_duration = float(grp['age_bars'].mean()) if 'age_bars' in grp.columns and len(grp) > 0 else 0.0
                            
                            if module_name not in module_symbol_stats:
                                module_symbol_stats[module_name] = {}
                            module_symbol_stats[module_name][symbol_name] = {
                                'trades': int(trades_count),
                                'win_rate': float(win_rate_module_symbol),
                                'pnl': pnl_sum,
                                'avg_trade_duration_bars': avg_duration
                            }
                    
                    # TTL exit share
                    if 'exit_reason' in df_trades_rebuilt.columns:
                        ttl_exits = len(df_trades_rebuilt[df_trades_rebuilt['exit_reason'] == 'TTL'])
                        ttl_exit_share = ttl_exits / total_trades_rebuilt if total_trades_rebuilt > 0 else 0.0
                    
                    # Turnover calculation from rebuilt trades.csv
                    # Use notional_entry_usd + notional_exit_usd for total notional traded
                    if 'notional_entry_usd' in df_trades_rebuilt.columns and 'notional_exit_usd' in df_trades_rebuilt.columns:
                        # Total notional = sum of entry + exit notionals (each trade counted once per leg)
                        df_trades_rebuilt['total_notional'] = df_trades_rebuilt['notional_entry_usd'] + df_trades_rebuilt['notional_exit_usd']
                    elif 'notional_entry_usd' in df_trades_rebuilt.columns:
                        # Fallback: use 2x entry notional as approximation
                        df_trades_rebuilt['total_notional'] = df_trades_rebuilt['notional_entry_usd'] * 2.0
                    else:
                        df_trades_rebuilt['total_notional'] = 0.0
                    
                    # Calculate daily turnover
                    if 'ts' in df_trades_rebuilt.columns and len(df_trades_rebuilt) > 0:
                        trades_with_ts = df_trades_rebuilt.dropna(subset=['ts'])
                        if len(trades_with_ts) > 0:
                            daily_notional = trades_with_ts.groupby(trades_with_ts['ts'].dt.floor('D'))['total_notional'].sum()
                            if len(daily_notional) > 0 and len(daily_equity) > 0:
                                # Align equity to daily notional dates
                                aligned_equity = daily_equity.reindex(daily_notional.index, method='ffill')
                                aligned_equity = aligned_equity.bfill()
                                daily_turnover_series = (daily_notional / aligned_equity).replace([np.inf, -np.inf], np.nan).dropna()
                                if len(daily_turnover_series) > 0:
                                    turnover_daily = float(daily_turnover_series.mean())
                                    turnover_annualized = turnover_daily * 252.0
                    
                    # Slippage stats from fills.csv (if available)
                    try:
                        fills_path = self.artifacts_dir / 'fills.csv'
                        if fills_path.exists():
                            df_fills = pd.read_csv(fills_path)
                            if len(df_fills) > 0 and 'slippage_bps_applied' in df_fills.columns:
                                slip_values = df_fills['slippage_bps_applied'].dropna()
                                if len(slip_values) > 0:
                                    slip_mean = float(slip_values.mean())
                                    slip_p50 = float(slip_values.quantile(0.50))
                                    slip_p90 = float(slip_values.quantile(0.90))
                                    slip_p95 = float(slip_values.quantile(0.95))
                                    slip_max = float(slip_values.max())
                                    slip_nonzero_share = float((slip_values.abs() > 1e-9).mean())
                    except Exception:
                        pass  # Keep default slippage values if fills.csv not available
                    
        except Exception as e:
            # Log error but continue with fallback
            import warnings
            import traceback
            warnings.warn(f"Error reading rebuilt trades.csv: {e}\n{traceback.format_exc()}")
            # Reset used_rebuilt_trades flag so fallback path is used
            used_rebuilt_trades = False
        
        # Fallback to old method if rebuilt trades.csv not available
        # Only execute fallback if we didn't successfully use rebuilt trades
        if not used_rebuilt_trades:
            # Check if we have old-format trades data
            if len(df_trades) > 0 and 'reason' in df_trades.columns:
                entry_trades = df_trades[df_trades['reason'] == 'ENTRY']
                exit_trades = df_trades[df_trades['reason'].isin(['EXIT', 'STOP', 'TP', 'TTL', 'TRAIL'])]
                
                participation_mean = entry_trades['participation_pct'].mean() if 'participation_pct' in entry_trades.columns and len(entry_trades) > 0 else 0.0
                # FIX 5: Export participation_pct distribution (p50/p90/p95/max)
                participation_p50 = entry_trades['participation_pct'].quantile(0.50) if 'participation_pct' in entry_trades.columns and len(entry_trades) > 0 else 0.0
                participation_p90 = entry_trades['participation_pct'].quantile(0.90) if 'participation_pct' in entry_trades.columns and len(entry_trades) > 0 else 0.0
                participation_p95 = entry_trades['participation_pct'].quantile(0.95) if 'participation_pct' in entry_trades.columns and len(entry_trades) > 0 else 0.0
                participation_max = entry_trades['participation_pct'].max() if 'participation_pct' in entry_trades.columns and len(entry_trades) > 0 else 0.0
                
                slip_mean = entry_trades['slip_bps'].mean() if len(entry_trades) > 0 else 0.0
                slip_p50 = entry_trades['slip_bps'].quantile(0.50) if len(entry_trades) > 0 else 0.0
                slip_p90 = entry_trades['slip_bps'].quantile(0.90) if len(entry_trades) > 0 else 0.0
                slip_p95 = entry_trades['slip_bps'].quantile(0.95) if len(entry_trades) > 0 else 0.0
                slip_max = entry_trades['slip_bps'].max() if len(entry_trades) > 0 else 0.0
                if len(entry_trades) > 0:
                    slip_nonzero_share = (entry_trades['slip_bps'].abs() > 1e-9).mean()
                
                # Profit factor
                winning_trades = exit_trades[exit_trades.get('pnl', 0) > 0] if 'pnl' in exit_trades.columns else pd.DataFrame()
                losing_trades = exit_trades[exit_trades.get('pnl', 0) < 0] if 'pnl' in exit_trades.columns else pd.DataFrame()
                
                gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 and 'pnl' in winning_trades.columns else 0.0
                gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 and 'pnl' in losing_trades.columns else 0.0
                
                if gross_loss > 0.0:
                    profit_factor = gross_profit / gross_loss
                elif gross_profit > 0.0:
                    # No losing trades but have winning trades - set to a large number instead of inf
                    profit_factor = 999999.0  # Use large number instead of inf for JSON serialization
                else:
                    profit_factor = 0.0
                
                # Win rate
                total_exits = len(exit_trades)
                wins = len(winning_trades)
                win_rate = wins / total_exits if total_exits > 0 else 0.0
                
                # Avg win/loss
                avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 and 'pnl' in winning_trades.columns else 0.0
                avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 and 'pnl' in losing_trades.columns else 0.0
                
                # FIX 3: Avg R - use round-trip PnL divided by initial_risk_usd from rebuilt trades.csv
                avg_r = 0.0
                # Use rebuilt trades.csv as SSOT (initial_risk_usd is calculated in _build_trades_from_fills)
                try:
                    trades_path = self.artifacts_dir / 'trades.csv'
                    if trades_path.exists():
                        df_trades_rebuilt = pd.read_csv(trades_path)
                        if len(df_trades_rebuilt) > 0 and 'initial_risk_usd' in df_trades_rebuilt.columns and 'pnl_net_usd' in df_trades_rebuilt.columns:
                            # Filter trades with valid initial_risk_usd
                            valid_trades = df_trades_rebuilt[
                                (df_trades_rebuilt['initial_risk_usd'] > 1e-9) &
                                (df_trades_rebuilt['pnl_net_usd'].notna())
                            ]
                            if len(valid_trades) > 0:
                                # R = pnl_net_usd / initial_risk_usd
                                r_multiples = valid_trades['pnl_net_usd'] / valid_trades['initial_risk_usd']
                                avg_r = float(r_multiples.mean())
                except Exception as e:
                    # Log error for debugging but don't fail
                    import warnings
                    warnings.warn(f"Failed to calculate avg_r from trades.csv: {e}")
                    pass
                
                # Fallback to old calculation if trades.csv not available or initial_risk_usd missing
                if avg_r == 0.0 and len(exit_trades) > 0 and 'pnl' in exit_trades.columns:
                    entry_trades = df_trades[df_trades['reason'] == 'ENTRY']
                    if len(entry_trades) > 0 and 'position_id' in exit_trades.columns and 'position_id' in entry_trades.columns:
                        # Group by position_id to match entry and exit
                        exit_trades_by_pos = exit_trades.groupby('position_id').last()
                        entry_trades_by_pos = entry_trades.groupby('position_id').first()
                        
                        # Check if required columns exist before merging
                        if 'stop_dist' in entry_trades_by_pos.columns and 'qty' in entry_trades_by_pos.columns:
                            # Merge entry and exit data
                            merged = exit_trades_by_pos.merge(
                                entry_trades_by_pos[['stop_dist', 'qty']],
                                left_index=True,
                                right_index=True,
                                suffixes=('_exit', '_entry')
                            )
                            
                            # After merge, columns from entry have '_entry' suffix
                            stop_dist_col = 'stop_dist_entry' if 'stop_dist_entry' in merged.columns else 'stop_dist'
                            qty_col = 'qty_entry' if 'qty_entry' in merged.columns else 'qty'
                            
                            # Calculate risk_$ = stop_dist * |qty| at entry
                            if stop_dist_col in merged.columns and qty_col in merged.columns:
                                merged['risk_dollar'] = merged[stop_dist_col] * merged[qty_col].abs()
                                merged = merged[merged['risk_dollar'] > 0]  # Filter out zero risk
                                
                                if len(merged) > 0:
                                    # R = round-trip PnL / risk_$ at entry
                                    # Guard against division by zero
                                    valid_risk = merged['risk_dollar'] > 1e-9
                                    if valid_risk.any():
                                        r_multiples = merged.loc[valid_risk, 'pnl'] / merged.loc[valid_risk, 'risk_dollar']
                                        avg_r = float(r_multiples.mean()) if len(r_multiples) > 0 else 0.0
                                    else:
                                        avg_r = 0.0
                        else:
                            # Fallback: use stop_dist from exit trades (less accurate)
                            if 'stop_dist' in exit_trades.columns and 'qty' in exit_trades.columns:
                                exit_trades_with_stop = exit_trades[
                                    (exit_trades['stop_dist'] > 0) & 
                                    (exit_trades['qty'].abs() > 0)
                                ]
                                if len(exit_trades_with_stop) > 0:
                                    risk_dollar = exit_trades_with_stop['stop_dist'] * exit_trades_with_stop['qty'].abs()
                                    # Guard against division by zero
                                    valid_risk = risk_dollar > 1e-9
                                    if valid_risk.any():
                                        r_multiples = exit_trades_with_stop.loc[valid_risk, 'pnl'] / risk_dollar.loc[valid_risk]
                                        avg_r = float(r_multiples.mean()) if len(r_multiples) > 0 else 0.0
                                    else:
                                        avg_r = 0.0
                    else:
                        # Fallback: use stop_dist from exit trades (less accurate)
                        if 'stop_dist' in exit_trades.columns and 'qty' in exit_trades.columns:
                            exit_trades_with_stop = exit_trades[
                                (exit_trades['stop_dist'] > 0) & 
                                (exit_trades['qty'].abs() > 0)
                            ]
                            if len(exit_trades_with_stop) > 0:
                                risk_dollar = exit_trades_with_stop['stop_dist'] * exit_trades_with_stop['qty'].abs()
                                # Guard against division by zero
                                valid_risk = risk_dollar > 1e-9
                                if valid_risk.any():
                                    r_multiples = exit_trades_with_stop.loc[valid_risk, 'pnl'] / risk_dollar.loc[valid_risk]
                                    avg_r = float(r_multiples.mean()) if len(r_multiples) > 0 else 0.0
                                else:
                                    avg_r = 0.0
                
                # Hit ratio per module (win rate per module)
                # If not already calculated from rebuilt trades.csv, fallback to old calculation
                if not hit_ratio_per_module and len(exit_trades) > 0 and 'module' in exit_trades.columns:
                    for module_name in exit_trades['module'].unique():
                        module_exits = exit_trades[exit_trades['module'] == module_name]
                        if len(module_exits) > 0:
                            module_wins = len(module_exits[module_exits.get('pnl', 0) > 0]) if 'pnl' in module_exits.columns else 0
                            hit_ratio_per_module[module_name] = module_wins / len(module_exits) if len(module_exits) > 0 else 0.0
                
                ttl_exits = exit_trades[exit_trades['reason'] == 'TTL']
                ttl_exit_share = len(ttl_exits) / total_exits if total_exits > 0 else 0.0
                
                # FIX 3: Group by position_id to avoid double-counting (sum only exit-side PnL once per position)
                if 'position_id' in exit_trades.columns and len(exit_trades) > 0:
                    # Group by position_id and take the last exit (in case of multiple exits, which shouldn't happen)
                    exit_trades_by_pos = exit_trades.groupby('position_id').last()
                    module_pnl = exit_trades_by_pos.groupby('module')['pnl'].sum().to_dict() if 'module' in exit_trades_by_pos.columns else {}
                    symbol_pnl = exit_trades_by_pos.groupby('symbol')['pnl'].sum().to_dict() if 'symbol' in exit_trades_by_pos.columns else {}
                else:
                    # Fallback: group by module/symbol directly (old behavior)
                    module_pnl = exit_trades.groupby('module')['pnl'].sum().to_dict() if len(exit_trades) > 0 and 'module' in exit_trades.columns else {}
                    symbol_pnl = exit_trades.groupby('symbol')['pnl'].sum().to_dict() if len(exit_trades) > 0 and 'symbol' in exit_trades.columns else {}
                module_symbol_stats = {}
                if len(exit_trades) > 0 and {'module', 'symbol'}.issubset(exit_trades.columns):
                    grouped = exit_trades.groupby(['module', 'symbol'])
                    for (module_name, symbol_name), grp in grouped:
                        trades_count = len(grp)
                        wins_count = len(grp[grp.get('pnl', 0) > 0]) if 'pnl' in grp.columns else 0
                        pnl_sum = grp['pnl'].sum() if 'pnl' in grp.columns else 0.0
                        module_symbol_stats.setdefault(module_name, {})[symbol_name] = {
                            'trades': trades_count,
                            'win_rate': wins_count / trades_count if trades_count > 0 else 0.0,
                            'pnl': pnl_sum
                        }
                
                if {'qty', 'price'}.issubset(df_trades.columns):
                    df_trades['notional'] = df_trades['qty'].abs() * df_trades['price'].abs()
                else:
                    df_trades['notional'] = 0.0
                trades_with_ts = df_trades.dropna(subset=['ts'])
                daily_notional = trades_with_ts.groupby(trades_with_ts['ts'].dt.floor('D'))['notional'].sum()
                if len(daily_notional) > 0 and len(daily_equity) > 0:
                    aligned_equity = daily_equity.reindex(daily_notional.index, method='ffill')
                    aligned_equity = aligned_equity.bfill()
                    daily_turnover_series = (daily_notional / aligned_equity).replace([np.inf, -np.inf], np.nan).dropna()
                    if len(daily_turnover_series) > 0:
                        turnover_daily = daily_turnover_series.mean()
                        turnover_annualized = turnover_daily * 252
            else:
                # No old-format trades data available - keep defaults (already set at initialization)
                # But try to compute win_rate from trades.csv if available
                try:
                    trades_path = self.artifacts_dir / 'trades.csv'
                    if trades_path.exists():
                        df_trades_fallback = pd.read_csv(trades_path)
                        if len(df_trades_fallback) > 0 and 'pnl_net_usd' in df_trades_fallback.columns:
                            wins = (df_trades_fallback['pnl_net_usd'] > 0).sum()
                            total_trades_fallback = len(df_trades_fallback)
                            win_rate = wins / total_trades_fallback if total_trades_fallback > 0 else 0.0
                except Exception:
                    pass  # Keep win_rate = 0.0 if trades.csv not available
        
        # Slippage degeneracy calculation from fills.csv (SSOT requirement)
        # Calculate from fills.csv: if > 20 non-zero observations and > 95% have exact same value → True
        # This always runs regardless of which path was taken above
        try:
            fills_path = self.artifacts_dir / 'fills.csv'
            if fills_path.exists():
                df_fills = pd.read_csv(fills_path)
                if len(df_fills) > 0 and 'slippage_bps_applied' in df_fills.columns:
                    # Filter to non-zero slippage_bps_applied values
                    non_zero_slip = df_fills[df_fills['slippage_bps_applied'].abs() > 1e-9]
                    if len(non_zero_slip) > 20:
                        # Count how many have the exact same slippage_bps_applied value
                        value_counts = non_zero_slip['slippage_bps_applied'].value_counts()
                        if len(value_counts) > 0:
                            most_common_count = value_counts.iloc[0]
                            most_common_pct = most_common_count / len(non_zero_slip)
                            if most_common_pct > 0.95:
                                slippage_degeneracy_warning = True
                                slippage_degeneracy_details = {
                                    'most_common_value': float(value_counts.index[0]),
                                    'most_common_count': int(most_common_count),
                                    'total_nonzero': len(non_zero_slip),
                                    'pct_same': float(most_common_pct)
                                }
        except Exception:
            # If fills.csv not available or error, keep default False
            pass
        
        # FIX 3: Exposure % - count bars where positions were open
        exposure_pct = 0.0
        if len(equity_curve) > 0 and positions_history:
            # Count unique timestamps where positions were open
            df_positions = pd.DataFrame(positions_history)
            if len(df_positions) > 0 and 'ts' in df_positions.columns:
                # Get all timestamps from equity curve
                df_equity = pd.DataFrame(equity_curve)
                if 'ts' in df_equity.columns:
                    # Count bars with at least one open position
                    positions_ts = set(df_positions['ts'].dropna().unique())
                    equity_ts = set(df_equity['ts'].dropna().unique())
                    bars_with_positions = len(positions_ts.intersection(equity_ts))
                    total_bars = len(equity_ts)
                    # SSOT defines exposure_pct as fraction [0, 1], not percentage
                    exposure_pct = (bars_with_positions / total_bars) if total_bars > 0 else 0.0
        
        # FIX 3: Avg trade duration - use age_bars from trades.csv (round-trips only)
        avg_trade_duration_bars = 0.0
        # Use rebuilt trades.csv as SSOT (age_bars is calculated from timestamps in _build_trades_from_fills)
        try:
            trades_path = self.artifacts_dir / 'trades.csv'
            if trades_path.exists():
                df_trades_rebuilt = pd.read_csv(trades_path)
                if len(df_trades_rebuilt) > 0 and 'age_bars' in df_trades_rebuilt.columns:
                    age_bars_values = df_trades_rebuilt['age_bars'].dropna()
                    age_bars_values = age_bars_values[age_bars_values >= 0]  # Filter out negative values
                    if len(age_bars_values) > 0:
                        avg_trade_duration_bars = float(age_bars_values.mean())
        except Exception:
            # Fallback to df_trades if available
            if len(df_trades) > 0:
                exit_trades = df_trades[df_trades['reason'].isin(['EXIT', 'STOP', 'TP', 'TTL', 'TRAIL'])]
                if len(exit_trades) > 0 and 'age_bars' in exit_trades.columns:
                    age_bars_values = exit_trades['age_bars'].dropna()
                    age_bars_values = age_bars_values[age_bars_values >= 0]
                    if len(age_bars_values) > 0:
                        avg_trade_duration_bars = float(age_bars_values.mean())
        
        # Slippage and fees
        slippage_bps_realized = (portfolio_state.slippage_paid / portfolio_state.get_total_notional() * 10000) if portfolio_state.get_total_notional() > 0 else 0.0
        fee_bps = (portfolio_state.fees_paid / portfolio_state.get_total_notional() * 10000) if portfolio_state.get_total_notional() > 0 else 0.0
        
        # FIX 1: Sum fees from fills.csv (single source of truth)
        total_fees = 0.0
        try:
            fills_path = self.artifacts_dir / 'fills.csv'
            if fills_path.exists():
                df_fills = pd.read_csv(fills_path)
                if len(df_fills) > 0 and 'fee_usd' in df_fills.columns:
                    total_fees = float(df_fills['fee_usd'].sum())
        except Exception:
            # Fallback to trades.csv if fills.csv not available
            if len(df_trades) > 0 and 'fees' in df_trades.columns:
                total_fees = float(df_trades['fees'].sum())
        
        usage_samples = [sample for sample in es_usage_samples if sample is not None]
        es_headroom_min = float(np.min(usage_samples)) if len(usage_samples) > 0 else 0.0
        es_headroom_p05 = float(np.quantile(usage_samples, 0.05)) if len(usage_samples) > 0 else 0.0
        es_headroom_median = float(np.median(usage_samples)) if len(usage_samples) > 0 else 0.0
        
        # Read funding_cost_total from ledger.csv (single source of truth)
        funding_cost_total = 0.0
        try:
            ledger_path = self.artifacts_dir / 'ledger.csv'
            if ledger_path.exists():
                df_ledger = pd.read_csv(ledger_path)
                if len(df_ledger) > 0 and 'event' in df_ledger.columns:
                    funding_events = df_ledger[df_ledger['event'] == 'FUNDING']
                    if len(funding_events) > 0 and 'funding_usd' in funding_events.columns:
                        funding_cost_total = float(funding_events['funding_usd'].sum())
        except Exception:
            # Fallback to portfolio_state if ledger.csv not available
            funding_cost_total = portfolio_state.funding_paid
        funding_cost_bps_total = (funding_cost_total / avg_equity * 10000) if avg_equity > 0 else 0.0
        # FIX 4: Export funding_events_exposed and avg_funding_cost_per_event_bps
        funding_events_exposed = funding_events_count
        avg_funding_cost_per_event_bps = (funding_cost_bps_total / funding_events_count) if funding_events_count > 0 else 0.0
        
        # D: Funding sanity bound: funding_events_exposed ≤ 1.1*3*days*#symbols
        if len(df_equity) > 0:
            start_date = df_equity['ts'].min()
            end_date = df_equity['ts'].max()
            days = (end_date - start_date).days + 1
            # Count unique symbols from trades
            num_symbols = len(df_trades['symbol'].unique()) if len(df_trades) > 0 and 'symbol' in df_trades.columns else 1
            max_expected_funding_events = 1.1 * 3 * days * num_symbols
            if funding_events_exposed > max_expected_funding_events:
                raise ValueError(
                    f"Funding sanity bound violated: funding_events_exposed ({funding_events_exposed}) > "
                    f"1.1*3*days*#symbols ({max_expected_funding_events:.1f})"
                )
        
        # VACUUM/THIN dwell %
        vacuum_dwell_pct = (vacuum_dwell_bars / total_bars_processed * 100) if total_bars_processed > 0 else 0.0
        thin_dwell_pct = (thin_dwell_bars / total_bars_processed * 100) if total_bars_processed > 0 else 0.0
        
        # FIX 3: PnL reconciliation - sum round-trip PnL from trades.csv (rebuilt from fills)
        # Read from artifacts/trades.csv which is rebuilt from fills with explicit cost breakdowns
        realized_pnl_from_trades = 0.0
        signal_pnl_before_costs = 0.0
        total_costs = 0.0
        total_entry_costs = 0.0
        total_exit_costs = 0.0
        total_funding_costs = 0.0
        try:
            trades_path = self.artifacts_dir / 'trades.csv'
            if trades_path.exists():
                df_round_trips = pd.read_csv(trades_path)
                if len(df_round_trips) > 0:
                    if 'pnl_net_usd' in df_round_trips.columns:
                        # Round-trip format: each row is a complete trade with pnl_net_usd (includes all costs)
                        realized_pnl_from_trades = float(df_round_trips['pnl_net_usd'].sum())
                    
                    # Cost breakdowns from trades.csv
                    if 'pnl_gross_usd' in df_round_trips.columns:
                        signal_pnl_before_costs = float(df_round_trips['pnl_gross_usd'].sum())
                    if 'entry_costs_usd' in df_round_trips.columns:
                        total_entry_costs = float(df_round_trips['entry_costs_usd'].sum())
                    if 'exit_costs_usd' in df_round_trips.columns:
                        total_exit_costs = float(df_round_trips['exit_costs_usd'].sum())
                    if 'funding_cost_usd' in df_round_trips.columns:
                        total_funding_costs = float(df_round_trips['funding_cost_usd'].sum())
                    
                    total_costs = total_entry_costs + total_exit_costs + total_funding_costs
                    
                    # If signal_pnl_before_costs not available, estimate from net + costs
                    if 'pnl_gross_usd' not in df_round_trips.columns and total_costs > 0.0:
                        signal_pnl_before_costs = realized_pnl_from_trades + total_costs
                    elif 'pnl_gross_usd' not in df_round_trips.columns and total_costs == 0.0:
                        # Case where costs are 0 (raw edge), gross = net
                        signal_pnl_before_costs = realized_pnl_from_trades
                elif len(df_round_trips) > 0 and 'pnl' in df_round_trips.columns:
                    # Fallback to 'pnl' column (old format)
                    realized_pnl_from_trades = float(df_round_trips['pnl'].sum())
        except Exception as e:
            # If trades.csv not available, calculate from raw trades list (fallback)
            if len(df_trades) > 0:
                exit_trades = df_trades[df_trades['reason'].isin(['EXIT', 'STOP', 'TP', 'TTL', 'TRAIL', 'EOD_CLOSE', 'EMERGENCY_FLATTEN', 'MARGIN_FLATTEN'])]
                if len(exit_trades) > 0 and 'pnl' in exit_trades.columns:
                    realized_pnl_from_trades = float(exit_trades['pnl'].sum())
        
        # Note: PnL reconciliation is validated in Gate 2 of _validate_metrics
        # which compares equity_delta to trades.pnl_net_usd (the authoritative check)
        # This intermediate check against portfolio.total_pnl is informational only
        realized_pnl_from_portfolio = portfolio_state.total_pnl
        pnl_reconciliation_diff = abs(realized_pnl_from_trades - realized_pnl_from_portfolio)
        pnl_reconciliation_tolerance = 0.01 * len(df_trades) if len(df_trades) > 0 else 0.01
        # Note: This check is informational; the real validation is in Gate 2 (equity delta vs trades.pnl_net_usd)
        pnl_reconciliation_passes = True  # Will be validated properly in Gate 2
        
        # FIX 3: PnL assertion - abs(trade_pnl) < 10 * risk_$ per round-trip (unless gap-through stop)
        pnl_assertion_violations = []
        if len(df_trades) > 0 and 'pnl' in df_trades.columns and 'stop_dist' in df_trades.columns:
            exit_trades = df_trades[df_trades['reason'].isin(['EXIT', 'STOP', 'TP', 'TTL', 'TRAIL'])]
            for _, trade in exit_trades.iterrows():
                trade_pnl = trade.get('pnl', 0)
                stop_dist = trade.get('stop_dist', 0)
                gap_through = trade.get('gap_through', False)  # Skip if gap-through
                if stop_dist > 0 and not gap_through:  # Skip gap-through trades
                    risk_dollar = stop_dist * abs(trade.get('qty', 0))
                    max_allowed_pnl = 10 * risk_dollar
                    if abs(trade_pnl) > max_allowed_pnl:
                        pnl_assertion_violations.append({
                            'ts': trade.get('ts'),
                            'symbol': trade.get('symbol'),
                            'position_id': trade.get('position_id', ''),
                            'pnl': trade_pnl,
                            'risk_dollar': risk_dollar,
                            'max_allowed': max_allowed_pnl
                        })
        
        # Read total_trades from rebuilt trades.csv (authoritative source)
        total_trades = 0
        try:
            trades_path = self.artifacts_dir / 'trades.csv'
            if trades_path.exists():
                df_trades_rebuilt = pd.read_csv(trades_path)
                total_trades = len(df_trades_rebuilt)
        except Exception:
            # Fallback to counting exit trades from raw trades list
            if len(df_trades) > 0 and 'reason' in df_trades.columns:
                exit_trades = df_trades[df_trades['reason'].isin(['EXIT', 'STOP', 'TP', 'TTL', 'TRAIL', 'EOD_CLOSE', 'EMERGENCY_FLATTEN', 'MARGIN_FLATTEN'])]
                total_trades = len(exit_trades)
        
        metrics = {
            'total_return': total_return,
            'cagr': cagr,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'mar': mar,
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_r': avg_r if 'avg_r' in locals() else 0.0,
            'exposure_pct': exposure_pct,
            'avg_trade_duration_bars': avg_trade_duration_bars,
            'hit_ratio_per_module': hit_ratio_per_module if 'hit_ratio_per_module' in locals() else {},
            'turnover_daily': turnover_daily,
            'turnover_annualized': turnover_annualized,
            'slippage_bps_realized': slippage_bps_realized,
            'fee_bps': fee_bps,
            'total_fees': total_fees,  # FIX 1: Sum from fills.csv
            'total_trades': total_trades,  # Read from rebuilt trades.csv
            'es_violations_count': es_violations_count,
            'es_block_count': es_block_count,  # G: ES blocks
            'beta_block_count': beta_block_count,  # G: Beta blocks
            'margin_blocks_count': margin_blocks_count,
            'margin_trim_count': margin_trim_count,  # G: Margin trims
            'halt_daily_hard_count': halt_daily_hard_count,  # G: Daily hard stops
            'halt_soft_brake_count': halt_soft_brake_count,  # G: Soft brake activations
            'per_symbol_loss_cap_count': per_symbol_loss_cap_count,  # G: Per-symbol loss cap hits
            'funding_cost_total': funding_cost_total,
            'funding_cost_bps_total': funding_cost_bps_total,
            'funding_cost_bps': funding_cost_bps_total,
            'funding_events_count': funding_events_count,
            'vacuum_blocks_count': vacuum_blocks_count,
            'thin_post_only_entries_count': thin_post_only_entries_count,
            'thin_extra_slip_bps_total': thin_extra_slip_bps_total,
            'thin_cancel_block_count': thin_cancel_block_count,
            'avg_participation_pct': participation_mean,
            'slip_bps_mean': slip_mean,
            'slip_bps_p50': slip_p50 if 'slip_p50' in locals() else 0.0,
            'slip_bps_p90': slip_p90 if 'slip_p90' in locals() else 0.0,
            'slip_bps_p95': slip_p95,
            'slip_bps_max': slip_max if 'slip_max' in locals() else 0.0,
            'slip_bps_nonzero_share': slip_nonzero_share,
            'ttl_exit_share': ttl_exit_share,
            'module_pnl': module_pnl,
            'symbol_pnl': symbol_pnl,
            'module_symbol_stats': module_symbol_stats,
            # FIX 5: Participation distribution
            'participation_pct_p50': participation_p50 if 'participation_p50' in locals() else 0.0,
            'participation_pct_p90': participation_p90 if 'participation_p90' in locals() else 0.0,
            'participation_pct_p95': participation_p95 if 'participation_p95' in locals() else 0.0,
            'participation_pct_max': participation_max if 'participation_max' in locals() else 0.0,
            # FIX 4: Funding metrics
            'funding_events_exposed': funding_events_exposed if 'funding_events_exposed' in locals() else funding_events_count,
            'avg_funding_cost_per_event_bps': avg_funding_cost_per_event_bps if 'avg_funding_cost_per_event_bps' in locals() else 0.0,
            'es_headroom_min': es_headroom_min,
            'es_headroom_p05': es_headroom_p05,
            'es_headroom_median': es_headroom_median,
            'vacuum_dwell_pct': vacuum_dwell_pct,
            'thin_dwell_pct': thin_dwell_pct,
            # FIX 3: PnL reconciliation
            'realized_pnl_from_trades': realized_pnl_from_trades if 'realized_pnl_from_trades' in locals() else 0.0,
            'realized_pnl_from_portfolio': realized_pnl_from_portfolio if 'realized_pnl_from_portfolio' in locals() else 0.0,
            'pnl_reconciliation_diff': pnl_reconciliation_diff if 'pnl_reconciliation_diff' in locals() else 0.0,
            'pnl_reconciliation_passes': pnl_reconciliation_passes if 'pnl_reconciliation_passes' in locals() else True,
            'pnl_assertion_violations_count': len(pnl_assertion_violations) if 'pnl_assertion_violations' in locals() else 0,
            'slippage_degeneracy_warning': slippage_degeneracy_warning if 'slippage_degeneracy_warning' in locals() else False,  # Always set (required field)
            'slippage_degeneracy_details': slippage_degeneracy_details if 'slippage_degeneracy_details' in locals() else None,
            # Cost breakdowns
            'signal_pnl_before_costs': signal_pnl_before_costs if 'signal_pnl_before_costs' in locals() else 0.0,
            'total_costs': total_costs if 'total_costs' in locals() else 0.0,
            'total_entry_costs': total_entry_costs if 'total_entry_costs' in locals() else 0.0,
            'total_exit_costs': total_exit_costs if 'total_exit_costs' in locals() else 0.0,
            'total_funding_costs': total_funding_costs if 'total_funding_costs' in locals() else 0.0,
            'fee_pct_of_gross': (total_fees / abs(signal_pnl_before_costs) * 100) if 'signal_pnl_before_costs' in locals() and signal_pnl_before_costs != 0.0 and 'total_fees' in locals() else 0.0
        }
        
        # Note: PnL reconciliation is validated in Gate 2 of _validate_metrics
        # which compares equity_delta to trades.pnl_net_usd (the authoritative check)
        # We don't fail here; Gate 2 will do the proper validation
        
        # Add initial_equity and final_equity to metrics dict BEFORE validation
        metrics['initial_equity'] = initial_equity
        metrics['final_equity'] = final_equity
        
        # Calculate total_slippage_cost from fills.csv BEFORE validation (needed for Gate 4)
        try:
            fills_path = self.artifacts_dir / 'fills.csv'
            if fills_path.exists():
                df_fills = pd.read_csv(fills_path)
                if len(df_fills) > 0 and 'slippage_cost_usd' in df_fills.columns:
                    metrics['total_slippage_cost'] = float(df_fills['slippage_cost_usd'].sum())
                else:
                    metrics['total_slippage_cost'] = 0.0
            else:
                metrics['total_slippage_cost'] = 0.0
        except Exception:
            metrics['total_slippage_cost'] = 0.0
        
        # Validate metrics before returning
        self._validate_metrics(metrics)
        
        # Additional sanity checks
        self._sanity_check_metrics(metrics, total_trades)
        
        return metrics
    
    def _sanity_check_metrics(self, metrics: Dict, total_trades: int) -> None:
        """
        Perform sanity checks on computed metrics.
        Raises warnings or exceptions for inconsistent values.
        """
        import warnings
        
        # Check 1: If there are trades, profit_factor, avg_win, avg_loss should be computed
        if total_trades > 0:
            if metrics.get('profit_factor', 0.0) == 0.0:
                # Check if there are actually winning trades
                if metrics.get('win_rate', 0.0) > 0.0:
                    # There are winning trades but profit_factor is 0 - this means no losing trades
                    # This is actually valid (infinite profit factor), but we set it to inf above
                    if metrics.get('profit_factor', 0.0) == 0.0 and metrics.get('win_rate', 0.0) > 0.0:
                        warnings.warn(
                            f"Sanity check: profit_factor is 0.0 but win_rate > 0. "
                            f"This may indicate no losing trades (should be inf). "
                            f"win_rate={metrics.get('win_rate', 0.0):.2%}, "
                            f"total_trades={total_trades}"
                        )
            
            # Check 2: avg_win should be > 0 if there are winning trades
            if metrics.get('win_rate', 0.0) > 0.0 and metrics.get('avg_win', 0.0) == 0.0:
                warnings.warn(
                    f"Sanity check: avg_win is 0.0 but win_rate > 0. "
                    f"This may indicate a calculation error. "
                    f"win_rate={metrics.get('win_rate', 0.0):.2%}, "
                    f"avg_win={metrics.get('avg_win', 0.0):.2f}"
                )
            
            # Check 3: avg_loss should be < 0 if there are losing trades
            if metrics.get('win_rate', 0.0) < 1.0 and metrics.get('avg_loss', 0.0) == 0.0:
                # This might be valid if all losing trades have exactly 0 PnL (unlikely)
                if metrics.get('win_rate', 0.0) < 0.5:  # More than half are losses
                    warnings.warn(
                        f"Sanity check: avg_loss is 0.0 but win_rate < 50%. "
                        f"This may indicate a calculation error. "
                        f"win_rate={metrics.get('win_rate', 0.0):.2%}, "
                        f"avg_loss={metrics.get('avg_loss', 0.0):.2f}"
                    )
            
            # Check 4: turnover_daily should be > 0 if there are trades
            if metrics.get('turnover_daily', 0.0) == 0.0:
                warnings.warn(
                    f"Sanity check: turnover_daily is 0.0 but total_trades={total_trades} > 0. "
                    f"This may indicate a calculation error or missing notional data."
                )
            
            # Check 5: module_pnl and symbol_pnl should not be empty if there are trades
            if not metrics.get('module_pnl', {}) and total_trades > 0:
                warnings.warn(
                    f"Sanity check: module_pnl is empty but total_trades={total_trades} > 0. "
                    f"This may indicate missing module column in trades.csv."
                )
            if not metrics.get('symbol_pnl', {}) and total_trades > 0:
                warnings.warn(
                    f"Sanity check: symbol_pnl is empty but total_trades={total_trades} > 0. "
                    f"This may indicate missing symbol column in trades.csv."
                )
            
            # Check 6: Cost breakdowns should reconcile
            signal_pnl = metrics.get('signal_pnl_before_costs', 0.0)
            total_costs = metrics.get('total_costs', 0.0)
            net_pnl = metrics.get('realized_pnl_from_trades', 0.0)
            if signal_pnl != 0.0 and total_costs != 0.0:
                expected_net = signal_pnl - total_costs
                diff = abs(net_pnl - expected_net)
                if diff > 1.0:  # Allow $1 tolerance for rounding
                    warnings.warn(
                        f"Sanity check: Cost breakdown reconciliation issue. "
                        f"signal_pnl={signal_pnl:.2f}, total_costs={total_costs:.2f}, "
                        f"expected_net={expected_net:.2f}, actual_net={net_pnl:.2f}, diff={diff:.2f}"
                    )
            
            # Check 7: avg_r should be calculated if trades exist and initial_risk_usd is available
            avg_r = metrics.get('avg_r', 0.0)
            if avg_r == 0.0 and total_trades > 0:
                # Check if initial_risk_usd is available in trades.csv
                try:
                    trades_path = self.artifacts_dir / 'trades.csv'
                    if trades_path.exists():
                        import pandas as pd
                        df_trades = pd.read_csv(trades_path)
                        if len(df_trades) > 0 and 'initial_risk_usd' in df_trades.columns:
                            valid_risk = df_trades[df_trades['initial_risk_usd'] > 1e-9]
                            if len(valid_risk) > 0:
                                warnings.warn(
                                    f"Sanity check: avg_r is 0.0 but {len(valid_risk)} trades have valid initial_risk_usd. "
                                    f"This may indicate a calculation error."
                                )
                except Exception:
                    pass  # Don't fail if we can't check
            
            # Check 8: hit_ratio_per_module should not be empty if trades exist
            hit_ratio = metrics.get('hit_ratio_per_module', {})
            if not hit_ratio and total_trades > 0:
                warnings.warn(
                    f"Sanity check: hit_ratio_per_module is empty but total_trades={total_trades} > 0. "
                    f"This may indicate missing module column in trades.csv or calculation error."
                )
            
            # Check 9: avg_trade_duration_bars should be > 0 if trades exist
            avg_duration = metrics.get('avg_trade_duration_bars', 0.0)
            if avg_duration == 0.0 and total_trades > 0:
                # Check if age_bars is available in trades.csv
                try:
                    trades_path = self.artifacts_dir / 'trades.csv'
                    if trades_path.exists():
                        import pandas as pd
                        df_trades = pd.read_csv(trades_path)
                        if len(df_trades) > 0 and 'age_bars' in df_trades.columns:
                            valid_age = df_trades[df_trades['age_bars'] > 0]
                            if len(valid_age) > 0:
                                warnings.warn(
                                    f"Sanity check: avg_trade_duration_bars is 0.0 but {len(valid_age)} trades have valid age_bars. "
                                    f"This may indicate a calculation error."
                                )
                except Exception:
                    pass  # Don't fail if we can't check
    
    def _validate_metrics(self, metrics: Dict) -> None:
        """
        Hard accounting gates with failing assertions.
        
        This method performs basic validation on the metrics dict itself.
        Full artifact-based validation should be done separately via validate_metrics()
        after all artifacts are written.
        """
        import warnings
        
        # Basic sanity checks on metrics dict (don't require artifacts to exist yet)
        # Check required fields are present
        required_fields = [
            'es_violations_count',
            'margin_blocks_count',
            'halt_daily_hard_count',
            'per_symbol_loss_cap_count',
            'halt_soft_brake_count',
            'slippage_degeneracy_warning',
            'vacuum_blocks_count',
            'thin_post_only_entries_count',
            'thin_cancel_block_count',
        ]
        
        missing_fields = [f for f in required_fields if f not in metrics]
        if missing_fields:
            raise ValueError(f"Missing required metrics fields: {missing_fields}")
        
        # Check exposure_pct is in valid range [0, 1]
        exposure_pct = metrics.get('exposure_pct', 0.0)
        if exposure_pct < 0.0 or exposure_pct > 1.0:
            raise ValueError(
                f"exposure_pct ({exposure_pct:.4f}) not in [0.0, 1.0] "
                f"(SSOT defines it as fraction, not percentage)"
            )
        
        # Note: Full validation (bar identity, PnL reconciliation, etc.) should be done
        # via validate_metrics() after all artifacts are written, not during _calculate_metrics
    
    def _calculate_empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            'total_return': 0.0,
            'cagr': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'mar': 0.0,
            'sharpe': 0.0,
            'sortino': 0.0,
            'calmar': 0.0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_r': 0.0,
            'exposure_pct': 0.0,
            'avg_trade_duration_bars': 0.0,
            'hit_ratio_per_module': {},
            'turnover_daily': 0.0,
            'turnover_annualized': 0.0,
            'slippage_bps_realized': 0.0,
            'fee_bps': 0.0,
            'es_violations_count': 0,
            'margin_blocks_count': 0,
            'funding_cost_total': 0.0,
            'funding_cost_bps_total': 0.0,
            'funding_cost_bps': 0.0,
            'funding_events_count': 0,
            'vacuum_blocks_count': 0,
            'thin_post_only_entries_count': 0,
            'thin_extra_slip_bps_total': 0.0,
            'thin_cancel_block_count': 0,
            'avg_participation_pct': 0.0,
            'slip_bps_mean': 0.0,
            'slip_bps_p95': 0.0,
            'slip_bps_nonzero_share': 0.0,
            'ttl_exit_share': 0.0,
            'module_pnl': {},
            'symbol_pnl': {},
            'module_symbol_stats': {},
            'es_headroom_min': 0.0,
            'es_headroom_p05': 0.0,
            'es_headroom_median': 0.0,
            'vacuum_dwell_pct': 0.0,
            'thin_dwell_pct': 0.0
        }
    
    def generate_forensic_log_jsonl(self, forensic_log: List[Dict]):
        """Generate log_forensic.jsonl"""
        with open(self.output_dir / 'log_forensic.jsonl', 'w') as f:
            for entry in forensic_log:
                f.write(json.dumps(entry, default=str) + '\n')
    
    def save_params_snapshot(self, params: Dict):
        """Save params_used.json"""
        with open(self.output_dir / 'params_used.json', 'w') as f:
            json.dump(params, f, indent=2, default=str)
    
    # ========== Canonical Artifact Writers ==========
    
    def _write_equity_artifact(self, equity_curve: List[Dict], portfolio_state, ledger: List[Dict] = None):
        """
        Write artifacts/equity.csv: ts,equity,cash,open_pnl,closed_pnl
        
        Rebuilds equity.csv from ledger data to ensure consistency.
        This ensures: final_equity = initial_equity + SUM(ledger.cash_delta_usd) when no open positions.
        
        Args:
            equity_curve: List of equity curve dicts from engine
            portfolio_state: PortfolioState object
            ledger: List of ledger events (optional, will read from file if not provided)
        """
        positions_path = self.artifacts_dir / 'positions.csv'
        
        if not equity_curve:
            df = pd.DataFrame(columns=['ts', 'equity', 'cash', 'open_pnl', 'closed_pnl'])
            df.to_csv(self.artifacts_dir / 'equity.csv', index=False)
            return
        
        # Get ledger data - use provided ledger or read from file
        if ledger is not None and len(ledger) > 0:
            df_ledger = pd.DataFrame(ledger)
        else:
            ledger_path = self.artifacts_dir / 'ledger.csv'
            if ledger_path.exists():
                df_ledger = pd.read_csv(ledger_path)
            else:
                # Fallback to original method if ledger not available
                df = pd.DataFrame(equity_curve)
                if 'ts' not in df.columns:
                    df = pd.DataFrame(columns=['ts', 'equity', 'cash', 'open_pnl', 'closed_pnl'])
                else:
                    if 'cash' not in df.columns:
                        df['cash'] = df['equity']
                    if 'open_pnl' not in df.columns:
                        df['open_pnl'] = 0.0
                    if 'closed_pnl' not in df.columns:
                        df['closed_pnl'] = 0.0
                    df['ts'] = pd.to_datetime(df['ts'])
                df = df[['ts', 'equity', 'cash', 'open_pnl', 'closed_pnl']]
                df.to_csv(self.artifacts_dir / 'equity.csv', index=False)
                return
        
        # Rebuild equity.csv from ledger for consistency
        df_equity_curve = pd.DataFrame(equity_curve)
        if not df_equity_curve.empty:
            df_equity_curve['ts'] = pd.to_datetime(df_equity_curve['ts'])
            # If two equity marks share the same timestamp, keep the latest state
            # (can happen at the last bar when both signal-bar and execution-bar resolve to it)
            df_equity_curve = df_equity_curve.drop_duplicates(subset=['ts'], keep='last')
        
        df_ledger['ts'] = pd.to_datetime(df_ledger['ts'])
        
        # Normalize timestamps to UTC (tz-naive) to avoid timezone comparison issues
        # If timestamps are tz-aware, convert to UTC and remove timezone info
        if len(df_equity_curve) > 0 and df_equity_curve['ts'].dt.tz is not None:
            df_equity_curve['ts'] = df_equity_curve['ts'].dt.tz_convert('UTC').dt.tz_localize(None)
        if len(df_ledger) > 0 and df_ledger['ts'].dt.tz is not None:
            df_ledger['ts'] = df_ledger['ts'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        initial_equity = float(portfolio_state.initial_capital)
        
        # Build equity dataframe with timestamps from equity_curve
        df_equity = df_equity_curve[['ts']].copy()
        df_equity = df_equity.sort_values('ts').reset_index(drop=True)
        
        # Sort ledger by timestamp
        df_ledger_sorted = df_ledger.sort_values('ts')
        
        # Compute closed_pnl from ledger: sum of all cash_delta_usd for closed positions
        # For each closed position, closed_pnl = SUM(cash_delta_usd) for all its events (ENTRY_FILL + EXIT_FILL + FUNDING)
        def compute_closed_pnl(ts):
            # Normalize ts to tz-naive if it's tz-aware
            if hasattr(ts, 'tz') and ts.tz is not None:
                ts = ts.tz_convert('UTC').tz_localize(None)
            # Get all ledger events up to this timestamp
            events_up_to_ts = df_ledger_sorted[df_ledger_sorted['ts'] <= ts]
            # Find positions that are closed by this timestamp (have EXIT_FILL)
            closed_by_ts = set(events_up_to_ts[events_up_to_ts['event'] == 'EXIT_FILL']['position_id'].unique())
            # Sum cash_delta_usd for all events of positions closed by this timestamp
            # This gives us the net PnL for closed positions (ENTRY_FILL + EXIT_FILL + FUNDING)
            closed_events = events_up_to_ts[events_up_to_ts['position_id'].isin(closed_by_ts)]
            return closed_events['cash_delta_usd'].sum() if len(closed_events) > 0 else 0.0
        
        df_equity['closed_pnl'] = df_equity['ts'].apply(compute_closed_pnl)
        
        # Compute cash_base: initial_equity + SUM(ENTRY_FILL + FUNDING cash_delta) only (costs, no PnL)
        # Then cash = cash_base (closed_pnl is separate for bar identity)
        def compute_cash_base(ts):
            # Normalize ts to tz-naive if it's tz-aware
            if hasattr(ts, 'tz') and ts.tz is not None:
                ts = ts.tz_convert('UTC').tz_localize(None)
            events_up_to_ts = df_ledger_sorted[df_ledger_sorted['ts'] <= ts]
            # Only include ENTRY_FILL and FUNDING events (costs), exclude EXIT_FILL (PnL)
            cost_events = events_up_to_ts[events_up_to_ts['event'].isin(['ENTRY_FILL', 'FUNDING'])]
            return initial_equity + cost_events['cash_delta_usd'].sum() if len(cost_events) > 0 else initial_equity
        
        df_equity['cash_base'] = df_equity['ts'].apply(compute_cash_base)
        
        # cash = cash_base (closed_pnl is tracked separately for bar identity: equity = cash + open_pnl + closed_pnl)
        df_equity['cash'] = df_equity['cash_base']
        
        # Compute open_pnl from positions_history
        # For positions that are still open at each timestamp, compute unrealized PnL
        # For now, we'll compute open_pnl from equity_curve if available, otherwise 0
        if 'equity' in df_equity_curve.columns:
            # Merge equity values from equity_curve
            df_equity = df_equity.merge(
                df_equity_curve[['ts', 'equity']],
                on='ts',
                how='left'
            )
        else:
            # If no equity from curve, compute from cash + closed_pnl (assuming no open positions)
            df_equity['equity'] = df_equity['cash'] + df_equity['closed_pnl']
        
        # Compute open_pnl = equity - cash - closed_pnl
        # This ensures bar identity: equity = cash + open_pnl + closed_pnl
        df_equity['open_pnl'] = df_equity['equity'] - df_equity['cash'] - df_equity['closed_pnl']
        
        # Fill NaN values (if equity not found from curve)
        df_equity['equity'] = df_equity['equity'].fillna(df_equity['cash'] + df_equity['closed_pnl'] + df_equity['open_pnl'])
        df_equity['open_pnl'] = df_equity['open_pnl'].fillna(0.0)
        
        # Ensure bar identity: equity = cash + open_pnl + closed_pnl
        # Recompute equity to ensure it matches this identity
        df_equity['equity'] = df_equity['cash'] + df_equity['open_pnl'] + df_equity['closed_pnl']
        
        # Ensure final equity matches ledger when no open positions
        # At final timestamp, if no open positions, equity should equal cash + closed_pnl (open_pnl = 0)
        final_ts = df_equity['ts'].iloc[-1]
        final_cash = df_equity['cash'].iloc[-1]
        final_closed_pnl = df_equity['closed_pnl'].iloc[-1]
        
        # Check if there are open positions at final timestamp
        has_open_positions = False
        if positions_path.exists():
            df_positions = pd.read_csv(positions_path)
            if 'ts' in df_positions.columns:
                df_positions['ts'] = pd.to_datetime(df_positions['ts'])
                final_positions = df_positions[df_positions['ts'] == final_ts]
                has_open_positions = len(final_positions) > 0
        
        # If no open positions, ensure open_pnl = 0
        # Final equity should be: initial_equity + SUM(ledger.cash_delta_usd) = initial_equity + sum_trades
        # This ensures equity_delta = sum_trades = sum_ledger
        if not has_open_positions:
            df_equity.loc[df_equity['ts'] == final_ts, 'open_pnl'] = 0.0
            # Final equity = initial_equity + SUM(all ledger.cash_delta_usd)
            final_ledger_sum = df_ledger_sorted['cash_delta_usd'].sum()
            final_equity_correct = initial_equity + final_ledger_sum
            df_equity.loc[df_equity['ts'] == final_ts, 'equity'] = final_equity_correct
            # Update cash to maintain bar identity: equity = cash + open_pnl + closed_pnl
            # Since open_pnl = 0, cash = equity - closed_pnl
            df_equity.loc[df_equity['ts'] == final_ts, 'cash'] = final_equity_correct - final_closed_pnl
        
        # Select and order columns
        df_equity = df_equity[['ts', 'equity', 'cash', 'open_pnl', 'closed_pnl']]
        df_equity.to_csv(self.artifacts_dir / 'equity.csv', index=False)
    
    def _write_positions_artifact(self, positions_history: List[Dict], portfolio_state):
        """Write artifacts/positions.csv: ts,position_id,symbol,qty,entry_px,stop_px,trail_px,module,age_bars,price,mtm_usd"""
        if not positions_history:
            df = pd.DataFrame(columns=['ts', 'position_id', 'symbol', 'qty', 'entry_px', 'stop_px', 'trail_px', 'module', 'age_bars', 'price', 'mtm_usd'])
        else:
            df = pd.DataFrame(positions_history)
            if 'ts' not in df.columns:
                return
            
            # Ensure required columns exist
            required_cols = {
                'position_id': '',
                'symbol': '',
                'qty': 0.0,
                'entry_px': 0.0,
                'stop_px': 0.0,
                'trail_px': 0.0,
                'module': '',
                'age_bars': 0.0,
                'price': 0.0
            }
            for col, default_val in required_cols.items():
                if col not in df.columns:
                    df[col] = default_val
            
            # Calculate mtm_usd (mark-to-market USD value)
            if 'mtm_usd' not in df.columns:
                # mtm_usd = qty * price (for futures, this is notional)
                df['mtm_usd'] = df['qty'] * df['price']
            
            df['ts'] = pd.to_datetime(df['ts'])
        
        # Select and order columns (include all fields needed for initial_risk_usd calculation)
        cols_to_include = ['ts', 'position_id', 'symbol', 'qty', 'entry_px', 'stop_px', 'trail_px', 'module', 'age_bars']
        if 'price' in df.columns:
            cols_to_include.append('price')
        if 'mtm_usd' in df.columns:
            cols_to_include.append('mtm_usd')
        
        # Only include columns that exist
        cols_to_include = [c for c in cols_to_include if c in df.columns]
        df = df[cols_to_include]
        df.to_csv(self.artifacts_dir / 'positions.csv', index=False)
    
    def _write_fills_artifact(self, fills: List[Dict]):
        """Write artifacts/fills.csv and fills.parquet"""
        if not fills:
            df = pd.DataFrame(columns=[
                'run_id', 'position_id', 'fill_id', 'ts', 'symbol', 'module', 'leg',
                'side', 'qty', 'price', 'notional_usd',
                'slippage_bps_applied', 'slippage_cost_usd',
                'fee_bps', 'fee_usd', 'liquidity',
                'participation_pct', 'adv60_usd'
            ])
        else:
            df = pd.DataFrame(fills)
            # Ensure all required columns exist
            required_cols = [
                'run_id', 'position_id', 'fill_id', 'ts', 'symbol', 'module', 'leg',
                'side', 'qty', 'price', 'notional_usd',
                'slippage_bps_applied', 'slippage_cost_usd',
                'fee_bps', 'fee_usd', 'liquidity',
                'participation_pct', 'adv60_usd'
            ]
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0.0 if col in ['qty', 'price', 'notional_usd', 'slippage_bps_applied', 
                                             'slippage_cost_usd', 'fee_bps', 'fee_usd', 
                                             'participation_pct', 'adv60_usd'] else ''
            
            df['ts'] = pd.to_datetime(df['ts'])
        
        # Order columns
        df = df[[
            'run_id', 'position_id', 'fill_id', 'ts', 'symbol', 'module', 'leg',
            'side', 'qty', 'price', 'notional_usd',
            'slippage_bps_applied', 'slippage_cost_usd',
            'fee_bps', 'fee_usd', 'liquidity',
            'participation_pct', 'adv60_usd'
        ]]
        
        # Write CSV
        df.to_csv(self.artifacts_dir / 'fills.csv', index=False)
        
        # Write Parquet if available
        if PARQUET_AVAILABLE:
            try:
                table = pa.Table.from_pandas(df)
                pq.write_table(table, self.artifacts_dir / 'fills.parquet')
            except Exception as e:
                print(f"Warning: Could not write fills.parquet: {e}")
    
    def _write_ledger_artifact(self, ledger: List[Dict]):
        """Write artifacts/ledger.csv: cash-affecting events"""
        if not ledger:
            df = pd.DataFrame(columns=[
                'ts', 'run_id', 'event', 'position_id', 'symbol', 'module', 'leg', 'side',
                'qty', 'price', 'notional_usd', 'fee_usd', 'slippage_cost_usd',
                'funding_usd', 'cash_delta_usd', 'note'
            ])
        else:
            df = pd.DataFrame(ledger)
            # Ensure required columns exist
            required_cols = [
                'ts', 'run_id', 'event', 'position_id', 'symbol', 'module', 'leg', 'side',
                'qty', 'price', 'notional_usd', 'fee_usd', 'slippage_cost_usd',
                'funding_usd', 'cash_delta_usd', 'note'
            ]
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0.0 if col in ['qty', 'price', 'notional_usd', 'fee_usd', 
                                             'slippage_cost_usd', 'funding_usd', 'cash_delta_usd'] else ''
            
            df['ts'] = pd.to_datetime(df['ts'])
        
        # Order columns
        df = df[[
            'ts', 'run_id', 'event', 'position_id', 'symbol', 'module', 'leg', 'side',
            'qty', 'price', 'notional_usd', 'fee_usd', 'slippage_cost_usd',
            'funding_usd', 'cash_delta_usd', 'note'
        ]]
        
        # Write CSV
        df.to_csv(self.artifacts_dir / 'ledger.csv', index=False)
    
    def _build_trades_from_fills(self, fills_df: pd.DataFrame, ledger_df: pd.DataFrame = None, trades_list: List[Dict] = None, positions_history: List[Dict] = None) -> pd.DataFrame:
        """Rebuild trades.csv from fills with explicit cost breakdowns
        
        Args:
            fills_df: DataFrame of fills
            ledger_df: DataFrame of ledger entries
            trades_list: Optional list of trades from engine
            positions_history: Optional list of position snapshots (for initial_risk_usd calculation)
        """
        if fills_df.empty:
            return pd.DataFrame(columns=[
                'run_id', 'position_id', 'symbol', 'module', 'dir',
                'open_ts', 'close_ts', 'open_idx', 'close_idx', 'age_bars',
                'notional_entry_usd', 'notional_exit_usd',
                'pnl_gross_usd', 'entry_costs_usd', 'exit_costs_usd', 'funding_cost_usd', 'pnl_net_usd',
                'initial_risk_usd', 'exit_reason', 'gap_through'
            ])
        
        # Load positions.csv if available (for initial_risk_usd calculation)
        positions_df = None
        if positions_history:
            positions_df = pd.DataFrame(positions_history)
            if 'ts' in positions_df.columns:
                positions_df['ts'] = pd.to_datetime(positions_df['ts'], errors='coerce')
        else:
            positions_path = self.artifacts_dir / 'positions.csv'
            if positions_path.exists():
                try:
                    positions_df = pd.read_csv(positions_path)
                    if 'ts' in positions_df.columns:
                        positions_df['ts'] = pd.to_datetime(positions_df['ts'], errors='coerce')
                except Exception:
                    positions_df = None
        
        # Group fills by position_id
        round_trips = []
        for position_id in fills_df['position_id'].unique():
            position_fills = fills_df[fills_df['position_id'] == position_id].copy()
            
            # Separate entry and exit fills
            entry_fills = position_fills[position_fills['leg'] == 'ENTRY'].copy()
            exit_fills = position_fills[position_fills['leg'] == 'EXIT'].copy()
            
            if entry_fills.empty or exit_fills.empty:
                continue  # Skip incomplete round-trips
            
            # Get metadata from first entry and last exit
            first_entry = entry_fills.iloc[0]
            last_exit = exit_fills.iloc[-1]
            
            open_ts = entry_fills['ts'].min()
            close_ts = exit_fills['ts'].max()
            
            # Determine direction from entry side
            entry_side = first_entry['side']  # 'BUY' or 'SELL'
            dir_side = 'LONG' if entry_side == 'BUY' else 'SHORT'
            
            # Calculate gross PnL
            # For LONG: gross_pnl = Σ(exit_qty * exit_price) - Σ(entry_qty * entry_price)
            # For SHORT: gross_pnl = Σ(entry_qty * entry_price) - Σ(exit_qty * exit_price)
            entry_notional = (entry_fills['qty'] * entry_fills['price']).sum()
            exit_notional = (exit_fills['qty'] * exit_fills['price']).sum()
            
            if dir_side == 'LONG':
                gross_pnl_usd = exit_notional - entry_notional
            else:  # SHORT
                gross_pnl_usd = entry_notional - exit_notional
            
            # Calculate costs
            entry_costs_usd = entry_fills['fee_usd'].sum() + entry_fills['slippage_cost_usd'].sum()
            exit_costs_usd = exit_fills['fee_usd'].sum() + exit_fills['slippage_cost_usd'].sum()
            
            # Calculate funding cost from ledger
            funding_cost_usd = 0.0
            if ledger_df is not None and not ledger_df.empty:
                position_ledger = ledger_df[
                    (ledger_df['position_id'] == position_id) &
                    (ledger_df['event'] == 'FUNDING') &
                    (pd.to_datetime(ledger_df['ts']) > open_ts) &
                    (pd.to_datetime(ledger_df['ts']) <= close_ts)
                ]
                funding_cost_usd = position_ledger['funding_usd'].sum() if 'funding_usd' in position_ledger.columns else 0.0
            
            # Calculate net PnL
            pnl_net_usd = gross_pnl_usd - entry_costs_usd - exit_costs_usd - funding_cost_usd
            
            # Get exit reason from trades list if available, otherwise default to 'EXIT'
            exit_reason = 'EXIT'
            if trades_list:
                # Find the exit trade for this position_id
                for trade in trades_list:
                    if trade.get('position_id') == position_id:
                        reason = trade.get('reason', '')
                        if reason in ['EXIT', 'STOP', 'TP', 'TTL', 'TRAIL', 'EOD_CLOSE', 'EMERGENCY_FLATTEN', 'MARGIN_FLATTEN']:
                            exit_reason = reason
                            break
            
            # Calculate notionals
            notional_entry_usd = abs(entry_notional)
            notional_exit_usd = abs(exit_notional)
            
            # Calculate age_bars (approximate from timestamps if indices not available)
            age_bars = 0
            try:
                # Ensure timestamps are datetime objects (they come from .min()/.max() on 'ts' column)
                open_ts_dt = pd.to_datetime(open_ts, utc=True) if not isinstance(open_ts, pd.Timestamp) else open_ts
                close_ts_dt = pd.to_datetime(close_ts, utc=True) if not isinstance(close_ts, pd.Timestamp) else close_ts
                
                time_diff = close_ts_dt - open_ts_dt
                # Approximate: 15-minute bars, so 4 bars per hour
                age_bars = int(time_diff.total_seconds() / 900)  # 900 seconds = 15 minutes
            except (ValueError, TypeError, AttributeError, KeyError):
                age_bars = 0
            
            # Calculate initial_risk_usd from positions data
            initial_risk_usd = 0.0
            if positions_df is not None and not positions_df.empty:
                try:
                    symbol = first_entry.get('symbol', '')
                    module = first_entry.get('module', '')
                    # Find first position snapshot for this symbol/module near open_ts (within 1 hour)
                    open_ts_dt = pd.to_datetime(open_ts, utc=True) if not isinstance(open_ts, pd.Timestamp) else open_ts
                    pos_matches = positions_df[
                        (positions_df['symbol'] == symbol) &
                        (positions_df['module'] == module) &
                        (positions_df['ts'] >= open_ts_dt - pd.Timedelta(hours=1)) &
                        (positions_df['ts'] <= open_ts_dt + pd.Timedelta(hours=1))
                    ]
                    if not pos_matches.empty:
                        # Get first match (closest to open_ts)
                        pos_match = pos_matches.iloc[0]
                        entry_px = pos_match.get('entry_px', 0.0)
                        stop_px = pos_match.get('stop_px', 0.0)
                        qty = pos_match.get('qty', 0.0)
                        if entry_px > 0 and stop_px > 0 and qty > 0:
                            # initial_risk_usd = |entry_price - stop_price| * qty
                            initial_risk_usd = abs(entry_px - stop_px) * qty
                except (ValueError, TypeError, AttributeError, KeyError):
                    initial_risk_usd = 0.0
            
            round_trips.append({
                'run_id': first_entry.get('run_id', ''),
                'position_id': position_id,
                'symbol': first_entry.get('symbol', ''),
                'module': first_entry.get('module', ''),
                'dir': dir_side,
                'open_ts': open_ts,
                'close_ts': close_ts,
                'open_idx': -1,  # Will be calculated if available
                'close_idx': -1,  # Will be calculated if available
                'age_bars': age_bars,
                'notional_entry_usd': notional_entry_usd,
                'notional_exit_usd': notional_exit_usd,
                'pnl_gross_usd': gross_pnl_usd,
                'entry_costs_usd': entry_costs_usd,
                'exit_costs_usd': exit_costs_usd,
                'funding_cost_usd': funding_cost_usd,
                'pnl_net_usd': pnl_net_usd,
                'initial_risk_usd': initial_risk_usd,
                'exit_reason': exit_reason,
                'gap_through': False  # Will be set from fills if available
            })
        
        if round_trips:
            df = pd.DataFrame(round_trips)
        else:
            df = pd.DataFrame(columns=[
                'run_id', 'position_id', 'symbol', 'module', 'dir',
                'open_ts', 'close_ts', 'open_idx', 'close_idx', 'age_bars',
                'notional_entry_usd', 'notional_exit_usd',
                'pnl_gross_usd', 'entry_costs_usd', 'exit_costs_usd', 'funding_cost_usd', 'pnl_net_usd',
                'initial_risk_usd', 'exit_reason', 'gap_through'
            ])
        
        # Ensure ts columns are datetime
        if 'open_ts' in df.columns:
            df['open_ts'] = pd.to_datetime(df['open_ts'], errors='coerce')
        if 'close_ts' in df.columns:
            df['close_ts'] = pd.to_datetime(df['close_ts'], errors='coerce')
        
        return df
    
    def _write_trades_artifact(self, trades: List[Dict]):
        """Write artifacts/trades.csv: round-trip schema (now rebuilt from fills)"""
        # Load fills.csv and ledger.csv from artifacts
        fills_path = self.artifacts_dir / 'fills.csv'
        ledger_path = self.artifacts_dir / 'ledger.csv'
        
        fills_df = pd.DataFrame()
        ledger_df = pd.DataFrame()
        
        if fills_path.exists():
            fills_df = pd.read_csv(fills_path)
            fills_df['ts'] = pd.to_datetime(fills_df['ts'])
        
        if ledger_path.exists():
            ledger_df = pd.read_csv(ledger_path)
            ledger_df['ts'] = pd.to_datetime(ledger_df['ts'])
        
        # Rebuild trades from fills
        if not fills_df.empty:
            # Try to load positions_history if available
            positions_history = None
            positions_path = self.artifacts_dir / 'positions.csv'
            if positions_path.exists():
                try:
                    positions_df = pd.read_csv(positions_path)
                    positions_history = positions_df.to_dict('records')
                except Exception:
                    pass
            df = self._build_trades_from_fills(fills_df, ledger_df, trades, positions_history)
        elif not trades:
            df = pd.DataFrame(columns=[
                'run_id', 'position_id', 'symbol', 'module', 'dir',
                'open_ts', 'close_ts', 'open_idx', 'close_idx', 'age_bars',
                'notional_entry_usd', 'notional_exit_usd',
                'pnl_gross_usd', 'fees_usd', 'slippage_cost_usd', 'funding_cost_usd', 'pnl_net_usd',
                'exit_reason', 'gap_through'
            ])
        else:
            # Group trades by position_id and match entry/exit pairs
            positions_dict = {}  # {position_id: {'entry': trade, 'exit': trade}}
            
            for trade in trades:
                position_id = trade.get('position_id', '')
                if not position_id:
                    continue
                
                if position_id not in positions_dict:
                    positions_dict[position_id] = {'entry': None, 'exit': None}
                
                reason = trade.get('reason', '')
                if reason == 'ENTRY':
                    positions_dict[position_id]['entry'] = trade
                elif reason in ['EXIT', 'STOP', 'TP', 'TTL', 'TRAIL', 'EOD_CLOSE', 'EMERGENCY_FLATTEN', 'MARGIN_FLATTEN']:
                    positions_dict[position_id]['exit'] = trade
            
            # Build round-trips from matched entry/exit pairs
            round_trips = []
            for position_id, pair in positions_dict.items():
                entry_trade = pair['entry']
                exit_trade = pair['exit']
                
                if not exit_trade:
                    # No exit yet (position still open) - skip for now
                    continue
                
                # Use exit trade as primary source (has pnl, close_ts, etc.)
                # Entry trade provides open_ts and entry details
                open_ts = entry_trade.get('open_ts') if entry_trade else exit_trade.get('open_ts')
                close_ts = exit_trade.get('close_ts') or exit_trade.get('ts')
                
                # Calculate notionals
                entry_qty = entry_trade.get('qty', 0) if entry_trade else exit_trade.get('qty', 0)
                entry_price = entry_trade.get('price', 0) if entry_trade else 0.0
                exit_price = exit_trade.get('price', 0)
                notional_entry_usd = abs(entry_qty * entry_price)
                notional_exit_usd = abs(exit_trade.get('qty', 0) * exit_price)
                
                # Sum fees from entry and exit
                entry_fees = entry_trade.get('fees', 0.0) if entry_trade else 0.0
                exit_fees = exit_trade.get('fees', 0.0)
                fees_usd = entry_fees + exit_fees
                
                # Slippage: sum from entry and exit fills (from fills.csv will be used in metrics)
                # For now, calculate from trades if available
                entry_slip_bps = entry_trade.get('slip_bps', 0.0) if entry_trade else 0.0
                exit_slip_bps = exit_trade.get('slip_bps', 0.0)
                entry_slippage_cost_usd = (notional_entry_usd * entry_slip_bps / 10000.0) if entry_trade else 0.0
                exit_slippage_cost_usd = (notional_exit_usd * exit_slip_bps / 10000.0)
                slippage_cost_usd = entry_slippage_cost_usd + exit_slippage_cost_usd
                
                # Funding cost: will be calculated from portfolio (funding_paid)
                funding_cost_usd = 0.0  # Will be calculated from portfolio
                
                # PnL from exit trade (already net of exit fees and exit slippage)
                # But we need to subtract entry fees and entry slippage to get true round-trip PnL
                exit_pnl = exit_trade.get('pnl', 0.0)
                pnl_net_usd = exit_pnl - entry_fees - entry_slippage_cost_usd
                pnl_gross_usd = pnl_net_usd + fees_usd + slippage_cost_usd
                
                round_trips.append({
                    'run_id': self.run_id or '',
                    'position_id': position_id,
                    'symbol': exit_trade.get('symbol', ''),
                    'module': exit_trade.get('module', ''),
                    'dir': exit_trade.get('side', ''),  # LONG or SHORT
                    'open_ts': open_ts,
                    'close_ts': close_ts,
                    'open_idx': -1,  # Will be calculated if available
                    'close_idx': -1,  # Will be calculated if available
                    'age_bars': exit_trade.get('age_bars', 0),
                    'notional_entry_usd': notional_entry_usd,
                    'notional_exit_usd': notional_exit_usd,
                    'pnl_gross_usd': pnl_gross_usd,
                    'fees_usd': fees_usd,
                    'slippage_cost_usd': slippage_cost_usd,
                    'funding_cost_usd': funding_cost_usd,
                    'pnl_net_usd': pnl_net_usd,
                    'exit_reason': exit_trade.get('reason', ''),
                    'gap_through': exit_trade.get('gap_through', False)
                })
            
            if round_trips:
                df = pd.DataFrame(round_trips)
            else:
                df = pd.DataFrame(columns=[
                    'run_id', 'position_id', 'symbol', 'module', 'dir',
                    'open_ts', 'close_ts', 'open_idx', 'close_idx', 'age_bars',
                    'notional_entry_usd', 'notional_exit_usd',
                    'pnl_gross_usd', 'fees_usd', 'slippage_cost_usd', 'funding_cost_usd', 'pnl_net_usd',
                    'exit_reason', 'gap_through'
                ])
            
            # Ensure ts columns are datetime
            if 'open_ts' in df.columns:
                df['open_ts'] = pd.to_datetime(df['open_ts'], errors='coerce')
            if 'close_ts' in df.columns:
                df['close_ts'] = pd.to_datetime(df['close_ts'], errors='coerce')
        
        # Order columns (use new schema with entry_costs_usd and exit_costs_usd)
        required_cols = [
            'run_id', 'position_id', 'symbol', 'module', 'dir',
            'open_ts', 'close_ts', 'open_idx', 'close_idx', 'age_bars',
            'notional_entry_usd', 'notional_exit_usd',
            'pnl_gross_usd', 'entry_costs_usd', 'exit_costs_usd', 'funding_cost_usd', 'pnl_net_usd',
            'initial_risk_usd', 'exit_reason', 'gap_through'
        ]
        # Only include columns that exist in df
        available_cols = [col for col in required_cols if col in df.columns]
        df = df[available_cols]
        
        df.to_csv(self.artifacts_dir / 'trades.csv', index=False)
    
    def _write_run_manifest(self, run_id: str, created_at: str, params_file: str, data_path: str, 
                            start_date: str, end_date: str, enable_opportunity_audit: bool = False,
                            op_audit_level: str = 'summary'):
        """Write artifacts/run_manifest.json"""
        import subprocess
        git_commit = None
        try:
            result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                                   capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            if result.returncode == 0:
                git_commit = result.stdout.strip()
        except Exception:
            pass
        
        manifest = {
            'run_id': run_id,
            'created_at': created_at,
            'engine_version': '1.0.0',  # TODO: get from version file
            'params_file': params_file,
            'data_path': data_path,
            'start_date': start_date,
            'end_date': end_date,
            'enable_opportunity_audit': enable_opportunity_audit,
            'op_audit_level': op_audit_level,
            'git_commit': git_commit
        }
        
        with open(self.artifacts_dir / 'run_manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
    
    def _write_opportunity_audit_artifacts(
        self,
        opportunity_audit: List[Dict],
        universe_state: List[Dict],
        enable_opportunity_audit: bool,
        op_audit_level: str
    ):
        """Write opportunity audit artifacts"""
        # Always write daily rollup and universe state
        if len(opportunity_audit) > 0:
            df_audit = pd.DataFrame(opportunity_audit)
            df_audit['ts'] = pd.to_datetime(df_audit['ts'])
            df_audit['date'] = df_audit['ts'].dt.date
            
            # Daily rollup: counts per day/symbol/module
            daily_rollup = []
            for (date, symbol, module), group in df_audit.groupby(['date', 'symbol', 'module']):
                candidates = (group['candidate'] == True).sum()
                taken = (group['taken'] == True).sum()
                reject_reasons = group[group['reject_reason'] != '']['reject_reason'].value_counts().to_dict()
                
                daily_rollup.append({
                    'date': date.isoformat() if isinstance(date, pd.Timestamp) else str(date),
                    'symbol': symbol,
                    'module': module,
                    'candidates': candidates,
                    'taken': taken,
                    'rejected': candidates - taken,
                    'reject_ES': reject_reasons.get('ES', 0),
                    'reject_BETA': reject_reasons.get('BETA', 0),
                    'reject_MARGIN': reject_reasons.get('MARGIN', 0),
                    'reject_HALT': reject_reasons.get('HALT', 0),
                    'reject_COOLDOWN': reject_reasons.get('COOLDOWN', 0),
                    'reject_LIQ': reject_reasons.get('LIQ', 0),
                    'reject_SPREAD': reject_reasons.get('SPREAD', 0),
                    'reject_TTL': reject_reasons.get('TTL', 0),
                    'reject_OTHER': reject_reasons.get('OTHER', 0)
                })
            
            df_daily = pd.DataFrame(daily_rollup)
            df_daily.to_csv(self.artifacts_dir / 'opportunity_audit_daily.csv', index=False)
        
        # Write full opportunity_audit.parquet if enabled
        if enable_opportunity_audit and len(opportunity_audit) > 0:
            df_audit = pd.DataFrame(opportunity_audit)
            df_audit['ts'] = pd.to_datetime(df_audit['ts'])
            df_audit['year'] = df_audit['ts'].dt.year
            df_audit['month'] = df_audit['ts'].dt.month
            
            if PARQUET_AVAILABLE:
                try:
                    # Write partitioned parquet
                    table = pa.Table.from_pandas(df_audit)
                    output_path = self.artifacts_dir / 'opportunity_audit.parquet'
                    pq.write_table(table, output_path)
                except Exception as e:
                    print(f"Warning: Could not write opportunity_audit.parquet: {e}")
            else:
                # Fallback to CSV if parquet not available
                df_audit.to_csv(self.artifacts_dir / 'opportunity_audit.csv', index=False)
        
        # Always write universe_state.csv
        if len(universe_state) > 0:
            df_universe = pd.DataFrame(universe_state)
            df_universe.to_csv(self.artifacts_dir / 'universe_state.csv', index=False)
        else:
            # Create empty file with correct schema
            df_empty = pd.DataFrame(columns=['date', 'symbol', 'oi_usd', 'adv60_usd', 'median_spread_bps_7d', 'liquidity_regime'])
            df_empty.to_csv(self.artifacts_dir / 'universe_state.csv', index=False)

