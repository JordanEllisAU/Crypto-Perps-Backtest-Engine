"""Main backtest engine orchestrator"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import timedelta
import sys
from pathlib import Path
import uuid
from collections import defaultdict


from engine_core.config.params_loader import ParamsLoader
from engine_core.src.data.loader import DataLoader
from engine_core.src.indicators.technical import compute_all_indicators
from engine_core.src.indicators.helpers import compute_helper_indicators
from engine_core.src.indicators.avwap import compute_avwap
from engine_core.src.modules.oracle import OracleModule
from engine_core.src.modules.deception import DeceptionModule, DeceptionSignal
from engine_core.src.risk.sizing import calculate_size_multiplier, calculate_position_size, calculate_max_possible_notional, get_module_factor
from engine_core.src.risk.es_guardrails import (
    calculate_final_es,
    check_es_constraint,
    calculate_ewhs_es,
    calculate_parametric_es,
    calculate_sigma_clip_es
)
from engine_core.src.risk.margin_guard import calculate_margin_ratio, check_margin_constraints, get_trim_precedence
from engine_core.src.risk.loss_halts import LossHaltState
from engine_core.src.risk.beta_controls import check_beta_caps
from engine_core.src.risk.engine_state import EngineStateManager, TradingState
from engine_core.src.liquidity.regimes import LiquidityRegimeDetector
from engine_core.src.liquidity.seasonal import SeasonalProfile
from engine_core.src.execution.fill_model import calculate_slippage, fill_stop_run, calculate_adv_60m
from engine_core.src.execution.constraints import validate_order_constraints
from engine_core.src.execution.funding_windows import check_funding_window
from engine_core.src.execution.sequencing import EventSequencer, OrderEvent
from engine_core.src.execution.fill_recorder import FillRecorder
from engine_core.src.execution.order_manager import OrderManager, PendingOrder
from engine_core.src.portfolio.state import PortfolioState
from engine_core.src.portfolio.universe import UniverseManager
from engine_core.src.reporting import ReportGenerator
from engine_core.src.indicators.technical import sma, ema




class ReportingMixin:
    def generate_reports(self, output_dir: str = "reports", run_id: str = None, start_ts: pd.Timestamp = None, end_ts: pd.Timestamp = None):
        """Generate output reports"""
        from engine_core.src.reporting import ReportGenerator
        from datetime import datetime, UTC
        
        if run_id:
            self.run_id = run_id
        
        report_gen = ReportGenerator(output_dir, run_id=self.run_id)
        
        # Generate all reports (legacy format for backward compatibility)
        report_gen.generate_trades_csv(self.trades)
        report_gen.generate_equity_curve_csv(self.equity_curve)
        report_gen.generate_positions_csv(self.positions_history)
        report_gen.generate_forensic_log_jsonl(self.forensic_log)
        
        # Write canonical artifacts FIRST (before metrics calculation, which reads from them)
        # Write ledger first so equity can use it for consistency
        report_gen._write_ledger_artifact(self.ledger)
        report_gen._write_equity_artifact(self.equity_curve, self.portfolio, ledger=self.ledger)
        report_gen._write_positions_artifact(self.positions_history, self.portfolio)
        report_gen._write_fills_artifact(self.fills)
        # Rebuild trades.csv from fills (must be called after fills and ledger are written)
        report_gen._write_trades_artifact(self.trades)
        
        # Generate metrics (now reads from rebuilt artifacts)
        params_snapshot = self.params.snapshot()
        report_gen.generate_metrics_json(
            self.portfolio,
            self.trades,
            self.equity_curve,
            self.positions_history,
            params_snapshot,
            es_violations_count=self.es_violations_count,
            es_block_count=self.es_block_count,
            beta_block_count=self.beta_block_count,
            margin_blocks_count=self.margin_blocks_count,
            margin_trim_count=self.margin_trim_count,
            halt_daily_hard_count=self.halt_daily_hard_count,
            halt_soft_brake_count=self.halt_soft_brake_count,
            per_symbol_loss_cap_count=self.per_symbol_loss_cap_count,
            vacuum_blocks_count=self.vacuum_blocks_count,
            thin_post_only_entries_count=self.thin_post_only_entries_count,
            thin_extra_slip_bps_total=self.thin_extra_slip_bps_total,
            thin_cancel_block_count=self.thin_cancel_block_count,
            funding_events_count=self.funding_events_count,
            es_usage_samples=self.es_usage_samples,
            vacuum_dwell_bars=self.vacuum_dwell_bars,
            thin_dwell_bars=self.thin_dwell_bars,
            total_bars_processed=self.total_bars_processed
        )
        
        # Write opportunity audit artifacts (always write daily rollup and universe state)
        report_gen._write_opportunity_audit_artifacts(
            self.opportunity_audit,
            self.universe_state,
            self.enable_opportunity_audit,
            self.op_audit_level
        )
        
        # Write run manifest
        created_at = datetime.now(UTC).isoformat()
        report_gen._write_run_manifest(
            run_id=self.run_id,
            created_at=created_at,
            params_file='params_used.json',
            data_path=str(self.data_loader.data_path),
            start_date=str(start_ts) if start_ts is not None else '',
            end_date=str(end_ts) if end_ts is not None else '',
            enable_opportunity_audit=self.enable_opportunity_audit,
            op_audit_level=self.op_audit_level
        )
        report_gen.save_params_snapshot(params_snapshot)
        
        # Write outlier log
        if hasattr(self, 'outlier_log'):
             import csv
             outlier_path = Path(output_dir) / "artifacts" / "outlier_trades.csv"
             if not outlier_path.parent.exists():
                 outlier_path.parent.mkdir(parents=True, exist_ok=True)
             
             headers = ['ts', 'symbol', 'module', 'leg', 'side', 'qty', 'price', 'intended_price', 
                        'slippage_bps_applied', 'outlier_threshold_bps', 'fee_bps', 'fee_usd', 
                        'notional_usd', 'run_id', 'position_id', 'fill_id']
             
             with open(outlier_path, 'w', newline='') as f:
                 writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
                 writer.writeheader()
                 if self.outlier_log:
                     writer.writerows(self.outlier_log)
        
        # Regenerate summary.txt from metrics.json
        from scripts.generate_summary import generate_summary
        generate_summary(str(report_gen.output_dir))
    def _print_profiling_summary(self):
        """Print profiling summary"""
        import time
        total_time = time.time() - self._start_time
        self._logger.info("\n" + "="*80)
        self._logger.info("PROFILING SUMMARY")
        self._logger.info("="*80)
        self._logger.info(f"Total runtime: {total_time:.2f}s")
        self._logger.info(f"Bars processed: {self._profile_counts['bars_processed']}")
        self._logger.info(f"Signals generated: {self._profile_counts['signals_generated']}")
        self._logger.info(f"Events collected: {self._profile_counts['events_collected']}")
        self._logger.info(f"Events executed: {self._profile_counts['events_executed']}")
        self._logger.info(f"ES checks: {self._profile_counts['es_checks']}")
        self._logger.info("\nTime breakdown:")
        for key, value in self._profile_time.items():
            pct = (value / total_time * 100) if total_time > 0 else 0
            self._logger.info(f"{key:<25}: {value:>8.2f}s ({pct:>5.1f}%)")
        self._logger.info("="*80)
        
        # Print entry block summary if forensic_log exists
        self._print_entry_block_summary()
    def _print_entry_block_summary(self):
        """Print summary of why entries were blocked"""
        if not hasattr(self, 'forensic_log') or not self.forensic_log:
            return
        
        block_reasons = {}
        for entry in self.forensic_log:
            event_type = entry.get('event', '')
            if 'BLOCK' in event_type or 'FAILED' in event_type or 'QTY_ZERO' in event_type or 'NO_SIGNAL' in event_type or 'HALT' in event_type:
                reason = event_type
                block_reasons[reason] = block_reasons.get(reason, 0) + 1
        
        if block_reasons:
            self._logger.info("\n" + "=" * 80)
            self._logger.info("ENTRY BLOCK SUMMARY")
            self._logger.info("=" * 80)
            for reason, count in sorted(block_reasons.items(), key=lambda x: x[1], reverse=True):
                self._logger.info(f"  {reason}: {count}")
            self._logger.info("=" * 80)
    def _record_opportunity_audit(
        self,
        ts: pd.Timestamp,
        symbol: str,
        module: str,
        candidate: bool,
        taken: bool,
        reject_reason: str = None,
        **kwargs
    ):
        """Record opportunity audit entry"""
        if not self.enable_opportunity_audit and self.op_audit_level == 'summary' and not candidate:
            # Skip non-candidates in summary mode
            return
        
        if self.op_audit_level == 'full' and not candidate:
            # In full mode, sample non-candidates if sampling is enabled
            if self.op_audit_sample and hash(f"{symbol}{ts}") % self.op_audit_sample != 0:
                return
        
        df = self.symbol_data.get(symbol)
        if df is None:
            return
        
        # Find bar index
        bar_idx = None
        if hasattr(self, 'symbol_ts_to_idx') and symbol in self.symbol_ts_to_idx:
            bar_idx = self.symbol_ts_to_idx[symbol].get(ts, None)
        
        if bar_idx is None:
            bar_df = df[df['ts'] == ts]
            if len(bar_df) > 0:
                bar_idx = bar_df.index[0]
            else:
                return
        
        if bar_idx >= len(df):
            return
        
        bar = df.iloc[bar_idx]
        
        # Get regime and indicators
        regime = bar.get('regime', 'UNCERTAIN')
        bb_width_pct = bar.get('bb_width', 0.0)
        bb_width_vs_mean = bar.get('bb_width_vs_mean', 0.0) if 'bb_width_vs_mean' in bar else 0.0
        donchian_len = 20  # Default, strategy-specific param not in base_params.json
        adx = bar.get('adx', 0.0)
        trend_slope_z = bar.get('slope_z', 0.0)
        rv_pct = bar.get('rv_pct', 0.0)
        
        # Get risk metrics
        es_headroom = kwargs.get('es_headroom', 0.0)
        beta_net = kwargs.get('beta_net', 0.0)
        margin_ratio = kwargs.get('margin_ratio', 0.0)
        cooldown_bars = kwargs.get('cooldown_bars', 0)
        adv60_usd = kwargs.get('adv60_usd', 0.0)
        participation_est = kwargs.get('participation_est', 0.0)
        notes = kwargs.get('notes', '')
        
        self.opportunity_audit.append({
            'ts': ts,
            'symbol': symbol,
            'module': module,
            'regime': regime,
            'bb_width_pct': bb_width_pct,
            'bb_width_vs_mean': bb_width_vs_mean,
            'donchian_len': donchian_len,
            'adx': adx,
            'trend_slope_z': trend_slope_z,
            'rv_pct': rv_pct,
            'candidate': candidate,
            'taken': taken,
            'reject_reason': reject_reason or '',
            'es_headroom': es_headroom,
            'beta_net': beta_net,
            'margin_ratio': margin_ratio,
            'cooldown_bars': cooldown_bars,
            'adv60_usd': adv60_usd,
            'participation_est': participation_est,
            'notes': notes
        })
    def _record_universe_state(self, ts: pd.Timestamp, symbol: str):
        """Record daily universe state (once per day per symbol)"""
        utc_date = ts.date()
        last_date = self._last_universe_state_date.get(symbol)
        
        # Only record once per day
        if last_date is not None and last_date == utc_date:
            return
        
        self._last_universe_state_date[symbol] = utc_date
        
        df = self.symbol_data.get(symbol)
        if df is None:
            return
        
        # Find bar index
        bar_idx = None
        if hasattr(self, 'symbol_ts_to_idx') and symbol in self.symbol_ts_to_idx:
            bar_idx = self.symbol_ts_to_idx[symbol].get(ts, None)
        
        if bar_idx is None:
            bar_df = df[df['ts'] == ts]
            if len(bar_df) > 0:
                bar_idx = bar_df.index[0]
            else:
                return
        
        if bar_idx >= len(df):
            return
        
        bar = df.iloc[bar_idx]
        
        # Get OI and ADV (if available)
        oi_usd = bar.get('oi_usd', 0.0) if 'oi_usd' in bar else 0.0
        adv60_usd = calculate_adv_60m(df['notional'], bar_idx) if 'notional' in df.columns else 0.0
        
        # Calculate median spread (7-day rolling if available)
        median_spread_bps_7d = 0.0
        if bar_idx >= 7 * 4:  # At least 7 days of data
            recent_bars = df.iloc[max(0, bar_idx - 7*4):bar_idx+1]
            if 'spread_bps' in recent_bars.columns:
                median_spread_bps_7d = recent_bars['spread_bps'].median()
        
        # Get liquidity regime
        liquidity_state = self.symbol_liquidity_state.get(symbol)
        liquidity_regime = liquidity_state.regime if liquidity_state else 'NORMAL'
        
        self.universe_state.append({
            'date': utc_date.isoformat(),
            'symbol': symbol,
            'oi_usd': oi_usd,
            'adv60_usd': adv60_usd,
            'median_spread_bps_7d': median_spread_bps_7d,
            'liquidity_regime': liquidity_regime
        })
