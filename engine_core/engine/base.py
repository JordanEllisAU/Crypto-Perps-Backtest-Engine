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




class BaseBacktestEngine:
    def __init__(self, data_loader: DataLoader, params: ParamsLoader, require_liquidity_data: bool = False, stress_fees: bool = False, stress_slip: bool = False, run_id: str = None, enable_opportunity_audit: bool = False, op_audit_level: str = 'summary', op_audit_sample: int = None):
        import time
        import uuid
        self.data_loader = data_loader
        self.params = params
        self.params_dict = params.get_all()
        self.require_liquidity_data = require_liquidity_data
        self.stress_fees = stress_fees
        self.stress_slip = stress_slip
        self.run_id = run_id if run_id else str(uuid.uuid4())
        self.enable_opportunity_audit = enable_opportunity_audit
        self.op_audit_level = op_audit_level
        self.op_audit_sample = op_audit_sample

        # Cost model toggle
        self.cost_model_enabled = self.params.get('cost_model', 'enabled', default=True)
        self.outlier_log: List[Dict] = []

        # Profiling counters
        self._profile_time = {
            'prepare_data': 0.0,
            'process_bar_t': 0.0,
            'process_bar_t_plus_1': 0.0,
            'generate_signals': 0.0,
            'collect_events': 0.0,
            'execute_events': 0.0,
            'update_equity': 0.0,
            'es_checks': 0.0,
            'indicator_calcs': 0.0
        }
        self._profile_counts = {
            'bars_processed': 0,
            'signals_generated': 0,
            'events_collected': 0,
            'events_executed': 0,
            'es_checks': 0
        }
        self._start_time = time.time()
        self._logger = logging.getLogger(__name__)

        # Initialize components
        self.portfolio = PortfolioState(
            initial_capital=self.params.get_default('general', 'initial_capital_usd')
        )
        self.universe = UniverseManager(self.params_dict)
        self.order_manager = OrderManager(self.params_dict)
        self.event_sequencer = EventSequencer()
        self.liquidity_detector = LiquidityRegimeDetector(self.params_dict)
        self.seasonal_profile = SeasonalProfile(self.params_dict)
        self.loss_halt_state = LossHaltState()
        
        # Debug invariants toggle
        self.debug_invariants = self.params.get('general', 'debug_invariants') or False
        
        # Launch Punch List – Blocker #5: global trading state machine
        # Initialize state manager
        self.state_manager = EngineStateManager(initial_state=TradingState.RUNNING)
        self._output_dir = None  # Will be set in run() method
        
        self.symbol_daily_pnl = defaultdict(float)
        self.symbol_prev_prices: Dict[str, float] = {}

        # Profiling: track per-symbol timing
        self._symbol_profile: Dict[str, Dict[str, float]] = {}
        self._symbol_counts: Dict[str, Dict[str, int]] = {}
        self.portfolio_returns: List[float] = []
        self.last_equity = self.portfolio.equity
        self.current_day: Optional[pd.Timestamp] = None
        self.day_start_equity = self.portfolio.equity
        beta_params = self.params_dict.get('beta_controls', {})
        self.beta_slow_priors = beta_params.get('beta_slow_priors', {})
        self.beta_cap_net = beta_params.get('cap_net_beta_abs', 1.0)
        self.beta_cap_gross = beta_params.get('cap_gross_beta', 2.2)
        
        # Trading modules - only Oracle for validation
        self.oracle_module = OracleModule(self.params_dict)
        # DeceptionModule (lazy-init via set_deception_signals())
        self.deception_module: Optional[DeceptionModule] = None

        # Per-symbol state
        self.symbol_data: Dict[str, pd.DataFrame] = {}
        # Engine-agnostic: no regime tracking
        self.symbol_master_side: Dict[str, str] = {}
        self.symbol_liquidity_state: Dict[str, any] = {}
        self.symbol_pending_signals: Dict[str, List] = {}  # Signals waiting for confirmation
        self.symbol_last_master_side_flip: Dict[str, pd.Timestamp] = {}

        # Results
        self.trades: List[Dict] = []
        self.fills: List[Dict] = []  # Track all fills separately (entry + exit)
        self.ledger: List[Dict] = []  # Cash ledger: all cash-affecting events
        self.equity_curve: List[Dict] = []
        self.positions_history: List[Dict] = []
        self.forensic_log: List[Dict] = []
        self._fill_counter: Dict[str, int] = {}  # Track fill sequence per position_id

        # Fill/ledger recorder (extracted to keep engine focused on orchestration)
        self.fill_recorder = FillRecorder(
            self.run_id,
            self._logger,
            self.fills,
            self.ledger,
            self.outlier_log,
            self._fill_counter,
        )
        
        # Opportunity audit tracking
        self.opportunity_audit: List[Dict] = []  # Full audit records
        self.universe_state: List[Dict] = []  # Daily universe state
        self._last_universe_state_date: Dict[str, pd.Timestamp] = {}  # Track last date per symbol
        
        # Trackers
        self.es_violations_count = 0
        self.es_block_count = 0  # G: Count when ES blocks an entry
        self.beta_block_count = 0  # G: Count when beta blocks an entry
        self.margin_blocks_count = 0
        self.margin_trim_count = 0  # G: Count when margin trims occur
        self.trim_count = 0
        self.vacuum_blocks_count = 0
        self.thin_post_only_entries_count = 0
        self.thin_extra_slip_bps_total = 0.0
        self.thin_cancel_block_count = 0
        self.thin_cancel_tracker: Dict[Tuple[str, pd.Timestamp], int] = {}
        self.funding_events_count = 0
        self.halt_daily_hard_count = 0  # G: Count daily hard stops
        self.halt_soft_brake_count = 0  # G: Count soft brake activations
        self.per_symbol_loss_cap_count = 0  # G: Count per-symbol loss cap hits
        # FIX 5: Deduplication sets for counter semantics
        self._halt_daily_hard_seen = set()  # {(utc_date,)} - unique UTC days
        self._halt_soft_brake_seen = set()  # {(utc_date,)} - unique UTC days
        self._per_symbol_loss_cap_seen = set()  # {(symbol, utc_date)} - distinct symbol×UTC-day
        self.es_usage_samples: List[float] = []
        self.vacuum_dwell_bars = 0
        self.thin_dwell_bars = 0
        self.total_bars_processed = 0
        
        # Performance optimization: cache master side per day (doesn't change every 15m)
        self._master_side_cache: Dict[str, str] = {}
        
        # Performance optimization: track if we have liquidity data to skip unnecessary calls
        self._has_liquidity_data: Dict[str, bool] = {}
        for symbol in self.data_loader.get_symbols():
            liq_df = self.data_loader.get_liquidity(symbol)
            self._has_liquidity_data[symbol] = (liq_df is not None and len(liq_df) > 0)
    def set_deception_signals(self, signals: List[DeceptionSignal]) -> None:
        """Attach a DeceptionModule loaded with the bot's signals.

        Call this BEFORE run(). When general.deception_mode is True, the engine
        will route generate_signals() to this module instead of OracleModule.
        """
        self.deception_module = DeceptionModule(self.params_dict, signals)
    def prepare_symbol_data(self, symbol: str):
        """Prepare and compute indicators for symbol"""
        df = self.data_loader.get_15m_bars(symbol)
        if df is None or len(df) == 0:
            return None
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Compute technical indicators
        import time
        start_indicators = time.time()
        df = compute_all_indicators(df)
        self._profile_time['indicator_calcs'] += time.time() - start_indicators
        
        # Compute helper indicators
        df = compute_helper_indicators(df)
        
        # Compute AVWAP
        import time
        start_avwap = time.time()
        avwap_series, reanchor_flags = compute_avwap(
            df,
            ema50=df.get('ema50'),
            vol_forecast=df.get('vol_forecast'),
            vol_fast_median=df.get('vol_fast_median'),
            avwap_drift_base_bps=self.params.get_default('sizing', 'avwap_drift_base_bps') or 10.0
        )
        df['avwap'] = avwap_series
        df['avwap_reanchor'] = reanchor_flags
        self._profile_time['indicator_calcs'] += time.time() - start_avwap
        
        # Engine-agnostic: no regime classification
        # Set default regime to UNCERTAIN (neutral)
        df['regime'] = 'UNCERTAIN'
        
        # Master side is computed dynamically per bar in process_bar_t
        # Don't pre-compute here
        
        # Compute higher TF indicators if available
        daily_df = self.data_loader.get_higher_tf(symbol, 'daily')
        if daily_df is not None and len(daily_df) > 0 and 'sma50' not in daily_df.columns:
            daily_df['sma50'] = sma(daily_df['close'], 50)
            # Update in loader
            if symbol not in self.data_loader._higher_tf:
                self.data_loader._higher_tf[symbol] = {}
            self.data_loader._higher_tf[symbol]['daily'] = daily_df
        
        h4_df = self.data_loader.get_higher_tf(symbol, '4h')
        if h4_df is not None and len(h4_df) > 0 and 'ema200' not in h4_df.columns:
            h4_df['ema200'] = ema(h4_df['close'], 200)
            # Update in loader
            if symbol not in self.data_loader._higher_tf:
                self.data_loader._higher_tf[symbol] = {}
            self.data_loader._higher_tf[symbol]['4h'] = h4_df
        
        self.symbol_data[symbol] = df
        # Engine-agnostic: no regime/master_side tracking
        self.symbol_master_side[symbol] = 'NEUTRAL'
        self.symbol_pending_signals[symbol] = []
        self.symbol_last_master_side_flip[symbol] = df['ts'].iloc[0] if len(df) > 0 else None
        
        return df
    def run(self, start_ts: Optional[pd.Timestamp] = None, end_ts: Optional[pd.Timestamp] = None, output_dir: str = "reports", run_id: str = None):
        """Run backtest"""
        # Launch Punch List – Blocker #5: global trading state machine
        # Store output_dir for state persistence
        self._output_dir = output_dir
        
        # Load state from file if it exists
        engine_params = self.params_dict.get('engine', {})
        state_path_template = engine_params.get('state_persistence_path', 'runs/{run_name}/engine_state.json')
        state_path = state_path_template.replace('{run_name}', Path(output_dir).name)
        self.state_manager.load_state(state_path)
        
        if run_id:
            self.run_id = run_id
        # Get time range
        if start_ts is None or end_ts is None:
            start_ts, end_ts = self.data_loader.get_time_range()
        
        # Prepare all symbols
        symbols = self.data_loader.get_symbols()
        for symbol in symbols:
            self.prepare_symbol_data(symbol)
        
        # Get common time index (15m bars)
        # For simplicity, use first symbol's timestamps
        if len(symbols) == 0:
            return
        
        first_symbol = symbols[0]
        time_index = self.symbol_data[first_symbol]['ts']
        
        # Filter to time range
        mask = (time_index >= start_ts) & (time_index <= end_ts)
        time_index = time_index[mask]
        
        # OPTIMIZATION: Build timestamp-to-index mapping for each symbol (O(1) lookups)
        # Store as instance variable so methods can use it
        self.symbol_ts_to_idx = {}
        for symbol in symbols:
            if symbol not in self.symbol_data:
                continue
            df = self.symbol_data[symbol]
            # Create mapping: timestamp -> integer index in DataFrame
            self.symbol_ts_to_idx[symbol] = {ts: idx for idx, ts in enumerate(df['ts'])}
        
        # Main loop: process each bar
        # Note: We process bar t (signal generation) and bar t+1 (order execution)
        debug_oracle = self.params.get('general', 'debug_oracle_flow', default=False)
        if debug_oracle:
            import sys
            sys.stdout.write(f"[ORACLE DEBUG] run: Starting main loop, time_index length={len(time_index)}, symbols={symbols}\n")
            sys.stdout.flush()

        # Seed equity curve with the initial state (before any executions)
        self.equity_curve.append({
            'ts': start_ts,
            'equity': self.portfolio.equity,
            'drawdown': self.portfolio.max_drawdown,
            'drawdown_pct': self.portfolio.max_drawdown_pct,
            'daily_pnl': 0.0
        })
        self.last_equity = self.portfolio.equity

        for bar_idx, current_ts in enumerate(time_index):
            # Process each symbol for signal generation (bar t)
            for symbol in symbols:
                if symbol not in self.symbol_data or symbol not in self.symbol_ts_to_idx:
                    continue
                
                # OPTIMIZATION: Use O(1) lookup instead of O(n) filter
                if current_ts not in self.symbol_ts_to_idx[symbol]:
                    continue
                
                idx = self.symbol_ts_to_idx[symbol][current_ts]
                df = self.symbol_data[symbol]
                
                # Process bar t (signal generation) - use previous bar's close
                # FIX: For Oracle signals on first bar (idx=0), we need to process idx=0, not skip it
                oracle_mode = self.params.get('general', 'oracle_mode')
                if idx > 0:
                    prev_idx = idx - 1
                    prev_ts = df['ts'].iloc[prev_idx]
                    self.process_bar_t(symbol, prev_idx, prev_ts)
                elif idx == 0 and oracle_mode:
                    # First bar - Oracle signals should be generated here (they're created on idx=0)
                    # For Oracle, we need to process the first bar directly
                    self.process_bar_t(symbol, 0, current_ts)
            
            # Process each symbol for order execution (bar t+1)
            execution_ts = None  # The timestamp at which fills actually occur this bar
            for symbol in symbols:
                if symbol not in self.symbol_data or symbol not in self.symbol_ts_to_idx:
                    continue
                
                # OPTIMIZATION: Use O(1) lookup instead of O(n) filter
                if current_ts not in self.symbol_ts_to_idx[symbol]:
                    continue
                
                idx = self.symbol_ts_to_idx[symbol][current_ts]
                df = self.symbol_data[symbol]
                
                # Determine the fill bar. If the next bar lies beyond the user-supplied end_ts,
                # fall back to the current bar (signals generated at this close would otherwise
                # be filled with future price action outside the run window).
                if idx < len(df) - 1:
                    next_idx = idx + 1
                    next_ts = df['ts'].iloc[next_idx]
                    if next_ts > end_ts:
                        # Don't fill beyond the requested end; drop any signals generated for
                        # a future bar that will not occur inside the run window.
                        if symbol in self.symbol_pending_signals:
                            self.symbol_pending_signals[symbol] = [
                                sig for sig in self.symbol_pending_signals[symbol]
                                if getattr(sig, 'signal_ts', pd.Timestamp.min) < current_ts
                            ]
                        next_idx = idx
                        next_ts = current_ts
                else:
                    next_idx = idx
                    next_ts = current_ts
                
                # Process bar t+1 (order execution) - use current bar
                self.process_bar_t_plus_1(symbol, idx, next_idx, next_ts)
                if execution_ts is None:
                    execution_ts = next_ts
            
            # Use the execution timestamp for equity marks so the equity curve aligns with the
            # bar on which orders are filled, not the signal bar.
            if execution_ts is None:
                execution_ts = current_ts
            
            # Track VACUUM/THIN dwell (once per bar, across all symbols)
            for symbol in symbols:
                if symbol in self.symbol_liquidity_state:
                    liquidity_state = self.symbol_liquidity_state[symbol]
                    if liquidity_state:
                        if liquidity_state.regime == 'VACUUM':
                            self.vacuum_dwell_bars += 1
                        elif liquidity_state.regime == 'THIN':
                            self.thin_dwell_bars += 1
            self.total_bars_processed += len(symbols)  # Count per symbol-bar
            
            # Update equity ONCE per bar for all positions (after all symbols processed)
            # Mark to the execution bar so the equity curve timestamps match fill timestamps.
            if self.portfolio.positions:
                symbol_prices = {}
                for pos_symbol in self.portfolio.positions.keys():
                    if pos_symbol in self.symbol_data and pos_symbol in self.symbol_ts_to_idx:
                        # OPTIMIZATION: Use O(1) lookup instead of O(n) filter
                        if execution_ts in self.symbol_ts_to_idx[pos_symbol]:
                            idx = self.symbol_ts_to_idx[pos_symbol][execution_ts]
                            pos_df = self.symbol_data[pos_symbol]
                            symbol_prices[pos_symbol] = pos_df['close'].iloc[idx]
                        else:
                            # Fallback: use most recent bar up to execution_ts
                            pos_df = self.symbol_data[pos_symbol]
                            pos_before = pos_df[pos_df['ts'] <= execution_ts]
                            if len(pos_before) > 0:
                                symbol_prices[pos_symbol] = pos_before.iloc[-1]['close']

                # Update equity once with all symbol prices
                if symbol_prices and len(symbol_prices) == len(self.portfolio.positions):
                    # Only update if we have prices for all positions
                    self.portfolio.update_equity_all_positions(symbol_prices, execution_ts)
                elif len(self.portfolio.positions) == 0:
                    # If no positions, equity should equal cash (unrealized PnL = 0)
                    self.portfolio.equity = self.portfolio.cash
                else:
                    # Partial prices - update what we can
                    partial_prices = {k: v for k, v in symbol_prices.items() if k in self.portfolio.positions}
                    if partial_prices:
                        self.portfolio.update_equity_all_positions(partial_prices, execution_ts)

                # Update per-symbol mark-to-market for intraday loss halts
                for pos_symbol, position in self.portfolio.positions.items():
                    current_price = symbol_prices.get(pos_symbol)
                    if current_price is None:
                        continue  # Skip this position if price not found
                    prev_price = self.symbol_prev_prices.get(pos_symbol, position.entry_price)
                    price_delta = current_price - prev_price
                    if position.side == 'LONG':
                        pnl_delta = price_delta * position.qty
                    else:
                        pnl_delta = -price_delta * position.qty
                    self.symbol_daily_pnl[pos_symbol] += pnl_delta
                    self.symbol_prev_prices[pos_symbol] = current_price

                # Check invariants after equity update (if debug enabled)
                if self.debug_invariants and symbol_prices:
                    self._check_invariants(execution_ts, symbol_prices)

            # Track portfolio returns for ES guardrails (even when no positions)
            if self.last_equity > 0:
                ret = (self.portfolio.equity - self.last_equity) / self.last_equity
                self.portfolio_returns.append(ret)
                if len(self.portfolio_returns) > 96 * 365:
                    self.portfolio_returns = self.portfolio_returns[-96 * 365:]
            self.last_equity = self.portfolio.equity
            
            # Daily reset handling
            current_day = current_ts.normalize()
            if self.current_day is None or current_day > self.current_day:
                self.current_day = current_day
                self.day_start_equity = self.portfolio.equity
                self.symbol_daily_pnl = defaultdict(float)
                # Reset prev prices for all symbols (even if no positions)
                for symbol in symbols:
                    if symbol in self.symbol_data and symbol in self.symbol_ts_to_idx:
                        if current_ts in self.symbol_ts_to_idx[symbol]:
                            idx = self.symbol_ts_to_idx[symbol][current_ts]
                            df = self.symbol_data[symbol]
                            self.symbol_prev_prices[symbol] = df['close'].iloc[idx]
            
            # Update loss halt telemetry
            self.portfolio.daily_pnl = self.portfolio.equity - self.day_start_equity
            self.portfolio.intraday_pnl = self.portfolio.daily_pnl
            self.loss_halt_state.daily_pnl = self.portfolio.daily_pnl
            self.loss_halt_state.intraday_pnl = self.portfolio.intraday_pnl
            
            # Launch Punch List – Blocker #3: robust daily loss kill-switch
            # Check daily kill-switch on each bar
            # Get bar_idx for vol_scale calculation (use first symbol as proxy)
            vol_scale = 1.0
            if len(symbols) > 0:
                first_symbol = symbols[0]
                if first_symbol in self.symbol_ts_to_idx and current_ts in self.symbol_ts_to_idx[first_symbol]:
                    vol_scale = self.get_vol_scale(first_symbol, self.symbol_ts_to_idx[first_symbol][current_ts])
            is_triggered, flatten_on_trigger, block_new_entries = self.loss_halt_state.check_daily_kill_switch(
                equity=self.portfolio.equity,
                initial_equity=self.day_start_equity,
                vol_scale=vol_scale,
                params=self.params_dict,
                current_ts=current_ts
            )
            
            if is_triggered:
                # Set state to RISK_HALT
                self.state_manager.set_state(TradingState.RISK_HALT, f"risk:daily_kill_switch_pnl_{self.portfolio.daily_pnl:.2f}", current_ts)
                
                # Flatten all positions if enabled
                if flatten_on_trigger and len(self.portfolio.positions) > 0:
                    symbol_prices_for_flatten = {}
                    for pos_symbol in self.portfolio.positions.keys():
                        if pos_symbol in self.symbol_data and pos_symbol in self.symbol_ts_to_idx:
                            if current_ts in self.symbol_ts_to_idx[pos_symbol]:
                                idx = self.symbol_ts_to_idx[pos_symbol][current_ts]
                                pos_df = self.symbol_data[pos_symbol]
                                symbol_prices_for_flatten[pos_symbol] = pos_df['close'].iloc[idx]
                    
                    # Flatten all positions (reuse margin flatten logic)
                    for pos_symbol in list(self.portfolio.positions.keys()):
                        if pos_symbol in symbol_prices_for_flatten:
                            exit_price = symbol_prices_for_flatten[pos_symbol]
                            pos = self.portfolio.positions[pos_symbol]
                            
                            # Calculate fees and slippage
                            notional = abs(pos.qty * exit_price)
                            fee_bps = self.params.get_default('general', 'taker_fee_bps')
                            if self.stress_fees:
                                fee_bps *= 1.5
                                
                            if not self.cost_model_enabled:
                                fee_bps = 0.0
                                
                            fees = notional * (fee_bps / 10000.0)
                            
                            # Calculate slippage
                            df = self.symbol_data.get(pos_symbol)
                            if df is not None:
                                if hasattr(self, 'symbol_ts_to_idx') and pos_symbol in self.symbol_ts_to_idx:
                                    fill_idx = self.symbol_ts_to_idx[pos_symbol].get(current_ts, -1)
                                else:
                                    # Fix pandas Series boolean ambiguity: ensure fill_idx is always a scalar integer
                                    ts_matches = df[df['ts'] == current_ts]
                                    if len(ts_matches) > 0:
                                        idx_result = ts_matches.index[0]
                                        fill_idx = int(idx_result) if hasattr(idx_result, '__iter__') and not isinstance(idx_result, str) else int(idx_result)
                                    else:
                                        fill_idx = -1
                                if fill_idx >= 0 and fill_idx < len(df):
                                    fill_bar = df.iloc[fill_idx]
                                    mid_price = (fill_bar['high'] + fill_bar['low']) / 2.0 if 'high' in fill_bar and 'low' in fill_bar else exit_price
                                    slippage_params = self.params_dict.get('slippage_costs', {})
                                    slippage_bps_applied = slippage_params.get('base_slip_bps_intercept', 2.0)
                                    adv60_usd = calculate_adv_60m(df['notional'], fill_idx) if fill_idx >= 0 else 0.0
                                else:
                                    mid_price = exit_price
                                    slippage_bps_applied = 2.0
                                    adv60_usd = 0.0
                            else:
                                mid_price = exit_price
                                slippage_bps_applied = 2.0
                                adv60_usd = 0.0
                            
                            if not self.cost_model_enabled:
                                slippage_bps_applied = 0.0
                            
                            slippage_cost_usd = notional * (slippage_bps_applied / 10000.0)
                            participation_pct = (notional / adv60_usd) if adv60_usd > 0 else 0.0
                            
                            # Record exit fill
                            self.fill_recorder.record_fill(
                                position_id=pos.position_id,
                                ts=current_ts,
                                symbol=pos_symbol,
                                module=pos.module,
                                leg='EXIT',
                                side='SELL' if pos.side == 'LONG' else 'BUY',
                                qty=pos.qty,
                                price=exit_price,
                                notional_usd=notional,
                                slippage_bps_applied=slippage_bps_applied,
                                slippage_cost_usd=slippage_cost_usd,
                                fee_bps=fee_bps,
                                fee_usd=fees,
                                liquidity='taker',
                                participation_pct=participation_pct,
                                adv60_usd=adv60_usd,
                                intended_price=exit_price
                            )
                            
                            # Close position
                            closed_pos, pnl = self.portfolio.close_position(
                                pos_symbol, exit_price, current_ts, 'KILL_SWITCH_FLATTEN', fees, slippage_cost_usd
                            )
                            
                            if closed_pos:
                                # Record ledger and trade (similar to margin flatten)
                                self.fill_recorder.record_ledger_event(
                                    ts=current_ts,
                                    event='EXIT_FILL',
                                    position_id=pos.position_id,
                                    symbol=pos_symbol,
                                    module=pos.module,
                                    leg='EXIT',
                                    side='SELL' if pos.side == 'LONG' else 'BUY',
                                    qty=pos.qty,
                                    price=exit_price,
                                    notional_usd=notional,
                                    fee_usd=fees,
                                    slippage_cost_usd=slippage_cost_usd,
                                    funding_usd=0.0,
                                    cash_delta_usd=pnl,
                                    note="Kill switch flatten"
                                )
                                self.trades.append({
                                    'ts': current_ts,
                                    'symbol': pos_symbol,
                                    'side': pos.side,
                                    'module': pos.module,
                                    'qty': pos.qty,
                                    'price': exit_price,
                                    'fees': fees,
                                    'slip_bps': slippage_bps_applied,
                                    'participation_pct': participation_pct,
                                    'post_only': False,
                                    'stop_dist': abs(pos.entry_price - pos.stop_price) if pos.stop_price > 0 else 0.0,
                                    'ES_used_before': 0.0,
                                    'ES_used_after': 0.0,
                                    'reason': 'KILL_SWITCH_FLATTEN',
                                    'pnl': pnl,
                                    'position_id': pos.position_id,
                                    'open_ts': pos.entry_ts,
                                    'close_ts': current_ts,
                                    'age_bars': pos.age_bars if hasattr(pos, 'age_bars') else 0,
                                    'gap_through': False
                                })
                
                # Log kill switch trigger
                self.forensic_log.append({
                    'ts': current_ts,
                    'event': 'DAILY_KILL_SWITCH_TRIGGERED',
                    'daily_pnl': self.portfolio.daily_pnl,
                    'daily_pnl_pct': (self.portfolio.daily_pnl / self.day_start_equity * 100.0) if self.day_start_equity > 0 else 0.0,
                    'flatten_on_trigger': flatten_on_trigger,
                    'block_new_entries': block_new_entries,
                    'positions_flattened': len(self.portfolio.positions) == 0
                })
            
            # Save loss halt state after each bar
            if self._output_dir:
                risk_state_path = Path(self._output_dir) / "risk_state.json"
                self.loss_halt_state.save_state(str(risk_state_path))
            
            # Record equity curve once per bar (even when no positions)
            self.equity_curve.append({
                'ts': execution_ts,
                'equity': self.portfolio.equity,
                'drawdown': self.portfolio.max_drawdown,
                'drawdown_pct': self.portfolio.max_drawdown_pct,
                'daily_pnl': self.portfolio.daily_pnl
            })
        
        # EOD Finalizer: Force-close all open positions at end_date
        if end_ts is not None and len(self.portfolio.positions) > 0:
            # Get current prices for all open positions
            symbol_prices = {}
            for symbol in self.portfolio.positions.keys():
                df = self.symbol_data.get(symbol)
                if df is not None and len(df) > 0:
                    # Use the last bar within the run window (no look-ahead)
                    valid_df = df[df['ts'] <= end_ts]
                    if len(valid_df) > 0:
                        last_bar = valid_df.iloc[-1]
                        symbol_prices[symbol] = last_bar['close']
            
            # Force-close all positions
            for pos_symbol in list(self.portfolio.positions.keys()):
                if pos_symbol in symbol_prices:
                    exit_price = symbol_prices[pos_symbol]
                    pos = self.portfolio.positions[pos_symbol]
                    
                    # Calculate fees and slippage for EOD close
                    notional = abs(pos.qty * exit_price)
                    fill_is_taker = True  # EOD closes are market orders
                    fee_bps = self.params.get_default('general', 'taker_fee_bps')
                    if self.stress_fees:
                        fee_bps *= 1.5
                        
                    if not self.cost_model_enabled:
                        fee_bps = 0.0
                        
                    fees = notional * (fee_bps / 10000.0)
                    
                    # Calculate slippage using the same run-window bar
                    close_idx = -1
                    df = self.symbol_data.get(pos_symbol)
                    if df is not None and len(df) > 0:
                        valid_df = df[df['ts'] <= end_ts]
                        if len(valid_df) > 0:
                            last_bar = valid_df.iloc[-1]
                            close_idx = int(valid_df.index[-1])
                            mid_price = (last_bar['high'] + last_bar['low']) / 2.0 if 'high' in last_bar and 'low' in last_bar else exit_price
                            slippage_params = self.params_dict.get('slippage_costs', {})
                            slippage_bps_applied = slippage_params.get('base_slip_bps_intercept', 2.0)
                            adv60_usd = calculate_adv_60m(df['notional'], close_idx) if close_idx >= 0 else 0.0
                        else:
                            mid_price = exit_price
                            slippage_bps_applied = 2.0
                            adv60_usd = 0.0
                    else:
                        mid_price = exit_price
                        slippage_bps_applied = 2.0
                        adv60_usd = 0.0
                    
                    if not self.cost_model_enabled:
                        slippage_bps_applied = 0.0
                    
                    slippage_cost_usd = notional * (slippage_bps_applied / 10000.0)
                    participation_pct = (notional / adv60_usd) if adv60_usd > 0 else 0.0
                    
                    # Record exit fill
                    self.fill_recorder.record_fill(
                        position_id=pos.position_id,
                        ts=end_ts,
                        symbol=pos_symbol,
                        module=pos.module,
                        leg='EXIT',
                        side='SELL' if pos.side == 'LONG' else 'BUY',
                        qty=pos.qty,
                        price=exit_price,
                        notional_usd=notional,
                        slippage_bps_applied=slippage_bps_applied,
                        slippage_cost_usd=slippage_cost_usd,
                        fee_bps=fee_bps,
                        fee_usd=fees,
                        liquidity='taker',
                        participation_pct=participation_pct,
                        adv60_usd=adv60_usd,
                        intended_price=exit_price
                    )
                    
                    # Close position
                    closed_pos, pnl = self.portfolio.close_position(
                        pos_symbol, exit_price, end_ts, 'EOD_CLOSE', fees, slippage_cost_usd
                    )
                    
                    # Record trade
                    if closed_pos:
                        # Record EXIT_FILL ledger event
                        # Note: pnl from close_position already has fees and slippage deducted
                        self.fill_recorder.record_ledger_event(
                            ts=end_ts,
                            event='EXIT_FILL',
                            position_id=pos.position_id,
                            symbol=pos_symbol,
                            module=pos.module,
                            leg='EXIT',
                            side='SELL' if pos.side == 'LONG' else 'BUY',
                            qty=pos.qty,
                            price=exit_price,
                            notional_usd=notional,
                            fee_usd=fees,
                            slippage_cost_usd=slippage_cost_usd,
                            funding_usd=0.0,
                            cash_delta_usd=pnl,  # PnL already has fees/slippage deducted
                            note="EOD close"
                        )
                        # Calculate age_bars
                        entry_idx = pos.entry_idx if pos.entry_idx >= 0 else -1
                        if entry_idx < 0 and df is not None:
                            # Fix pandas Series boolean ambiguity: ensure entry_idx is always a scalar integer
                            ts_matches = df[df['ts'] == pos.entry_ts]
                            if len(ts_matches) > 0:
                                idx_result = ts_matches.index[0]
                                entry_idx = int(idx_result) if hasattr(idx_result, '__iter__') and not isinstance(idx_result, str) else int(idx_result)
                            else:
                                entry_idx = -1
                        # close_idx was set from the run-window bar above
                        age_bars = (close_idx - entry_idx) if entry_idx >= 0 and close_idx >= 0 else pos.age_bars if hasattr(pos, 'age_bars') else 0
                        
                        self.trades.append({
                            'ts': end_ts,
                            'symbol': pos_symbol,
                            'side': pos.side,
                            'module': pos.module,
                            'qty': pos.qty,
                            'price': exit_price,
                            'fees': fees,
                            'slip_bps': slippage_bps_applied,
                            'participation_pct': participation_pct,
                            'post_only': False,
                            'stop_dist': abs(pos.entry_price - pos.stop_price) if pos.stop_price > 0 else 0.0,
                            'ES_used_before': 0.0,
                            'ES_used_after': 0.0,
                            'reason': 'EOD_CLOSE',
                            'pnl': pnl,
                            'position_id': pos.position_id,
                            'open_ts': pos.entry_ts,
                            'close_ts': end_ts,
                            'age_bars': age_bars,
                            'gap_through': False
                        })
            
            # After EOD finalizer closes all positions, update equity one final time
            # Since all positions are closed, equity should equal cash
            if len(self.portfolio.positions) == 0:
                self.portfolio.equity = self.portfolio.cash
                # Record final equity in equity curve
                if len(self.equity_curve) > 0:
                    # Update the last entry or add a new one
                    last_entry = self.equity_curve[-1]
                    if last_entry['ts'] == end_ts:
                        last_entry['equity'] = self.portfolio.equity
                    else:
                        self.equity_curve.append({
                            'ts': end_ts,
                            'equity': self.portfolio.equity,
                            'drawdown': self.portfolio.max_drawdown,
                            'drawdown_pct': self.portfolio.max_drawdown_pct,
                            'daily_pnl': self.portfolio.daily_pnl
                        })
        
        # Final equity update before generating reports
        if len(self.portfolio.positions) == 0:
            self.portfolio.equity = self.portfolio.cash
        
        # Launch Punch List – Blocker #5: global trading state machine
        # Save state at end of run
        if self._output_dir:
            engine_params = self.params_dict.get('engine', {})
            state_path_template = engine_params.get('state_persistence_path', 'runs/{run_name}/engine_state.json')
            state_path = state_path_template.replace('{run_name}', Path(self._output_dir).name)
            self.state_manager.save_state(state_path)
        
        # Generate reports
        self.generate_reports(output_dir, run_id=run_id, start_ts=start_ts, end_ts=end_ts)

        # Print profiling summary
        self._print_profiling_summary()
    def process_bar_t(self, symbol: str, idx: int, current_ts: pd.Timestamp):
        """Process bar t (decision bar at close) - generate signals"""
        import time
        start_time = time.time()

        df = self.symbol_data[symbol]
        if idx >= len(df):
            return

        bar = df.iloc[idx]

        # Engine-agnostic: always use NEUTRAL
        master_side = 'NEUTRAL'
        df.at[df.index[idx], 'master_side'] = master_side

        # Update liquidity regime (skip if no liquidity data to avoid unnecessary calculations)
        if self._has_liquidity_data.get(symbol, False):
            start_liq = time.time()
            self.update_liquidity_regime(symbol, idx, current_ts)
            self._profile_time['update_liquidity'] = self._profile_time.get('update_liquidity', 0) + (time.time() - start_liq)

        # Check loss halts
        start_loss = time.time()
        vol_scale = self.get_vol_scale(symbol, idx)
        self.loss_halt_state.update_daily_pnl(
            current_ts, self.portfolio.daily_pnl, vol_scale, self.params_dict
        )
        self._profile_time['loss_halt_checks'] = self._profile_time.get('loss_halt_checks', 0) + (time.time() - start_loss)

        # Generate signals (will be evaluated on t+1)
        start_signals = time.time()
        # Engine-agnostic: generate signals if not halted (Oracle mode always enabled for validation)
        oracle_mode = self.params.get('general', 'oracle_mode')
        debug_oracle = self.params.get('general', 'debug_oracle_flow', default=False)
        if debug_oracle:
            import sys
            msg = f"[ORACLE DEBUG] process_bar_t: symbol={symbol}, idx={idx}, master_side={master_side}, oracle_mode={oracle_mode}, halt_manual={self.loss_halt_state.halt_manual}\n"
            sys.stdout.write(msg)
            sys.stdout.flush()
            # Also write to file for debugging
            try:
                with open('artifacts/oracle_debug.log', 'a') as f:
                    f.write(msg)
            except OSError:
                self._logger.debug("Failed to write oracle debug log")
        if not self.loss_halt_state.halt_manual:
            if debug_oracle:
                self._logger.info(f"[ORACLE DEBUG] process_bar_t: Calling generate_signals")
            self.generate_signals(symbol, idx, current_ts, master_side)
        elif debug_oracle:
            self._logger.info(f"[ORACLE DEBUG] process_bar_t: SKIPPING generate_signals (halted)")
        self._profile_time['signal_generation'] = self._profile_time.get('signal_generation', 0) + (time.time() - start_signals)

        self._profile_time['process_bar_t'] += time.time() - start_time
        self._profile_counts['bars_processed'] += 1
    def process_bar_t_plus_1(
        self, symbol: str, signal_idx: int, fill_idx: int, fill_ts: pd.Timestamp
    ):
        """Process bar t+1 (order simulation bar) - execute orders using t+1 OHLC"""
        df = self.symbol_data[symbol]
        if fill_idx >= len(df):
            return
        
        fill_bar = df.iloc[fill_idx]
        
        # Check funding windows
        funding_throttle = self.params.get_default('general', 'funding_throttle_minutes')
        squeeze_disable = self.params.get_default('general', 'squeeze_disable_minutes')
        
        funding_window = check_funding_window(fill_ts, funding_throttle, squeeze_disable)
        
        # Collect all events for this symbol
        events = []
        
        # 1. Stops first (adverse_first)
        events.extend(self.collect_stop_events(symbol, fill_bar, fill_ts))

        # 1b. DeceptionModule exit lattice (TP1/TP2/TSL) — checked after SL
        # because same-bar pessimism is handled inside the method.
        events.extend(self.collect_deception_exit_events(symbol, fill_bar, fill_ts, fill_idx))
        
        # 2. New entries (ORACLE signals only in Model-1)
        # ORACLE signals bypass funding window blocks
        oracle_mode = self.params.get('general', 'oracle_mode')
        if not funding_window['block_entries'] or oracle_mode:
            try:
                new_entry_events = self.collect_new_entry_events(
                    symbol, signal_idx, fill_idx, fill_bar, fill_ts, funding_window
                )
                if new_entry_events is not None:
                    events.extend(new_entry_events)
            except Exception as e:
                self._logger.info(f"WARNING: Error collecting new entry events for {symbol}: {e}")
                # Continue without new entry events
        
        # 3. Trails: tighten only
        events.extend(self.collect_trail_events(symbol, fill_bar, fill_ts))
        
        # 4. TTL/Expiry: generic TTL handling
        events.extend(self.collect_ttl_events(symbol, fill_idx, fill_ts))
        
        # 5. Stale: unfilled entries aged > 3 bars → cancel
        events.extend(self.collect_stale_events(symbol, fill_idx, fill_ts))
        
        # Sequence and execute events
        sequenced_events = self.event_sequencer.sequence_events(events)
        self.execute_events(symbol, sequenced_events, fill_bar, fill_ts)
        
        # FIX 1: Enforce equity >= 0 after event batch
        # Update portfolio equity for ALL positions using correct prices for each symbol
        # Collect current prices for all symbols with open positions
        symbol_prices = {}
        for pos_symbol in self.portfolio.positions.keys():
            if pos_symbol in self.symbol_data:
                pos_df = self.symbol_data[pos_symbol]
                # OPTIMIZATION: Use O(1) lookup if mapping available
                if hasattr(self, 'symbol_ts_to_idx') and pos_symbol in self.symbol_ts_to_idx:
                    if fill_ts in self.symbol_ts_to_idx[pos_symbol]:
                        idx = self.symbol_ts_to_idx[pos_symbol][fill_ts]
                        if idx < len(pos_df):
                            symbol_prices[pos_symbol] = pos_df['close'].iloc[idx]
                        else:
                            valid_df = pos_df[pos_df['ts'] <= fill_ts]
                            symbol_prices[pos_symbol] = valid_df.iloc[-1]['close'] if len(valid_df) > 0 else 0.0
                    else:
                        # Fallback: use most recent bar at or before fill_ts
                        valid_df = pos_df[pos_df['ts'] <= fill_ts]
                        symbol_prices[pos_symbol] = valid_df.iloc[-1]['close'] if len(valid_df) > 0 else 0.0
                else:
                    # Fallback: DataFrame lookup
                    pos_bar = pos_df[pos_df['ts'] == fill_ts]
                    if len(pos_bar) > 0:
                        symbol_prices[pos_symbol] = pos_bar.iloc[0]['close']
                    else:
                        # Fallback: use most recent bar at or before fill_ts
                        valid_df = pos_df[pos_df['ts'] <= fill_ts]
                        if len(valid_df) > 0:
                            symbol_prices[pos_symbol] = valid_df.iloc[-1]['close']
        
        # Update equity once with all symbol prices
        if symbol_prices:
            self.portfolio.update_equity_all_positions(symbol_prices, fill_ts)
        else:
            # If no positions, equity should equal cash (unrealized PnL = 0)
            if len(self.portfolio.positions) == 0:
                self.portfolio.equity = self.portfolio.cash
        
        # Check accounting invariants if debug mode enabled
        if self.debug_invariants:
            self._check_invariants(fill_ts, symbol_prices)
        
        # FIX 1: Enforce equity >= 0 (critical invariant)
        if self.portfolio.equity < 0:
            # Emergency: equity went negative, flatten all positions
            self.forensic_log.append({
                'ts': fill_ts,
                'symbol': symbol,
                'event': 'EQUITY_NEGATIVE_EMERGENCY_FLATTEN',
                'equity_before': self.portfolio.equity,
                'positions': list(self.portfolio.positions.keys())
            })
            # Flatten all positions
            for pos_symbol in list(self.portfolio.positions.keys()):
                if pos_symbol in symbol_prices:
                    exit_price = symbol_prices[pos_symbol]
                    pos = self.portfolio.positions[pos_symbol]
                    
                    # Calculate fees and slippage for emergency flatten
                    notional = abs(pos.qty * exit_price)
                    fill_is_taker = True  # Emergency exits are market orders
                    fee_bps = self.params.get_default('general', 'taker_fee_bps')
                    if self.stress_fees:
                        fee_bps *= 1.5
                        
                    if not self.cost_model_enabled:
                        fee_bps = 0.0
                        
                    fees = notional * (fee_bps / 10000.0)
                    
                    # Calculate slippage (minimal for emergency)
                    df = self.symbol_data.get(pos_symbol)
                    if df is not None:
                        if hasattr(self, 'symbol_ts_to_idx') and pos_symbol in self.symbol_ts_to_idx:
                            fill_idx = self.symbol_ts_to_idx[pos_symbol].get(fill_ts, -1)
                        else:
                            # Fix pandas Series boolean ambiguity: ensure fill_idx is always a scalar integer
                            ts_matches = df[df['ts'] == fill_ts]
                            if len(ts_matches) > 0:
                                idx_result = ts_matches.index[0]
                                fill_idx = int(idx_result) if hasattr(idx_result, '__iter__') and not isinstance(idx_result, str) else int(idx_result)
                            else:
                                fill_idx = -1
                        if fill_idx >= 0 and fill_idx < len(df):
                            fill_bar = df.iloc[fill_idx]
                            mid_price = (fill_bar['high'] + fill_bar['low']) / 2.0 if 'high' in fill_bar and 'low' in fill_bar else exit_price
                            slippage_params = self.params_dict.get('slippage_costs', {})
                            slippage_bps_applied = slippage_params.get('base_slip_bps_intercept', 2.0)
                            adv60_usd = calculate_adv_60m(df['notional'], fill_idx) if fill_idx >= 0 else 0.0
                        else:
                            mid_price = exit_price
                            slippage_bps_applied = 2.0
                            adv60_usd = 0.0
                    else:
                        mid_price = exit_price
                        slippage_bps_applied = 2.0
                        adv60_usd = 0.0
                    
                    if not self.cost_model_enabled:
                        slippage_bps_applied = 0.0
                    
                    slippage_cost_usd = notional * (slippage_bps_applied / 10000.0)
                    participation_pct = (notional / adv60_usd) if adv60_usd > 0 else 0.0
                    
                    # Record exit fill
                    self.fill_recorder.record_fill(
                        position_id=pos.position_id,
                        ts=fill_ts,
                        symbol=pos_symbol,
                        module=pos.module,
                        leg='EXIT',
                        side='SELL' if pos.side == 'LONG' else 'BUY',
                        qty=pos.qty,
                        price=exit_price,
                        notional_usd=notional,
                        slippage_bps_applied=slippage_bps_applied,
                        slippage_cost_usd=slippage_cost_usd,
                        fee_bps=fee_bps,
                        fee_usd=fees,
                        liquidity='taker',
                        participation_pct=participation_pct,
                        adv60_usd=adv60_usd,
                        intended_price=exit_price
                    )
                    
                    closed_pos, pnl = self.portfolio.close_position(
                        pos_symbol, exit_price, fill_ts, 'EMERGENCY_FLATTEN', fees, slippage_cost_usd
                    )
                    # Record trade for reconciliation
                    if closed_pos:
                        # Record EXIT_FILL ledger event
                        # Note: pnl from close_position already has fees and slippage deducted
                        self.fill_recorder.record_ledger_event(
                            ts=fill_ts,
                            event='EXIT_FILL',
                            position_id=pos.position_id,
                            symbol=pos_symbol,
                            module=pos.module,
                            leg='EXIT',
                            side='SELL' if pos.side == 'LONG' else 'BUY',
                            qty=pos.qty,
                            price=exit_price,
                            notional_usd=notional,
                            fee_usd=fees,
                            slippage_cost_usd=slippage_cost_usd,
                            funding_usd=0.0,
                            cash_delta_usd=pnl,  # PnL already has fees/slippage deducted
                            note="Emergency flatten"
                        )
                        self.trades.append({
                            'ts': fill_ts,
                            'symbol': pos_symbol,
                            'side': pos.side,
                            'module': pos.module,
                            'qty': pos.qty,
                            'price': exit_price,
                            'fees': fees,
                            'slip_bps': slippage_bps_applied,
                            'participation_pct': participation_pct,
                            'post_only': False,
                            'stop_dist': abs(pos.entry_price - pos.stop_price) if pos.stop_price > 0 else 0.0,
                            'ES_used_before': 0.0,
                            'ES_used_after': 0.0,
                            'reason': 'EMERGENCY_FLATTEN',
                            'pnl': pnl,
                            'position_id': pos.position_id,
                            'open_ts': pos.entry_ts,
                            'close_ts': fill_ts,
                            'age_bars': pos.age_bars if hasattr(pos, 'age_bars') else 0,
                            'gap_through': False
                        })
            # Recalculate equity
            self.portfolio.equity = max(0.0, self.portfolio.cash)
            self.loss_halt_state.halt_manual = True
        
        # Launch Punch List – Blocker #1: trim deadlock safety
        from engine_core.src.risk.margin_guard import calculate_margin_ratio, check_margin_constraints, trim_with_deadlock_safety
        margin_ratio = calculate_margin_ratio(self.portfolio.positions, self.portfolio.equity)
        margin_action, should_act = check_margin_constraints(
            margin_ratio,
            block_ratio=self.params.get('margin', 'block_new_entries_ratio_pct') / 100.0,
            trim_ratio=self.params.get('margin', 'trim_target_ratio_pct') / 100.0,
            flatten_ratio=self.params.get('margin', 'flatten_ratio_pct') / 100.0
        )
        
        # If TRIM action, use bounded trim loop with deadlock safety
        if margin_action == 'TRIM' and should_act:
            margin_params = self.params_dict.get('margin', {})
            
            # Create callback to close a position
            def close_position_for_trim(symbol_to_close: str) -> bool:
                if symbol_to_close not in self.portfolio.positions:
                    return False
                if symbol_to_close not in symbol_prices:
                    return False
                
                exit_price = symbol_prices[symbol_to_close]
                pos = self.portfolio.positions[symbol_to_close]
                
                # Calculate fees and slippage
                notional = abs(pos.qty * exit_price)
                fee_bps = self.params.get_default('general', 'taker_fee_bps')
                if self.stress_fees:
                    fee_bps *= 1.5
                    
                if not self.cost_model_enabled:
                    fee_bps = 0.0
                    
                fees = notional * (fee_bps / 10000.0)
                
                # Calculate slippage
                df = self.symbol_data.get(symbol_to_close)
                if df is not None:
                    if hasattr(self, 'symbol_ts_to_idx') and symbol_to_close in self.symbol_ts_to_idx:
                        fill_idx = self.symbol_ts_to_idx[symbol_to_close].get(fill_ts, -1)
                    else:
                        # Fix pandas Series boolean ambiguity: ensure fill_idx is always a scalar integer
                        ts_matches = df[df['ts'] == fill_ts]
                        if len(ts_matches) > 0:
                            idx_result = ts_matches.index[0]
                            fill_idx = int(idx_result) if hasattr(idx_result, '__iter__') and not isinstance(idx_result, str) else int(idx_result)
                        else:
                            fill_idx = -1
                    if fill_idx >= 0 and fill_idx < len(df):
                        fill_bar = df.iloc[fill_idx]
                        mid_price = (fill_bar['high'] + fill_bar['low']) / 2.0 if 'high' in fill_bar and 'low' in fill_bar else exit_price
                        slippage_params = self.params_dict.get('slippage_costs', {})
                        slippage_bps_applied = slippage_params.get('base_slip_bps_intercept', 2.0)
                        adv60_usd = calculate_adv_60m(df['notional'], fill_idx) if fill_idx >= 0 else 0.0
                    else:
                        mid_price = exit_price
                        slippage_bps_applied = 2.0
                        adv60_usd = 0.0
                else:
                    mid_price = exit_price
                    slippage_bps_applied = 2.0
                    adv60_usd = 0.0
                
                if not self.cost_model_enabled:
                    slippage_bps_applied = 0.0
                
                slippage_cost_usd = notional * (slippage_bps_applied / 10000.0)
                participation_pct = (notional / adv60_usd) if adv60_usd > 0 else 0.0
                
                # Record exit fill
                self.fill_recorder.record_fill(
                    position_id=pos.position_id,
                    ts=fill_ts,
                    symbol=symbol_to_close,
                    module=pos.module,
                    leg='EXIT',
                    side='SELL' if pos.side == 'LONG' else 'BUY',
                    qty=pos.qty,
                    price=exit_price,
                    notional_usd=notional,
                    slippage_bps_applied=slippage_bps_applied,
                    slippage_cost_usd=slippage_cost_usd,
                    fee_bps=fee_bps,
                    fee_usd=fees,
                    liquidity='taker',
                    participation_pct=participation_pct,
                    adv60_usd=adv60_usd,
                    intended_price=exit_price
                )
                
                # Close position
                closed_pos, pnl = self.portfolio.close_position(
                    symbol_to_close, exit_price, fill_ts, 'MARGIN_TRIM', fees, slippage_cost_usd
                )
                
                if closed_pos:
                    # Record ledger event
                    self.fill_recorder.record_ledger_event(
                        ts=fill_ts,
                        event='EXIT_FILL',
                        position_id=pos.position_id,
                        symbol=symbol_to_close,
                        module=pos.module,
                        leg='EXIT',
                        side='SELL' if pos.side == 'LONG' else 'BUY',
                        qty=pos.qty,
                        price=exit_price,
                        notional_usd=notional,
                        fee_usd=fees,
                        slippage_cost_usd=slippage_cost_usd,
                        funding_usd=0.0,
                        cash_delta_usd=pnl,
                        note="Margin trim"
                    )
                    # Record trade
                    self.trades.append({
                        'ts': fill_ts,
                        'symbol': symbol_to_close,
                        'side': pos.side,
                        'module': pos.module,
                        'qty': pos.qty,
                        'price': exit_price,
                        'fees': fees,
                        'slip_bps': slippage_bps_applied,
                        'participation_pct': participation_pct,
                        'post_only': False,
                        'stop_dist': abs(pos.entry_price - pos.stop_price) if pos.stop_price > 0 else 0.0,
                        'ES_used_before': 0.0,
                        'ES_used_after': 0.0,
                        'reason': 'MARGIN_TRIM',
                        'pnl': pnl,
                        'position_id': pos.position_id,
                        'open_ts': pos.entry_ts,
                        'close_ts': fill_ts,
                        'age_bars': pos.age_bars if hasattr(pos, 'age_bars') else 0,
                        'gap_through': False
                    })
                    self.margin_trim_count += 1
                    return True
                return False
            
            # Get ES contributions for trim precedence
            es_contributions = {}  # TODO: Calculate actual ES contributions per symbol if needed
            
            # Run trim loop with deadlock safety
            should_flatten, trim_count, margin_ratio_before, margin_ratio_after = trim_with_deadlock_safety(
                self.portfolio.positions,
                self.portfolio.equity,
                margin_params,
                es_contributions=es_contributions,
                close_position_callback=close_position_for_trim
            )
            
            # Log trim result
            self.forensic_log.append({
                'ts': fill_ts,
                'symbol': symbol,
                'event': 'MARGIN_TRIM_LOOP',
                'margin_ratio_before': margin_ratio_before,
                'margin_ratio_after': margin_ratio_after,
                'trim_count': trim_count,
                'should_flatten': should_flatten
            })
            
            # If deadlock occurred, flatten all and set state to RISK_HALT
            if should_flatten:
                margin_action = 'FLATTEN'
                # Set state to RISK_HALT
                self.state_manager.set_state(TradingState.RISK_HALT, f"risk:trim_deadlock_after_{trim_count}_trims", fill_ts)
        
        if margin_action == 'FLATTEN' and should_act:
            # Launch Punch List – Blocker #1: trim deadlock safety
            # Flatten all positions and set HALT_MANUAL + RISK_HALT state
            self.forensic_log.append({
                'ts': fill_ts,
                'symbol': symbol,
                'event': 'MARGIN_FLATTEN',
                'margin_ratio': margin_ratio,
                'positions': list(self.portfolio.positions.keys())
            })
            # Set state to RISK_HALT
            self.state_manager.set_state(TradingState.RISK_HALT, f"risk:margin_flatten_margin_ratio_{margin_ratio:.4f}", fill_ts)
            for pos_symbol in list(self.portfolio.positions.keys()):
                if pos_symbol in symbol_prices:
                    exit_price = symbol_prices[pos_symbol]
                    pos = self.portfolio.positions[pos_symbol]
                    
                    # Calculate fees and slippage for margin flatten
                    notional = abs(pos.qty * exit_price)
                    fill_is_taker = True  # Margin exits are market orders
                    fee_bps = self.params.get_default('general', 'taker_fee_bps')
                    if self.stress_fees:
                        fee_bps *= 1.5
                        
                    if not self.cost_model_enabled:
                        fee_bps = 0.0
                        
                    fees = notional * (fee_bps / 10000.0)
                    
                    # Calculate slippage (minimal for margin flatten)
                    df = self.symbol_data.get(pos_symbol)
                    if df is not None:
                        if hasattr(self, 'symbol_ts_to_idx') and pos_symbol in self.symbol_ts_to_idx:
                            fill_idx = self.symbol_ts_to_idx[pos_symbol].get(fill_ts, -1)
                        else:
                            # Fix pandas Series boolean ambiguity: ensure fill_idx is always a scalar integer
                            ts_matches = df[df['ts'] == fill_ts]
                            if len(ts_matches) > 0:
                                idx_result = ts_matches.index[0]
                                fill_idx = int(idx_result) if hasattr(idx_result, '__iter__') and not isinstance(idx_result, str) else int(idx_result)
                            else:
                                fill_idx = -1
                        if fill_idx >= 0 and fill_idx < len(df):
                            fill_bar = df.iloc[fill_idx]
                            mid_price = (fill_bar['high'] + fill_bar['low']) / 2.0 if 'high' in fill_bar and 'low' in fill_bar else exit_price
                            slippage_params = self.params_dict.get('slippage_costs', {})
                            slippage_bps_applied = slippage_params.get('base_slip_bps_intercept', 2.0)
                            adv60_usd = calculate_adv_60m(df['notional'], fill_idx) if fill_idx >= 0 else 0.0
                        else:
                            mid_price = exit_price
                            slippage_bps_applied = 2.0
                            adv60_usd = 0.0
                    else:
                        mid_price = exit_price
                        slippage_bps_applied = 2.0
                        adv60_usd = 0.0
                    
                    if not self.cost_model_enabled:
                        slippage_bps_applied = 0.0
                    
                    slippage_cost_usd = notional * (slippage_bps_applied / 10000.0)
                    participation_pct = (notional / adv60_usd) if adv60_usd > 0 else 0.0
                    
                    # Record exit fill
                    self.fill_recorder.record_fill(
                        position_id=pos.position_id,
                        ts=fill_ts,
                        symbol=pos_symbol,
                        module=pos.module,
                        leg='EXIT',
                        side='SELL' if pos.side == 'LONG' else 'BUY',
                        qty=pos.qty,
                        price=exit_price,
                        notional_usd=notional,
                        slippage_bps_applied=slippage_bps_applied,
                        slippage_cost_usd=slippage_cost_usd,
                        fee_bps=fee_bps,
                        fee_usd=fees,
                        liquidity='taker',
                        participation_pct=participation_pct,
                        adv60_usd=adv60_usd,
                        intended_price=exit_price
                    )
                    
                    closed_pos, pnl = self.portfolio.close_position(
                        pos_symbol, exit_price, fill_ts, 'MARGIN_FLATTEN', fees, slippage_cost_usd
                    )
                    # Record trade for reconciliation
                    if closed_pos:
                        # Record EXIT_FILL ledger event
                        # Note: pnl from close_position already has fees and slippage deducted
                        self.fill_recorder.record_ledger_event(
                            ts=fill_ts,
                            event='EXIT_FILL',
                            position_id=pos.position_id,
                            symbol=pos_symbol,
                            module=pos.module,
                            leg='EXIT',
                            side='SELL' if pos.side == 'LONG' else 'BUY',
                            qty=pos.qty,
                            price=exit_price,
                            notional_usd=notional,
                            fee_usd=fees,
                            slippage_cost_usd=slippage_cost_usd,
                            funding_usd=0.0,
                            cash_delta_usd=pnl,  # PnL already has fees/slippage deducted
                            note="Margin flatten"
                        )
                        self.trades.append({
                            'ts': fill_ts,
                            'symbol': pos_symbol,
                            'side': pos.side,
                            'module': pos.module,
                            'qty': pos.qty,
                            'price': exit_price,
                            'fees': fees,
                            'slip_bps': slippage_bps_applied,
                            'participation_pct': participation_pct,
                            'post_only': False,
                            'stop_dist': abs(pos.entry_price - pos.stop_price) if pos.stop_price > 0 else 0.0,
                            'ES_used_before': 0.0,
                            'ES_used_after': 0.0,
                            'reason': 'MARGIN_FLATTEN',
                            'pnl': pnl,
                            'position_id': pos.position_id,
                            'open_ts': pos.entry_ts,
                            'close_ts': fill_ts,
                            'age_bars': pos.age_bars if hasattr(pos, 'age_bars') else 0,
                            'gap_through': False
                        })
            self.loss_halt_state.halt_manual = True
            # Recalculate equity after flattening
            if symbol_prices:
                self.portfolio.update_equity_all_positions(symbol_prices, fill_ts)
        
        # Update position age and extremes
        # OPTIMIZATION: Use fill_idx parameter and stored entry_idx instead of DataFrame lookups
        if symbol in self.portfolio.positions:
            pos = self.portfolio.positions[symbol]
            df = self.symbol_data[symbol]
            # Use fill_idx parameter (already available, no need to look up)
            if fill_idx is not None and fill_idx < len(df):
                # Use stored entry_idx if available, otherwise fallback to lookup
                if pos.entry_idx >= 0:
                    entry_idx = pos.entry_idx
                else:
                    # Fallback: lookup entry_idx (should rarely happen)
                    # Fix pandas Series boolean ambiguity: ensure entry_idx is always a scalar integer
                    ts_matches = df[df['ts'] == pos.entry_ts]
                    if len(ts_matches) > 0:
                        idx_result = ts_matches.index[0]
                        entry_idx = int(idx_result) if hasattr(idx_result, '__iter__') and not isinstance(idx_result, str) else int(idx_result)
                    else:
                        entry_idx = None
                
                if entry_idx is not None and entry_idx < len(df):
                    age_bars = fill_idx - entry_idx
                    self.portfolio.update_position_age(symbol, age_bars)
                    
                    # Update extremes
                    highest = df['close'].iloc[entry_idx:fill_idx+1].max() if entry_idx <= fill_idx else fill_bar['close']
                    lowest = df['close'].iloc[entry_idx:fill_idx+1].min() if entry_idx <= fill_idx else fill_bar['close']
                    self.portfolio.update_position_extremes(symbol, highest, lowest)
        
        # Apply funding costs
        self.apply_funding_costs(symbol, fill_ts)
        
        # Check invariants after funding costs (if debug enabled)
        if self.debug_invariants:
            # Recalculate symbol prices for invariant check
            check_symbol_prices = {}
            for pos_symbol in self.portfolio.positions.keys():
                if pos_symbol in symbol_prices:
                    check_symbol_prices[pos_symbol] = symbol_prices[pos_symbol]
                elif pos_symbol in self.symbol_data:
                    pos_df = self.symbol_data[pos_symbol]
                    if hasattr(self, 'symbol_ts_to_idx') and pos_symbol in self.symbol_ts_to_idx:
                        if fill_ts in self.symbol_ts_to_idx[pos_symbol]:
                            idx = self.symbol_ts_to_idx[pos_symbol][fill_ts]
                            if idx < len(pos_df):
                                check_symbol_prices[pos_symbol] = pos_df['close'].iloc[idx]
            if check_symbol_prices:
                self.portfolio.update_equity_all_positions(check_symbol_prices, fill_ts)
            self._check_invariants(fill_ts, check_symbol_prices if check_symbol_prices else symbol_prices)
        
        # Note: Equity curve is now recorded once per bar in main loop, not per symbol
        
        # Record positions
        if symbol in self.portfolio.positions:
            pos = self.portfolio.positions[symbol]
            self.positions_history.append({
                'ts': fill_ts,
                'symbol': symbol,
                'qty': pos.qty,
                'entry_px': pos.entry_price,
                'stop_px': pos.stop_price,
                'trail_px': pos.trail_price,
                'module': pos.module,
                'age_bars': pos.age_bars
            })
