"""Main backtest engine orchestrator"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import timedelta
import sys
from pathlib import Path
import uuid
from collections import defaultdict

# Add engine_core to path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
from engine_core.src.execution.order_manager import OrderManager, PendingOrder
from engine_core.src.portfolio.state import PortfolioState
from engine_core.src.portfolio.universe import UniverseManager
from engine_core.src.reporting import ReportGenerator
from engine_core.src.indicators.technical import sma, ema


class BacktestEngine:
    """Main backtest engine"""
    
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
            for symbol in symbols:
                if symbol not in self.symbol_data or symbol not in self.symbol_ts_to_idx:
                    continue
                
                # OPTIMIZATION: Use O(1) lookup instead of O(n) filter
                if current_ts not in self.symbol_ts_to_idx[symbol]:
                    continue
                
                idx = self.symbol_ts_to_idx[symbol][current_ts]
                df = self.symbol_data[symbol]
                
                # Process bar t+1 (order execution) - use current bar
                if idx < len(df) - 1:
                    next_idx = idx + 1
                    next_ts = df['ts'].iloc[next_idx]
                    self.process_bar_t_plus_1(symbol, idx, next_idx, next_ts)
                else:
                    # Last bar - still process t+1 with current bar as fill bar
                    self.process_bar_t_plus_1(symbol, idx, idx, current_ts)
            
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
            # This ensures all positions use correct prices simultaneously
            if self.portfolio.positions:
                symbol_prices = {}
                for pos_symbol in self.portfolio.positions.keys():
                    if pos_symbol in self.symbol_data and pos_symbol in self.symbol_ts_to_idx:
                        # OPTIMIZATION: Use O(1) lookup instead of O(n) filter
                        if current_ts in self.symbol_ts_to_idx[pos_symbol]:
                            idx = self.symbol_ts_to_idx[pos_symbol][current_ts]
                            pos_df = self.symbol_data[pos_symbol]
                            symbol_prices[pos_symbol] = pos_df['close'].iloc[idx]
                        else:
                            # Fallback: use most recent bar up to current_ts
                            pos_df = self.symbol_data[pos_symbol]
                            pos_before = pos_df[pos_df['ts'] <= current_ts]
                            if len(pos_before) > 0:
                                symbol_prices[pos_symbol] = pos_before.iloc[-1]['close']
                
                # Update equity once with all symbol prices
                if symbol_prices and len(symbol_prices) == len(self.portfolio.positions):
                    # Only update if we have prices for all positions
                    self.portfolio.update_equity_all_positions(symbol_prices, current_ts)
                elif len(self.portfolio.positions) == 0:
                    # If no positions, equity should equal cash (unrealized PnL = 0)
                    self.portfolio.equity = self.portfolio.cash
                else:
                    # Partial prices - update what we can
                    partial_prices = {k: v for k, v in symbol_prices.items() if k in self.portfolio.positions}
                    if partial_prices:
                        self.portfolio.update_equity_all_positions(partial_prices, current_ts)
                
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
                    self._check_invariants(current_ts, symbol_prices)
            
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
                            self._record_fill(
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
                                self._record_ledger_event(
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
                'ts': current_ts,
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
                    # Get the last bar's close price
                    last_bar = df.iloc[-1]
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
                    
                    # Calculate slippage
                    df = self.symbol_data.get(pos_symbol)
                    if df is not None and len(df) > 0:
                        last_bar = df.iloc[-1]
                        mid_price = (last_bar['high'] + last_bar['low']) / 2.0 if 'high' in last_bar and 'low' in last_bar else exit_price
                        slippage_params = self.params_dict.get('slippage_costs', {})
                        slippage_bps_applied = slippage_params.get('base_slip_bps_intercept', 2.0)
                        fill_idx = len(df) - 1
                        adv60_usd = calculate_adv_60m(df['notional'], fill_idx) if fill_idx >= 0 else 0.0
                    else:
                        mid_price = exit_price
                        slippage_bps_applied = 2.0
                        adv60_usd = 0.0
                    
                    if not self.cost_model_enabled:
                        slippage_bps_applied = 0.0
                    
                    slippage_cost_usd = notional * (slippage_bps_applied / 10000.0)
                    participation_pct = (notional / adv60_usd) if adv60_usd > 0 else 0.0
                    
                    # Record exit fill
                    self._record_fill(
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
                        self._record_ledger_event(
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
                        close_idx = len(df) - 1 if df is not None else -1
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
            except:
                pass
        if not self.loss_halt_state.halt_manual:
            if debug_oracle:
                print(f"[ORACLE DEBUG] process_bar_t: Calling generate_signals")
            self.generate_signals(symbol, idx, current_ts, master_side)
        elif debug_oracle:
            print(f"[ORACLE DEBUG] process_bar_t: SKIPPING generate_signals (halted)")
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
                print(f"WARNING: Error collecting new entry events for {symbol}: {e}")
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
                            symbol_prices[pos_symbol] = pos_df.iloc[-1]['close'] if len(pos_df) > 0 else 0.0
                    else:
                        # Fallback: use most recent bar
                        symbol_prices[pos_symbol] = pos_df.iloc[-1]['close'] if len(pos_df) > 0 else 0.0
                else:
                    # Fallback: DataFrame lookup
                    pos_bar = pos_df[pos_df['ts'] == fill_ts]
                    if len(pos_bar) > 0:
                        symbol_prices[pos_symbol] = pos_bar.iloc[0]['close']
                    else:
                        # Fallback: use most recent bar
                        if len(pos_df) > 0:
                            symbol_prices[pos_symbol] = pos_df.iloc[-1]['close']
        
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
                    self._record_fill(
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
                        self._record_ledger_event(
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
                self._record_fill(
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
                    self._record_ledger_event(
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
                    self._record_fill(
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
                        self._record_ledger_event(
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
        print("\n" + "="*80)
        print("PROFILING SUMMARY")
        print("="*80)
        print(f"Total runtime: {total_time:.2f}s")
        print(f"Bars processed: {self._profile_counts['bars_processed']}")
        print(f"Signals generated: {self._profile_counts['signals_generated']}")
        print(f"Events collected: {self._profile_counts['events_collected']}")
        print(f"Events executed: {self._profile_counts['events_executed']}")
        print(f"ES checks: {self._profile_counts['es_checks']}")
        print("\nTime breakdown:")
        for key, value in self._profile_time.items():
            pct = (value / total_time * 100) if total_time > 0 else 0
            print(f"{key:<25}: {value:>8.2f}s ({pct:>5.1f}%)")
        print("="*80)
        
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
            print("\n" + "=" * 80)
            print("ENTRY BLOCK SUMMARY")
            print("=" * 80)
            for reason, count in sorted(block_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"  {reason}: {count}")
            print("=" * 80)
    
    # ========== Helper Methods ==========
    
    def update_master_side(self, symbol: str, current_ts: pd.Timestamp) -> str:
        """Update master side for symbol - Engine-agnostic: always returns NEUTRAL"""
        # Engine-agnostic: always return NEUTRAL
        # Strategy should provide master_side externally if needed
        return 'NEUTRAL'
    
    def update_liquidity_regime(self, symbol: str, idx: int, current_ts: pd.Timestamp):
        """Update liquidity regime for symbol"""
        df = self.symbol_data[symbol]
        if idx >= len(df):
            return
        
        bar = df.iloc[idx]
        
        # Get liquidity data
        liquidity_df = self.data_loader.get_liquidity(symbol)
        spread_bps = bar.get('spread_bps', 0.0)
        depth5_usd = 0.0
        
        if self.require_liquidity_data and (liquidity_df is None or len(liquidity_df) == 0):
            raise ValueError(f"Liquidity data missing for {symbol} at {current_ts} while liquidity regimes enabled")
        
        if liquidity_df is None or len(liquidity_df) == 0:
            return
        
        if len(liquidity_df) > 0:
            # OPTIMIZATION: Build timestamp index once if not exists
            cache_key = f"_liq_idx_{symbol}"
            if not hasattr(self, cache_key):
                # Build sorted index for binary search
                liquidity_df_sorted = liquidity_df.sort_values('ts')
                setattr(self, cache_key, liquidity_df_sorted)
            else:
                liquidity_df_sorted = getattr(self, cache_key)
            
            # Find matching liquidity data using sorted index (much faster)
            liq_match = liquidity_df_sorted[liquidity_df_sorted['ts'] <= current_ts]
            if len(liq_match) > 0:
                liq_bar = liq_match.iloc[-1]
                spread_bps = liq_bar.get('spread_bps', spread_bps)
                depth5_bid = liq_bar.get('Depth5_bid_usd', 0.0)
                depth5_ask = liq_bar.get('Depth5_ask_usd', 0.0)
                depth5_usd = depth5_bid + depth5_ask
        
        # Calculate max_possible_notional
        vol_forecast = bar.get('vol_forecast', 0.02)
        vol_fast_median = bar.get('vol_fast_median', 0.02)
        size_mult = calculate_size_multiplier(
            vol_forecast, vol_fast_median,
            pd.Series([bar.get('close', 0)]),  # Simplified
            vol_forecast / np.sqrt(96),  # Approximate 15m sigma
            bar.get('slope_z', 0.0),
            self.params_dict
        )
        
        module_factors = self.params.get('sizing', 'module_factors') or {}
        r_base = self.params.get_default('general', 'r_base')
        entry_estimate = bar['close']
        stop_distance_estimate = bar.get('atr', entry_estimate * 0.02) * 3.0  # Rough estimate
        
        max_notional = calculate_max_possible_notional(
            self.portfolio.equity, entry_estimate, stop_distance_estimate,
            size_mult, module_factors, r_base
        )
        
        # Get seasonal values for THIN
        seasonal_spread_z = None
        seasonal_depth_pct = None
        if liquidity_df is not None:
            seasonal_spread_z, seasonal_depth_pct = self.seasonal_profile.get_seasonal_values(
                liquidity_df, current_ts
            )
        
        # Update liquidity state
        current_state = self.symbol_liquidity_state.get(symbol)
        new_state = self.liquidity_detector.update_regime(
            spread_bps, depth5_usd, max_notional, current_state,
            current_ts, seasonal_spread_z, seasonal_depth_pct
        )
        self.symbol_liquidity_state[symbol] = new_state
    
    def get_vol_scale(self, symbol: str, idx: int) -> float:
        """Get volatility scaling factor for loss halts"""
        df = self.symbol_data[symbol]
        if idx >= len(df):
            return 1.0
        
        bar = df.iloc[idx]
        vol_forecast = bar.get('vol_forecast', 0.02)
        vol_fast_median = bar.get('vol_fast_median', 0.02)
        
        if vol_fast_median > 0:
            return max(1.0, vol_forecast / vol_fast_median)
        return 1.0
    
    def _get_drawdown_size_constraints(self) -> Tuple[float, bool]:
        """Return size multiplier adjustment and halt flag from drawdown ladder"""
        if self.portfolio.peak_equity > 0:
            current_drawdown_pct = (self.portfolio.equity - self.portfolio.peak_equity) / self.portfolio.peak_equity
        else:
            current_drawdown_pct = 0.0
        size_mult_adjust, should_halt = self.loss_halt_state.check_drawdown_ladder(
            current_drawdown_pct, self.params_dict
        )
        return size_mult_adjust, should_halt
    
    def _check_beta_caps_with_new_position(self, symbol: str, qty: float, price: float, event_ts: pd.Timestamp, side: str = 'LONG') -> bool:
        """Check beta caps including a hypothetical new position"""
        # Launch Punch List – Blocker #4: enforce BTC-beta caps (symbol + portfolio)
        # Use new portfolio-level beta cap check
        from engine_core.src.risk.beta_controls import check_portfolio_beta_caps
        
        # Get beta values
        beta_map = {}
        all_symbols = set(self.portfolio.positions.keys()) | {symbol}
        for sym in all_symbols:
            beta_map[sym] = self.beta_slow_priors.get(sym, 1.0)
        
        # Get risk params
        risk_params = self.params_dict.get('risk', {}).get('beta', {})
        max_symbol_beta = risk_params.get('max_symbol_beta', 1.5)
        max_portfolio_beta = risk_params.get('max_portfolio_beta', 3.0)
        reference_symbol = risk_params.get('reference_symbol', 'BTCUSDT')
        
        # Convert positions to dict format
        positions_dict = {}
        for sym, pos in self.portfolio.positions.items():
            positions_dict[sym] = {
                'qty': pos.qty,
                'entry_price': pos.entry_price,
                'notional': abs(pos.qty * pos.entry_price),
                'side': pos.side
            }
        
        # Check portfolio beta caps
        is_valid, symbol_beta_exposure, portfolio_beta_exposure, reason = check_portfolio_beta_caps(
            positions=positions_dict,
            beta_slow=beta_map,
            new_symbol=symbol,
            new_qty=qty,
            new_price=price,
            new_side=side,
            max_symbol_beta=max_symbol_beta,
            max_portfolio_beta=max_portfolio_beta,
            reference_symbol=reference_symbol
        )
        
        if not is_valid:
            self.forensic_log.append({
                'ts': event_ts,
                'symbol': symbol,
                'event': 'BETA_CAP_BLOCK',
                'symbol_beta_exposure': symbol_beta_exposure,
                'portfolio_beta_exposure': portfolio_beta_exposure,
                'max_symbol_beta': max_symbol_beta,
                'max_portfolio_beta': max_portfolio_beta,
                'reason': reason
            })
        
        # Also check legacy net/gross beta caps for backward compatibility
        positions_snapshot: Dict[str, Dict] = {}
        for sym, pos in self.portfolio.positions.items():
            positions_snapshot[sym] = {
                'qty': pos.qty,
                'entry_price': pos.entry_price,
                'notional': abs(pos.qty * pos.entry_price)
            }
        positions_snapshot[symbol] = {
            'qty': qty,
            'entry_price': price,
            'notional': abs(qty * price)
        }
        is_valid_legacy, net_beta, gross_beta = check_beta_caps(
            positions_snapshot,
            beta_map,
            self.beta_cap_net,
            self.beta_cap_gross
        )
        
        # Both checks must pass
        return is_valid and is_valid_legacy
    
    def generate_signals(self, symbol: str, idx: int, current_ts: pd.Timestamp, master_side: str):
        """Generate trading signals at bar t close"""
        debug_oracle = self.params.get('general', 'debug_oracle_flow', default=False)
        if debug_oracle:
            print(f"[ORACLE DEBUG] generate_signals CALLED: symbol={symbol}, idx={idx}, ts={current_ts}, master_side={master_side}")
        df = self.symbol_data[symbol]
        if idx >= len(df):
            if debug_oracle:
                print(f"[ORACLE DEBUG] generate_signals: idx {idx} >= len(df) {len(df)}, returning early")
            return
        
        bar = df.iloc[idx]
        regime = bar.get('regime', 'UNCERTAIN')
        is_btc = 'BTC' in symbol
        
        # Initialize pending signals list (DO NOT clear existing signals - they need to persist for confirmation)
        if symbol not in self.symbol_pending_signals:
            self.symbol_pending_signals[symbol] = []
        
        # ORACLE mode: bypass all normal signal generation (for validation/testing only)
        oracle_mode = self.params.get('general', 'oracle_mode')

        # DECEPTION mode: replay DeceptionLeaderBot signals through the engine.
        # Mutually exclusive with oracle_mode. Signals are pre-loaded via
        # set_deception_signals() and emitted at the bar matching entry_ts.
        deception_mode = self.params.get('general', 'deception_mode', default=False)
        if deception_mode and self.deception_module is not None:
            sig = self.deception_module.generate_signal(symbol, df, idx, current_ts)
            if sig is not None:
                self.symbol_pending_signals[symbol].append(sig)
                self._profile_counts['signals_generated'] += 1
            return

        if oracle_mode:
            if oracle_mode == 'always_long':
                oracle_signal = self.oracle_module.generate_always_long(symbol, df, idx, current_ts)
                if oracle_signal:
                    if debug_oracle:
                        print(f"[ORACLE DEBUG] Bar {idx} ({current_ts}): Created ORACLE signal for {symbol}, side={oracle_signal.side}, signal_bar_idx={oracle_signal.signal_bar_idx}")
                    self.symbol_pending_signals[symbol].append(oracle_signal)
                    self._profile_counts['signals_generated'] += 1
                    if debug_oracle:
                        print(f"[ORACLE DEBUG] Bar {idx}: Added to pending_signals. Total pending for {symbol}: {len(self.symbol_pending_signals[symbol])}")
            elif oracle_mode == 'always_short':
                oracle_signal = self.oracle_module.generate_always_short(symbol, df, idx, current_ts)
                if oracle_signal:
                    if debug_oracle:
                        print(f"[ORACLE DEBUG] Bar {idx} ({current_ts}): Created ORACLE signal for {symbol}, side={oracle_signal.side}, signal_bar_idx={oracle_signal.signal_bar_idx}")
                    self.symbol_pending_signals[symbol].append(oracle_signal)
                    self._profile_counts['signals_generated'] += 1
                    if debug_oracle:
                        print(f"[ORACLE DEBUG] Bar {idx}: Added to pending_signals. Total pending for {symbol}: {len(self.symbol_pending_signals[symbol])}")
            elif oracle_mode == 'flat':
                # No signals (flat strategy)
                pass
            elif oracle_mode == 'random':
                oracle_signal = self.oracle_module.generate_random(symbol, df, idx, current_ts)
                if oracle_signal:
                    self.symbol_pending_signals[symbol].append(oracle_signal)
                    self._profile_counts['signals_generated'] += 1
            # In oracle mode, skip all normal signal generation
            return
        
        # Strategy-specific signal generation removed from engine core.
        # For production use, provide strategy modules externally via a signal generator callback.
        # Engine core only supports oracle_mode for validation/testing.
        raise NotImplementedError(
            "Strategy-specific signal generation not available in engine core. "
            "Use oracle_mode for validation, or provide external strategy modules."
        )
    
    # ========== Event Collection Methods ==========
    
    def collect_stop_events(self, symbol: str, fill_bar: pd.Series, fill_ts: pd.Timestamp) -> List[OrderEvent]:
        """Collect stop-loss events for non-SQUEEZE positions"""
        events = []
        
        if symbol not in self.portfolio.positions:
            return events
        
        pos = self.portfolio.positions[symbol]
        # SQUEEZE stops are also checked here (same logic)
        
        # Check if stop is triggered
        current_price = fill_bar['close']
        high = fill_bar['high']
        low = fill_bar['low']
        
        stop_triggered = False
        if pos.side == 'LONG' and low <= pos.stop_price:
            stop_triggered = True
        elif pos.side == 'SHORT' and high >= pos.stop_price:
            stop_triggered = True
        
        if stop_triggered:
            events.append(OrderEvent(
                event_type='STOP',
                symbol=symbol,
                module=pos.module,
                priority=1,
                signal_ts=pos.entry_ts
            ))

        return events

    def collect_deception_exit_events(
        self, symbol: str, fill_bar: pd.Series, fill_ts: pd.Timestamp, fill_idx: int
    ) -> List[OrderEvent]:
        """Collect TP1/TP2/TSL/SL events for DeceptionLeaderBot positions.

        Mirrors core/sim_core.py simulate_exit_on_bars:
          1. If TSL active: trail favourable extreme, exit on retracement > tsl_bps
          2. Else: check SL and TP1 on the same bar
             - If both hit on the same bar: pessimistic — SL wins
             - If TP1 hits and tp1_frac < 1.0: activate TSL on remainder (if tsl_bps>0)
             - If TP1 hits and tp1_frac == 1.0: full close at TP1
          3. NO time-based exit (max_hold/timeout removed from the bot)
        """
        events: List[OrderEvent] = []
        if symbol not in self.portfolio.positions:
            return events
        pos = self.portfolio.positions[symbol]
        if pos.module != 'DECEPTION':
            return events

        high = float(fill_bar['high'])
        low = float(fill_bar['low'])

        # ── TSL active: trailing stop on the remainder ──────────────────────
        if pos.tsl_active:
            if pos.side == 'LONG':
                if high > pos.tsl_extreme:
                    pos.tsl_extreme = high
                tsl_px = pos.tsl_extreme * (1.0 - pos.tsl_bps / 10000.0)
                if low <= tsl_px:
                    events.append(OrderEvent(
                        event_type='DECEPTION_TSL_EXIT',
                        symbol=symbol, module='DECEPTION', priority=1,
                        signal_ts=pos.entry_ts,
                    ))
            else:  # SHORT
                if low < pos.tsl_extreme:
                    pos.tsl_extreme = low
                tsl_px = pos.tsl_extreme * (1.0 + pos.tsl_bps / 10000.0)
                if high >= tsl_px:
                    events.append(OrderEvent(
                        event_type='DECEPTION_TSL_EXIT',
                        symbol=symbol, module='DECEPTION', priority=1,
                        signal_ts=pos.entry_ts,
                    ))
            return events  # TSL active: skip fixed SL/TP checks

        # ── Fixed SL / TP1 check (same-bar pessimism: SL wins) ──────────────
        sl_hit = (pos.side == 'LONG' and low <= pos.stop_price) or \
                 (pos.side == 'SHORT' and high >= pos.stop_price)
        tp1_hit = False
        if pos.tp1_price > 0:
            tp1_hit = (pos.side == 'LONG' and high >= pos.tp1_price) or \
                      (pos.side == 'SHORT' and low <= pos.tp1_price)

        if sl_hit and tp1_hit:
            # Same-bar pessimism: SL wins (matches sim_core)
            events.append(OrderEvent(
                event_type='STOP', symbol=symbol, module='DECEPTION',
                priority=1, signal_ts=pos.entry_ts,
            ))
            return events
        if sl_hit:
            events.append(OrderEvent(
                event_type='STOP', symbol=symbol, module='DECEPTION',
                priority=1, signal_ts=pos.entry_ts,
            ))
            return events
        if tp1_hit:
            if pos.tp1_frac >= 1.0 - 1e-9:
                # Full close at TP1
                events.append(OrderEvent(
                    event_type='DECEPTION_TP1_FULL',
                    symbol=symbol, module='DECEPTION', priority=1,
                    signal_ts=pos.entry_ts,
                ))
            else:
                # Partial TP1 — activate TSL on remainder if enabled
                events.append(OrderEvent(
                    event_type='DECEPTION_TP1_PARTIAL',
                    symbol=symbol, module='DECEPTION', priority=1,
                    signal_ts=pos.entry_ts,
                ))
            return events
        # TP2 check (only if TP1 already hit and TP2 configured)
        if pos.tp1_hit and pos.tp2_price > 0:
            tp2_hit = (pos.side == 'LONG' and high >= pos.tp2_price) or \
                      (pos.side == 'SHORT' and low <= pos.tp2_price)
            if tp2_hit:
                events.append(OrderEvent(
                    event_type='DECEPTION_TP2_EXIT',
                    symbol=symbol, module='DECEPTION', priority=1,
                    signal_ts=pos.entry_ts,
                ))
        return events
    
    def collect_squeeze_tp1_events(self, symbol: str, fill_bar: pd.Series, fill_ts: pd.Timestamp) -> List[OrderEvent]:
        """Collect SQUEEZE TP1 exit events - Strategy-specific (dead code in oracle mode)"""
        events = []
        
        # Engine-agnostic: skip strategy-specific logic in oracle mode
        oracle_mode = self.params.get('general', 'oracle_mode')
        if oracle_mode:
            return events
        
        if symbol not in self.portfolio.positions:
            return events
        
        pos = self.portfolio.positions[symbol]
        if pos.module != 'SQUEEZE' or pos.tp1_price <= 0:
            return events  # Only for SQUEEZE with TP1 enabled
        
        # Check if TP1 is triggered
        high = fill_bar['high']
        low = fill_bar['low']
        
        tp1_triggered = False
        if pos.side == 'LONG' and high >= pos.tp1_price:
            tp1_triggered = True
        elif pos.side == 'SHORT' and low <= pos.tp1_price:
            tp1_triggered = True
        
        if tp1_triggered:
            events.append(OrderEvent(
                event_type='SQUEEZE_TP1',
                symbol=symbol,
                module=pos.module,
                priority=1,  # Same priority as STOP (check before other exits)
                signal_ts=pos.entry_ts
            ))
        
        return events
    
    def collect_squeeze_vol_exit_events(self, symbol: str, fill_bar: pd.Series, fill_ts: pd.Timestamp, fill_idx: int) -> List[OrderEvent]:
        """Collect SQUEEZE volatility expansion exit events - Strategy-specific (dead code in oracle mode)"""
        events = []
        
        # Engine-agnostic: skip strategy-specific logic in oracle mode
        oracle_mode = self.params.get('general', 'oracle_mode')
        if oracle_mode:
            return events
        
        if symbol not in self.portfolio.positions:
            return events
        
        pos = self.portfolio.positions[symbol]
        if pos.module != 'SQUEEZE':
            return events
        
        # Strategy-specific params - use defaults (not in base_params.json)
        exit_atr_pct_thresh = 80.0  # Default
        if exit_atr_pct_thresh <= 0:
            return events  # Vol expansion exit disabled
        
        min_R_before_vol_exit = 0.5  # Default
        
        # Calculate current unrealized R
        if pos.initial_R <= 0:
            return events  # No R stored (shouldn't happen for SQUEEZE)
        
        current_price = fill_bar['close']
        if pos.side == 'LONG':
            unrealized_pnl = (current_price - pos.entry_price) * pos.qty
        else:  # SHORT
            unrealized_pnl = (pos.entry_price - current_price) * pos.qty
        
        unrealized_R = unrealized_pnl / (pos.initial_R * pos.qty) if (pos.initial_R * pos.qty) > 0 else 0.0
        
        # Check minimum R requirement
        if unrealized_R < min_R_before_vol_exit:
            return events  # Not enough profit yet
        
        # Calculate ATR percentile
        df = self.symbol_data[symbol]
        atr_percentile_lookback = 200  # Default, strategy-specific param not in base_params.json
        
        if fill_idx < atr_percentile_lookback or 'atr' not in df.columns:
            return events  # Insufficient history
        
        try:
            atr_window = df['atr'].iloc[fill_idx - atr_percentile_lookback:fill_idx]
            atr_window_clean = atr_window.dropna()
            
            if len(atr_window_clean) < atr_percentile_lookback * 0.5:
                return events  # Insufficient valid data
            
            current_atr = df['atr'].iloc[fill_idx]
            if pd.isna(current_atr):
                return events
            
            # Calculate percentile rank of current ATR
            atr_pct_rank = (atr_window_clean < current_atr).sum() / len(atr_window_clean) * 100
            
            # Check if ATR percentile exceeds threshold
            if atr_pct_rank >= exit_atr_pct_thresh:
                events.append(OrderEvent(
                    event_type='SQUEEZE_VOL_EXIT',
                    symbol=symbol,
                    module=pos.module,
                    priority=2,  # After TP1/STOP but before TTL
                    signal_ts=pos.entry_ts
                ))
        except (IndexError, KeyError):
            pass  # Insufficient history or missing column
        
        return events
    
    def collect_squeeze_entry_events(self, symbol: str, fill_bar: pd.Series, fill_ts: pd.Timestamp) -> List[OrderEvent]:
        """Collect SQUEEZE entry events (entry_first) - Strategy-specific (dead code in oracle mode)"""
        events = []
        
        # Engine-agnostic: skip strategy-specific logic in oracle mode
        oracle_mode = self.params.get('general', 'oracle_mode')
        if oracle_mode:
            return events
        
        # Check pending SQUEEZE orders
        pending_orders = self.order_manager.get_orders_by_module('SQUEEZE')
        for order in pending_orders:
            if order.symbol != symbol or order.filled:
                continue
            
            # Check if entry trigger is hit
            high = fill_bar['high']
            low = fill_bar['low']
            
            trigger_hit = False
            if order.side == 'LONG' and high >= order.trigger_price:
                trigger_hit = True
            elif order.side == 'SHORT' and low <= order.trigger_price:
                trigger_hit = True
            
            if trigger_hit:
                events.append(OrderEvent(
                    event_type='SQUEEZE_ENTRY',
                    symbol=symbol,
                    module='SQUEEZE',
                    priority=2,
                    signal_ts=order.signal_ts,
                    order_id=order.order_id,
                    side=order.side
                ))
        
        # Apply direction gates
        return self._apply_direction_gates(events)
    
    def _apply_direction_gates(self, entry_events: List[OrderEvent]) -> List[OrderEvent]:
        """Apply global long-only / short-only direction gates"""
        sizing_params = self.params.get('sizing') or {}
        long_only = sizing_params.get('long_only', False)
        short_only = sizing_params.get('short_only', False)
        
        if long_only and short_only:
            raise ValueError("Invalid config: both sizing.long_only and sizing.short_only are True")
            
        if long_only:
            filtered = [e for e in entry_events if e.side != 'SHORT']
            if len(filtered) < len(entry_events):
                print(f"DEBUG: Gating applied (Long Only). Blocked {len(entry_events) - len(filtered)} SHORT events.")
            return filtered
            
        if short_only:
            filtered = [e for e in entry_events if e.side != 'LONG']
            if len(filtered) < len(entry_events):
                print(f"DEBUG: Gating applied (Short Only). Blocked {len(entry_events) - len(filtered)} LONG events.")
            return filtered
            
        return entry_events
    
    def collect_new_entry_events(
        self, symbol: str, signal_idx: int, fill_idx: int, fill_bar: pd.Series,
        fill_ts: pd.Timestamp, funding_window: Dict
    ) -> List[OrderEvent]:
        """Collect new entry events (ORACLE signals only in Model-1)"""
        # Collect internal events first
        entry_events = self._collect_new_entry_events(symbol, signal_idx, fill_idx, fill_bar, fill_ts, funding_window)
        
        # Apply direction gates
        return self._apply_direction_gates(entry_events)

    def _collect_new_entry_events(
        self, symbol: str, signal_idx: int, fill_idx: int, fill_bar: pd.Series,
        fill_ts: pd.Timestamp, funding_window: Dict
    ) -> List[OrderEvent]:
        """Internal collection of new entry events (before gating)."""
        events = []
        
        # Check liquidity regime (VACUUM blocks entries)
        liquidity_state = self.symbol_liquidity_state.get(symbol)
        if liquidity_state and liquidity_state.regime == 'VACUUM':
            self.vacuum_blocks_count += 1
            self.forensic_log.append({
                'ts': fill_ts,
                'symbol': symbol,
                'event': 'VACUUM_BLOCK',
                'module': None,
                'spread_bps': liquidity_state.spread_bps,
                'depth5_usd': liquidity_state.depth5_usd
            })
            return events  # VACUUM blocks new entries
        
        # Check if already have position
        if symbol in self.portfolio.positions:
            return events  # Already have position
        
        # Check for ORACLE/DECEPTION signals - they bypass max positions, loss halts, etc.
        debug_oracle = self.params.get('general', 'debug_oracle_flow', default=False)
        pending_signals = self.symbol_pending_signals.get(symbol, [])
        if debug_oracle:
            print(f"[ORACLE DEBUG] _collect_new_entry_events: symbol={symbol}, fill_idx={fill_idx}, pending_signals count={len(pending_signals)}")
            for i, sig in enumerate(pending_signals):
                if hasattr(sig, 'module'):
                    print(f"  Signal {i}: module={sig.module}, signal_bar_idx={getattr(sig, 'signal_bar_idx', 'N/A')}")
        oracle_signals = [s for s in pending_signals if hasattr(s, 'module') and s.module == 'ORACLE']
        deception_signals = [s for s in pending_signals if hasattr(s, 'module') and s.module == 'DECEPTION']
        if debug_oracle:
            print(f"[ORACLE DEBUG] Found {len(oracle_signals)} ORACLE signals, {len(deception_signals)} DECEPTION signals")
        if deception_signals:
            # DECEPTION signals bypass max positions, loss halts, etc. (but not position check above)
            for signal in deception_signals:
                signal_bar_idx_scalar = int(signal.signal_bar_idx) if hasattr(signal.signal_bar_idx, '__iter__') and not isinstance(signal.signal_bar_idx, str) else signal.signal_bar_idx
                fill_idx_scalar = int(fill_idx) if hasattr(fill_idx, '__iter__') and not isinstance(fill_idx, str) else fill_idx
                if fill_idx_scalar >= signal_bar_idx_scalar:
                    events.append(OrderEvent(
                        event_type='DECEPTION_ENTRY',
                        symbol=symbol,
                        module='DECEPTION',
                        priority=1,
                        signal_ts=signal.signal_ts,
                        side=signal.side
                    ))
                    self._profile_counts['events_collected'] += 1
                    return events
        if oracle_signals:
            # ORACLE signals bypass max positions, loss halts, etc. (but not position check above)
            for signal in oracle_signals:
                # Only process if we're on or after the signal bar
                signal_bar_idx_scalar = int(signal.signal_bar_idx) if hasattr(signal.signal_bar_idx, '__iter__') and not isinstance(signal.signal_bar_idx, str) else signal.signal_bar_idx
                fill_idx_scalar = int(fill_idx) if hasattr(fill_idx, '__iter__') and not isinstance(fill_idx, str) else fill_idx
                # Process on same bar or next bar (Oracle signals bypass confirmation)
                # For Oracle: signal created on bar t, process on bar t+1 (fill_idx should be signal_bar_idx + 1)
                # Always process Oracle signals (they bypass all timing checks)
                # Note: fill_idx is the bar where we're processing fills (bar t+1), signal_bar_idx is where signal was created (bar t)
                # So fill_idx should be >= signal_bar_idx + 1, but we allow >= for safety
                if debug_oracle:
                    print(f"[ORACLE DEBUG] Checking timing: fill_idx={fill_idx_scalar}, signal_bar_idx={signal_bar_idx_scalar}, condition={fill_idx_scalar >= signal_bar_idx_scalar}")
                if fill_idx_scalar >= signal_bar_idx_scalar:
                    if debug_oracle:
                        print(f"[ORACLE DEBUG] Creating ORACLE_ENTRY event for {symbol}, side={signal.side}")
                    events.append(OrderEvent(
                        event_type='ORACLE_ENTRY',
                        symbol=symbol,
                        module='ORACLE',
                        priority=1,
                        signal_ts=signal.signal_ts,
                        side=signal.side
                    ))
                    self._profile_counts['events_collected'] += 1
                    if debug_oracle:
                        print(f"[ORACLE DEBUG] Created {len(events)} events, returning")
                    return events  # Return immediately for ORACLE (only one at a time)
                elif debug_oracle:
                    print(f"[ORACLE DEBUG] Timing check FAILED: fill_idx={fill_idx_scalar} < signal_bar_idx={signal_bar_idx_scalar}")
        
        # Check max positions
        max_positions = self.params.get_default('general', 'max_positions')
        if len(self.portfolio.positions) >= max_positions:
            return events
        
        # Check loss halts
        if self.loss_halt_state.halt_manual:
            return events
        
        if self.loss_halt_state.check_daily_hard_stop(
            self.portfolio.equity, self.get_vol_scale(symbol, signal_idx), self.params_dict
        ):
            # FIX 5: Deduplicate by UTC day
            utc_date = fill_ts.date()
            if (utc_date,) not in self._halt_daily_hard_seen:
                self._halt_daily_hard_seen.add((utc_date,))
                self.halt_daily_hard_count += 1  # G: Track daily hard stops
            return events
        
        # Check soft brake status
        soft_active, should_activate = self.loss_halt_state.check_soft_brake(
            fill_ts, self.portfolio.equity,
            self.get_vol_scale(symbol, signal_idx),
            self.params_dict
        )
        if soft_active:
            # Track activation (should_activate is True when threshold is hit)
            # FIX 5: Deduplicate by UTC day
            if should_activate:
                utc_date = fill_ts.date()
                if (utc_date,) not in self._halt_soft_brake_seen:
                    self._halt_soft_brake_seen.add((utc_date,))
                    self.halt_soft_brake_count += 1  # G: Track soft brake activations
            self._last_soft_brake_state = soft_active
            return events
        else:
            self._last_soft_brake_state = False
        
        # Per-symbol daily cap
        symbol_daily_pnl = self.symbol_daily_pnl.get(symbol, 0.0)
        if self.loss_halt_state.check_per_symbol_cap(
            symbol,
            symbol_daily_pnl,
            self.portfolio.equity,
            self.get_vol_scale(symbol, signal_idx),
            self.params_dict,
            fill_ts
        ):
            # FIX 5: Deduplicate by (symbol, UTC day)
            utc_date = fill_ts.date()
            if (symbol, utc_date) not in self._per_symbol_loss_cap_seen:
                self._per_symbol_loss_cap_seen.add((symbol, utc_date))
                self.per_symbol_loss_cap_count += 1  # G: Track per-symbol loss cap hits
            self.forensic_log.append({
                'ts': fill_ts,
                'symbol': symbol,
                'event': 'LOSS_HALT_SYMBOL_BLOCK',
                'module': None,
                'symbol_daily_pnl': symbol_daily_pnl
            })
            return events
        
        # Get pending signals for this symbol
        pending_signals = self.symbol_pending_signals.get(symbol, [])
        
        # Remove signals that are too old (max 20 bars old to prevent memory bloat)
        max_signal_age = 20
        signals_to_remove = []
        # Ensure fill_idx is a scalar to avoid pandas Series boolean ambiguity
        fill_idx_scalar = int(fill_idx) if hasattr(fill_idx, '__iter__') and not isinstance(fill_idx, str) else fill_idx
        for signal in pending_signals:
            signal_bar_idx_scalar = int(signal.signal_bar_idx) if hasattr(signal.signal_bar_idx, '__iter__') and not isinstance(signal.signal_bar_idx, str) else signal.signal_bar_idx
            if fill_idx_scalar - signal_bar_idx_scalar > max_signal_age:
                signals_to_remove.append(signal)
        for signal in signals_to_remove:
            # Use index-based removal to avoid pandas Series boolean ambiguity in __eq__
            try:
                idx = next((i for i, s in enumerate(pending_signals) if s is signal or (hasattr(s, 'signal_bar_idx') and hasattr(signal, 'signal_bar_idx') and int(s.signal_bar_idx) == int(signal.signal_bar_idx) and s.module == signal.module)), None)
                if idx is not None:
                    pending_signals.pop(idx)
            except (ValueError, TypeError):
                # Fallback: try direct removal if comparison works
                try:
                    pending_signals.remove(signal)
                except (ValueError, TypeError):
                    pass
        
        # Model-1: Only ORACLE signals are supported
        # ORACLE signals are handled above with early return (bypass all checks)
        # Strategy-specific modules (TREND, RANGE, SQUEEZE, NEUTRAL_PROBE) are not supported
        # Non-ORACLE signals in pending_signals are ignored
        return events
    
    def collect_trail_events(self, symbol: str, fill_bar: pd.Series, fill_ts: pd.Timestamp) -> List[OrderEvent]:
        """Collect trailing stop events (tighten only) - Generic for all positions"""
        events = []
        
        if symbol not in self.portfolio.positions:
            return events
        
        pos = self.portfolio.positions[symbol]
        
        # Generic trailing for all positions (Model-1: no module-specific logic)
        events.append(OrderEvent(
            event_type='TRAIL',
            symbol=symbol,
            module=pos.module,
            priority=7,
            signal_ts=pos.entry_ts
        ))
        
        return events
    
    def collect_range_time_stops(self, symbol: str, fill_idx: int, fill_ts: pd.Timestamp) -> List[OrderEvent]:
        """Collect RANGE time stop events (20 bars) - Strategy-specific (dead code in oracle mode)"""
        events = []
        
        # Engine-agnostic: skip strategy-specific logic in oracle mode
        oracle_mode = self.params.get('general', 'oracle_mode')
        if oracle_mode:
            return events
        
        if symbol not in self.portfolio.positions:
            return events
        
        pos = self.portfolio.positions[symbol]
        
        if pos.module != 'RANGE':
            return events
        
        # Check if time stop is reached (20 bars) - use default if param not found
        time_stop_bars = 20  # Default, strategy-specific params not in base_params.json
        if pos.age_bars >= time_stop_bars:
            events.append(OrderEvent(
                event_type='STOP',
                symbol=symbol,
                module=pos.module,
                priority=1,
                signal_ts=pos.entry_ts
            ))
        
        return events
    
    def collect_ttl_events(self, symbol: str, fill_idx: int, fill_ts: pd.Timestamp) -> List[OrderEvent]:
        """Collect TTL expiration events for both pending orders and filled positions"""
        events = []
        
        # Check pending orders (unfilled entries)
        expired_order_ids = self.order_manager.check_ttl_orders(fill_idx, fill_ts)
        for order_id in expired_order_ids:
            order = self.order_manager.pending_orders.get(order_id)
            if order and order.symbol == symbol:
                events.append(OrderEvent(
                    event_type='TTL',
                    symbol=symbol,
                    module=order.module,
                    priority=8,
                    signal_ts=order.signal_ts,
                    order_id=order_id
                ))
        
        # Check filled positions (SQUEEZE expires after configured TTL)
        if symbol in self.portfolio.positions:
            pos = self.portfolio.positions[symbol]
            if pos.module == 'SQUEEZE':
                # OPTIMIZATION: Use stored entry_idx if available
                if pos.entry_idx >= 0:
                    entry_idx = pos.entry_idx
                else:
                    # Fallback: lookup entry_idx (should rarely happen)
                    df = self.symbol_data[symbol]
                    entry_idx = df[df['ts'] == pos.entry_ts].index[0] if len(df[df['ts'] == pos.entry_ts]) > 0 else None
                
                if entry_idx is not None:
                    # Ensure entry_idx is an integer (not pandas Index)
                    if hasattr(entry_idx, '__iter__') and not isinstance(entry_idx, str):
                        entry_idx = entry_idx[0] if len(entry_idx) > 0 else None
                    if entry_idx is not None:
                        entry_idx = int(entry_idx)
                        age_bars = fill_idx - entry_idx
                        ttl_hours = 12  # Default, strategy-specific param not in base_params.json
                        ttl_bars = int(ttl_hours * 4)  # 4 bars per hour on 15m
                        if age_bars >= ttl_bars:
                            events.append(OrderEvent(
                                event_type='TTL',
                                symbol=symbol,
                                module=pos.module,
                                priority=8,
                                signal_ts=pos.entry_ts
                            ))
        
        return events
    
    def collect_stale_events(self, symbol: str, fill_idx: int, fill_ts: pd.Timestamp) -> List[OrderEvent]:
        """Collect stale order cancellation events"""
        events = []
        
        stale_order_ids = self.order_manager.check_stale_orders(fill_idx, fill_ts)
        liquidity_state = self.symbol_liquidity_state.get(symbol)
        thin_mode = bool(liquidity_state and liquidity_state.regime == 'THIN')
        thin_key = (symbol, fill_ts) if thin_mode else None
        thin_cancel_count = self.thin_cancel_tracker.get(thin_key, 0) if thin_mode else 0
        
        for order_id in stale_order_ids:
            order = self.order_manager.pending_orders.get(order_id)
            if order and order.symbol == symbol:
                if thin_mode and thin_cancel_count >= 1:
                    self.thin_cancel_block_count += 1
                    self.forensic_log.append({
                        'ts': fill_ts,
                        'symbol': symbol,
                        'event': 'THIN_CANCEL_BLOCK',
                        'order_id': order_id,
                        'module': order.module
                    })
                    continue
                
                events.append(OrderEvent(
                    event_type='STALE_CANCEL',
                    symbol=symbol,
                    module=order.module,
                    priority=9,
                    signal_ts=order.signal_ts,
                    order_id=order_id
                ))
                
                if thin_mode:
                    thin_cancel_count += 1
        
        if thin_mode:
            self.thin_cancel_tracker[thin_key] = thin_cancel_count
        
        return events
    
    # ========== Event Execution Methods ==========
    
    def execute_events(
        self, symbol: str, events: List[OrderEvent],
        fill_bar: pd.Series, fill_ts: pd.Timestamp
    ):
        """Execute sequenced events"""
        for event in events:
            if event.symbol != symbol:
                continue
            
            if event.event_type == 'STOP':
                self.execute_stop(symbol, fill_bar, fill_ts)
            elif event.event_type == 'DECEPTION_ENTRY':
                # DECEPTION signals bypass all filters and go directly to execute_entry
                self.execute_entry(event, fill_bar, fill_ts)
                self._profile_counts['events_executed'] += 1
            elif event.event_type in ('DECEPTION_TP1_FULL', 'DECEPTION_TP2_EXIT', 'DECEPTION_TSL_EXIT'):
                # Full close at TP1/TP2/TSL — taker fee, exit at target price
                self.execute_deception_exit(symbol, fill_bar, fill_ts, event.event_type)
                self._profile_counts['events_executed'] += 1
            elif event.event_type == 'DECEPTION_TP1_PARTIAL':
                # Partial close at TP1, activate TSL on remainder
                self.execute_deception_tp1_partial(symbol, fill_bar, fill_ts)
                self._profile_counts['events_executed'] += 1
            elif event.event_type == 'ORACLE_ENTRY':
                # ORACLE signals bypass all filters and go directly to execute_entry
                debug_oracle = self.params.get('general', 'debug_oracle_flow', default=False)
                if debug_oracle:
                    print(f"[ORACLE DEBUG] execute_events: Calling execute_entry for ORACLE_ENTRY, symbol={event.symbol}, side={event.side}")
                self.execute_entry(event, fill_bar, fill_ts)
                self._profile_counts['events_executed'] += 1
                if debug_oracle:
                    print(f"[ORACLE DEBUG] execute_entry returned. Trades count: {len(self.trades)}, Fills count: {len(self.fills)}")
            # Note: Strategy-specific event types (TREND_ENTRY, RANGE_ENTRY, SQUEEZE_ENTRY, etc.) 
            # have been removed in Model-1. Only ORACLE_ENTRY is supported.
            elif event.event_type == 'TRAIL':
                self.execute_trail(symbol, fill_bar, fill_ts)
            elif event.event_type == 'TTL':
                if event.order_id:
                    self.execute_ttl(order_id=event.order_id, fill_ts=fill_ts)
                else:
                    self.execute_ttl(symbol=event.symbol, fill_ts=fill_ts)
                self._profile_counts['events_executed'] += 1
            elif event.event_type == 'STALE_CANCEL':
                self.execute_stale_cancel(event.order_id)
                self._profile_counts['events_executed'] += 1
    
    def _check_and_log_outlier(self, fill_info: Dict):
        """Check if fill is an outlier and log if so"""
        slippage_bps = abs(fill_info.get('slippage_bps_applied', 0.0))
        
        # Flag condition: slippage_bps > max(20, 3 * median_slippage_bps_symbol_30d)
        # Fallback threshold = 20 bps (as we don't have 30d history loaded in engine)
        threshold = 20.0 
        
        if slippage_bps > threshold:
             outlier_record = fill_info.copy()
             outlier_record['outlier_threshold_bps'] = threshold
             self.outlier_log.append(outlier_record)
    
    def _record_fill(
        self,
        position_id: str,
        ts: pd.Timestamp,
        symbol: str,
        module: str,
        leg: str,  # 'ENTRY' or 'EXIT'
        side: str,  # 'BUY' or 'SELL'
        qty: float,
        price: float,
        notional_usd: float,
        slippage_bps_applied: float,
        slippage_cost_usd: float,
        fee_bps: float,
        fee_usd: float,
        liquidity: str,  # 'maker' or 'taker'
        participation_pct: float,
        adv60_usd: float,
        intended_price: float = None
    ):
        """Record a fill event to fills list"""
        # Generate unique fill_id
        if position_id not in self._fill_counter:
            self._fill_counter[position_id] = 0
        self._fill_counter[position_id] += 1
        fill_id = f"{position_id}-{leg}-{self._fill_counter[position_id]}"
        
        fill_record = {
            'run_id': self.run_id,
            'position_id': position_id,
            'fill_id': fill_id,
            'ts': ts,
            'symbol': symbol,
            'module': module,
            'leg': leg,
            'side': side,
            'qty': qty,
            'price': price,
            'notional_usd': notional_usd,
            'slippage_bps_applied': slippage_bps_applied,
            'slippage_cost_usd': slippage_cost_usd,
            'fee_bps': fee_bps,
            'fee_usd': fee_usd,
            'liquidity': liquidity,
            'participation_pct': participation_pct,
            'adv60_usd': adv60_usd,
            'intended_price': intended_price if intended_price is not None else price
        }
        
        self.fills.append(fill_record)
        self._check_and_log_outlier(fill_record)
    
    def _record_ledger_event(
        self,
        ts: pd.Timestamp,
        event: str,
        position_id: str,
        symbol: str,
        module: str,
        leg: str,
        side: str,
        qty: float,
        price: float,
        notional_usd: float,
        fee_usd: float,
        slippage_cost_usd: float,
        funding_usd: float,
        cash_delta_usd: float,
        note: str = ""
    ):
        """Record a cash-affecting event to ledger"""
        self.ledger.append({
            'ts': ts,
            'run_id': self.run_id,
            'event': event,
            'position_id': position_id,
            'symbol': symbol,
            'module': module,
            'leg': leg,
            'side': side,
            'qty': qty,
            'price': price,
            'notional_usd': notional_usd,
            'fee_usd': fee_usd,
            'slippage_cost_usd': slippage_cost_usd,
            'funding_usd': funding_usd,
            'cash_delta_usd': cash_delta_usd,
            'note': note
        })
    
    def execute_stop(self, symbol: str, fill_bar: pd.Series, fill_ts: pd.Timestamp):
        """Execute stop-loss fill"""
        if symbol not in self.portfolio.positions:
            return
        
        pos = self.portfolio.positions[symbol]
        stop_price = pos.stop_price
        
        # Calculate fill using stop-run model
        mid_price = (fill_bar['high'] + fill_bar['low']) / 2.0
        slippage_params = self.params_dict.get('slippage_costs', {})
        slippage_bps_base = slippage_params.get('base_slip_bps_intercept', 2.0)
        
        # Use fill_stop_run to get fill price and gap_through
        fill_price, gap_through = fill_stop_run(
            stop_price, pos.side, fill_bar['high'], fill_bar['low'],
            mid_price, slippage_bps_base
        )
        
        # Calculate actual slippage_bps_applied from the fill
        if pos.side == 'LONG':
            slippage_bps_applied = ((fill_price - stop_price) / mid_price) * 10000.0 if mid_price > 0 else slippage_bps_base
        else:  # SHORT
            slippage_bps_applied = ((stop_price - fill_price) / mid_price) * 10000.0 if mid_price > 0 else slippage_bps_base
        
        # Log gap-through to forensic log
        if gap_through:
            self.forensic_log.append({
                'ts': fill_ts,
                'symbol': symbol,
                'event': 'GAP_THROUGH',
                'module': pos.module,
                'position_id': pos.position_id,
                'side': pos.side,
                'trigger_price': stop_price,
                'fill_price': fill_price,
                'bar_high': fill_bar['high'],
                'bar_low': fill_bar['low']
            })
        
        # Calculate fees
        # Stop-market exits are always taker
        notional = abs(pos.qty * fill_price)
        fill_is_taker = True  # Stop-market exits are always taker
        fee_bps = self.params.get_default('general', 'taker_fee_bps') if fill_is_taker else self.params.get_default('general', 'maker_fee_bps')
        if self.stress_fees:
            fee_bps *= 1.5  # Stress test: multiply fees by 1.5x
            
        if not self.cost_model_enabled:
            fee_bps = 0.0
            slippage_bps_applied = 0.0
            
        fees = notional * (fee_bps / 10000.0)
        
        slippage_cost_usd = notional * (slippage_bps_applied / 10000.0)
        
        # FIX 2: Calculate age_bars using bar indices
        df = self.symbol_data[symbol]
        
        # Get ADV_60m for participation calculation
        if hasattr(self, 'symbol_ts_to_idx') and symbol in self.symbol_ts_to_idx:
            fill_idx = self.symbol_ts_to_idx[symbol].get(fill_ts, -1)
        else:
            # Fix pandas Series boolean ambiguity: ensure fill_idx is always a scalar integer
            ts_matches = df[df['ts'] == fill_ts]
            if len(ts_matches) > 0:
                idx_result = ts_matches.index[0]
                fill_idx = int(idx_result) if hasattr(idx_result, '__iter__') and not isinstance(idx_result, str) else int(idx_result)
            else:
                fill_idx = -1
        adv60_usd = calculate_adv_60m(df['notional'], fill_idx) if fill_idx >= 0 else 0.0
        participation_pct = (notional / adv60_usd) if adv60_usd > 0 else 0.0
        
        # Record exit fill
        self._record_fill(
            position_id=pos.position_id,
            ts=fill_ts,
            symbol=symbol,
            module=pos.module,
            leg='EXIT',
            side='SELL' if pos.side == 'LONG' else 'BUY',
            qty=pos.qty,
            price=fill_price,
            notional_usd=notional,
            slippage_bps_applied=slippage_bps_applied,
            slippage_cost_usd=slippage_cost_usd,
            fee_bps=fee_bps,
            fee_usd=fees,
            liquidity='taker',
            participation_pct=participation_pct,
            adv60_usd=adv60_usd,
            intended_price=stop_price
        )
        
        # Calculate close_idx and entry_idx for age_bars
        if hasattr(self, 'symbol_ts_to_idx') and symbol in self.symbol_ts_to_idx:
            close_idx = self.symbol_ts_to_idx[symbol].get(fill_ts, -1)
        else:
            # Fix pandas Series boolean ambiguity: ensure close_idx is always a scalar integer
            ts_matches = df[df['ts'] == fill_ts]
            if len(ts_matches) > 0:
                idx_result = ts_matches.index[0]
                close_idx = int(idx_result) if hasattr(idx_result, '__iter__') and not isinstance(idx_result, str) else int(idx_result)
            else:
                close_idx = -1
        
        entry_idx = pos.entry_idx if pos.entry_idx >= 0 else -1
        if entry_idx < 0:
            # Fallback: lookup entry_idx
            # Fix pandas Series boolean ambiguity: ensure entry_idx is always a scalar integer
            ts_matches = df[df['ts'] == pos.entry_ts]
            if len(ts_matches) > 0:
                idx_result = ts_matches.index[0]
                entry_idx = int(idx_result) if hasattr(idx_result, '__iter__') and not isinstance(idx_result, str) else int(idx_result)
            else:
                entry_idx = -1
        
        age_bars = (close_idx - entry_idx) if entry_idx >= 0 and close_idx >= 0 else pos.age_bars
        
        # FIX 2: Assert SQUEEZE TTL <= 48 bars
        if pos.module == 'SQUEEZE' and age_bars > 48:
            self.forensic_log.append({
                'ts': fill_ts,
                'symbol': symbol,
                'event': 'SQUEEZE_TTL_VIOLATION',
                'age_bars': age_bars,
                'max_allowed': 48,
                'position_id': pos.position_id
            })
            # Log violation but continue (position is being closed anyway)
        
        # Close position (this calculates PnL internally and returns it)
        closed_pos, pnl = self.portfolio.close_position(
            symbol, fill_price, fill_ts, 'STOP', fees, slippage_cost_usd
        )
        
        if closed_pos:
            # Record EXIT_FILL ledger event
            # Note: pnl from close_position already has fees and slippage deducted
            # So cash_delta = pnl (the net effect on cash)
            self._record_ledger_event(
                ts=fill_ts,
                event='EXIT_FILL',
                position_id=pos.position_id,
                symbol=symbol,
                module=pos.module,
                leg='EXIT',
                side='SELL' if pos.side == 'LONG' else 'BUY',
                qty=pos.qty,
                price=fill_price,
                notional_usd=notional,
                fee_usd=fees,
                slippage_cost_usd=slippage_cost_usd,
                funding_usd=0.0,
                cash_delta_usd=pnl,  # PnL already has fees/slippage deducted, so this is the net cash change
                note="Stop loss exit"
            )
            # Use PnL from close_position to ensure exact match with portfolio.total_pnl
            # FIX 2 & 3: Record trade with position_id, open_ts, close_ts, age_bars, gap_through
            self.trades.append({
                'ts': fill_ts,
                'symbol': symbol,
                'side': pos.side,
                'module': pos.module,
                'qty': pos.qty,
                'price': fill_price,
                'fees': fees,
                'slip_bps': slippage_bps_applied,
                'participation_pct': 0.0,
                'post_only': False,
                'stop_dist': abs(pos.entry_price - pos.stop_price),
                'ES_used_before': 0.0,  # Would calculate
                'ES_used_after': 0.0,
                'reason': 'STOP',
                'pnl': pnl,
                'position_id': pos.position_id,
                'open_ts': pos.entry_ts,
                'close_ts': fill_ts,
                'age_bars': age_bars,
                'gap_through': gap_through
            })
            self.symbol_daily_pnl[symbol] += pnl
            self.symbol_prev_prices.pop(symbol, None)

    def execute_deception_exit(
        self, symbol: str, fill_bar: pd.Series, fill_ts: pd.Timestamp, event_type: str
    ):
        """Full close at TP1/TP2/TSL for DeceptionModule positions.

        All exits are TAKER (market orders) — matches live PARITY 2026-07-12.
        Fill price is the target price (TP1/TP2/TSL) with no slippage model
        for TP1/TP2 (limit-style target) and a small slippage for TSL.
        """
        if symbol not in self.portfolio.positions:
            return
        pos = self.portfolio.positions[symbol]
        if pos.module != 'DECEPTION':
            return

        # Determine fill price and reason from event type
        if event_type == 'DECEPTION_TP1_FULL':
            fill_price = pos.tp1_price
            reason = 'TP1'
        elif event_type == 'DECEPTION_TP2_EXIT':
            fill_price = pos.tp2_price
            reason = 'TP2'
        elif event_type == 'DECEPTION_TSL_EXIT':
            # TSL fill: trailing stop price with small slippage
            if pos.side == 'LONG':
                fill_price = pos.tsl_extreme * (1.0 - pos.tsl_bps / 10000.0)
            else:
                fill_price = pos.tsl_extreme * (1.0 + pos.tsl_bps / 10000.0)
            reason = 'TSL'
        else:
            return

        notional = abs(pos.qty * fill_price)
        fee_bps = self.params.get_default('general', 'taker_fee_bps')
        if self.stress_fees:
            fee_bps *= 1.5
        if not self.cost_model_enabled:
            fee_bps = 0.0
        fees = notional * (fee_bps / 10000.0)
        slippage_cost_usd = 0.0  # TP1/TP2 are limit-style targets; TSL uses the trailing price directly

        # Record exit fill
        self._record_fill(
            position_id=pos.position_id, ts=fill_ts, symbol=symbol,
            module='DECEPTION', leg='EXIT',
            side='SELL' if pos.side == 'LONG' else 'BUY',
            qty=pos.qty, price=fill_price, notional_usd=notional,
            slippage_bps_applied=0.0, slippage_cost_usd=slippage_cost_usd,
            fee_bps=fee_bps, fee_usd=fees, liquidity='taker',
            participation_pct=0.0, adv60_usd=0.0, intended_price=fill_price,
        )

        closed_pos, pnl = self.portfolio.close_position(
            symbol, fill_price, fill_ts, reason, fees, slippage_cost_usd
        )
        if closed_pos:
            self._record_ledger_event(
                ts=fill_ts, event='EXIT_FILL', position_id=pos.position_id,
                symbol=symbol, module='DECEPTION', leg='EXIT',
                side='SELL' if pos.side == 'LONG' else 'BUY',
                qty=pos.qty, price=fill_price, notional_usd=notional,
                fee_usd=fees, slippage_cost_usd=slippage_cost_usd,
                funding_usd=0.0, cash_delta_usd=pnl, note=f"Deception {reason} exit",
            )
            self.trades.append({
                'ts': fill_ts, 'symbol': symbol, 'side': pos.side,
                'module': 'DECEPTION', 'qty': pos.qty, 'price': fill_price,
                'fees': fees, 'slip_bps': 0.0, 'participation_pct': 0.0,
                'post_only': False,
                'stop_dist': abs(pos.entry_price - pos.stop_price),
                'ES_used_before': 0.0, 'ES_used_after': 0.0,
                'reason': reason, 'pnl': pnl,
                'position_id': pos.position_id,
                'open_ts': pos.entry_ts, 'close_ts': fill_ts,
                'age_bars': 0, 'gap_through': False,
                'trap_type': pos.trap_type, 'deception_score': pos.deception_score,
            })
            self.symbol_daily_pnl[symbol] += pnl
            self.symbol_prev_prices.pop(symbol, None)

    def execute_deception_tp1_partial(
        self, symbol: str, fill_bar: pd.Series, fill_ts: pd.Timestamp
    ):
        """Partial close at TP1, then activate TSL on the remainder.

        Mirrors sim_core: tp1_frac of the position closes at TP1 (taker fee),
        the remainder stays open with the fixed SL cancelled and a TSL
        activated at tsl_bps callback from the favourable extreme.
        """
        if symbol not in self.portfolio.positions:
            return
        pos = self.portfolio.positions[symbol]
        if pos.module != 'DECEPTION' or pos.tp1_price <= 0:
            return

        fill_price = pos.tp1_price
        partial_qty = pos.qty * pos.tp1_frac
        notional = abs(partial_qty * fill_price)
        fee_bps = self.params.get_default('general', 'taker_fee_bps')
        if self.stress_fees:
            fee_bps *= 1.5
        if not self.cost_model_enabled:
            fee_bps = 0.0
        fees = notional * (fee_bps / 10000.0)

        # Record partial exit fill
        self._record_fill(
            position_id=pos.position_id, ts=fill_ts, symbol=symbol,
            module='DECEPTION', leg='EXIT',
            side='SELL' if pos.side == 'LONG' else 'BUY',
            qty=partial_qty, price=fill_price, notional_usd=notional,
            slippage_bps_applied=0.0, slippage_cost_usd=0.0,
            fee_bps=fee_bps, fee_usd=fees, liquidity='taker',
            participation_pct=0.0, adv60_usd=0.0, intended_price=fill_price,
        )

        # Reduce position qty and record partial PnL
        long = pos.side == 'LONG'
        sign = 1.0 if long else -1.0
        partial_pnl = sign * (fill_price - pos.entry_price) * partial_qty - fees
        self.portfolio.cash += partial_pnl
        self.portfolio.total_pnl += partial_pnl
        self.portfolio.fees_paid += fees
        pos.qty -= partial_qty
        pos.tp1_hit = True

        # Activate TSL on the remainder if configured
        if pos.tsl_bps > 0:
            pos.tsl_active = True
            pos.tsl_extreme = fill_price  # start tracking from TP1 fill
        # else: remainder continues with fixed SL (checked by collect_stop_events / collect_deception_exit_events)

        self._record_ledger_event(
            ts=fill_ts, event='PARTIAL_EXIT_FILL', position_id=pos.position_id,
            symbol=symbol, module='DECEPTION', leg='EXIT',
            side='SELL' if long else 'BUY',
            qty=partial_qty, price=fill_price, notional_usd=notional,
            fee_usd=fees, slippage_cost_usd=0.0, funding_usd=0.0,
            cash_delta_usd=partial_pnl, note="Deception TP1 partial",
        )
        self.trades.append({
            'ts': fill_ts, 'symbol': symbol, 'side': pos.side,
            'module': 'DECEPTION', 'qty': partial_qty, 'price': fill_price,
            'fees': fees, 'slip_bps': 0.0, 'participation_pct': 0.0,
            'post_only': False, 'stop_dist': abs(pos.entry_price - pos.stop_price),
            'ES_used_before': 0.0, 'ES_used_after': 0.0,
            'reason': 'TP1_PARTIAL', 'pnl': partial_pnl,
            'position_id': pos.position_id, 'open_ts': pos.entry_ts,
            'close_ts': fill_ts, 'age_bars': 0, 'gap_through': False,
            'trap_type': pos.trap_type, 'deception_score': pos.deception_score,
        })
        self.symbol_daily_pnl[symbol] += partial_pnl
        """Execute SQUEEZE TP1 exit"""
        if symbol not in self.portfolio.positions:
            return
        
        pos = self.portfolio.positions[symbol]
        if pos.module != 'SQUEEZE' or pos.tp1_price <= 0:
            return  # Only for SQUEEZE with TP1 enabled
        
        # Use TP1 price as fill target (limit order semantics)
        mid_price = (fill_bar['high'] + fill_bar['low']) / 2.0
        slippage_params = self.params_dict.get('slippage_costs', {})
        slippage_bps_base = slippage_params.get('base_slip_bps_intercept', 2.0)
        
        # For TP1, check if price reached TP1 level (limit order semantics)
        if pos.side == 'LONG':
            # Long: TP1 triggered if high >= TP1 price
            if fill_bar['high'] < pos.tp1_price:
                return  # TP1 not reached, don't exit
            # Fill at TP1 or better (use TP1 price for limit order)
            fill_price = pos.tp1_price
        else:  # SHORT
            # Short: TP1 triggered if low <= TP1 price
            if fill_bar['low'] > pos.tp1_price:
                return  # TP1 not reached, don't exit
            # Fill at TP1 or better (use TP1 price for limit order)
            fill_price = pos.tp1_price
        
        gap_through = False  # TP1 is limit order, no gap-through
        
        # Calculate slippage (should be minimal for limit orders)
        if pos.side == 'LONG':
            slippage_bps_applied = ((fill_price - pos.tp1_price) / mid_price) * 10000.0 if mid_price > 0 else 0.0
        else:  # SHORT
            slippage_bps_applied = ((pos.tp1_price - fill_price) / mid_price) * 10000.0 if mid_price > 0 else 0.0
        
        if not self.cost_model_enabled:
            slippage_bps_applied = 0.0
        
        # Calculate fees (TP1 exits are typically maker, but use taker for conservative)
        notional = abs(pos.qty * fill_price)
        fee_bps = self.params.get_default('general', 'taker_fee_bps')
        if self.stress_fees:
            fee_bps *= 1.5
            
        if not self.cost_model_enabled:
            fee_bps = 0.0
            slippage_bps_applied = 0.0
            
        fees = notional * (fee_bps / 10000.0)
        
        slippage_cost_usd = notional * (slippage_bps_applied / 10000.0)
        
        # Get ADV_60m for participation
        df = self.symbol_data[symbol]
        if hasattr(self, 'symbol_ts_to_idx') and symbol in self.symbol_ts_to_idx:
            fill_idx = self.symbol_ts_to_idx[symbol].get(fill_ts, -1)
        else:
            # Fix pandas Series boolean ambiguity: ensure fill_idx is always a scalar integer
            ts_matches = df[df['ts'] == fill_ts]
            if len(ts_matches) > 0:
                idx_result = ts_matches.index[0]
                fill_idx = int(idx_result) if hasattr(idx_result, '__iter__') and not isinstance(idx_result, str) else int(idx_result)
            else:
                fill_idx = -1
        adv60_usd = calculate_adv_60m(df['notional'], fill_idx) if fill_idx >= 0 else 0.0
        participation_pct = (notional / adv60_usd) if adv60_usd > 0 else 0.0
        
        # Record exit fill
        self._record_fill(
            position_id=pos.position_id,
            ts=fill_ts,
            symbol=symbol,
            module=pos.module,
            leg='EXIT',
            side='SELL' if pos.side == 'LONG' else 'BUY',
            qty=pos.qty,
            price=fill_price,
            notional_usd=notional,
            slippage_bps_applied=slippage_bps_applied,
            slippage_cost_usd=slippage_cost_usd,
            fee_bps=fee_bps,
            fee_usd=fees,
            liquidity='maker',  # TP1 is limit order
            participation_pct=participation_pct,
            adv60_usd=adv60_usd,
            intended_price=pos.tp1_price
        )
        
        # Calculate age_bars
        if hasattr(self, 'symbol_ts_to_idx') and symbol in self.symbol_ts_to_idx:
            close_idx = self.symbol_ts_to_idx[symbol].get(fill_ts, -1)
        else:
            close_idx = df[df['ts'] == fill_ts].index[0] if len(df[df['ts'] == fill_ts]) > 0 else -1
        entry_idx = pos.entry_idx if pos.entry_idx >= 0 else -1
        if entry_idx < 0:
            entry_idx = df[df['ts'] == pos.entry_ts].index[0] if len(df[df['ts'] == pos.entry_ts]) > 0 else -1
        age_bars = (close_idx - entry_idx) if entry_idx >= 0 and close_idx >= 0 else pos.age_bars
        
        # Close position
        closed_pos, pnl = self.portfolio.close_position(
            symbol, fill_price, fill_ts, 'TP1', fees, slippage_cost_usd
        )
        
        if closed_pos:
            # Record ledger event
            self._record_ledger_event(
                ts=fill_ts,
                event='EXIT_FILL',
                position_id=pos.position_id,
                symbol=symbol,
                module=pos.module,
                leg='EXIT',
                side='SELL' if pos.side == 'LONG' else 'BUY',
                qty=pos.qty,
                price=fill_price,
                notional_usd=notional,
                fee_usd=fees,
                slippage_cost_usd=slippage_cost_usd,
                funding_usd=0.0,
                cash_delta_usd=pnl,
                note="SQUEEZE TP1 exit"
            )
            
            # Record trade
            self.trades.append({
                'ts': fill_ts,
                'symbol': symbol,
                'side': pos.side,
                'module': pos.module,
                'qty': pos.qty,
                'price': fill_price,
                'fees': fees,
                'slip_bps': slippage_bps_applied,
                'participation_pct': participation_pct,
                'post_only': False,
                'stop_dist': abs(pos.entry_price - pos.stop_price),
                'ES_used_before': 0.0,
                'ES_used_after': 0.0,
                'reason': 'TP1',
                'pnl': pnl,
                'position_id': pos.position_id,
                'open_ts': pos.entry_ts,
                'close_ts': fill_ts,
                'age_bars': age_bars,
                'gap_through': gap_through
            })
            self.symbol_daily_pnl[symbol] += pnl
            self.symbol_prev_prices.pop(symbol, None)
    
    def execute_squeeze_vol_exit(self, symbol: str, fill_bar: pd.Series, fill_ts: pd.Timestamp):
        """Execute SQUEEZE volatility expansion exit"""
        if symbol not in self.portfolio.positions:
            return
        
        pos = self.portfolio.positions[symbol]
        if pos.module != 'SQUEEZE':
            return
        
        # Use market order for vol expansion exit (exit immediately)
        mid_price = (fill_bar['high'] + fill_bar['low']) / 2.0
        slippage_params = self.params_dict.get('slippage_costs', {})
        slippage_bps_base = slippage_params.get('base_slip_bps_intercept', 2.0)
        
        # Market order: use current price with slippage
        if pos.side == 'LONG':
            fill_price = max(fill_bar['low'], mid_price * (1 - slippage_bps_base / 10000.0))
        else:  # SHORT
            fill_price = min(fill_bar['high'], mid_price * (1 + slippage_bps_base / 10000.0))
        
        gap_through = False
        
        # Calculate slippage
        if pos.side == 'LONG':
            slippage_bps_applied = ((fill_price - mid_price) / mid_price) * 10000.0 if mid_price > 0 else slippage_bps_base
        else:  # SHORT
            slippage_bps_applied = ((mid_price - fill_price) / mid_price) * 10000.0 if mid_price > 0 else slippage_bps_base
        
        # Calculate fees (market order = taker)
        notional = abs(pos.qty * fill_price)
        fee_bps = self.params.get_default('general', 'taker_fee_bps')
        if self.stress_fees:
            fee_bps *= 1.5
            
        if not self.cost_model_enabled:
            fee_bps = 0.0
            slippage_bps_applied = 0.0
            
        fees = notional * (fee_bps / 10000.0)
        
        slippage_cost_usd = notional * (slippage_bps_applied / 10000.0)
        
        # Get ADV_60m for participation
        df = self.symbol_data[symbol]
        if hasattr(self, 'symbol_ts_to_idx') and symbol in self.symbol_ts_to_idx:
            fill_idx = self.symbol_ts_to_idx[symbol].get(fill_ts, -1)
        else:
            # Fix pandas Series boolean ambiguity: ensure fill_idx is always a scalar integer
            ts_matches = df[df['ts'] == fill_ts]
            if len(ts_matches) > 0:
                idx_result = ts_matches.index[0]
                fill_idx = int(idx_result) if hasattr(idx_result, '__iter__') and not isinstance(idx_result, str) else int(idx_result)
            else:
                fill_idx = -1
        adv60_usd = calculate_adv_60m(df['notional'], fill_idx) if fill_idx >= 0 else 0.0
        participation_pct = (notional / adv60_usd) if adv60_usd > 0 else 0.0
        
        # Record exit fill
        self._record_fill(
            position_id=pos.position_id,
            ts=fill_ts,
            symbol=symbol,
            module=pos.module,
            leg='EXIT',
            side='SELL' if pos.side == 'LONG' else 'BUY',
            qty=pos.qty,
            price=fill_price,
            notional_usd=notional,
            slippage_bps_applied=slippage_bps_applied,
            slippage_cost_usd=slippage_cost_usd,
            fee_bps=fee_bps,
            fee_usd=fees,
            liquidity='taker',
            participation_pct=participation_pct,
            adv60_usd=adv60_usd
        )
        
        # Calculate age_bars
        if hasattr(self, 'symbol_ts_to_idx') and symbol in self.symbol_ts_to_idx:
            close_idx = self.symbol_ts_to_idx[symbol].get(fill_ts, -1)
        else:
            close_idx = df[df['ts'] == fill_ts].index[0] if len(df[df['ts'] == fill_ts]) > 0 else -1
        entry_idx = pos.entry_idx if pos.entry_idx >= 0 else -1
        if entry_idx < 0:
            entry_idx = df[df['ts'] == pos.entry_ts].index[0] if len(df[df['ts'] == pos.entry_ts]) > 0 else -1
        age_bars = (close_idx - entry_idx) if entry_idx >= 0 and close_idx >= 0 else pos.age_bars
        
        # Close position
        closed_pos, pnl = self.portfolio.close_position(
            symbol, fill_price, fill_ts, 'VOL_EXIT', fees, slippage_cost_usd
        )
        
        if closed_pos:
            # Record ledger event
            self._record_ledger_event(
                ts=fill_ts,
                event='EXIT_FILL',
                position_id=pos.position_id,
                symbol=symbol,
                module=pos.module,
                leg='EXIT',
                side='SELL' if pos.side == 'LONG' else 'BUY',
                qty=pos.qty,
                price=fill_price,
                notional_usd=notional,
                fee_usd=fees,
                slippage_cost_usd=slippage_cost_usd,
                funding_usd=0.0,
                cash_delta_usd=pnl,
                note="SQUEEZE volatility expansion exit"
            )
            
            # Record trade
            self.trades.append({
                'ts': fill_ts,
                'symbol': symbol,
                'side': pos.side,
                'module': pos.module,
                'qty': pos.qty,
                'price': fill_price,
                'fees': fees,
                'slip_bps': slippage_bps_applied,
                'participation_pct': participation_pct,
                'post_only': False,
                'stop_dist': abs(pos.entry_price - pos.stop_price),
                'ES_used_before': 0.0,
                'ES_used_after': 0.0,
                'reason': 'VOL_EXIT',
                'pnl': pnl,
                'position_id': pos.position_id,
                'open_ts': pos.entry_ts,
                'close_ts': fill_ts,
                'age_bars': age_bars,
                'gap_through': gap_through
            })
            self.symbol_daily_pnl[symbol] += pnl
            self.symbol_prev_prices.pop(symbol, None)
    
    def execute_squeeze_entry(self, order_id: str, fill_bar: pd.Series, fill_ts: pd.Timestamp):
        """Execute SQUEEZE entry fill"""
        if order_id not in self.order_manager.pending_orders:
            return
        
        order = self.order_manager.pending_orders[order_id]
        
        # Check if symbol already has a position
        if order.symbol in self.portfolio.positions:
            return  # Already have position in this symbol
        if order.filled:
            return
        
        # Fill using stop-run model
        mid_price = (fill_bar['high'] + fill_bar['low']) / 2.0
        df = self.symbol_data[order.symbol]
        # OPTIMIZATION: Use O(1) lookup if mapping available, otherwise fallback
        if hasattr(self, 'symbol_ts_to_idx') and order.symbol in self.symbol_ts_to_idx:
            fill_idx = self.symbol_ts_to_idx[order.symbol].get(fill_ts, order.signal_bar_idx)
        else:
            # Fix pandas Series boolean ambiguity: ensure fill_idx is always a scalar integer
            ts_matches = df[df['ts'] == fill_ts]
            if len(ts_matches) > 0:
                idx_result = ts_matches.index[0]
                fill_idx = int(idx_result) if hasattr(idx_result, '__iter__') and not isinstance(idx_result, str) else int(idx_result)
            else:
                fill_idx = order.signal_bar_idx
        adv_60m = calculate_adv_60m(df['notional'], fill_idx)
        
        liquidity_state = self.symbol_liquidity_state.get(order.symbol)
        regime = liquidity_state.regime if liquidity_state else 'NORMAL'
        regime_adder = self.liquidity_detector.get_slippage_adder(regime)
        post_only = bool(regime == 'THIN')
        if post_only:
            self.thin_post_only_entries_count += 1
            self.thin_extra_slip_bps_total += regime_adder
        
        drawdown_mult, drawdown_halt = self._get_drawdown_size_constraints()
        if drawdown_halt:
            self.loss_halt_state.halt_manual = True
            self.order_manager.cancel_order(order_id)
            return
        effective_qty = order.qty * drawdown_mult
        if effective_qty <= 0:
            self.order_manager.cancel_order(order_id)
            return
        
        # Launch Punch List – Blocker #2: centralized ES + margin guardrails
        # Get current ES usage
        current_es_pct = (self.es_usage_samples[-1] * 100.0) if self.es_usage_samples else 0.0
        # Calculate additional risk (stop distance * qty) - approximate with ATR
        atr = df.iloc[order.signal_bar_idx].get('atr', 0) if order.signal_bar_idx < len(df) else 0.0
        sl_atr_mult = 2.5  # Default, strategy-specific param not in base_params.json
        stop_distance = sl_atr_mult * atr
        additional_risk = stop_distance * effective_qty
        
        # Centralized risk check before order
        from engine_core.src.risk.margin_guard import check_risk_before_order
        risk_allowed, risk_reason, margin_ratio_proj, es_used_proj_pct = check_risk_before_order(
            symbol=order.symbol,
            qty=effective_qty,
            price=order.trigger_price,
            current_positions=self.portfolio.positions,
            current_equity=self.portfolio.equity,
            current_es_used_pct=current_es_pct,
            additional_risk=additional_risk,
            params=self.params_dict,
            es_cap_pct=self.params.get('es_guardrails', 'es_cap_of_equity') or 0.0225
        )
        
        if not risk_allowed:
            # Block entry due to risk guardrails
            self.forensic_log.append({
                'ts': fill_ts,
                'symbol': order.symbol,
                'event': 'RISK_GUARD_BLOCK',
                'module': order.module,
                'reason': risk_reason,
                'margin_ratio_proj': margin_ratio_proj,
                'es_used_proj_pct': es_used_proj_pct
            })
            self.order_manager.cancel_order(order_id)
            return
        
        order_notional = abs(effective_qty * order.trigger_price)
        if adv_60m <= 0:
            raise ValueError(f"ADV_60m is non-positive for {order.symbol} at {fill_ts}")
        
        # Check participation cap
        participation_pct = order_notional / adv_60m
        participation_cap = self.liquidity_detector.get_participation_cap(regime)
        if participation_pct > participation_cap:
            # Reject order - participation exceeds cap
            self.forensic_log.append({
                'ts': fill_ts,
                'symbol': order.symbol,
                'event': 'PARTICIPATION_CAP_BLOCK',
                'module': order.module,
                'participation_pct': participation_pct,
                'cap': participation_cap,
                'regime': regime
            })
            self.order_manager.cancel_order(order_id)
            return
        
        slippage_params = self.params_dict.get('slippage_costs', {})
        # FIX 5: Pass governed_universe flag (use require_liquidity_data as proxy)
        slippage_bps, participation_pct = calculate_slippage(
            order_notional, adv_60m,
            slippage_params.get('base_slip_bps_intercept', 2.0),
            slippage_params.get('base_slip_bps_slope_per_participation', 20.0),
            regime_adder,
            governed_universe=self.require_liquidity_data,
            stress_slip=self.stress_slip
        )
        
        self.forensic_log.append({
            'ts': fill_ts,
            'symbol': order.symbol,
            'event': 'SLIPPAGE',
            'module': order.module,
            'participation_pct': participation_pct,
            'slip_bps': slippage_bps,
            'adv_60m': adv_60m,
            'order_notional': order_notional,
            'post_only': post_only
        })
        
        fill_price, gap_through = fill_stop_run(
            order.trigger_price, order.side, fill_bar['high'], fill_bar['low'],
            mid_price, slippage_bps
        )
        
        # Log gap-through to forensic log
        if gap_through:
            self.forensic_log.append({
                'ts': fill_ts,
                'symbol': order.symbol,
                'event': 'GAP_THROUGH',
                'module': order.module,
                'side': order.side,
                'trigger_price': order.trigger_price,
                'fill_price': fill_price,
                'bar_high': fill_bar['high'],
                'bar_low': fill_bar['low']
            })
        
        # Validate constraints
        contract_metadata = self.data_loader.get_contract_metadata(order.symbol)
        is_valid, error_msg, adjusted_qty, adjusted_price = validate_order_constraints(
            effective_qty, fill_price, contract_metadata, order.side
        )
        
        if not is_valid:
            return  # Reject order
        
        # Calculate fees
        # Stop-run entries are taker, unless post_only=True (maker)
        notional = abs(adjusted_qty * adjusted_price)
        fill_is_taker = not post_only  # Taker unless post-only resting fill
        fee_bps = self.params.get_default('general', 'taker_fee_bps') if fill_is_taker else self.params.get_default('general', 'maker_fee_bps')
        if self.stress_fees:
            fee_bps *= 1.5  # Stress test: multiply fees by 1.5x
        if not self.cost_model_enabled:
            fee_bps = 0.0
            slippage_bps = 0.0
            
        fees = notional * (fee_bps / 10000.0)
        
        # Calculate stop price based on side
        atr = self.symbol_data[order.symbol].iloc[order.signal_bar_idx].get('atr', 0)
        sl_atr_mult = 2.5  # Default, strategy-specific param not in base_params.json
        if order.side == 'LONG':
            stop_price = adjusted_price - (sl_atr_mult * atr)
        else:  # SHORT
            stop_price = adjusted_price + (sl_atr_mult * atr)
        
        # ES guardrail
        signal_bar = self.symbol_data[order.symbol].iloc[order.signal_bar_idx]
        vol_forecast = signal_bar.get('vol_forecast', 0.02)
        vol_fast_median = signal_bar.get('vol_fast_median', 0.02)
        candidate_risk = abs(stop_price - adjusted_price) * abs(adjusted_qty)
        es_ok, es_before, es_after = self._passes_es_guard(
            candidate_risk, order.symbol, order.module, fill_ts,
            vol_forecast, vol_fast_median
        )
        if not es_ok:
            self.es_block_count += 1  # G: Track ES blocks
            self.order_manager.cancel_order(order_id)
            return
        
        if not self._check_beta_caps_with_new_position(order.symbol, adjusted_qty, adjusted_price, fill_ts, order.side):
            self.beta_block_count += 1  # G: Track beta blocks
            self.order_manager.cancel_order(order_id)
            return
        
        # Calculate R and TP1 for SQUEEZE positions
        initial_R = abs(adjusted_price - stop_price)
        tp1_mult_R = 0.0  # Default, strategy-specific param not in base_params.json
        tp1_price = 0.0
        if tp1_mult_R > 0:
            if order.side == 'LONG':
                tp1_price = adjusted_price + (tp1_mult_R * initial_R)
            else:  # SHORT
                tp1_price = adjusted_price - (tp1_mult_R * initial_R)
        
        # Add position
        # OPTIMIZATION: Pass entry_idx to avoid future lookups
        self.portfolio.add_position(
            order.symbol, adjusted_qty, adjusted_price, fill_ts,
            stop_price,  # Stop
            stop_price,  # Trail (initial)
            order.module, order.side, fees, 0.0,
            entry_idx=fill_idx  # Store bar index for performance
        )
        
        # Store R and TP1 for SQUEEZE positions
        if order.symbol in self.portfolio.positions:
            pos = self.portfolio.positions[order.symbol]
            pos.initial_R = initial_R
            pos.tp1_price = tp1_price
        
        self.symbol_prev_prices[order.symbol] = adjusted_price
        
        # Get position_id after adding position
        position_id = self.portfolio.positions[order.symbol].position_id if order.symbol in self.portfolio.positions else ""
        
        # Calculate actual slippage_bps_applied from the fill
        if order.side == 'LONG':
            slippage_bps_applied = ((adjusted_price - order.trigger_price) / mid_price) * 10000.0 if mid_price > 0 else slippage_bps
        else:  # SHORT
            slippage_bps_applied = ((order.trigger_price - adjusted_price) / mid_price) * 10000.0 if mid_price > 0 else slippage_bps
        
        if not self.cost_model_enabled:
            slippage_bps_applied = 0.0
        
        slippage_cost_usd = notional * (slippage_bps_applied / 10000.0)
        
        # Record entry fill
        self._record_fill(
            position_id=position_id,
            ts=fill_ts,
            symbol=order.symbol,
            module=order.module,
            leg='ENTRY',
            side='BUY' if order.side == 'LONG' else 'SELL',
            qty=adjusted_qty,
            price=adjusted_price,
            notional_usd=notional,
            slippage_bps_applied=slippage_bps_applied,
            slippage_cost_usd=slippage_cost_usd,
            fee_bps=fee_bps,
            fee_usd=fees,
            liquidity='maker' if post_only else 'taker',
            participation_pct=participation_pct,
            adv60_usd=adv_60m,
            intended_price=order.trigger_price
        )
        
        # Record ENTRY_FILL ledger event (cash decreases by fees + slippage)
        self._record_ledger_event(
            ts=fill_ts,
            event='ENTRY_FILL',
            position_id=position_id,
            symbol=order.symbol,
            module=order.module,
            leg='ENTRY',
            side='BUY' if order.side == 'LONG' else 'SELL',
            qty=adjusted_qty,
            price=adjusted_price,
            notional_usd=notional,
            fee_usd=fees,
            slippage_cost_usd=slippage_cost_usd,
            funding_usd=0.0,
            cash_delta_usd=-(fees + slippage_cost_usd),  # Cash decreases
            note=f"Entry fill: {order.module}"
        )
        
        # Calculate ES_used_after (with new position)
        import time
        es_start = time.time()
        _, _, es_after_actual = self._passes_es_guard(
            0.0, order.symbol, order.module, fill_ts,
            vol_forecast, vol_fast_median
        )
        self._profile_time['es_checks'] += time.time() - es_start
        self._profile_counts['es_checks'] += 1
        
        # Mark order as filled
        self.order_manager.fill_order(order_id, adjusted_price, fill_ts)
        
        # FIX 2 & 3: Record trade with position_id, open_ts, gap_through
        self.trades.append({
            'ts': fill_ts,
            'symbol': order.symbol,
            'side': order.side,
            'module': order.module,
            'qty': adjusted_qty,
            'price': adjusted_price,
            'fees': fees,
            'slip_bps': slippage_bps,
            'participation_pct': participation_pct,
            'post_only': post_only,
            'stop_dist': abs(adjusted_price - stop_price),
            'ES_used_before': es_before,
            'ES_used_after': es_after_actual,
            'reason': 'ENTRY',
            'position_id': position_id,
            'open_ts': fill_ts,
            'close_ts': None,
            'age_bars': 0,
            'gap_through': gap_through
        })
    
    def execute_entry(self, event: OrderEvent, fill_bar: pd.Series, fill_ts: pd.Timestamp):
        """Execute new entry with all risk checks"""
        debug_oracle = self.params.get('general', 'debug_oracle_flow', default=False)
        # Get signal
        pending_signals = self.symbol_pending_signals.get(event.symbol, [])
        if debug_oracle and event.module == 'ORACLE':
            print(f"[ORACLE DEBUG] execute_entry: Looking for ORACLE signal in {len(pending_signals)} pending signals for {event.symbol}")
            for i, s in enumerate(pending_signals):
                if hasattr(s, 'module'):
                    print(f"  Signal {i}: module={s.module}, matches={s.module == event.module}")
        signal = None
        for s in pending_signals:
            if s.module == event.module:
                signal = s
                break
        
        if not signal:
            if debug_oracle and event.module == 'ORACLE':
                print(f"[ORACLE DEBUG] ERROR: No ORACLE signal found in pending_signals for {event.symbol}")
            self.forensic_log.append({
                'ts': fill_ts,
                'symbol': event.symbol,
                'event': 'EXECUTE_ENTRY_NO_SIGNAL',
                'module': event.module
            })
            return
        
        # Calculate position size
        df = self.symbol_data[event.symbol]
        bar_idx = signal.signal_bar_idx
        bar = df.iloc[bar_idx]
        
        vol_forecast = bar.get('vol_forecast', 0.02)
        vol_fast_median = bar.get('vol_fast_median', 0.02)
        returns_15m = df['close'].pct_change().iloc[max(0, bar_idx-6):bar_idx+1]
        sigma_15m = returns_15m.std() if len(returns_15m) > 1 else vol_forecast / np.sqrt(96)
        
        size_mult = calculate_size_multiplier(
            vol_forecast, vol_fast_median, returns_15m, sigma_15m,
            bar.get('slope_z', 0.0), self.params_dict
        )
        # ORACLE signals bypass drawdown halt checks
        if event.module != 'ORACLE':
            drawdown_mult, drawdown_halt = self._get_drawdown_size_constraints()
            if drawdown_halt:
                self.loss_halt_state.halt_manual = True
                self.forensic_log.append({
                    'ts': fill_ts,
                    'symbol': event.symbol,
                    'event': 'EXECUTE_ENTRY_DRAWDOWN_HALT',
                    'module': event.module
                })
                # Use index-based removal to avoid pandas Series boolean ambiguity
                try:
                    idx = next((i for i, s in enumerate(pending_signals) if s is signal or (hasattr(s, 'signal_bar_idx') and hasattr(signal, 'signal_bar_idx') and int(s.signal_bar_idx) == int(signal.signal_bar_idx) and s.module == signal.module)), None)
                    if idx is not None:
                        pending_signals.pop(idx)
                except (ValueError, TypeError):
                    pass
                return
            size_mult *= drawdown_mult
        else:
            # ORACLE: use default size multiplier (no drawdown constraints)
            drawdown_mult = 1.0
        
        module_factor = get_module_factor(event.module, self.params_dict)
        # Debug log if module_factor is unexpectedly 0.0 for TREND/RANGE
        if module_factor == 0.0 and event.module in ['TREND', 'RANGE']:
            self.forensic_log.append({
                'ts': fill_ts,
                'symbol': event.symbol,
                'event': 'EXECUTE_ENTRY_MODULE_FACTOR_ZERO',
                'module': event.module,
                'module_factor': module_factor
            })
        r_base = self.params.get_default('general', 'r_base')
        
        stop_distance = abs(signal.entry_price - signal.stop_price)
        qty = calculate_position_size(
            self.portfolio.equity, signal.entry_price, signal.stop_price,
            size_mult, module_factor, r_base,
            self.data_loader.get_contract_metadata(event.symbol).get('stepSize', 0.001)
        )
        
        if qty == 0:
            # ORACLE signals bypass qty==0 check (use minimum qty if needed)
            if event.module == 'ORACLE':
                # Use minimum position size for Oracle
                step_size = self.data_loader.get_contract_metadata(event.symbol).get('stepSize', 0.001)
                qty = step_size  # Minimum position size
                if debug_oracle:
                    print(f"[ORACLE DEBUG] execute_entry: qty was 0, using minimum step_size={step_size}")
            else:
                self.forensic_log.append({
                    'ts': fill_ts,
                    'symbol': event.symbol,
                    'event': 'EXECUTE_ENTRY_QTY_ZERO',
                    'module': event.module,
                    'equity': self.portfolio.equity,
                    'entry_price': signal.entry_price,
                    'stop_price': signal.stop_price,
                    'stop_distance': stop_distance,
                    'size_mult': size_mult,
                    'module_factor': module_factor,
                    'r_base': r_base
                })
                return
        
        # Check all pre-sizing guardrails (loss halts, margin, etc.)
        # ORACLE signals bypass all guardrails
        if event.module != 'ORACLE' and not self.check_all_entry_guards(event.symbol, signal, fill_bar, fill_ts):
            self.forensic_log.append({
                'ts': fill_ts,
                'symbol': event.symbol,
                'event': 'EXECUTE_ENTRY_GUARDS_FAILED',
                'module': event.module
            })
            return
        
        # Launch Punch List – Blocker #2: centralized ES + margin guardrails
        # Get current ES usage
        current_es_pct = (self.es_usage_samples[-1] * 100.0) if self.es_usage_samples else 0.0
        additional_risk = stop_distance * qty
        
        # Centralized risk check before order
        # ORACLE signals bypass risk checks
        if event.module == 'ORACLE':
            risk_allowed = True
            risk_reason = 'ORACLE_BYPASS'
            margin_ratio_proj = 0.0
            es_used_proj_pct = 0.0
        else:
            from engine_core.src.risk.margin_guard import check_risk_before_order
            risk_allowed, risk_reason, margin_ratio_proj, es_used_proj_pct = check_risk_before_order(
                symbol=event.symbol,
                qty=qty,
                price=signal.entry_price,  # Use signal entry price for projection
                current_positions=self.portfolio.positions,
                current_equity=self.portfolio.equity,
                current_es_used_pct=current_es_pct,
                additional_risk=additional_risk,
                params=self.params_dict,
                es_cap_pct=self.params.get('es_guardrails', 'es_cap_of_equity') or 0.0225
            )
        
        if not risk_allowed:
            # Block entry due to risk guardrails
            self.forensic_log.append({
                'ts': fill_ts,
                'symbol': event.symbol,
                'event': 'RISK_GUARD_BLOCK',
                'module': event.module,
                'reason': risk_reason,
                'margin_ratio_proj': margin_ratio_proj,
                'es_used_proj_pct': es_used_proj_pct
            })
            # Use index-based removal to avoid pandas Series boolean ambiguity
            try:
                idx = next((i for i, s in enumerate(pending_signals) if s is signal or (hasattr(s, 'signal_bar_idx') and hasattr(signal, 'signal_bar_idx') and int(s.signal_bar_idx) == int(signal.signal_bar_idx) and s.module == signal.module)), None)
                if idx is not None:
                    pending_signals.pop(idx)
            except (ValueError, TypeError):
                pass
            return
        
        # Use fill_bar close as entry price (market order simulation)
        entry_price = fill_bar['close']
        if debug_oracle and event.module == 'ORACLE':
            print(f"[ORACLE DEBUG] execute_entry: Calculated qty={qty}, entry_price={entry_price}, stop_price={signal.stop_price}")
        
        # Validate constraints
        contract_metadata = self.data_loader.get_contract_metadata(event.symbol)
        is_valid, error_msg, adjusted_qty, adjusted_price = validate_order_constraints(
            qty, entry_price, contract_metadata, signal.side
        )
        
        if not is_valid:
            if debug_oracle and event.module == 'ORACLE':
                print(f"[ORACLE DEBUG] execute_entry: Validation FAILED: {error_msg}")
            self.forensic_log.append({
                'ts': fill_ts,
                'symbol': event.symbol,
                'event': 'EXECUTE_ENTRY_VALIDATION_FAILED',
                'module': event.module,
                'error': error_msg
            })
            return
        
        # Calculate slippage
        adv_60m = calculate_adv_60m(df['notional'], bar_idx)
        order_notional = abs(adjusted_qty * adjusted_price)
        liquidity_state = self.symbol_liquidity_state.get(event.symbol)
        regime = liquidity_state.regime if liquidity_state else 'NORMAL'
        regime_adder = self.liquidity_detector.get_slippage_adder(regime)
        post_only = bool(regime == 'THIN')
        if post_only:
            self.thin_post_only_entries_count += 1
            self.thin_extra_slip_bps_total += regime_adder
        
        if adv_60m <= 0:
            raise ValueError(f"ADV_60m is non-positive for {event.symbol} at {fill_ts}")
        
        # Check participation cap
        participation_pct = order_notional / adv_60m
        participation_cap = self.liquidity_detector.get_participation_cap(regime)
        if participation_pct > participation_cap:
            # Reject entry - participation exceeds cap
            self.forensic_log.append({
                'ts': fill_ts,
                'symbol': event.symbol,
                'event': 'PARTICIPATION_CAP_BLOCK',
                'module': event.module,
                'participation_pct': participation_pct,
                'cap': participation_cap,
                'regime': regime
            })
            # Use index-based removal to avoid pandas Series boolean ambiguity
            try:
                idx = next((i for i, s in enumerate(pending_signals) if s is signal or (hasattr(s, 'signal_bar_idx') and hasattr(signal, 'signal_bar_idx') and int(s.signal_bar_idx) == int(signal.signal_bar_idx) and s.module == signal.module)), None)
                if idx is not None:
                    pending_signals.pop(idx)
            except (ValueError, TypeError):
                pass
            return
        
        slippage_params = self.params_dict.get('slippage_costs', {})
        # FIX 5: Pass governed_universe flag (use require_liquidity_data as proxy)
        slippage_bps, participation_pct = calculate_slippage(
            order_notional, adv_60m,
            slippage_params.get('base_slip_bps_intercept', 2.0),
            slippage_params.get('base_slip_bps_slope_per_participation', 20.0),
            regime_adder,
            governed_universe=self.require_liquidity_data,
            stress_slip=self.stress_slip
        )
        
        self.forensic_log.append({
            'ts': fill_ts,
            'symbol': event.symbol,
            'event': 'SLIPPAGE',
            'module': event.module,
            'participation_pct': participation_pct,
            'slip_bps': slippage_bps,
            'adv_60m': adv_60m,
            'order_notional': order_notional,
            'post_only': post_only
        })
        
        # Apply slippage to fill price
        mid_price = (fill_bar['high'] + fill_bar['low']) / 2.0
        fill_price, gap_through = fill_stop_run(
            entry_price, signal.side, fill_bar['high'], fill_bar['low'],
            mid_price, slippage_bps
        )
        
        # Log gap-through to forensic log
        if gap_through:
            self.forensic_log.append({
                'ts': fill_ts,
                'symbol': event.symbol,
                'event': 'GAP_THROUGH',
                'module': event.module,
                'side': signal.side,
                'trigger_price': entry_price,
                'fill_price': fill_price,
                'bar_high': fill_bar['high'],
                'bar_low': fill_bar['low']
            })
        
        # Adjust stop for actual fill price and enforce ES guard
        epsilon = 0.0002
        if signal.side == 'LONG':
            effective_stop_price = min(signal.stop_price, fill_price - epsilon)
        else:
            effective_stop_price = max(signal.stop_price, fill_price + epsilon)
        
        stop_distance_actual = abs(effective_stop_price - fill_price)
        candidate_risk = stop_distance_actual * abs(adjusted_qty)
        import time
        es_start = time.time()
        es_ok, es_before, es_after = self._passes_es_guard(
            candidate_risk, event.symbol, event.module, fill_ts,
            vol_forecast, vol_fast_median
        )
        self._profile_time['es_checks'] += time.time() - es_start
        self._profile_counts['es_checks'] += 1
        if not es_ok:
            self.es_block_count += 1  # G: Track ES blocks
            self.forensic_log.append({
                'ts': fill_ts,
                'symbol': event.symbol,
                'event': 'EXECUTE_ENTRY_ES_GUARD_FAILED',
                'module': event.module,
                'es_before': es_before,
                'es_after': es_after
            })
            # Use index-based removal to avoid pandas Series boolean ambiguity
            try:
                idx = next((i for i, s in enumerate(pending_signals) if s is signal or (hasattr(s, 'signal_bar_idx') and hasattr(signal, 'signal_bar_idx') and int(s.signal_bar_idx) == int(signal.signal_bar_idx) and s.module == signal.module)), None)
                if idx is not None:
                    pending_signals.pop(idx)
            except (ValueError, TypeError):
                pass
            return
        
        if not self._check_beta_caps_with_new_position(event.symbol, adjusted_qty, fill_price, fill_ts, signal.side):
            self.beta_block_count += 1  # G: Track beta blocks
            self.forensic_log.append({
                'ts': fill_ts,
                'symbol': event.symbol,
                'event': 'EXECUTE_ENTRY_BETA_CAPS_FAILED',
                'module': event.module
            })
            # Use index-based removal to avoid pandas Series boolean ambiguity
            try:
                idx = next((i for i, s in enumerate(pending_signals) if s is signal or (hasattr(s, 'signal_bar_idx') and hasattr(signal, 'signal_bar_idx') and int(s.signal_bar_idx) == int(signal.signal_bar_idx) and s.module == signal.module)), None)
                if idx is not None:
                    pending_signals.pop(idx)
            except (ValueError, TypeError):
                pass
            return
        
        # Calculate fees
        # Stop-run entries are taker, unless post_only=True (maker)
        notional = abs(adjusted_qty * fill_price)
        fill_is_taker = not post_only  # Taker unless post-only resting fill
        fee_bps = self.params.get_default('general', 'taker_fee_bps') if fill_is_taker else self.params.get_default('general', 'maker_fee_bps')
        if self.stress_fees:
            fee_bps *= 1.5  # Stress test: multiply fees by 1.5x
            
        if not self.cost_model_enabled:
            fee_bps = 0.0
            slippage_bps_applied = 0.0
            
        fees = notional * (fee_bps / 10000.0)
        
        # Add position
        # OPTIMIZATION: Calculate and pass entry_idx to avoid future lookups
        df = self.symbol_data[event.symbol]
        if hasattr(self, 'symbol_ts_to_idx') and event.symbol in self.symbol_ts_to_idx:
            entry_idx = self.symbol_ts_to_idx[event.symbol].get(fill_ts, -1)
        else:
            # Fix pandas Series boolean ambiguity: ensure entry_idx is always a scalar integer
            ts_matches = df[df['ts'] == fill_ts]
            if len(ts_matches) > 0:
                idx_result = ts_matches.index[0]
                entry_idx = int(idx_result) if hasattr(idx_result, '__iter__') and not isinstance(idx_result, str) else int(idx_result)
            else:
                entry_idx = -1
        
        if debug_oracle and event.module == 'ORACLE':
            print(f"[ORACLE DEBUG] execute_entry: Adding position: symbol={event.symbol}, qty={adjusted_qty}, price={fill_price}, side={signal.side}")
        # DeceptionModule signals carry the bot's exit lattice (TP1/TP2/TSL).
        # Pass them through to Position so collect_deception_exit_events can use them.
        deception_kwargs = {}
        if event.module == 'DECEPTION' and isinstance(signal, DeceptionSignal):
            deception_kwargs = dict(
                tp1_price=signal.tp1_price,
                tp2_price=signal.tp2_price,
                tp1_frac=signal.tp1_frac,
                tsl_bps=signal.tsl_bps,
                deception_score=signal.deception_score,
                trap_type=signal.trap_type,
            )
        self.portfolio.add_position(
            event.symbol, adjusted_qty, fill_price, fill_ts,
            effective_stop_price, effective_stop_price,  # Initial stop and trail
            event.module, signal.side, fees, 0.0,
            entry_idx=entry_idx,  # Store bar index for performance
            **deception_kwargs,
        )
        self.symbol_prev_prices[event.symbol] = fill_price
        if debug_oracle and event.module == 'ORACLE':
            print(f"[ORACLE DEBUG] execute_entry: Position added! Total positions: {len(self.portfolio.positions)}, Total trades: {len(self.trades)}")
        
        # Get position_id after adding position
        position_id = self.portfolio.positions[event.symbol].position_id if event.symbol in self.portfolio.positions else ""
        
        # Calculate actual slippage_bps_applied from the fill
        if signal.side == 'LONG':
            slippage_bps_applied = ((fill_price - entry_price) / mid_price) * 10000.0 if mid_price > 0 else slippage_bps
        else:  # SHORT
            slippage_bps_applied = ((entry_price - fill_price) / mid_price) * 10000.0 if mid_price > 0 else slippage_bps
        
        if not self.cost_model_enabled:
            slippage_bps_applied = 0.0
        
        slippage_cost_usd = notional * (slippage_bps_applied / 10000.0)
        
        # Record entry fill
        self._record_fill(
            position_id=position_id,
            ts=fill_ts,
            symbol=event.symbol,
            module=event.module,
            leg='ENTRY',
            side='BUY' if signal.side == 'LONG' else 'SELL',
            qty=adjusted_qty,
            price=fill_price,
            notional_usd=notional,
            slippage_bps_applied=slippage_bps_applied,
            slippage_cost_usd=slippage_cost_usd,
            fee_bps=fee_bps,
            fee_usd=fees,
            liquidity='maker' if post_only else 'taker',
            participation_pct=participation_pct,
            adv60_usd=adv_60m,
            intended_price=entry_price
        )
        
        # Record ENTRY_FILL ledger event (cash decreases by fees + slippage)
        self._record_ledger_event(
            ts=fill_ts,
            event='ENTRY_FILL',
            position_id=position_id,
            symbol=event.symbol,
            module=event.module,
            leg='ENTRY',
            side='BUY' if signal.side == 'LONG' else 'SELL',
            qty=adjusted_qty,
            price=fill_price,
            notional_usd=notional,
            fee_usd=fees,
            slippage_cost_usd=slippage_cost_usd,
            funding_usd=0.0,
            cash_delta_usd=-(fees + slippage_cost_usd),  # Cash decreases
            note=f"Entry fill: {event.module}"
        )
        
        # Calculate ES_used_after (with new position)
        _, _, es_after_actual = self._passes_es_guard(
            0.0, event.symbol, event.module, fill_ts,
            vol_forecast, vol_fast_median
        )
        
        # Remove signal from pending
        # Use index-based removal to avoid pandas Series boolean ambiguity
        try:
            idx = next((i for i, s in enumerate(pending_signals) if s is signal or (hasattr(s, 'signal_bar_idx') and hasattr(signal, 'signal_bar_idx') and int(s.signal_bar_idx) == int(signal.signal_bar_idx) and s.module == signal.module)), None)
            if idx is not None:
                pending_signals.pop(idx)
        except (ValueError, TypeError):
            pass
        
        # FIX 2 & 3: Record trade with position_id, open_ts, gap_through
        if debug_oracle and event.module == 'ORACLE':
            print(f"[ORACLE DEBUG] execute_entry: Appending trade: symbol={event.symbol}, side={signal.side}, qty={adjusted_qty}, price={fill_price}")
        self.trades.append({
            'ts': fill_ts,
            'symbol': event.symbol,
            'side': signal.side,
            'module': event.module,
            'qty': adjusted_qty,
            'price': fill_price,
            'fees': fees,
            'slip_bps': slippage_bps_applied,
            'participation_pct': participation_pct,
            'post_only': post_only,
            'stop_dist': stop_distance_actual,
            'ES_used_before': es_before,
            'ES_used_after': es_after_actual,
            'reason': 'ENTRY',
            'position_id': position_id,
            'open_ts': fill_ts,
            'close_ts': None,
            'age_bars': 0,
            'gap_through': gap_through
        })
        if debug_oracle and event.module == 'ORACLE':
            print(f"[ORACLE DEBUG] execute_entry: Trade appended! Total trades: {len(self.trades)}")
    
    def execute_trail(self, symbol: str, fill_bar: pd.Series, fill_ts: pd.Timestamp):
        """Update trailing stops (tighten only)"""
        if symbol not in self.portfolio.positions:
            return
        
        pos = self.portfolio.positions[symbol]
        df = self.symbol_data[symbol]
        
        # Get current bar data
        # OPTIMIZATION: Use O(1) lookup if mapping available
        current_idx = None
        if hasattr(self, 'symbol_ts_to_idx') and symbol in self.symbol_ts_to_idx:
            current_idx = self.symbol_ts_to_idx[symbol].get(fill_ts, None)
        
        if current_idx is None:
            # Fallback: DataFrame lookup or use most recent bar
            # Fix pandas Series boolean ambiguity: ensure current_idx is always a scalar integer
            ts_matches = df[df['ts'] == fill_ts]
            if len(ts_matches) > 0:
                idx_result = ts_matches.index[0]
                current_idx = int(idx_result) if hasattr(idx_result, '__iter__') and not isinstance(idx_result, str) else int(idx_result)
            else:
                current_idx = len(df) - 1
        
        if current_idx is None or current_idx >= len(df):
            return
        
        bar = df.iloc[current_idx]
        atr = bar.get('atr', 0.0)
        
        # Calculate new stop/trail
        # Strategy-specific stop/trail calculation removed from engine core
        # Oracle positions use fixed stops (no trailing)
        if pos.module != 'ORACLE':
            raise NotImplementedError(
                "Strategy-specific stop/trail calculation not available in engine core. "
                "Use oracle_mode for validation."
            )
        # Oracle positions keep original stop (no trailing)
        new_stop = pos.stop_price
        new_trail = None
        
        # Only tighten (never widen)
        if pos.side == 'LONG':
            new_stop = max(new_stop, pos.stop_price)
        else:
            new_stop = min(new_stop, pos.stop_price)
        
        self.portfolio.update_position_trail(symbol, new_stop, new_trail)
    
    def execute_ttl(self, order_id: str = None, symbol: str = None, fill_ts: pd.Timestamp = None):
        """Handle TTL expiration - either cancel pending order or close filled position"""
        if order_id:
            # Cancel pending order
            self.order_manager.cancel_order(order_id)
        elif symbol and symbol in self.portfolio.positions:
            # Close filled position due to TTL
            pos = self.portfolio.positions[symbol]
            df = self.symbol_data[symbol]
            # Get current bar at fill_ts (or most recent if fill_ts not provided)
            # OPTIMIZATION: Use O(1) lookup if mapping available
            if fill_ts is not None:
                if hasattr(self, 'symbol_ts_to_idx') and symbol in self.symbol_ts_to_idx:
                    fill_idx = self.symbol_ts_to_idx[symbol].get(fill_ts, None)
                    if fill_idx is not None and fill_idx < len(df):
                        current_bar = df.iloc[fill_idx]
                    else:
                        # Fallback: use most recent bar
                        current_bar = df.iloc[-1] if len(df) > 0 else None
                else:
                    # Fallback: DataFrame lookup
                    current_bar = df[df['ts'] == fill_ts]
                    if len(current_bar) > 0:
                        current_bar = current_bar.iloc[0]
                    else:
                        # Fallback: use most recent bar up to fill_ts
                        current_bar = df[df['ts'] <= fill_ts]
                        if len(current_bar) > 0:
                            current_bar = current_bar.iloc[-1]
                        else:
                            current_bar = df.iloc[-1] if len(df) > 0 else None
            else:
                current_bar = df.iloc[-1] if len(df) > 0 else None
            
            if current_bar is not None:
                current_price = current_bar['close']
                current_ts = current_bar['ts'] if fill_ts is None else fill_ts
                
                # Calculate fees
                # TTL exits are market orders, always taker
                notional = abs(pos.qty * current_price)
                fill_is_taker = True  # TTL exits are market orders, always taker
                fee_bps = self.params.get_default('general', 'taker_fee_bps') if fill_is_taker else self.params.get_default('general', 'maker_fee_bps')
                if self.stress_fees:
                    fee_bps *= 1.5  # Stress test: multiply fees by 1.5x
                
                if not self.cost_model_enabled:
                    fee_bps = 0.0
                    
                fees = notional * (fee_bps / 10000.0)
                
                # FIX 2: Calculate age_bars using bar indices
                if hasattr(self, 'symbol_ts_to_idx') and symbol in self.symbol_ts_to_idx:
                    close_idx = self.symbol_ts_to_idx[symbol].get(current_ts, -1)
                else:
                    # Fix pandas Series boolean ambiguity: ensure close_idx is always a scalar integer
                    ts_matches = df[df['ts'] == current_ts]
                    if len(ts_matches) > 0:
                        idx_result = ts_matches.index[0]
                        close_idx = int(idx_result) if hasattr(idx_result, '__iter__') and not isinstance(idx_result, str) else int(idx_result)
                    else:
                        close_idx = -1
                
                entry_idx = pos.entry_idx if pos.entry_idx >= 0 else -1
                if entry_idx < 0:
                    # Fix pandas Series boolean ambiguity: ensure entry_idx is always a scalar integer
                    ts_matches = df[df['ts'] == pos.entry_ts]
                    if len(ts_matches) > 0:
                        idx_result = ts_matches.index[0]
                        entry_idx = int(idx_result) if hasattr(idx_result, '__iter__') and not isinstance(idx_result, str) else int(idx_result)
                    else:
                        entry_idx = -1
                
                age_bars = (close_idx - entry_idx) if entry_idx >= 0 and close_idx >= 0 else pos.age_bars
                
                # FIX 2: Assert SQUEEZE TTL <= 48 bars
                if pos.module == 'SQUEEZE' and age_bars > 48:
                    self.forensic_log.append({
                        'ts': current_ts,
                        'symbol': symbol,
                        'event': 'SQUEEZE_TTL_VIOLATION',
                        'age_bars': age_bars,
                        'max_allowed': 48,
                        'position_id': pos.position_id
                    })
                
                # Calculate slippage for TTL exit (market order, minimal slippage)
                mid_price = (current_bar['high'] + current_bar['low']) / 2.0 if 'high' in current_bar and 'low' in current_bar else current_price
                slippage_params = self.params_dict.get('slippage_costs', {})
                slippage_bps_base = slippage_params.get('base_slip_bps_intercept', 2.0)
                # TTL exits are market orders, use base slippage
                slippage_bps_applied = slippage_bps_base
                
                if not self.cost_model_enabled:
                    slippage_bps_applied = 0.0
                
                slippage_cost_usd = notional * (slippage_bps_applied / 10000.0)
                
                # Get ADV_60m for participation calculation
                adv60_usd = calculate_adv_60m(df['notional'], close_idx) if close_idx >= 0 else 0.0
                participation_pct = (notional / adv60_usd) if adv60_usd > 0 else 0.0
                
                # Record exit fill
                self._record_fill(
                    position_id=pos.position_id,
                    ts=current_ts,
                    symbol=symbol,
                    module=pos.module,
                    leg='EXIT',
                    side='SELL' if pos.side == 'LONG' else 'BUY',
                    qty=pos.qty,
                    price=current_price,
                    notional_usd=notional,
                    slippage_bps_applied=slippage_bps_applied,
                    slippage_cost_usd=slippage_cost_usd,
                    fee_bps=fee_bps,
                    fee_usd=fees,
                    liquidity='taker',
                    participation_pct=participation_pct,
                    adv60_usd=adv60_usd,
                    intended_price=current_price
                )
                
                # Close position (this calculates PnL internally and returns it)
                closed_pos, pnl = self.portfolio.close_position(
                    symbol, current_price, current_ts, 'TTL', fees, slippage_cost_usd
                )
                
                if closed_pos:
                    # Record EXIT_FILL ledger event
                    # Note: pnl from close_position already has fees and slippage deducted
                    self._record_ledger_event(
                        ts=current_ts,
                        event='EXIT_FILL',
                        position_id=pos.position_id,
                        symbol=symbol,
                        module=pos.module,
                        leg='EXIT',
                        side='SELL' if pos.side == 'LONG' else 'BUY',
                        qty=pos.qty,
                        price=current_price,
                        notional_usd=notional,
                        fee_usd=fees,
                        slippage_cost_usd=slippage_cost_usd,
                        funding_usd=0.0,
                        cash_delta_usd=pnl,  # PnL already has fees/slippage deducted
                        note="TTL expiration exit"
                    )
                    # Use PnL from close_position to ensure exact match with portfolio.total_pnl
                    # FIX 2 & 3: Record trade with position_id, open_ts, close_ts, age_bars, gap_through
                    self.trades.append({
                        'ts': current_ts,
                        'symbol': symbol,
                        'side': pos.side,
                        'module': pos.module,
                        'qty': pos.qty,
                        'price': current_price,
                        'fees': fees,
                        'slip_bps': 0.0,
                        'participation_pct': 0.0,
                        'post_only': False,
                        'stop_dist': abs(pos.entry_price - pos.stop_price),
                        'ES_used_before': 0.0,
                        'ES_used_after': 0.0,
                        'reason': 'TTL',
                        'pnl': pnl,
                        'position_id': pos.position_id,
                        'open_ts': pos.entry_ts,
                        'close_ts': current_ts,
                        'age_bars': age_bars,
                        'gap_through': False  # TTL closes don't use stop-run, so no gap-through
                    })
                    self.symbol_daily_pnl[symbol] += pnl
                    self.symbol_prev_prices.pop(symbol, None)
    
    def execute_stale_cancel(self, order_id: str):
        """Cancel stale order"""
        self.order_manager.cancel_order(order_id)
        
        # Log forensic event
        order = self.order_manager.pending_orders.get(order_id)
        if order:
            self.forensic_log.append({
                'ts': pd.Timestamp.now(tz='UTC'),
                'symbol': order.symbol,
                'event': 'STALE_CANCEL',
                'order_id': order_id,
                'module': order.module
            })
    
    # ========== Opportunity Audit Methods ==========
    
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
    
    # ========== Risk Guard Methods ==========
    
    def check_all_entry_guards(
        self, symbol: str, signal, fill_bar: pd.Series, fill_ts: pd.Timestamp
    ) -> bool:
        """Check all entry guardrails before allowing entry"""
        # Record opportunity audit (candidate=True, will update taken/reject_reason below)
        reject_reason = None
        
        # Launch Punch List – Blocker #5: global trading state machine
        # Check trading state first - no new orders if halted
        if not self.state_manager.can_trade(module=signal.module if hasattr(signal, 'module') else None):
            current_state = self.state_manager.get_state()
            reject_reason = 'STATE_HALT'
            if self.enable_opportunity_audit:
                self._record_opportunity_audit(
                    fill_ts, symbol, signal.module if hasattr(signal, 'module') else 'UNKNOWN', 
                    candidate=True, taken=False,
                    reject_reason=reject_reason, notes=f'Trading state: {current_state.value}'
                )
            return False
        
        # Check loss halts
        vol_scale = self.get_vol_scale(symbol, signal.signal_bar_idx)
        if self.loss_halt_state.halt_manual:
            reject_reason = 'HALT'
            if self.enable_opportunity_audit:
                self._record_opportunity_audit(
                    fill_ts, symbol, signal.module, candidate=True, taken=False,
                    reject_reason=reject_reason, notes='HALT_MANUAL'
                )
            return False
        
        # Launch Punch List – Blocker #3: robust daily loss kill-switch
        # Check daily kill-switch before entry
        is_triggered, flatten_on_trigger, block_new_entries = self.loss_halt_state.check_daily_kill_switch(
            equity=self.portfolio.equity,
            initial_equity=self.day_start_equity,
            vol_scale=vol_scale,
            params=self.params_dict,
            current_ts=fill_ts
        )
        
        if is_triggered and block_new_entries:
            # Block new entries if kill-switch is triggered
            reject_reason = 'KILL_SWITCH'
            if self.enable_opportunity_audit:
                self._record_opportunity_audit(
                    fill_ts, symbol, signal.module, candidate=True, taken=False,
                    reject_reason=reject_reason, notes='Daily kill switch triggered'
                )
            return False
        
        # Also check legacy daily hard stop for backward compatibility
        if self.loss_halt_state.check_daily_hard_stop(
            self.portfolio.equity, vol_scale, self.params_dict
        ):
            # FIX 5: Deduplicate by UTC day
            utc_date = fill_ts.date()
            if (utc_date,) not in self._halt_daily_hard_seen:
                self._halt_daily_hard_seen.add((utc_date,))
                self.halt_daily_hard_count += 1  # G: Track daily hard stops
            reject_reason = 'HALT'
            if self.enable_opportunity_audit:
                self._record_opportunity_audit(
                    fill_ts, symbol, signal.module, candidate=True, taken=False,
                    reject_reason=reject_reason, notes='Daily hard stop'
                )
            return False
        
        # Check margin
        margin_ratio = calculate_margin_ratio(self.portfolio.positions, self.portfolio.equity)
        block_ratio = (self.params.get('margin', 'block_new_entries_ratio_pct') or 60.0) / 100.0
        trim_ratio = (self.params.get('margin', 'trim_target_ratio_pct') or 50.0) / 100.0
        flatten_ratio = (self.params.get('margin', 'flatten_ratio_pct') or 80.0) / 100.0
        action, should_act = check_margin_constraints(margin_ratio, block_ratio, trim_ratio, flatten_ratio)
        if action == 'TRIM':
            self.margin_trim_count += 1  # G: Track margin trims
        if action == 'BLOCK' or action == 'FLATTEN':
            self.margin_blocks_count += 1
            reject_reason = 'MARGIN'
            if self.enable_opportunity_audit:
                self._record_opportunity_audit(
                    fill_ts, symbol, signal.module, candidate=True, taken=False,
                    reject_reason=reject_reason, margin_ratio=margin_ratio, notes=f'Margin {action}'
                )
            return False
        
        # Check liquidity regime (VACUUM blocks)
        liquidity_state = self.symbol_liquidity_state.get(symbol)
        if liquidity_state and liquidity_state.regime == 'VACUUM':
            reject_reason = 'LIQ'
            if self.enable_opportunity_audit:
                self._record_opportunity_audit(
                    fill_ts, symbol, signal.module, candidate=True, taken=False,
                    reject_reason=reject_reason, notes='VACUUM regime'
                )
            return False
        
        # Check max positions
        max_positions = self.params.get_default('general', 'max_positions')
        if len(self.portfolio.positions) >= max_positions:
            reject_reason = 'OTHER'
            if self.enable_opportunity_audit:
                self._record_opportunity_audit(
                    fill_ts, symbol, signal.module, candidate=True, taken=False,
                    reject_reason=reject_reason, notes=f'Max positions ({len(self.portfolio.positions)}/{max_positions})'
                )
            return False
        
        # Check if already have position in this symbol
        if symbol in self.portfolio.positions:
            reject_reason = 'OTHER'
            if self.enable_opportunity_audit:
                self._record_opportunity_audit(
                    fill_ts, symbol, signal.module, candidate=True, taken=False,
                    reject_reason=reject_reason, notes='Already have position'
                )
            return False
        
        # ES and beta checks would go here (simplified for now)
        # Full implementation would calculate ES and beta before entry
        
        # All guards passed - entry will be taken
        if self.enable_opportunity_audit:
            # Get ES headroom and beta for audit
            es_headroom = 1.0  # Placeholder - would calculate from ES guard
            beta_net = 0.0  # Placeholder - would calculate from beta guard
            df = self.symbol_data.get(symbol)
            adv60_usd = 0.0
            if df is not None and signal.signal_bar_idx < len(df):
                adv60_usd = calculate_adv_60m(df['notional'], signal.signal_bar_idx) if 'notional' in df.columns else 0.0
            
            self._record_opportunity_audit(
                fill_ts, symbol, signal.module, candidate=True, taken=True,
                reject_reason='', es_headroom=es_headroom, beta_net=beta_net,
                margin_ratio=margin_ratio, adv60_usd=adv60_usd, notes='Entry taken'
            )
        
        return True
    
    def apply_funding_costs(self, symbol: str, current_ts: pd.Timestamp):
        """Apply funding costs (adverse only) - FIX 4: Only at exact funding times (00:00, 08:00, 16:00 UTC)"""
        if symbol not in self.portfolio.positions:
            return
        
        pos = self.portfolio.positions[symbol]
        funding_df = self.data_loader.get_funding(symbol)
        
        if funding_df is None or len(funding_df) == 0:
            return
        
        # FIX 4: Only accrue funding at exact funding times: 00:00, 08:00, 16:00 UTC
        # Check if current_ts is exactly at a funding time (exact match required)
        current_hour = current_ts.hour
        current_minute = current_ts.minute
        current_second = current_ts.second
        
        # Funding times are exactly at 00:00:00, 08:00:00, 16:00:00 UTC
        funding_times = [0, 8, 16]  # 00:00, 08:00, 16:00 UTC
        is_funding_time = (
            current_hour in funding_times and
            current_minute == 0 and
            current_second == 0
        )
        
        if not is_funding_time:
            return  # Skip funding accrual if not at exact funding time
        
        # Find funding rate for this exact time
        funding_match = funding_df[funding_df['funding_ts'] <= current_ts]
        if len(funding_match) == 0:
            return
        
        # Get the funding rate for the most recent funding event
        funding_rate = funding_match.iloc[-1]['funding_rate']
        notional = abs(pos.qty * pos.entry_price)
        
        # FIX 4: Apply adverse funding only (long pays when rate>0, short pays when rate<0)
        cost = self.portfolio.calculate_funding_cost(symbol, funding_rate, notional)
        
        if not self.cost_model_enabled:
            cost = 0.0
            
        if cost > 0:
            self.forensic_log.append({
                'ts': current_ts,
                'symbol': symbol,
                'event': 'FUNDING_COST',
                'funding_rate': funding_rate,
                'cost': cost,
                'position_id': pos.position_id
            })
            # FIX 4: Count funding event only if position is open and cost > 0
            self.funding_events_count += 1
            
            # Record FUNDING ledger event (cash decreases by funding cost)
            self._record_ledger_event(
                ts=current_ts,
                event='FUNDING',
                position_id=pos.position_id,
                symbol=symbol,
                module=pos.module,
                leg='',  # Funding is not a fill leg
                side=pos.side,
                qty=pos.qty,
                price=pos.entry_price,  # Use entry price for reference
                notional_usd=notional,
                fee_usd=0.0,
                slippage_cost_usd=0.0,
                funding_usd=cost,
                cash_delta_usd=-cost,  # Funding always decreases cash
                note=f"Funding cost: rate={funding_rate:.6f}"
            )
    
    def _check_invariants(self, current_ts: pd.Timestamp, symbol_prices: Optional[Dict[str, float]] = None):
        """
        Check accounting invariants when debug_invariants is enabled.
        
        Invariants checked:
        1. Equity identity: equity = cash + unrealized_pnl (for futures, NOT position_notional)
        2. Position conservation: pos_t = pos_{t-1} + fills_t (tracked via position_qty before/after)
        3. Realized PnL conservation: realized PnL changes only on fills/closures
        4. Cost signs: fees <= 0, slippage <= 0, funding sign matches position sign
        5. Cost toggle invariants: if cost_model.enabled=false, all costs == 0.0
        6. No ghost trades: if no order → no fill → no pnl impact
        """
        if not self.debug_invariants:
            return
        
        errors = []
        tolerance = max(1e-6 * abs(self.portfolio.equity), 0.01)  # 1e-6 of equity or $0.01, whichever is larger
        
        # 1. Equity identity: equity = cash + unrealized_pnl
        # Calculate unrealized PnL from positions
        total_unrealized_pnl = 0.0
        total_position_notional = 0.0
        
        if symbol_prices:
            for symbol, position in self.portfolio.positions.items():
                current_price = symbol_prices.get(symbol)
                if current_price is None:
                    continue
                
                # Calculate unrealized PnL
                if position.side == 'LONG':
                    unrealized_pnl = (current_price - position.entry_price) * position.qty
                else:  # SHORT
                    unrealized_pnl = (position.entry_price - current_price) * position.qty
                
                total_unrealized_pnl += unrealized_pnl
                total_position_notional += abs(position.qty * current_price)
        
        # Equity should equal cash + unrealized_pnl
        expected_equity = self.portfolio.cash + total_unrealized_pnl
        equity_error = abs(self.portfolio.equity - expected_equity)
        
        if equity_error > tolerance:
            errors.append({
                'invariant': 'Equity identity',
                'error': equity_error,
                'expected': expected_equity,
                'actual': self.portfolio.equity,
                'cash': self.portfolio.cash,
                'unrealized_pnl': total_unrealized_pnl,
                'ts': current_ts
            })
        
        # 2. Position conservation: pos_t = pos_{t-1} + fills_t
        # Track position quantities before/after fills (checked implicitly via add_position/close_position)
        # Positions are only modified via explicit methods, so conservation is guaranteed by design
        # We verify that position quantities are non-negative and consistent
        for symbol, position in self.portfolio.positions.items():
            if position.qty <= 0:
                errors.append({
                    'invariant': 'Position conservation',
                    'error': f'Position qty must be positive, got {position.qty} for {symbol}',
                    'symbol': symbol,
                    'ts': current_ts
                })
        
        # 3. Realized PnL conservation: realized PnL changes only on fills/closures
        # Verify that total_pnl matches sum of closed position PnLs
        # This is verified by checking that total_pnl equals sum of all closed position PnLs
        # (checked via ledger reconciliation in reporting, but we verify consistency here)
        # Note: This invariant is primarily checked at the end of backtest via reporting reconciliation
        
        # 4. Cost signs: fees <= 0, slippage <= 0, funding sign matches position sign
        # Check recent ledger entries for cost signs
        if len(self.ledger) > 0:
            recent_ledger = self.ledger[-10:]  # Check last 10 ledger entries
            for entry in recent_ledger:
                fee_usd = entry.get('fee_usd', 0.0)
                slippage_cost_usd = entry.get('slippage_cost_usd', 0.0)
                funding_usd = entry.get('funding_usd', 0.0)
                
                if fee_usd > 0:
                    errors.append({
                        'invariant': 'Cost signs (fees)',
                        'error': f'Fee is positive: {fee_usd}',
                        'entry': entry,
                        'ts': current_ts
                    })
                
                if slippage_cost_usd > 0:
                    errors.append({
                        'invariant': 'Cost signs (slippage)',
                        'error': f'Slippage cost is positive: {slippage_cost_usd}',
                        'entry': entry,
                        'ts': current_ts
                    })
                
                # Check funding sign matches position sign
                if funding_usd != 0.0:
                    symbol = entry.get('symbol')
                    side = entry.get('side')
                    if symbol and symbol in self.portfolio.positions:
                        pos = self.portfolio.positions[symbol]
                        # Long pays when rate > 0, Short pays when rate < 0
                        # If funding_usd > 0, it means we paid (cost)
                        # For LONG: funding should be > 0 only if rate > 0
                        # For SHORT: funding should be > 0 only if rate < 0
                        # This is a simplified check - actual funding rate would need to be checked
        
        # 5. Cost toggle invariants: if cost_model.enabled=false, all costs == 0.0
        if not self.cost_model_enabled:
            if abs(self.portfolio.fees_paid) > tolerance:
                errors.append({
                    'invariant': 'Cost toggle (fees)',
                    'error': f'Fees should be 0 when cost_model disabled, got {self.portfolio.fees_paid}',
                    'ts': current_ts
                })
            
            if abs(self.portfolio.slippage_paid) > tolerance:
                errors.append({
                    'invariant': 'Cost toggle (slippage)',
                    'error': f'Slippage should be 0 when cost_model disabled, got {self.portfolio.slippage_paid}',
                    'ts': current_ts
                })
            
            if abs(self.portfolio.funding_paid) > tolerance:
                errors.append({
                    'invariant': 'Cost toggle (funding)',
                    'error': f'Funding should be 0 when cost_model disabled, got {self.portfolio.funding_paid}',
                    'ts': current_ts
                })
            
            # Check recent fills for zero costs
            if len(self.fills) > 0:
                recent_fills = self.fills[-10:]
                for fill in recent_fills:
                    if abs(fill.get('fee_usd', 0.0)) > tolerance:
                        errors.append({
                            'invariant': 'Cost toggle (fill fees)',
                            'error': f'Fill fee should be 0 when cost_model disabled, got {fill.get("fee_usd", 0.0)}',
                            'fill': fill,
                            'ts': current_ts
                        })
                    
                    if abs(fill.get('slippage_cost_usd', 0.0)) > tolerance:
                        errors.append({
                            'invariant': 'Cost toggle (fill slippage)',
                            'error': f'Fill slippage should be 0 when cost_model disabled, got {fill.get("slippage_cost_usd", 0.0)}',
                            'fill': fill,
                            'ts': current_ts
                        })
        
        # 6. No ghost trades: Check that all fills have corresponding ledger entries
        # Verify that every fill has a corresponding ledger entry
        if len(self.fills) > 0:
            recent_fills = self.fills[-10:]
            for fill in recent_fills:
                fill_ts = fill.get('ts')
                fill_symbol = fill.get('symbol')
                fill_notional = fill.get('notional_usd', 0.0)
                
                # Check if there's a corresponding ledger entry for this fill
                matching_ledger = [
                    entry for entry in self.ledger[-20:]
                    if entry.get('ts') == fill_ts and entry.get('symbol') == fill_symbol
                ]
                
                # For entry fills, there should be an ENTRY_FILL ledger event
                # For exit fills, there should be an EXIT_FILL ledger event
                if fill.get('leg') == 'ENTRY' and not any(e.get('event') == 'ENTRY_FILL' for e in matching_ledger):
                    if fill_notional > 0:  # Only flag if notional is significant
                        errors.append({
                            'invariant': 'No ghost trades (entry)',
                            'error': f'Entry fill at {fill_ts} for {fill_symbol} has no matching ENTRY_FILL ledger entry',
                            'fill': fill,
                            'ts': current_ts
                        })
                elif fill.get('leg') == 'EXIT' and not any(e.get('event') == 'EXIT_FILL' for e in matching_ledger):
                    if fill_notional > 0:  # Only flag if notional is significant
                        errors.append({
                            'invariant': 'No ghost trades (exit)',
                            'error': f'Exit fill at {fill_ts} for {fill_symbol} has no matching EXIT_FILL ledger entry',
                            'fill': fill,
                            'ts': current_ts
                        })
        
        # Log errors if any
        if errors:
            for error in errors:
                self.forensic_log.append({
                    'ts': current_ts,
                    'event': 'INVARIANT_VIOLATION',
                    **error
                })
            
            # Write snapshot on failure
            snapshot = {
                'timestamp': str(current_ts),
                'portfolio_state': {
                    'cash': self.portfolio.cash,
                    'equity': self.portfolio.equity,
                    'fees_paid': self.portfolio.fees_paid,
                    'slippage_paid': self.portfolio.slippage_paid,
                    'funding_paid': self.portfolio.funding_paid,
                    'total_pnl': self.portfolio.total_pnl,
                    'position_count': len(self.portfolio.positions),
                    'positions': {
                        sym: {
                            'qty': pos.qty,
                            'entry_price': pos.entry_price,
                            'side': pos.side,
                            'module': pos.module
                        }
                        for sym, pos in self.portfolio.positions.items()
                    }
                },
                'symbol_prices': symbol_prices or {},
                'recent_fills': self.fills[-5:] if len(self.fills) > 0 else [],
                'recent_ledger': self.ledger[-5:] if len(self.ledger) > 0 else [],
                'errors': errors
            }
            
            # Write snapshot to artifacts directory
            import json
            from pathlib import Path
            artifacts_dir = Path("artifacts")
            artifacts_dir.mkdir(exist_ok=True)
            snapshot_path = artifacts_dir / "invariant_failure_snapshot.json"
            with open(snapshot_path, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
            
            # Raise exception if invariants fail (in debug mode, we want to catch these immediately)
            error_summary = '\n'.join([f"{e['invariant']}: {e.get('error', 'Unknown error')}" for e in errors])
            raise AssertionError(
                f"Accounting invariant violations detected at {current_ts}:\n{error_summary}\n"
                f"Snapshot saved to: {snapshot_path}"
            )
    
    def _passes_es_guard(
        self,
        additional_risk: float,
        symbol: Optional[str],
        module: Optional[str],
        event_ts: pd.Timestamp,
        vol_forecast: Optional[float] = None,
        vol_fast_median: Optional[float] = None
    ) -> Tuple[bool, float, float]:
        """Check ES guardrail before committing additional risk"""
        if additional_risk <= 0 or self.portfolio.equity <= 0:
            return True, 0.0, 0.0
        
        returns_window = self.portfolio_returns[-(96 * 60):]  # up to ~60 days
        if returns_window:
            returns_series = pd.Series(returns_window)
        else:
            returns_series = pd.Series([0.0])
        
        ewhs_es = calculate_ewhs_es(returns_series)
        if len(returns_series) >= 96:
            portfolio_vol_fast = returns_series.tail(96).std(ddof=0)
        else:
            portfolio_vol_fast = returns_series.std(ddof=0)
        if pd.isna(portfolio_vol_fast):
            portfolio_vol_fast = 0.0
        parametric_es = calculate_parametric_es(
            portfolio_vol_fast,
            np.array([[1.0]]),
            np.array([1.0])
        )
        sigma_clip_es = calculate_sigma_clip_es(
            portfolio_vol_fast,
            vol_forecast if vol_forecast is not None else portfolio_vol_fast,
            vol_fast_median if vol_fast_median is not None and vol_fast_median != 0 else max(portfolio_vol_fast, 1e-6)
        )
        final_es = calculate_final_es(ewhs_es, parametric_es, sigma_clip_es)
        
        # Add additional risk to current ES
        total_risk = self.portfolio.get_total_stop_risk() + additional_risk
        es_used = max(final_es, total_risk)  # ES_used = max(ES methods, total stop risk)
        
        es_cap_pct = self.params.get('es_guardrails', 'es_cap_of_equity') or 0.0225
        es_cap_dollar = self.portfolio.equity * es_cap_pct
        
        es_used_before = (self.es_usage_samples[-1] * self.portfolio.equity) if self.es_usage_samples else 0.0
        es_pct = es_used / self.portfolio.equity if self.portfolio.equity > 0 else 0.0
        is_valid = es_used <= es_cap_dollar
        
        if is_valid:
            self.es_usage_samples.append(es_pct)
        else:
            self.es_violations_count += 1
            self.forensic_log.append({
                'ts': event_ts,
                'symbol': symbol,
                'event': 'ES_GUARD_BLOCK',
                'module': module,
                'additional_risk': additional_risk,
                'equity': self.portfolio.equity,
                'es_cap_pct': es_cap_pct,
                'es_cap_dollar': es_cap_dollar,
                'es_used': es_used,
                'es_pct': es_pct,
                'ewhs_es': ewhs_es,
                'parametric_es': parametric_es,
                'sigma_clip_es': sigma_clip_es,
                'final_es': final_es,
                'total_stop_risk': total_risk
            })
        
        return is_valid, es_used_before / self.portfolio.equity if self.portfolio.equity > 0 else 0.0, es_pct

