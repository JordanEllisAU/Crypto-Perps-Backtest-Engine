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




class EventMixin:
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
                self._logger.info(f"DEBUG: Gating applied (Long Only). Blocked {len(entry_events) - len(filtered)} SHORT events.")
            return filtered
            
        if short_only:
            filtered = [e for e in entry_events if e.side != 'LONG']
            if len(filtered) < len(entry_events):
                self._logger.info(f"DEBUG: Gating applied (Short Only). Blocked {len(entry_events) - len(filtered)} LONG events.")
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
            self._logger.info(f"[ORACLE DEBUG] _collect_new_entry_events: symbol={symbol}, fill_idx={fill_idx}, pending_signals count={len(pending_signals)}")
            for i, sig in enumerate(pending_signals):
                if hasattr(sig, 'module'):
                    self._logger.info(f"  Signal {i}: module={sig.module}, signal_bar_idx={getattr(sig, 'signal_bar_idx', 'N/A')}")
        oracle_signals = [s for s in pending_signals if hasattr(s, 'module') and s.module == 'ORACLE']
        deception_signals = [s for s in pending_signals if hasattr(s, 'module') and s.module == 'DECEPTION']
        if debug_oracle:
            self._logger.info(f"[ORACLE DEBUG] Found {len(oracle_signals)} ORACLE signals, {len(deception_signals)} DECEPTION signals")
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
                    self._logger.info(f"[ORACLE DEBUG] Checking timing: fill_idx={fill_idx_scalar}, signal_bar_idx={signal_bar_idx_scalar}, condition={fill_idx_scalar >= signal_bar_idx_scalar}")
                if fill_idx_scalar >= signal_bar_idx_scalar:
                    if debug_oracle:
                        self._logger.info(f"[ORACLE DEBUG] Creating ORACLE_ENTRY event for {symbol}, side={signal.side}")
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
                        self._logger.info(f"[ORACLE DEBUG] Created {len(events)} events, returning")
                    return events  # Return immediately for ORACLE (only one at a time)
                elif debug_oracle:
                    self._logger.info(f"[ORACLE DEBUG] Timing check FAILED: fill_idx={fill_idx_scalar} < signal_bar_idx={signal_bar_idx_scalar}")
        
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
                    self._logger.debug("Failed to remove stale pending signal")
        
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
                    self._logger.info(f"[ORACLE DEBUG] execute_events: Calling execute_entry for ORACLE_ENTRY, symbol={event.symbol}, side={event.side}")
                self.execute_entry(event, fill_bar, fill_ts)
                self._profile_counts['events_executed'] += 1
                if debug_oracle:
                    self._logger.info(f"[ORACLE DEBUG] execute_entry returned. Trades count: {len(self.trades)}, Fills count: {len(self.fills)}")
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
