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




class ExecutionMixin:
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
        
        # Stop exits use the opposite trade side to the held position.
        # Cost-model toggle removes fill-price slippage.
        if not self.cost_model_enabled:
            slippage_bps_base = 0.0
        trade_side = 'SHORT' if pos.side == 'LONG' else 'LONG'
        fill_price, gap_through = fill_stop_run(
            stop_price, trade_side, fill_bar['high'], fill_bar['low'],
            mid_price, slippage_bps_base
        )
        
        # Positive slippage for a stop exit: fill is worse than the stop trigger
        if pos.side == 'LONG':  # selling
            slippage_bps_applied = ((stop_price - fill_price) / mid_price) * 10000.0 if mid_price > 0 else slippage_bps_base
        else:  # SHORT, buying back
            slippage_bps_applied = ((fill_price - stop_price) / mid_price) * 10000.0 if mid_price > 0 else slippage_bps_base
        
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
        
        # Calculate fees and slippage using the intended stop trigger price
        # Stop-market exits are always taker
        notional = abs(pos.qty * stop_price)
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
        self.fill_recorder.record_fill(
            position_id=pos.position_id,
            ts=fill_ts,
            symbol=symbol,
            module=pos.module,
            leg='EXIT',
            side='SELL' if pos.side == 'LONG' else 'BUY',
            qty=pos.qty,
            price=stop_price,
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
        
        # Close position at the intended stop price with explicit slippage cost
        closed_pos, pnl = self.portfolio.close_position(
            symbol, stop_price, fill_ts, 'STOP', fees, slippage_cost_usd
        )
        
        if closed_pos:
            # Record EXIT_FILL ledger event
            # Note: pnl from close_position already has fees and slippage deducted
            # So cash_delta = pnl (the net effect on cash)
            self.fill_recorder.record_ledger_event(
                ts=fill_ts,
                event='EXIT_FILL',
                position_id=pos.position_id,
                symbol=symbol,
                module=pos.module,
                leg='EXIT',
                side='SELL' if pos.side == 'LONG' else 'BUY',
                qty=pos.qty,
                price=stop_price,
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
                'price': stop_price,
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
        Fill price is the target price (TP1/TP2/TSL) with no additional slippage model.
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
            # TSL fill: the trailing stop price itself (extreme adjusted by the callback)
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
        slippage_cost_usd = 0.0  # TP1/TP2 are limit-style targets; TSL uses the trailing stop price directly

        # Record exit fill
        self.fill_recorder.record_fill(
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
            self.fill_recorder.record_ledger_event(
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
        self.fill_recorder.record_fill(
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

        self.fill_recorder.record_ledger_event(
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
        self.fill_recorder.record_fill(
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
            self.fill_recorder.record_ledger_event(
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
    def execute_entry(self, event: OrderEvent, fill_bar: pd.Series, fill_ts: pd.Timestamp):
        """Execute new entry with all risk checks"""
        debug_oracle = self.params.get('general', 'debug_oracle_flow', default=False)
        # Get signal
        pending_signals = self.symbol_pending_signals.get(event.symbol, [])
        if debug_oracle and event.module == 'ORACLE':
            self._logger.info(f"[ORACLE DEBUG] execute_entry: Looking for ORACLE signal in {len(pending_signals)} pending signals for {event.symbol}")
            for i, s in enumerate(pending_signals):
                if hasattr(s, 'module'):
                    self._logger.info(f"  Signal {i}: module={s.module}, matches={s.module == event.module}")
        signal = None
        for s in pending_signals:
            if s.module == event.module:
                signal = s
                break
        
        if not signal:
            if debug_oracle and event.module == 'ORACLE':
                self._logger.info(f"[ORACLE DEBUG] ERROR: No ORACLE signal found in pending_signals for {event.symbol}")
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
                    self._logger.debug("Failed to remove stale pending signal")
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
                    self._logger.info(f"[ORACLE DEBUG] execute_entry: qty was 0, using minimum step_size={step_size}")
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
                self._logger.debug("Failed to remove stale pending signal")
            return
        
        # Use fill_bar close as entry price (market order simulation)
        entry_price = fill_bar['close']
        if debug_oracle and event.module == 'ORACLE':
            self._logger.info(f"[ORACLE DEBUG] execute_entry: Calculated qty={qty}, entry_price={entry_price}, stop_price={signal.stop_price}")
        
        # Validate constraints
        contract_metadata = self.data_loader.get_contract_metadata(event.symbol)
        is_valid, error_msg, adjusted_qty, adjusted_price = validate_order_constraints(
            qty, entry_price, contract_metadata, signal.side
        )
        
        if not is_valid:
            if debug_oracle and event.module == 'ORACLE':
                self._logger.info(f"[ORACLE DEBUG] execute_entry: Validation FAILED: {error_msg}")
            self.forensic_log.append({
                'ts': fill_ts,
                'symbol': event.symbol,
                'event': 'EXECUTE_ENTRY_VALIDATION_FAILED',
                'module': event.module,
                'error': error_msg
            })
            return

        # ORACLE test module: size to the full equity slice when allowed by ES guardrails,
        # otherwise scale down to the maximum risk the current ES cap permits. This lets
        # buy-and-hold benchmarks track the naive close-to-close return without tripping
        # the ES guard on default risk parameters.
        if event.module == 'ORACLE':
            step_size = contract_metadata.get('stepSize', 0.001)
            min_qty = contract_metadata.get('minQty', 0.001)
            min_notional = contract_metadata.get('minNotional', 5.0)
            full_equity_qty = np.floor((self.portfolio.equity / adjusted_price) / step_size) * step_size
            # Respect ES cap: candidate_risk = qty * |entry - stop| must fit within es_cap_of_equity * equity
            es_cap_pct = self.params.get('es_guardrails', 'es_cap_of_equity') or 0.0225
            stop_distance_for_oracle = abs(adjusted_price - signal.stop_price)
            if stop_distance_for_oracle > 0:
                es_capped_qty = (self.portfolio.equity * es_cap_pct * 0.95) / stop_distance_for_oracle
                full_equity_qty = min(full_equity_qty, np.floor((es_capped_qty / step_size)) * step_size)
            if full_equity_qty >= min_qty and full_equity_qty * adjusted_price >= min_notional:
                adjusted_qty = full_equity_qty
        
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
                self._logger.debug("Failed to remove stale pending signal")
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
        
        # Cost-model toggle removes fill-price slippage as well as fees
        if not self.cost_model_enabled:
            slippage_bps = 0.0
        
        # Apply slippage to the rounded intended entry price
        mid_price = (fill_bar['high'] + fill_bar['low']) / 2.0
        fill_price, gap_through = fill_stop_run(
            adjusted_price, signal.side, fill_bar['high'], fill_bar['low'],
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
                self._logger.debug("Failed to remove stale pending signal")
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
                self._logger.debug("Failed to remove stale pending signal")
            return
        
        # Calculate fees and slippage using the intended (tick-aligned) entry price
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
        
        # Actual slippage experienced relative to intended entry
        if signal.side == 'LONG':
            slippage_bps_applied = ((fill_price - adjusted_price) / mid_price) * 10000.0 if mid_price > 0 else slippage_bps
        else:  # SHORT
            slippage_bps_applied = ((adjusted_price - fill_price) / mid_price) * 10000.0 if mid_price > 0 else slippage_bps
        
        if not self.cost_model_enabled:
            slippage_bps_applied = 0.0
        
        slippage_cost_usd = notional * (slippage_bps_applied / 10000.0)
        
        # Add position using intended entry price; costs already deducted from cash
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
            self._logger.info(f"[ORACLE DEBUG] execute_entry: Adding position: symbol={event.symbol}, qty={adjusted_qty}, price={entry_price}, side={signal.side}")
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
            event.symbol, adjusted_qty, adjusted_price, fill_ts,
            effective_stop_price, effective_stop_price,  # Initial stop and trail
            event.module, signal.side, fees, slippage_cost_usd,
            entry_idx=entry_idx,  # Store bar index for performance
            **deception_kwargs,
        )
        self.symbol_prev_prices[event.symbol] = fill_price
        if debug_oracle and event.module == 'ORACLE':
            self._logger.info(f"[ORACLE DEBUG] execute_entry: Position added! Total positions: {len(self.portfolio.positions)}, Total trades: {len(self.trades)}")
        
        # Get position_id after adding position
        position_id = self.portfolio.positions[event.symbol].position_id if event.symbol in self.portfolio.positions else ""
        
        # Record entry fill
        self.fill_recorder.record_fill(
            position_id=position_id,
            ts=fill_ts,
            symbol=event.symbol,
            module=event.module,
            leg='ENTRY',
            side='BUY' if signal.side == 'LONG' else 'SELL',
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
            intended_price=adjusted_price
        )
        
        # Record ENTRY_FILL ledger event (cash decreases by fees + slippage)
        self.fill_recorder.record_ledger_event(
            ts=fill_ts,
            event='ENTRY_FILL',
            position_id=position_id,
            symbol=event.symbol,
            module=event.module,
            leg='ENTRY',
            side='BUY' if signal.side == 'LONG' else 'SELL',
            qty=adjusted_qty,
            price=adjusted_price,
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
            self._logger.debug("Failed to remove stale pending signal")
        
        # FIX 2 & 3: Record trade with position_id, open_ts, gap_through
        if debug_oracle and event.module == 'ORACLE':
            self._logger.info(f"[ORACLE DEBUG] execute_entry: Appending trade: symbol={event.symbol}, side={signal.side}, qty={adjusted_qty}, price={fill_price}")
        self.trades.append({
            'ts': fill_ts,
            'symbol': event.symbol,
            'side': signal.side,
            'module': event.module,
            'qty': adjusted_qty,
            'price': adjusted_price,
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
            self._logger.info(f"[ORACLE DEBUG] execute_entry: Trade appended! Total trades: {len(self.trades)}")
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
        # Oracle and Deception positions use fixed stops (no trailing)
        if pos.module not in ('ORACLE', 'DECEPTION'):
            raise NotImplementedError(
                "Strategy-specific stop/trail calculation not available in engine core. "
                "Use oracle_mode for validation."
            )
        # Oracle/Deception positions keep original stop (no trailing)
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
                        # Fallback: use most recent bar at or before fill_ts
                        valid_df = df[df['ts'] <= fill_ts]
                        current_bar = valid_df.iloc[-1] if len(valid_df) > 0 else None
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
                            current_bar = None
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
                self.fill_recorder.record_fill(
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
                    self.fill_recorder.record_ledger_event(
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
                        'slip_bps': slippage_bps_applied,
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
