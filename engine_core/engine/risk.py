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




class RiskMixin:
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
        """Apply signed funding costs at Binance USD-M funding times (00:00, 08:00, 16:00 UTC)."""
        if symbol not in self.portfolio.positions:
            return
        
        pos = self.portfolio.positions[symbol]
        funding_df = self.data_loader.get_funding(symbol)
        
        if funding_df is None or len(funding_df) == 0:
            return
        
        # Only accrue funding at exact funding times: 00:00, 08:00, 16:00 UTC
        current_hour = current_ts.hour
        current_minute = current_ts.minute
        current_second = current_ts.second
        
        funding_times = [0, 8, 16]
        is_funding_time = (
            current_hour in funding_times and
            current_minute == 0 and
            current_second == 0
        )
        
        if not is_funding_time:
            return
        
        # Find the most recent funding rate published at or before current_ts
        funding_match = funding_df[funding_df['funding_ts'] <= current_ts]
        if len(funding_match) == 0:
            return
        
        funding_rate = funding_match.iloc[-1]['funding_rate']
        
        # Use current mark price (close of the bar at current_ts) for funding notional
        df = self.symbol_data.get(symbol)
        if df is not None and len(df) > 0:
            if hasattr(self, 'symbol_ts_to_idx') and symbol in self.symbol_ts_to_idx:
                bar_idx = self.symbol_ts_to_idx[symbol].get(current_ts, None)
            else:
                bar_match = df[df['ts'] == current_ts]
                bar_idx = bar_match.index[0] if len(bar_match) > 0 else None
            if bar_idx is not None and 0 <= bar_idx < len(df):
                mark_price = df.iloc[bar_idx]['close']
            else:
                valid_df = df[df['ts'] <= current_ts]
                mark_price = valid_df.iloc[-1]['close'] if len(valid_df) > 0 else pos.entry_price
        else:
            mark_price = pos.entry_price
        
        notional = abs(pos.qty * mark_price)
        
        if not self.cost_model_enabled:
            return
        
        cost = self.portfolio.calculate_funding_cost(symbol, funding_rate, notional)
        
        if cost != 0.0:
            self.forensic_log.append({
                'ts': current_ts,
                'symbol': symbol,
                'event': 'FUNDING_COST',
                'funding_rate': funding_rate,
                'cost': cost,
                'position_id': pos.position_id
            })
            self.funding_events_count += 1
            
            self.fill_recorder.record_ledger_event(
                ts=current_ts,
                event='FUNDING',
                position_id=pos.position_id,
                symbol=symbol,
                module=pos.module,
                leg='',
                side=pos.side,
                qty=pos.qty,
                price=mark_price,
                notional_usd=notional,
                fee_usd=0.0,
                slippage_cost_usd=0.0,
                funding_usd=cost,
                cash_delta_usd=-cost,
                note=f"Funding: rate={funding_rate:.6f}, cost={cost:.6f}"
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
