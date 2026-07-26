"""Loss halts: vol-adjusted daily/intraday halts"""
import logging
import pandas as pd
import json
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
from engine_core.src.risk.logging import log_risk_event

LOGGER = logging.getLogger(__name__)


class LossHaltState:
    """Track loss halt state"""
    
    def __init__(self):
        self.daily_pnl_start = None  # UTC day start
        self.daily_pnl = 0.0
        self.intraday_pnl = 0.0
        self.soft_brake_active = False
        self.soft_brake_until = None
        self.disabled_symbols = {}  # {symbol: disable_until_ts}
        self.drawdown_ladder_triggered = []  # List of triggered thresholds
        self.halt_manual = False
        # Launch Punch List – Blocker #3: robust daily loss kill-switch
        self.kill_switch_triggered = False
        self.kill_switch_triggered_date = None
    
    def update_daily_pnl(self, current_ts: pd.Timestamp, pnl: float, vol_scale: float, params: dict):
        """Update daily PnL and check hard stop"""
        # Check if new UTC day
        current_day = current_ts.normalize()
        
        if self.daily_pnl_start is None or current_day > self.daily_pnl_start:
            # New day - reset
            self.daily_pnl_start = current_day
            self.daily_pnl = 0.0
            self.soft_brake_active = False
            self.soft_brake_until = None
        
        # Update daily PnL
        self.daily_pnl = pnl
        
        # Check daily hard stop
        daily_hard_stop_pct = params.get('daily_hard_stop_pct', -0.025)
        threshold = daily_hard_stop_pct * vol_scale
        
        # Hard stop blocks new entries until next day
        # (handled by checking if daily_pnl < threshold)
    
    def check_daily_hard_stop(self, equity: float, vol_scale: float, params: dict) -> bool:
        """Check if daily hard stop is triggered"""
        if equity == 0:
            return False
        
        daily_hard_stop_pct = params.get('daily_hard_stop_pct', -0.025)
        threshold_pct = daily_hard_stop_pct * vol_scale
        threshold_dollar = equity * threshold_pct
        
        return self.daily_pnl < threshold_dollar
    
    def check_soft_brake(self, current_ts: pd.Timestamp, equity: float, vol_scale: float, params: dict) -> Tuple[bool, bool]:
        """
        Check soft brake.
        
        Returns:
            (is_active, should_activate)
        """
        soft_brake_pct = params.get('soft_brake_intraday_pct', -0.015)
        threshold_pct = soft_brake_pct * vol_scale
        threshold_dollar = equity * threshold_pct
        
        should_activate = self.intraday_pnl < threshold_dollar
        
        if should_activate and not self.soft_brake_active:
            # Activate for 6 hours
            self.soft_brake_active = True
            self.soft_brake_until = current_ts + timedelta(hours=6)
        elif self.soft_brake_active and current_ts >= self.soft_brake_until:
            # Deactivate
            self.soft_brake_active = False
            self.soft_brake_until = None
        
        return self.soft_brake_active, should_activate
    
    def check_per_symbol_cap(self, symbol: str, symbol_pnl: float, equity: float, vol_scale: float, params: dict, current_ts: pd.Timestamp) -> bool:
        """Check per-symbol daily loss cap"""
        per_symbol_cap_pct = params.get('per_symbol_daily_cap_pct', -0.008)
        threshold_pct = per_symbol_cap_pct * vol_scale
        threshold_dollar = equity * threshold_pct
        
        if symbol_pnl < threshold_dollar:
            # Disable symbol for 24h
            self.disabled_symbols[symbol] = current_ts + timedelta(hours=24)
            return True
        
        # Check if still disabled
        if symbol in self.disabled_symbols:
            if current_ts < self.disabled_symbols[symbol]:
                return True
            else:
                # Re-enable
                del self.disabled_symbols[symbol]
        
        return False
    
    def check_drawdown_ladder(self, drawdown_pct: float, params: dict) -> Tuple[float, bool]:
        """
        Check drawdown ladder.
        
        -10% → sizes ×0.5
        -20% → sizes ×0.1 + HALT_MANUAL (no auto-flatten)
        
        Returns:
            (size_multiplier, should_halt)
        """
        drawdown_ladder = params.get('drawdown_ladder', [])
        
        size_mult = 1.0
        should_halt = False
        
        for threshold in drawdown_ladder:
            dd_threshold = threshold.get('dd_threshold_pct', 0)
            size_mult_threshold = threshold.get('size_mult', 1.0)
            halt_flag = threshold.get('halt', False)
            
            if drawdown_pct <= dd_threshold and dd_threshold not in self.drawdown_ladder_triggered:
                size_mult = min(size_mult, size_mult_threshold)
                if halt_flag:
                    should_halt = True
                    self.halt_manual = True
                self.drawdown_ladder_triggered.append(dd_threshold)
        
        return size_mult, should_halt
    
    # Launch Punch List – Blocker #3: robust daily loss kill-switch
    def check_daily_kill_switch(
        self,
        equity: float,
        initial_equity: float,
        vol_scale: float,
        params: dict,
        current_ts: pd.Timestamp
    ) -> Tuple[bool, bool, bool]:
        """
        Check daily kill-switch with both pct and USD thresholds.
        
        Args:
            equity: Current equity
            initial_equity: Starting equity for the day
            vol_scale: Volatility scale factor
            params: Parameters dict with kill_switch settings
            current_ts: Current timestamp
        
        Returns:
            (is_triggered, flatten_on_trigger, block_new_entries)
        """
        kill_switch_params = params.get('risk', {}).get('kill_switch', {})
        max_daily_loss_pct = kill_switch_params.get('max_daily_loss_pct', 4.0)
        max_daily_loss_usd = kill_switch_params.get('max_daily_loss_usd', 2000.0)
        flatten_on_trigger = kill_switch_params.get('flatten_on_trigger', True)
        block_new_entries = kill_switch_params.get('block_new_entries', True)
        
        if equity == 0 or initial_equity == 0:
            return False, False, False
        
        # Calculate daily PnL (realized + unrealized)
        # For simplicity, use daily_pnl which tracks realized PnL
        # In a full implementation, this would include unrealized PnL from open positions
        daily_pnl_dollar = self.daily_pnl
        daily_pnl_pct = (daily_pnl_dollar / initial_equity) * 100.0 if initial_equity > 0 else 0.0
        
        # Apply vol scale to thresholds
        threshold_pct = max_daily_loss_pct * vol_scale
        threshold_usd = max_daily_loss_usd * vol_scale
        
        # Check if either threshold is breached
        is_triggered = (daily_pnl_pct <= -threshold_pct) or (daily_pnl_dollar <= -threshold_usd)
        
        if is_triggered and not self.kill_switch_triggered:
            # First time trigger - log loudly
            self.kill_switch_triggered = True
            self.kill_switch_triggered_date = current_ts.date() if hasattr(current_ts, 'date') else current_ts
            
            # Launch Punch List – Quick Win #1: structured risk logging
            log_risk_event(
                'daily_kill',
                {
                    'timestamp': current_ts.isoformat() if hasattr(current_ts, 'isoformat') else str(current_ts),
                    'daily_pnl_dollar': daily_pnl_dollar,
                    'daily_pnl_pct': daily_pnl_pct,
                    'threshold_usd': -threshold_usd,
                    'threshold_pct': -threshold_pct,
                    'flatten_on_trigger': flatten_on_trigger,
                    'block_new_entries': block_new_entries,
                    'equity': equity,
                    'initial_equity': initial_equity
                },
                timestamp=current_ts if hasattr(current_ts, 'isoformat') else None
            )
        
        return is_triggered, flatten_on_trigger, block_new_entries
    
    def save_state(self, filepath: str):
        """Save state to JSON file"""
        state_data = {
            'daily_pnl_start': self.daily_pnl_start.isoformat() if self.daily_pnl_start else None,
            'daily_pnl': self.daily_pnl,
            'intraday_pnl': self.intraday_pnl,
            'soft_brake_active': self.soft_brake_active,
            'soft_brake_until': self.soft_brake_until.isoformat() if self.soft_brake_until else None,
            'disabled_symbols': {
                sym: ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
                for sym, ts in self.disabled_symbols.items()
            },
            'drawdown_ladder_triggered': self.drawdown_ladder_triggered,
            'halt_manual': self.halt_manual,
            'kill_switch_triggered': self.kill_switch_triggered,
            'kill_switch_triggered_date': str(self.kill_switch_triggered_date) if self.kill_switch_triggered_date else None
        }
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    def load_state(self, filepath: str) -> bool:
        """
        Load state from JSON file
        
        Returns:
            True if state was loaded, False if file doesn't exist
        """
        path = Path(filepath)
        if not path.exists():
            return False
        
        try:
            with open(path, 'r') as f:
                state_data = json.load(f)
            
            # Restore state
            if state_data.get('daily_pnl_start'):
                self.daily_pnl_start = pd.Timestamp(state_data['daily_pnl_start'])
            self.daily_pnl = state_data.get('daily_pnl', 0.0)
            self.intraday_pnl = state_data.get('intraday_pnl', 0.0)
            self.soft_brake_active = state_data.get('soft_brake_active', False)
            if state_data.get('soft_brake_until'):
                self.soft_brake_until = pd.Timestamp(state_data['soft_brake_until'])
            self.disabled_symbols = {
                sym: pd.Timestamp(ts) if isinstance(ts, str) else ts
                for sym, ts in state_data.get('disabled_symbols', {}).items()
            }
            self.drawdown_ladder_triggered = state_data.get('drawdown_ladder_triggered', [])
            self.halt_manual = state_data.get('halt_manual', False)
            self.kill_switch_triggered = state_data.get('kill_switch_triggered', False)
            if state_data.get('kill_switch_triggered_date'):
                from datetime import date
                self.kill_switch_triggered_date = date.fromisoformat(state_data['kill_switch_triggered_date'])
            
            return True
        except Exception as e:
            LOGGER.warning("Failed to load state from %s: %s", filepath, e)
            return False

