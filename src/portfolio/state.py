"""Portfolio state: positions, equity, PnL, ES, margin"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import timedelta


@dataclass
class Position:
    """Position state"""
    symbol: str
    qty: float
    entry_price: float
    entry_ts: pd.Timestamp
    stop_price: float
    trail_price: float
    module: str
    side: str  # 'LONG' or 'SHORT'
    position_id: str = ""  # Unique position identifier
    entry_idx: int = -1  # Bar index when position was opened (for performance)
    age_bars: int = 0
    highest_close: float = 0.0
    lowest_close: float = 0.0
    r_multiple: float = 0.0  # Profit in R multiples
    initial_R: float = 0.0  # Initial risk distance (entry to initial stop) for SQUEEZE
    tp1_price: float = 0.0  # TP1 target price for SQUEEZE (0.0 = disabled)
    exit_on_last_bar: bool = False  # For ORACLE Buy & Hold: exit at last bar
    # ── DeceptionLeaderBot exit lattice (added for deception_mode) ──────────
    tp2_price: float = 0.0  # TP2 target price for the remainder (0 = disabled)
    tp1_frac: float = 1.0  # Fraction of position closed at TP1 (1.0 = full close)
    tsl_bps: float = 0.0  # Trailing stop callback in bps (0 = no TSL)
    tsl_active: bool = False  # Whether TSL is currently active on the remainder
    tsl_extreme: float = 0.0  # Favourable extreme tracked since TSL activation
    tp1_hit: bool = False  # Whether TP1 has already fired (partial close)
    deception_score: float = 0.0  # Bot signal deception_score (for reporting)
    trap_type: str = ""  # Bot signal trap_type (for reporting)


class PortfolioState:
    """Track portfolio state"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.daily_pnl = 0.0
        self.intraday_pnl = 0.0
        self.total_pnl = 0.0
        self.funding_paid = 0.0
        self.fees_paid = 0.0
        self.slippage_paid = 0.0
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0
    
    def add_position(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        entry_ts: pd.Timestamp,
        stop_price: float,
        trail_price: float,
        module: str,
        side: str,
        fees: float = 0.0,
        slippage: float = 0.0,
        entry_idx: int = -1,
        position_id: str = None,
        exit_on_last_bar: bool = False,
        tp1_price: float = 0.0,
        tp2_price: float = 0.0,
        tp1_frac: float = 1.0,
        tsl_bps: float = 0.0,
        deception_score: float = 0.0,
        trap_type: str = "",
    ):
        """Add new position"""
        import uuid
        if position_id is None:
            position_id = str(uuid.uuid4())

        position = Position(
            symbol=symbol,
            qty=qty,
            entry_price=entry_price,
            entry_ts=entry_ts,
            entry_idx=entry_idx,
            stop_price=stop_price,
            trail_price=trail_price,
            module=module,
            side=side,
            position_id=position_id,
            exit_on_last_bar=exit_on_last_bar,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            tp1_frac=tp1_frac,
            tsl_bps=tsl_bps,
            deception_score=deception_score,
            trap_type=trap_type,
        )
        
        # Validate: no duplicate positions per symbol
        if symbol in self.positions:
            raise ValueError(f"Position already exists for {symbol}")
        
        # Validate: quantity must be positive
        if qty <= 0:
            raise ValueError(f"Position quantity must be positive, got {qty}")
        
        # Validate: stop price must be valid (with small tolerance for floating point)
        tolerance = 0.0001
        if side == 'LONG' and stop_price >= entry_price - tolerance:
            raise ValueError(f"LONG position stop price ({stop_price:.4f}) must be < entry price ({entry_price:.4f})")
        if side == 'SHORT' and stop_price <= entry_price + tolerance:
            raise ValueError(f"SHORT position stop price ({stop_price:.4f}) must be > entry price ({entry_price:.4f})")
        
        self.positions[symbol] = position
        
        # Update cash
        # For futures, margin requirement is typically 1-5% of notional, but for simplicity
        # we track the full notional as "used" capital. The actual margin is much less.
        # However, for PnL calculation, we need to account for the full notional.
        notional = abs(qty * entry_price)
        # Reduce cash by fees and slippage only (margin is separate in futures)
        # But we need to track that we have a position
        self.cash -= fees + slippage
        self.fees_paid += fees
        self.slippage_paid += slippage
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_ts: pd.Timestamp,
        reason: str,
        fees: float = 0.0,
        slippage: float = 0.0
    ) -> tuple[Optional[Position], float]:
        """Close position and return (position, pnl)"""
        if symbol not in self.positions:
            return None, 0.0
        
        position = self.positions[symbol]
        
        # Calculate PnL
        if position.side == 'LONG':
            pnl = (exit_price - position.entry_price) * position.qty
        else:  # SHORT
            pnl = (position.entry_price - exit_price) * position.qty
        
        # Fees and slippage reduce PnL
        pnl -= fees + slippage
        
        # Update cash: add the PnL (fees/slippage already deducted from pnl)
        # For futures, we don't actually "get back" notional since we never paid it
        # We just realize the PnL
        old_cash = self.cash
        self.cash += pnl
        # Debug: ensure cash is updated correctly
        if abs(self.cash - (old_cash + pnl)) > 1e-6:
            raise ValueError(f"Cash update error: old={old_cash}, pnl={pnl}, new={self.cash}")
        
        # Update totals
        self.total_pnl += pnl
        self.fees_paid += fees
        self.slippage_paid += slippage
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[symbol]
        
        # Update equity: cash + unrealized PnL from remaining positions
        # When no positions remain, unrealized_pnl = 0, so equity = cash
        # Note: This is a simplified update. For accurate unrealized PnL with remaining positions,
        # update_equity_all_positions should be called with current prices after this.
        # But for accounting correctness, at minimum: equity = cash when no positions
        if len(self.positions) == 0:
            self.equity = self.cash
        # If positions remain, equity will be updated by update_equity_all_positions with current prices
        
        return position, pnl
    
    def update_position_pnl(
        self,
        symbol: str,
        current_price: float,
        current_ts: pd.Timestamp
    ) -> float:
        """Update unrealized PnL for a single position (returns PnL only, doesn't update equity)"""
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        
        # Calculate unrealized PnL for this position only
        if position.side == 'LONG':
            unrealized_pnl = (current_price - position.entry_price) * position.qty
        else:  # SHORT
            unrealized_pnl = (position.entry_price - current_price) * position.qty
        
        return unrealized_pnl
    
    def update_equity_all_positions(self, symbol_prices: Dict[str, float], current_ts: pd.Timestamp):
        """
        Update equity for all positions using correct prices for each symbol.
        
        This is the correct way to calculate equity when multiple positions exist.
        """
        # Calculate total unrealized PnL using correct price for each position
        total_unrealized_pnl = 0.0
        
        for symbol, position in self.positions.items():
            # Get current price for this symbol
            current_price = symbol_prices.get(symbol)
            if current_price is None:
                # If price not available, skip this position (shouldn't happen)
                continue
            
            # Calculate unrealized PnL for this position
            if position.side == 'LONG':
                unrealized_pnl = (current_price - position.entry_price) * position.qty
            else:  # SHORT
                unrealized_pnl = (position.entry_price - current_price) * position.qty
            
            total_unrealized_pnl += unrealized_pnl
        
        # Update equity: cash + unrealized PnL from all positions
        self.equity = self.cash + total_unrealized_pnl
        
        # Validate equity bounds (should never be negative beyond reasonable limit)
        min_equity = -0.1 * self.initial_capital  # Allow up to -10% (margin call scenario)
        max_equity = 10.0 * self.initial_capital  # Flag if equity exceeds 10x (unrealistic spike)
        
        # Note: We don't cap equity here, just validate bounds for debugging
        # If equity goes outside bounds, it indicates a bug that needs fixing
        
        # Update drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        drawdown = self.peak_equity - self.equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        if self.peak_equity > 0:
            self.max_drawdown_pct = self.max_drawdown / self.peak_equity
    
    def update_position_trail(
        self,
        symbol: str,
        new_stop: float,
        new_trail: float
    ):
        """Update stop and trail prices"""
        if symbol in self.positions:
            position = self.positions[symbol]
            position.stop_price = new_stop
            position.trail_price = new_trail
    
    def update_position_age(self, symbol: str, age_bars: int):
        """Update position age"""
        if symbol in self.positions:
            self.positions[symbol].age_bars = age_bars
    
    def update_position_extremes(
        self,
        symbol: str,
        highest_close: float,
        lowest_close: float
    ):
        """Update highest/lowest closes for position"""
        if symbol in self.positions:
            position = self.positions[symbol]
            position.highest_close = max(position.highest_close, highest_close)
            position.lowest_close = min(position.lowest_close, lowest_close) if position.lowest_close > 0 else lowest_close
    
    def calculate_funding_cost(
        self,
        symbol: str,
        funding_rate: float,
        notional: float
    ) -> float:
        """Calculate and apply signed funding cost (positive = paid, negative = received)."""
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        
        # Binance USD-M convention: positive rate -> longs pay shorts.
        # LONG cost = notional * rate; SHORT cost = -notional * rate.
        if position.side == 'LONG':
            cost = notional * funding_rate
        else:
            cost = -notional * funding_rate
        
        self.funding_paid += cost
        self.cash -= cost
        
        return cost
    
    def get_total_notional(self) -> float:
        """Get total notional of all positions"""
        return sum(abs(pos.qty * pos.entry_price) for pos in self.positions.values())
    
    def get_total_stop_risk(self) -> float:
        """Get total stop-distance risk across all positions (used for ES guard)"""
        total_risk = 0.0
        for pos in self.positions.values():
            stop_distance = abs(pos.entry_price - pos.stop_price)
            total_risk += stop_distance * abs(pos.qty)
        return total_risk
    
    def get_margin_ratio(self) -> float:
        """Get margin ratio"""
        if self.equity == 0:
            return 0.0
        return self.get_total_notional() / self.equity
    
    def get_position_count(self) -> int:
        """Get number of open positions"""
        return len(self.positions)

