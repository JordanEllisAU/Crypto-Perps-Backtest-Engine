"""Margin guard: cross-margin ratio gates"""
import pandas as pd
from typing import Dict, List, Tuple, Optional
from engine_core.src.risk.logging import log_risk_event


def calculate_margin_ratio(
    positions,  # Dict[str, Position] or Dict[str, Dict]
    equity: float
) -> float:
    """
    Calculate cross-margin ratio.
    
    Margin ratio = total_notional / equity
    """
    if equity == 0:
        return 0.0
    
    total_notional = 0.0
    for pos in positions.values():
        if hasattr(pos, 'qty'):
            # Position object
            module = pos.module if hasattr(pos, 'module') else ''
            qty = pos.qty
            entry_price = pos.entry_price
        elif isinstance(pos, dict):
            # Dict
            module = pos.get('module', '')
            qty = pos.get('qty', 0)
            entry_price = pos.get('entry_price', 0)
        else:
            continue
        if module == 'ORACLE':
            continue
        total_notional += abs(qty * entry_price)
    
    margin_ratio = total_notional / equity
    
    return margin_ratio


def check_margin_constraints(
    margin_ratio: float,
    block_ratio: float = 0.60,
    trim_ratio: float = 0.50,
    flatten_ratio: float = 0.80
) -> Tuple[str, bool]:
    """
    Check margin constraints.
    
    Returns:
        (action, should_act)
        action: 'NONE', 'BLOCK', 'TRIM', 'FLATTEN'
    """
    if margin_ratio >= flatten_ratio:
        return 'FLATTEN', True
    elif margin_ratio >= block_ratio:
        return 'BLOCK', True
    elif margin_ratio >= trim_ratio:
        return 'TRIM', True
    else:
        return 'NONE', False


def get_trim_precedence(
    positions,  # Dict[str, Position] or Dict[str, Dict]
    es_contributions: Dict[str, float] = None
) -> List[Tuple[str, Dict]]:
    """
    Get trim precedence order.
    
    Order: largest notional → highest ES contribution → oldest → symbol A→Z
    """
    # Convert to list of (symbol, position) tuples with metadata
    pos_list = []
    for sym, pos in positions.items():
        if hasattr(pos, 'qty'):
            # Position object
            qty = pos.qty
            entry_price = pos.entry_price
            age_bars = getattr(pos, 'age_bars', 0)
        elif isinstance(pos, dict):
            # Dict
            qty = pos.get('qty', 0)
            entry_price = pos.get('entry_price', 0)
            age_bars = pos.get('age_bars', 0)
        else:
            continue
        
        if qty != 0:
            pos_meta = {
                '_notional': abs(qty * entry_price),
                '_es_contrib': es_contributions.get(sym, 0.0) if es_contributions else 0.0,
                '_age': age_bars
            }
            pos_list.append((sym, pos_meta))
    
    # Sort: largest notional → highest ES → oldest → lexicographic
    pos_list.sort(key=lambda x: (
        -x[1]['_notional'],  # Largest first
        -x[1]['_es_contrib'],  # Highest ES first
        x[1]['_age'],  # Oldest first
        x[0]  # Lexicographic
    ))
    
    return pos_list


def handle_margin_deadlock(
    trim_count: int,
    max_trim_count: int = 3
) -> Tuple[bool, bool]:
    """
    Check for margin deadlock.
    
    After 3 trims without resolving → flatten all & persist HALT_MANUAL
    
    Returns:
        (is_deadlock, should_flatten)
    """
    is_deadlock = trim_count >= max_trim_count
    
    return is_deadlock, is_deadlock


# Launch Punch List – Blocker #1: trim deadlock safety
def trim_with_deadlock_safety(
    positions,  # Dict[str, Position] or Dict[str, Dict]
    equity: float,
    params: dict,
    es_contributions: Dict[str, float] = None,
    close_position_callback=None  # Callback function to close a position: close_position_callback(symbol) -> bool
) -> Tuple[bool, int, float, float]:
    """
    Bounded trim loop with deadlock safety.
    
    Implements: while margin_ratio >= trim_threshold and trim_count < max_trim_count → trim once, recompute, increment.
    If after max_trim_count trims constraints still breach and trim_flatten_on_fail=True → return flatten flag.
    
    Args:
        positions: Current positions dict
        equity: Current equity
        params: Parameters dict with margin thresholds and safety settings
        es_contributions: Optional ES contributions per symbol
        close_position_callback: Function to close a position: close_position_callback(symbol) -> bool (True if closed)
    
    Returns:
        (should_flatten, trim_count, margin_ratio_before, margin_ratio_after)
    """
    max_trim_count = params.get('max_trim_count', 3)
    trim_flatten_on_fail = params.get('trim_flatten_on_fail', True)
    trim_threshold = params.get('trim_target_ratio_pct', 50.0) / 100.0
    
    trim_count = 0
    margin_ratio_before = calculate_margin_ratio(positions, equity)
    
    # Log initial state
    print(f"[TRIM_DEADLOCK_SAFETY] Starting trim loop: margin_ratio={margin_ratio_before:.4f}, trim_threshold={trim_threshold:.4f}, max_trim_count={max_trim_count}")
    
    # Early exit: if margin_ratio < trim_threshold, no action needed
    if margin_ratio_before < trim_threshold:
        print(f"[TRIM_DEADLOCK_SAFETY] Margin ratio {margin_ratio_before:.4f} below threshold {trim_threshold:.4f}, no trim needed")
        return False, 0, margin_ratio_before, margin_ratio_before
    
    # Bounded loop: while margin_ratio >= trim_threshold AND trim_count < max_trim_count
    current_margin_ratio = margin_ratio_before
    while current_margin_ratio >= trim_threshold and trim_count < max_trim_count:
        # Get trim precedence
        pos_list = get_trim_precedence(positions, es_contributions)
        if len(pos_list) == 0:
            print(f"[TRIM_DEADLOCK_SAFETY] No positions to trim")
            break
        
        # Close largest position (simplified: close entire position, one at a time)
        symbol_to_close, pos_meta = pos_list[0]
        
        # Get current position info for logging
        pos = positions.get(symbol_to_close)
        if pos:
            if hasattr(pos, 'qty'):
                current_qty = pos.qty
                entry_price = pos.entry_price
            else:
                current_qty = pos.get('qty', 0)
                entry_price = pos.get('entry_price', 0)
        else:
            current_qty = 0
            entry_price = 0
        
        # Call close callback if provided
        if close_position_callback:
            closed = close_position_callback(symbol_to_close)
            if not closed:
                print(f"[TRIM_DEADLOCK_SAFETY] Close callback failed for {symbol_to_close}, treating as deadlock")
                break  # Treat callback failure as deadlock - break and check if we should flatten
        else:
            # Default: remove from positions dict (simplified - actual engine should handle this properly)
            if symbol_to_close in positions:
                del positions[symbol_to_close]
        
        trim_count += 1
        # Recalculate margin ratio with updated positions
        current_margin_ratio = calculate_margin_ratio(positions, equity)
        
        print(f"[TRIM_DEADLOCK_SAFETY] Trim #{trim_count}: Closed {symbol_to_close} (qty={current_qty:.6f}), margin_ratio={current_margin_ratio:.4f}")
    
    # After loop: check if still breached
    margin_ratio_after = calculate_margin_ratio(positions, equity)
    
    # Determine if we should flatten
    should_flatten = False
    if margin_ratio_after >= trim_threshold:
        if trim_flatten_on_fail:
            should_flatten = True
            print(f"[TRIM_DEADLOCK_SAFETY] DEADLOCK: After {trim_count} trims, margin_ratio={margin_ratio_after:.4f} still breaches threshold {trim_threshold:.4f}. FLATTEN required.")
            # Launch Punch List – Quick Win #1: structured risk logging
            log_risk_event(
                'trim_fail',
                {
                    'trim_count': trim_count,
                    'margin_ratio_before': margin_ratio_before,
                    'margin_ratio_after': margin_ratio_after,
                    'max_trim_count': max_trim_count,
                    'trim_threshold': trim_threshold,
                    'action': 'FLATTEN_REQUIRED'
                }
            )
        else:
            print(f"[TRIM_DEADLOCK_SAFETY] WARNING: After {trim_count} trims, margin_ratio={margin_ratio_after:.4f} still breaches threshold {trim_threshold:.4f}, but flatten_on_fail=False")
            # Launch Punch List – Quick Win #1: structured risk logging
            log_risk_event(
                'trim_fail',
                {
                    'trim_count': trim_count,
                    'margin_ratio_before': margin_ratio_before,
                    'margin_ratio_after': margin_ratio_after,
                    'max_trim_count': max_trim_count,
                    'trim_threshold': trim_threshold,
                    'action': 'WARNING_ONLY'
                }
            )
    else:
        print(f"[TRIM_DEADLOCK_SAFETY] Trim loop resolved after {trim_count} trims: margin_ratio={margin_ratio_after:.4f} < threshold {trim_threshold:.4f}")
    
    return should_flatten, trim_count, margin_ratio_before, margin_ratio_after


# Launch Punch List – Blocker #2: centralized ES + margin guardrails
def check_risk_before_order(
    symbol: str,
    qty: float,
    price: float,
    current_positions,  # Dict[str, Position] or Dict[str, Dict]
    current_equity: float,
    current_es_used_pct: float,  # Current ES usage as % of equity
    additional_risk: float,  # Additional risk from this order (stop distance * qty)
    params: dict,
    es_cap_pct: float = 0.0225  # ES cap as % of equity
) -> Tuple[bool, str, float, float]:
    """
    Centralized risk check before submitting any order.
    
    Computes post-trade margin ratio and ES usage, and blocks if any cap would be breached.
    
    Args:
        symbol: Symbol for the order
        qty: Order quantity
        price: Order price
        current_positions: Current positions dict
        current_equity: Current equity
        current_es_used_pct: Current ES usage as % of equity
        additional_risk: Additional risk from this order
        params: Parameters dict with risk limits
        es_cap_pct: ES cap as % of equity (default 2.25%)
    
    Returns:
        (is_allowed, reason, margin_ratio_proj, es_used_proj_pct)
        - is_allowed: True if order is allowed, False if blocked
        - reason: Reason for block (empty if allowed)
        - margin_ratio_proj: Projected margin ratio after order
        - es_used_proj_pct: Projected ES usage as % of equity after order
    """
    risk_params = params.get('risk', {}).get('limits', {})
    min_margin_ratio_pct = risk_params.get('min_margin_ratio_pct', 9.0)  # Minimum margin headroom
    max_es_pct = risk_params.get('max_es_pct', 2.25)  # Max ES usage
    
    # Calculate current margin ratio
    current_margin_ratio = calculate_margin_ratio(current_positions, current_equity)
    
    # Project post-trade margin ratio
    # Add new position notional to total
    new_notional = abs(qty * price)
    current_total_notional = sum(
        abs(pos.qty * (pos.entry_price if hasattr(pos, 'entry_price') else pos.get('entry_price', 0)))
        for pos in current_positions.values()
    )
    
    # If symbol already has a position, we need to account for it
    existing_pos_notional = 0.0
    if symbol in current_positions:
        pos = current_positions[symbol]
        if hasattr(pos, 'qty'):
            existing_pos_notional = abs(pos.qty * pos.entry_price)
        else:
            existing_pos_notional = abs(pos.get('qty', 0) * pos.get('entry_price', 0))
    
    # Projected total notional: current - existing (if any) + new
    projected_total_notional = current_total_notional - existing_pos_notional + new_notional
    margin_ratio_proj = projected_total_notional / current_equity if current_equity > 0 else 0.0
    
    # Project post-trade ES usage
    # ES_used = max(ES methods, total stop risk)
    # We approximate: current_es_used + additional_risk
    additional_es_pct = (additional_risk / current_equity) if current_equity > 0 else 0.0
    es_used_proj_pct = current_es_used_pct + additional_es_pct
    
    # Check margin constraint
    # We want margin_ratio to be below block threshold (typically 60%)
    block_ratio = params.get('block_new_entries_ratio_pct', 60.0) / 100.0
    if margin_ratio_proj >= block_ratio:
        reason = f"MARGIN: projected margin_ratio={margin_ratio_proj:.4f} >= block_ratio={block_ratio:.4f}"
        # Launch Punch List – Quick Win #1: structured risk logging
        log_risk_event(
            'es_margin_block',
            {
                'symbol': symbol,
                'reason': reason,
                'margin_ratio_proj': margin_ratio_proj,
                'block_ratio': block_ratio,
                'es_used_proj_pct': es_used_proj_pct
            }
        )
        return False, reason, margin_ratio_proj, es_used_proj_pct
    
    # Check ES constraint
    if es_used_proj_pct > max_es_pct:
        reason = f"ES: projected es_used={es_used_proj_pct:.4f}% > max_es={max_es_pct:.4f}%"
        # Launch Punch List – Quick Win #1: structured risk logging
        log_risk_event(
            'es_block',
            {
                'symbol': symbol,
                'reason': reason,
                'es_used_proj_pct': es_used_proj_pct,
                'max_es_pct': max_es_pct,
                'margin_ratio_proj': margin_ratio_proj
            }
        )
        return False, reason, margin_ratio_proj, es_used_proj_pct
    
    # Check minimum margin headroom (optional safety check)
    if min_margin_ratio_pct > 0:
        margin_headroom_pct = (1.0 - margin_ratio_proj) * 100.0
        if margin_headroom_pct < min_margin_ratio_pct:
            reason = f"MARGIN_HEADROOM: margin_headroom={margin_headroom_pct:.2f}% < min={min_margin_ratio_pct:.2f}%"
            # Launch Punch List – Quick Win #1: structured risk logging
            log_risk_event(
                'margin_headroom_block',
                {
                    'symbol': symbol,
                    'reason': reason,
                    'margin_headroom_pct': margin_headroom_pct,
                    'min_margin_ratio_pct': min_margin_ratio_pct,
                    'margin_ratio_proj': margin_ratio_proj
                }
            )
            return False, reason, margin_ratio_proj, es_used_proj_pct
    
    # All checks passed
    return True, "", margin_ratio_proj, es_used_proj_pct

