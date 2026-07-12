"""DECEPTION module: replays DeceptionLeaderBot signals through the engine.

This is NOT an oracle — it feeds real signals from the bot's pipeline into the
engine's accounting/risk/execution framework to independently verify PnL math,
position conservation, and directional accuracy using the bot's exact exit
lattice (SL at signal SL, TP1 partial at signal TP1, remainder to TP2 or TSL).

Signal schema (CSV columns):
    symbol, side, entry_ts, entry_price, sl_price, tp1_price, tp2_price,
    tp1_frac, tsl_bps, qty, trap_type, deception_score

Exit lattice (mirrors core/sim_core.py simulate_exit_on_bars):
    1. Liquidation check (cross-margin, N-concurrent) — checked first
    2. If TSL active: trail favourable extreme, exit on retracement > tsl_callback
    3. Else: check SL and TP1 on the same bar
        - If both hit on the same bar: pessimistic — SL wins
        - If TP1 hits and tp1_frac < 1.0: activate TSL on remainder (if tsl_bps>0)
        - If TP1 hits and tp1_frac == 1.0: full close at TP1
    4. All exits are TAKER (market orders) — matches live PARITY 2026-07-12
    5. NO time-based exit (max_hold/timeout removed from the bot)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class DeceptionSignal:
    """A single DeceptionLeaderBot signal replayed through the engine."""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    stop_price: float  # SL price (signal's sl_price)
    signal_bar_idx: int
    signal_ts: pd.Timestamp
    module: str = 'DECEPTION'
    # Bot-specific exit lattice fields
    tp1_price: float = 0.0  # TP1 target (0 = disabled, fall back to single TP)
    tp2_price: float = 0.0  # TP2 target for remainder (0 = disabled)
    tp1_frac: float = 1.0  # Fraction of position closed at TP1 (1.0 = full close)
    tsl_bps: float = 0.0  # Trailing stop callback in bps (0 = no TSL)
    qty: float = 0.0  # Position size (in base units)
    trap_type: str = ""
    deception_score: float = 0.0
    exit_on_last_bar: bool = True  # Bot closes at end of available data


def load_signals_csv(csv_path: str | Path) -> List[DeceptionSignal]:
    """Load signals from a CSV file into DeceptionSignal objects.

    Expected columns: symbol, side, entry_ts, entry_price, sl_price,
    tp1_price, tp2_price, tp1_frac, tsl_bps, qty, trap_type, deception_score

    `side` is normalized to UPPER. `entry_ts` is parsed as UTC.
    """
    signals: List[DeceptionSignal] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row["entry_ts"]
            ts = pd.Timestamp(ts_str)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            side = row["side"].strip().upper()
            if side in ("BUY", "LONG"):
                side = "LONG"
            elif side in ("SELL", "SHORT"):
                side = "SHORT"
            signals.append(DeceptionSignal(
                symbol=row["symbol"],
                side=side,
                entry_price=float(row["entry_price"]),
                stop_price=float(row["sl_price"]),
                signal_bar_idx=-1,  # Resolved against df at runtime
                signal_ts=ts,
                tp1_price=float(row.get("tp1_price", 0.0) or 0.0),
                tp2_price=float(row.get("tp2_price", 0.0) or 0.0),
                tp1_frac=float(row.get("tp1_frac", 1.0) or 1.0),
                tsl_bps=float(row.get("tsl_bps", 0.0) or 0.0),
                qty=float(row.get("qty", 0.0) or 0.0),
                trap_type=row.get("trap_type", ""),
                deception_score=float(row.get("deception_score", 0.0) or 0.0),
            ))
    return signals


class DeceptionModule:
    """Replays DeceptionLeaderBot signals through the engine.

    The module is stateless per-bar: it emits a DECEPTION signal at the bar
    whose timestamp matches (or is the first bar at-or-after) the signal's
    entry_ts. The engine's existing pending_signals mechanism routes the
    signal to execution on the next bar.
    """

    def __init__(self, params: dict, signals: Optional[List[DeceptionSignal]] = None):
        self.params = params
        # Index signals by (symbol, bar_ts_floor) for fast lookup.
        # bar_ts_floor is the 15m (or 1m) floor of entry_ts.
        self._signal_index: Dict[tuple, List[DeceptionSignal]] = {}
        # Track which signals have been emitted (one-shot per signal)
        self._emitted: set = set()
        if signals:
            self._index_signals(signals)

    def _index_signals(self, signals: List[DeceptionSignal]) -> None:
        """Group signals by (symbol, floored_ts) for bar-by-bar lookup."""
        for i, sig in enumerate(signals):
            # Floor to the nearest minute (engine bar size is configurable).
            # The engine will resolve the actual bar index.
            bar_ts = sig.signal_ts.floor("1min")
            key = (sig.symbol, bar_ts)
            if key not in self._signal_index:
                self._signal_index[key] = []
            # Tag with original index for one-shot tracking
            sig_copy = DeceptionSignal(
                symbol=sig.symbol, side=sig.side,
                entry_price=sig.entry_price, stop_price=sig.stop_price,
                signal_bar_idx=-1, signal_ts=sig.signal_ts,
                tp1_price=sig.tp1_price, tp2_price=sig.tp2_price,
                tp1_frac=sig.tp1_frac, tsl_bps=sig.tsl_bps,
                qty=sig.qty, trap_type=sig.trap_type,
                deception_score=sig.deception_score,
                exit_on_last_bar=sig.exit_on_last_bar,
            )
            sig_copy._sig_id = i  # type: ignore[attr-defined]
            self._signal_index[key].append(sig_copy)

    def generate_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        idx: int,
        current_ts: pd.Timestamp,
    ) -> Optional[DeceptionSignal]:
        """Emit a DECEPTION signal for the current bar if one matches.

        Resolves signal_bar_idx against the df so the engine can route it
        to execution on bar idx+1.
        """
        # Try exact floor match first, then walk back 1-2 minutes for tolerance
        for offset_min in (0, 1, 2):
            bar_ts = current_ts.floor("1min") - pd.Timedelta(minutes=offset_min)
            key = (symbol, bar_ts)
            if key not in self._signal_index:
                continue
            for sig in self._signal_index[key]:
                sig_id = getattr(sig, "_sig_id", id(sig))
                if sig_id in self._emitted:
                    continue
                self._emitted.add(sig_id)
                # Resolve the bar index in df for this signal
                sig.signal_bar_idx = idx
                return sig
        return None
