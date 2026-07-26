"""Fill and ledger recorder for the backtest engine.

Extracted from ``src/engine.py`` so the engine does not own low-level
bookkeeping details.  The recorder keeps references to the engine's
``fills``, ``ledger`` and ``outlier_log`` lists and updates them in place.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    import pandas as pd


class FillRecorder:
    """Record fill and ledger events and flag outlier fills."""

    def __init__(
        self,
        run_id: str,
        logger: logging.Logger,
        fills: List[Dict[str, Any]],
        ledger: List[Dict[str, Any]],
        outlier_log: List[Dict[str, Any]],
        fill_counter: Dict[str, int],
    ) -> None:
        self.run_id = run_id
        self._logger = logger
        self.fills = fills
        self.ledger = ledger
        self.outlier_log = outlier_log
        self._fill_counter = fill_counter

    def _check_and_log_outlier(self, fill_info: Dict[str, Any]) -> None:
        """Check if fill is an outlier and log if so."""
        slippage_bps = abs(fill_info.get("slippage_bps_applied", 0.0))

        # Flag condition: slippage_bps > max(20, 3 * median_slippage_bps_symbol_30d)
        # Fallback threshold = 20 bps (as we don't have 30d history loaded in engine)
        threshold = 20.0

        if slippage_bps > threshold:
            outlier_record = fill_info.copy()
            outlier_record["outlier_threshold_bps"] = threshold
            self.outlier_log.append(outlier_record)

    def record_fill(
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
        intended_price: float | None = None,
    ) -> None:
        """Record a fill event to the fills list."""
        if position_id not in self._fill_counter:
            self._fill_counter[position_id] = 0
        self._fill_counter[position_id] += 1
        fill_id = f"{position_id}-{leg}-{self._fill_counter[position_id]}"

        fill_record = {
            "run_id": self.run_id,
            "position_id": position_id,
            "fill_id": fill_id,
            "ts": ts,
            "symbol": symbol,
            "module": module,
            "leg": leg,
            "side": side,
            "qty": qty,
            "price": price,
            "notional_usd": notional_usd,
            "slippage_bps_applied": slippage_bps_applied,
            "slippage_cost_usd": slippage_cost_usd,
            "fee_bps": fee_bps,
            "fee_usd": fee_usd,
            "liquidity": liquidity,
            "participation_pct": participation_pct,
            "adv60_usd": adv60_usd,
            "intended_price": intended_price if intended_price is not None else price,
        }

        self.fills.append(fill_record)
        self._check_and_log_outlier(fill_record)

    def record_ledger_event(
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
        note: str = "",
    ) -> None:
        """Record a cash-affecting event to the ledger."""
        self.ledger.append(
            {
                "ts": ts,
                "run_id": self.run_id,
                "event": event,
                "position_id": position_id,
                "symbol": symbol,
                "module": module,
                "leg": leg,
                "side": side,
                "qty": qty,
                "price": price,
                "notional_usd": notional_usd,
                "fee_usd": fee_usd,
                "slippage_cost_usd": slippage_cost_usd,
                "funding_usd": funding_usd,
                "cash_delta_usd": cash_delta_usd,
                "note": note,
            }
        )
