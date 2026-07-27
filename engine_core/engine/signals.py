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




class SignalsMixin:
    def generate_signals(self, symbol: str, idx: int, current_ts: pd.Timestamp, master_side: str):
        """Generate trading signals at bar t close"""
        debug_oracle = self.params.get('general', 'debug_oracle_flow', default=False)
        if debug_oracle:
            self._logger.info(f"[ORACLE DEBUG] generate_signals CALLED: symbol={symbol}, idx={idx}, ts={current_ts}, master_side={master_side}")
        df = self.symbol_data[symbol]
        if idx >= len(df):
            if debug_oracle:
                self._logger.info(f"[ORACLE DEBUG] generate_signals: idx {idx} >= len(df) {len(df)}, returning early")
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
                        self._logger.info(f"[ORACLE DEBUG] Bar {idx} ({current_ts}): Created ORACLE signal for {symbol}, side={oracle_signal.side}, signal_bar_idx={oracle_signal.signal_bar_idx}")
                    self.symbol_pending_signals[symbol].append(oracle_signal)
                    self._profile_counts['signals_generated'] += 1
                    if debug_oracle:
                        self._logger.info(f"[ORACLE DEBUG] Bar {idx}: Added to pending_signals. Total pending for {symbol}: {len(self.symbol_pending_signals[symbol])}")
            elif oracle_mode == 'always_short':
                oracle_signal = self.oracle_module.generate_always_short(symbol, df, idx, current_ts)
                if oracle_signal:
                    if debug_oracle:
                        self._logger.info(f"[ORACLE DEBUG] Bar {idx} ({current_ts}): Created ORACLE signal for {symbol}, side={oracle_signal.side}, signal_bar_idx={oracle_signal.signal_bar_idx}")
                    self.symbol_pending_signals[symbol].append(oracle_signal)
                    self._profile_counts['signals_generated'] += 1
                    if debug_oracle:
                        self._logger.info(f"[ORACLE DEBUG] Bar {idx}: Added to pending_signals. Total pending for {symbol}: {len(self.symbol_pending_signals[symbol])}")
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
