"""Compatibility re-export for the refactored BacktestEngine.

The engine implementation has moved to the engine_core.engine package;
this file is kept so existing imports of engine_core.src.engine continue
to work.
"""
from engine_core.engine import BacktestEngine

__all__ = ["BacktestEngine"]
