"""Technical indicators: ADX, EMA, MACD, Bollinger, RSI, ATR, Volume SMA"""
import logging
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

LOGGER = logging.getLogger(__name__)


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=period).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD (Moving Average Convergence Divergence)
    Returns DataFrame with columns: macd, signal, histogram
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    })


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands
    Returns DataFrame with columns: upper, middle, lower, width
    """
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    width = (upper - lower) / middle * 100  # Width as percentage
    
    return pd.DataFrame({
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'width': width
    })


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index"""
    # Calculate +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # True Range
    tr = atr(high, low, close, period=1)
    
    # Smoothed values
    atr_smooth = tr.rolling(window=period).sum()
    plus_di = 100 * (plus_dm.rolling(window=period).sum() / atr_smooth)
    minus_di = 100 * (minus_dm.rolling(window=period).sum() / atr_smooth)
    
    # DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # ADX is smoothed DX
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return adx


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators for a 15m bar dataframe
    Adds columns: adx, ema50, ema200, macd, macd_signal, macd_histogram,
                  bb_upper, bb_middle, bb_lower, bb_width, rsi, atr, volume_sma20
    """
    result = df.copy()

    # Define indicator computation tasks that can run in parallel
    def compute_adx_indicator():
        return ('adx', adx(result['high'], result['low'], result['close'], period=14))

    def compute_emas():
        close = result['close']
        return [
            ('ema50', ema(close, 50)),
            ('ema200', ema(close, 200))
        ]

    def compute_macd_indicator():
        macd_df = macd(result['close'], fast=12, slow=26, signal=9)
        return [
            ('macd', macd_df['macd']),
            ('macd_signal', macd_df['signal']),
            ('macd_histogram', macd_df['histogram'])
        ]

    def compute_bb_indicator():
        bb_df = bollinger_bands(result['close'], period=20, num_std=2.0)
        return [
            ('bb_upper', bb_df['upper']),
            ('bb_middle', bb_df['middle']),
            ('bb_lower', bb_df['lower']),
            ('bb_width', bb_df['width'])
        ]

    def compute_rsi_indicator():
        return ('rsi', rsi(result['close'], period=14))

    def compute_atr_indicator():
        return ('atr', atr(result['high'], result['low'], result['close'], period=14))

    def compute_volume_sma():
        return ('volume_sma20', sma(result['volume'], 20))

    # Run computations in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit tasks
        futures = [
            executor.submit(compute_adx_indicator),
            executor.submit(compute_emas),
            executor.submit(compute_macd_indicator),
            executor.submit(compute_bb_indicator),
            executor.submit(compute_rsi_indicator),
            executor.submit(compute_atr_indicator),
            executor.submit(compute_volume_sma)
        ]

        # Collect results
        for future in as_completed(futures):
            try:
                result_data = future.result()
                if isinstance(result_data, tuple):
                    # Single indicator result: ('name', series)
                    name, series = result_data
                    result[name] = series
                elif isinstance(result_data, list):
                    # Multiple indicator results: [('name1', series1), ('name2', series2), ...]
                    for name, series in result_data:
                        result[name] = series
            except Exception as exc:
                LOGGER.warning('Indicator computation failed: %s', exc)
                raise

    return result

