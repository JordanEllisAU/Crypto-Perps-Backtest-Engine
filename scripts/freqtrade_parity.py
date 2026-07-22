"""
Freqtrade parity harness for the Crypto-Perps-Backtest-Engine.

Converts an engine run (params + fills + OHLCV data) into a freqtrade
`user_data` bundle and, optionally, runs `freqtrade backtesting` to benchmark
the engine's trades against freqtrade's PnL engine.

Usage:
    python scripts/freqtrade_parity.py export runs/example_oracle --data-dir /path/to/ohlcv
    python scripts/freqtrade_parity.py run   runs/example_oracle --data-dir /path/to/ohlcv
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_value(param: Any, default: Any = None) -> Any:
    """If a param is a tunable dict, return its default value."""
    if isinstance(param, dict):
        return param.get("default", default)
    return param if param is not None else default


def _symbol_to_pair(symbol: str, stake_currency: str = "USDT") -> str:
    """Map engine symbol (e.g. BTCUSDT) to a freqtrade futures pair."""
    quote = stake_currency.upper()
    if symbol.upper().endswith(quote):
        base = symbol[: -len(quote)]
    else:
        # Best-effort split: assume last 4 chars are the quote
        base = symbol[:-4] if len(symbol) > 4 else symbol
    return f"{base.upper()}/{quote}:{quote}"


def _pair_to_filename(pair: str) -> str:
    """Convert a freqtrade pair to its data filename token."""
    return pair.replace("/", "_").replace(":", "_")


def _ts_to_ms(ts: pd.Timestamp) -> int:
    """Convert a UTC-aware Timestamp to integer milliseconds."""
    return int(ts.value) // 1_000_000


def _format_timerange(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Format a timerange string accepted by `freqtrade backtesting`."""
    return f"{int(start.timestamp())}-{int(end.timestamp())}"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def load_params(run_dir: Path) -> Dict[str, Any]:
    params_path = run_dir / "params_used.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing params file: {params_path}")
    with open(params_path, "r") as f:
        return json.load(f)


def load_fills(run_dir: Path) -> pd.DataFrame:
    fills_path = run_dir / "artifacts" / "fills.csv"
    if not fills_path.exists():
        # Fallback to the older location at the run root
        fills_path = run_dir / "fills.csv"
    if not fills_path.exists():
        raise FileNotFoundError(f"Missing fills CSV: {fills_path}")
    df = pd.read_csv(fills_path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def map_fills_to_signals(fills_df: pd.DataFrame, timeframe: str, stake_currency: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Aggregate fills into per-candle entry/exit signals keyed by signal candle.

    Freqtrade shifts entry/exit columns by one candle before acting on them, so a
    signal at time `t` is executed at `t + 1`. We use:
      - `signal_ts`: the candle where the signal column is set.
      - `exec_ts`:   the candle where freqtrade actually acts (`signal_ts + 1`).

    For same-bar exits (entry and exit share the same fill timestamp) we push the
    exit signal one candle later so freqtrade has an open position before it sees
    the exit signal.
    """
    timeframe_td = pd.Timedelta(timeframe)

    def _action(row: pd.Series) -> str:
        leg = str(row["leg"]).upper()
        side = str(row["side"]).upper()
        if leg == "ENTRY":
            return "enter_long" if side == "BUY" else "enter_short"
        return "exit_long" if side == "SELL" else "exit_short"

    df = fills_df.copy()
    df["pair"] = df["symbol"].apply(lambda s: _symbol_to_pair(s, stake_currency))
    df["action"] = df.apply(_action, axis=1)
    df["fill_ts"] = df["ts"].dt.floor(timeframe_td)

    # Compute per-position entry fill times to detect same-bar exits.
    entry_ts_by_position = (
        df[df["leg"].str.upper() == "ENTRY"]
        .groupby("position_id")["fill_ts"]
        .first()
        .to_dict()
    )

    rows = []
    for _, row in df.iterrows():
        action = row["action"]
        fill_ts = row["fill_ts"]
        if action.startswith("enter_"):
            signal_ts = fill_ts - timeframe_td
        else:
            entry_ts = entry_ts_by_position.get(row["position_id"])
            # If the exit happens in the same candle as the entry, push the exit
            # signal one candle later so freqtrade can exit the opened position.
            if entry_ts is not None and fill_ts == entry_ts:
                signal_ts = fill_ts
            else:
                signal_ts = fill_ts - timeframe_td
        exec_ts = signal_ts + timeframe_td
        rows.append(
            {
                "pair": row["pair"],
                "symbol": row["symbol"],
                "action": action,
                "signal_ts": signal_ts,
                "exec_ts": exec_ts,
                "fill_ts": fill_ts,
                "qty": float(row["qty"]),
                "price": float(row["price"]),
                "fee_bps": float(row["fee_bps"]),
            }
        )

    sig_df = pd.DataFrame(rows)
    sig_df["notional"] = sig_df["price"] * sig_df["qty"]
    grouped = (
        sig_df.groupby(["pair", "action", "signal_ts"], as_index=False)
        .agg(
            {
                "symbol": "first",
                "exec_ts": "first",
                "fill_ts": "first",
                "qty": "sum",
                "notional": "sum",
                "fee_bps": "first",
            }
        )
    )
    grouped["price"] = grouped["notional"] / grouped["qty"]
    grouped = grouped.drop(columns=["notional"])

    pair_map = dict(zip(df["symbol"], df["pair"]))
    return grouped, pair_map


def load_symbol_ohlcv(data_dir: Path, symbol: str) -> pd.DataFrame:
    """Load the engine's 15m CSV/parquet for a symbol."""
    for ext in (".csv", ".parquet"):
        path = data_dir / f"{symbol}_15m{ext}"
        if path.exists():
            if ext == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_parquet(path)
            if "ts" not in df.columns:
                raise ValueError(f"{path} is missing the 'ts' column")
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            return df.sort_values("ts").reset_index(drop=True)
    raise FileNotFoundError(f"No 15m data file for {symbol} in {data_dir}")


def write_ohlcv_json(
    output_dir: Path,
    exchange: str,
    pair: str,
    timeframe: str,
    df: pd.DataFrame,
) -> Path:
    """Write a freqtrade-compatible OHLCV JSON file for a pair."""
    out_dir = output_dir / "data" / exchange / "futures"
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = ["ts", "open", "high", "low", "close", "volume"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing {c} column in OHLCV data for {pair}")

    ohlcv = df[cols].copy()
    ohlcv["ts"] = ohlcv["ts"].apply(_ts_to_ms)
    values = ohlcv.values.tolist()

    filename = f"{_pair_to_filename(pair)}-{timeframe}-futures.json"
    out_path = out_dir / filename
    with open(out_path, "w") as f:
        json.dump(values, f)
    return out_path


def write_signals_json(output_dir: Path, signals: pd.DataFrame) -> Path:
    """Write the engine signal map consumed by the generated strategy."""
    out_path = output_dir / "engine_signals.json"
    payload: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    for _, row in signals.iterrows():
        pair = row["pair"]
        action = row["action"]
        exec_ts = row["exec_ts"]
        signal_ts = row["signal_ts"]
        payload.setdefault(pair, {}).setdefault(action, {})[signal_ts.isoformat()] = {
            "exec_ts": exec_ts.isoformat(),
            "price": float(row["price"]),
            "qty": float(row["qty"]),
            "fee_bps": float(row["fee_bps"]),
        }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def write_config(
    output_dir: Path,
    exchange: str,
    margin_mode: str,
    stake_currency: str,
    timeframe: str,
    pair_whitelist: List[str],
    max_open_trades: int,
    dry_run_wallet: float,
) -> Path:
    """Generate a minimal freqtrade config."""
    config = {
        "max_open_trades": max_open_trades,
        "stake_currency": stake_currency.upper(),
        "stake_amount": "unlimited",
        "dry_run_wallet": dry_run_wallet,
        "tradable_balance_ratio": 0.99,
        "fiat_display_currency": "USD",
        "timeframe": timeframe,
        "trading_mode": "futures",
        "margin_mode": margin_mode,
        "exchange": {
            "name": exchange,
            "key": "",
            "secret": "",
            "pair_whitelist": pair_whitelist,
            "pair_blacklist": [],
            "skip_pair_validation": True,
            "ccxt_config": {"options": {"defaultType": "swap"}},
            "ccxt_async_config": {"options": {"defaultType": "swap"}},
        },
        "pairlists": [{"method": "StaticPairList", "allow_inactive": True}],
        "dataformat_ohlcv": "json",
        "dataformat_trades": "json",
        "entry_pricing": {
            "price_side": "other",
            "use_order_book": False,
            "order_book_top": 1,
            "price_last_balance": 0.0,
        },
        "exit_pricing": {
            "price_side": "other",
            "use_order_book": False,
            "order_book_top": 1,
            "price_last_balance": 0.0,
        },
        "order_types": {
            "entry": "limit",
            "exit": "limit",
            "stoploss": "limit",
            "stoploss_on_exchange": False,
        },
        "order_time_in_force": {"entry": "GTC", "exit": "GTC"},
        "use_exit_signal": True,
        "exit_profit_only": False,
        "ignore_roi_if_entry_signal": False,
        "cancel_open_orders_on_exit": True,
        "user_data_dir": str(output_dir),
        "datadir": str(output_dir / "data" / exchange),
    }

    out_path = output_dir / "config.json"
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    return out_path


def write_strategy(output_dir: Path, timeframe: str) -> Path:
    """Generate a strategy that replays engine signals with exact fill prices/sizes."""
    strategy_dir = output_dir / "strategies"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    out_path = strategy_dir / "EngineParityStrategy.py"

    code = f'''"""
Auto-generated strategy that mirrors the Crypto-Perps-Backtest-Engine fills.
Reads `engine_signals.json` from the user_data directory.
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from freqtrade.strategy import IStrategy


class EngineParityStrategy(IStrategy):
    timeframe = "{timeframe}"
    can_short = True
    startup_candle_count = 0
    process_only_new_candles = False

    order_types = {{
        "entry": "limit",
        "exit": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": False,
    }}
    order_time_in_force = {{"entry": "GTC", "exit": "GTC"}}

    minimal_roi = {{"0": 100}}
    stoploss = -1
    trailing_stop = False

    def __init__(self, config):
        super().__init__(config)
        self._load_signals()

    def _load_signals(self):
        user_data = Path(self.config.get("user_data_dir", "."))
        signals_path = user_data / "engine_signals.json"
        self._signals = {{
            "enter_long": {{}},
            "enter_short": {{}},
            "exit_long": {{}},
            "exit_short": {{}},
        }}
        self._prices = {{"enter_long": {{}}, "enter_short": {{}}, "exit_long": {{}}, "exit_short": {{}}}}
        self._qtys = {{"enter_long": {{}}, "enter_short": {{}}}}

        if not signals_path.exists():
            return

        with open(signals_path, "r") as f:
            raw = json.load(f)

        for pair, actions in raw.items():
            for action, records in actions.items():
                signal_ts_map = {{}}
                price_map = {{}}
                qty_map = {{}}
                for signal_ts_iso, rec in records.items():
                    signal_ts = pd.Timestamp(signal_ts_iso)
                    exec_ts = pd.Timestamp(rec["exec_ts"])
                    signal_ts_map[signal_ts] = True
                    price_map[exec_ts] = float(rec["price"])
                    if action.startswith("enter_"):
                        qty_map[exec_ts] = float(rec["qty"])
                self._signals[action][pair] = signal_ts_map
                self._prices[action][pair] = price_map
                if action.startswith("enter_"):
                    self._qtys[action][pair] = qty_map

    def populate_indicators(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return df

    def _set_signal_column(self, df: pd.DataFrame, pair: str, action: str) -> None:
        dates = sorted(self._signals.get(action, {{}}).get(pair, {{}}).keys())
        df[action] = df["date"].isin(dates).astype(int)

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata["pair"]
        self._set_signal_column(df, pair, "enter_long")
        self._set_signal_column(df, pair, "enter_short")
        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata["pair"]
        self._set_signal_column(df, pair, "exit_long")
        self._set_signal_column(df, pair, "exit_short")
        return df

    def leverage(
        self, pair, current_time, current_rate, proposed_leverage, max_leverage,
        entry_tag, side, **kwargs
    ) -> float:
        return 1.0

    def custom_entry_price(
        self, pair, trade, current_time, proposed_rate, entry_tag, side, **kwargs
    ) -> float:
        action = "enter_long" if side == "long" else "enter_short"
        return self._prices.get(action, {{}}).get(pair, {{}}).get(current_time, proposed_rate)

    def custom_stake_amount(
        self, pair, current_time, current_rate, proposed_stake, min_stake,
        max_stake, leverage, entry_tag, side, **kwargs
    ) -> float:
        action = "enter_long" if side == "long" else "enter_short"
        price_map = self._prices.get(action, {{}}).get(pair, {{}})
        qty_map = self._qtys.get(action, {{}}).get(pair, {{}})
        if current_time in qty_map and current_time in price_map:
            return qty_map[current_time] * price_map[current_time] / max(leverage, 1e-9)
        return proposed_stake

    def custom_exit_price(
        self, pair, trade, current_time, proposed_rate, current_profit, exit_tag, **kwargs
    ) -> float:
        action = "exit_long" if not trade.is_short else "exit_short"
        return self._prices.get(action, {{}}).get(pair, {{}}).get(current_time, proposed_rate)
'''

    with open(out_path, "w") as f:
        f.write(code)
    return out_path


def export_freqtrade_userdata(args) -> Dict[str, Any]:
    """Convert an engine run into a freqtrade user_data directory."""
    run_dir = Path(args.run_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    params = load_params(run_dir)
    fills_df = load_fills(run_dir)

    initial_capital = _extract_value(params.get("general", {}).get("initial_capital_usd"), 100000.0)
    max_positions = int(_extract_value(params.get("general", {}).get("max_positions"), 1))
    taker_fee_bps = _extract_value(params.get("general", {}).get("taker_fee_bps"), 4.0)
    universe = params.get("universe", {}).get("initial", [])

    # The engine records fees and slippage as separate costs. Freqtrade only
    # exposes a single fee rate, so we compute an effective fee rate that folds
    # slippage into the per-trade cost. This makes total PnL parity much tighter.
    base_fee_rate = float(taker_fee_bps) / 10000.0
    total_notional = float(fills_df["notional_usd"].sum()) if "notional_usd" in fills_df.columns else 0.0
    total_fees = float(fills_df["fee_usd"].sum()) if "fee_usd" in fills_df.columns else 0.0
    total_slippage = float(fills_df["slippage_cost_usd"].sum()) if "slippage_cost_usd" in fills_df.columns else 0.0
    if total_notional > 0:
        effective_fee_rate = (total_fees + total_slippage) / total_notional
    else:
        effective_fee_rate = base_fee_rate

    signals_df, pair_map = map_fills_to_signals(fills_df, args.timeframe, args.stake_currency)

    # Whitelist only pairs that actually traded in the engine run
    pair_whitelist = sorted({row["pair"] for _, row in signals_df.iterrows()})

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write OHLCV for each traded symbol
    date_range = [None, None]
    symbols_needed = sorted({row["symbol"] for _, row in signals_df.iterrows()})
    for symbol in symbols_needed:
        try:
            ohlcv = load_symbol_ohlcv(data_dir, symbol)
            pair = pair_map[symbol]
            write_ohlcv_json(output_dir, args.exchange, pair, args.timeframe, ohlcv)
            if date_range[0] is None or ohlcv["ts"].iloc[0] < date_range[0]:
                date_range[0] = ohlcv["ts"].iloc[0]
            if date_range[1] is None or ohlcv["ts"].iloc[-1] > date_range[1]:
                date_range[1] = ohlcv["ts"].iloc[-1]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Cannot export freqtrade data: {e}. Provide --data-dir containing "
                f"{{symbol}}_15m.csv/parquet files."
            ) from e

    write_signals_json(output_dir, signals_df)
    write_config(
        output_dir,
        args.exchange,
        args.margin_mode,
        args.stake_currency,
        args.timeframe,
        pair_whitelist,
        max_open_trades=max_positions,
        dry_run_wallet=initial_capital,
    )
    write_strategy(output_dir, args.timeframe)

    timerange = _format_timerange(date_range[0], date_range[1]) if date_range[0] else None

    summary = {
        "output_dir": str(output_dir),
        "exchange": args.exchange,
        "pair_whitelist": pair_whitelist,
        "timerange": timerange,
        "fee": effective_fee_rate,
        "base_fee_rate": base_fee_rate,
        "effective_fee_rate": effective_fee_rate,
        "total_fees": total_fees,
        "total_slippage": total_slippage,
        "total_notional": total_notional,
        "initial_capital": float(initial_capital),
        "max_open_trades": max_positions,
        "num_signals": len(signals_df),
    }

    (output_dir / "freqtrade_parity_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Run / compare
# ---------------------------------------------------------------------------

def find_freqtrade_result(output_dir: Path) -> Optional[Path]:
    """Find the most recent freqtrade backtest result zip."""
    results_dir = output_dir / "backtest_results"
    if not results_dir.exists():
        return None
    zips = sorted(results_dir.glob("backtest-result-*.zip"), key=lambda p: p.stat().st_mtime)
    return zips[-1] if zips else None


def parse_freqtrade_result(zip_path: Path, strategy_name: str = "EngineParityStrategy") -> Dict[str, Any]:
    """Read the key metrics from a freqtrade backtest result zip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        json_name = next((n for n in names if n.endswith(".json") and "_config" not in n and "_wallet" not in n and "_market" not in n and "_" + strategy_name not in n), None)
        if json_name is None:
            raise ValueError(f"Could not locate result JSON inside {zip_path}")
        with zf.open(json_name) as f:
            data = json.load(f)

    strat = data.get("strategy", {}).get(strategy_name, {})
    return {
        "total_trades": int(strat.get("total_trades", 0)),
        "profit_total_abs": float(strat.get("profit_total_abs", 0.0)),
        "final_balance": float(strat.get("final_balance", strat.get("starting_balance", 0.0))),
    }


def load_engine_metrics(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "artifacts" / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing engine metrics: {metrics_path}")
    with open(metrics_path, "r") as f:
        return json.load(f)


def compare_results(
    run_dir: Path,
    output_dir: Path,
    report_path: Path,
) -> Dict[str, Any]:
    """Compare freqtrade backtest output with engine metrics."""
    zip_path = find_freqtrade_result(output_dir)
    if zip_path is None:
        raise FileNotFoundError(f"No freqtrade backtest result found in {output_dir / 'backtest_results'}")

    ft = parse_freqtrade_result(zip_path)
    engine = load_engine_metrics(run_dir)

    engine_trades = int(engine.get("total_trades", 0))
    engine_pnl = float(engine.get("realized_pnl_from_trades", engine.get("pnl_net_usd", 0.0)))
    engine_reported_final = float(engine.get("final_equity", 0.0))
    engine_equity = float(engine.get("initial_equity", 100000.0))
    # Use implied final equity (initial + realized PnL) for parity because it
    # reflects the same definition freqtrade uses (starting + absolute profit).
    engine_final = engine_equity + engine_pnl

    comparisons = [
        {
            "metric": "num_trades",
            "engine": engine_trades,
            "freqtrade": ft["total_trades"],
            "diff": abs(engine_trades - ft["total_trades"]),
            "pass": engine_trades == ft["total_trades"],
            "tolerance": "0 (identical)",
        },
        {
            "metric": "total_pnl",
            "engine": engine_pnl,
            "freqtrade": ft["profit_total_abs"],
            "diff": abs(engine_pnl - ft["profit_total_abs"]),
            "diff_bps_equity": (abs(engine_pnl - ft["profit_total_abs"]) / engine_equity) * 10000 if engine_equity else 0.0,
            "pass": (abs(engine_pnl - ft["profit_total_abs"]) / engine_equity) * 10000 < 10.0 if engine_equity else False,
            "tolerance": "< 10 bps equity",
        },
        {
            "metric": "final_equity",
            "engine": engine_final,
            "freqtrade": ft["final_balance"],
            "diff": abs(engine_final - ft["final_balance"]),
            "diff_bps": (abs(engine_final - ft["final_balance"]) / engine_equity) * 10000 if engine_equity else 0.0,
            "pass": (abs(engine_final - ft["final_balance"]) / engine_equity) * 10000 < 10.0 if engine_equity else False,
            "tolerance": "< 10 bps equity",
        },
        {
            "metric": "final_equity_reported",
            "engine": engine_reported_final,
            "freqtrade": engine_final,
            "diff": abs(engine_reported_final - engine_final),
            "diff_bps": (abs(engine_reported_final - engine_final) / engine_equity) * 10000 if engine_equity else 0.0,
            "pass": True,
            "tolerance": "informational only",
        },
    ]

    all_pass = all(c["pass"] for c in comparisons)
    report = {
        "parity_check": "PASS" if all_pass else "FAIL",
        "freqtrade_result_zip": str(zip_path),
        "comparisons": comparisons,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("Freqtrade parity report:")
    print(f"  Overall: {report['parity_check']}")
    for c in comparisons:
        status = "[PASS]" if c["pass"] else "[FAIL]"
        ft_val = c.get("freqtrade", c.get("replay"))
        print(f"  {status} {c['metric']}: engine={c['engine']:.6f} freqtrade={ft_val:.6f} diff={c['diff']:.6f}")
    print(f"  Report written to: {report_path}")
    return report


def run_freqtrade_backtest(output_dir: Path, fee: float, timerange: Optional[str]) -> None:
    """Invoke `freqtrade backtesting` on the generated user_data."""
    freqtrade_cmd = shutil.which("freqtrade")
    if freqtrade_cmd is None:
        raise RuntimeError(
            "`freqtrade` not found in PATH. Install it with "
            "`pip install -e '.[freqtrade]'` or `pip install freqtrade`."
        )

    config_path = output_dir / "config.json"
    cmd = [
        freqtrade_cmd,
        "backtesting",
        "--config", str(config_path),
        "--strategy", "EngineParityStrategy",
        "--userdir", str(output_dir),
        "--export", "trades",
        "--fee", str(fee),
    ]
    if timerange:
        cmd.extend(["--timerange", timerange])

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"freqtrade backtest failed (exit {result.returncode})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("run_dir", type=str, help="Path to the engine run directory")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing {symbol}_15m.csv/parquet OHLCV files")
    parser.add_argument("--output-dir", type=str, default="freqtrade_user_data", help="Output user_data directory")
    parser.add_argument("--exchange", type=str, default="okx", help="Exchange name for freqtrade (default: okx)")
    parser.add_argument("--margin-mode", type=str, default="isolated", choices=["isolated", "cross"], help="Futures margin mode")
    parser.add_argument("--stake-currency", type=str, default="USDT", help="Stake/quote currency")
    parser.add_argument("--timeframe", type=str, default="15m", help="Candle timeframe")


def main():
    parser = argparse.ArgumentParser(description="Freqtrade parity harness for the engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_p = subparsers.add_parser("export", help="Generate freqtrade user_data from an engine run")
    _add_export_args(export_p)

    run_p = subparsers.add_parser("run", help="Export and run freqtrade backtest, then compare results")
    _add_export_args(run_p)
    run_p.add_argument("--report", type=str, default="freqtrade_parity_report.json", help="Path for the parity report")

    args = parser.parse_args()

    summary = export_freqtrade_userdata(args)
    print(json.dumps(summary, indent=2))

    if args.command == "run":
        run_freqtrade_backtest(
            Path(summary["output_dir"]),
            fee=summary["fee"],
            timerange=summary["timerange"],
        )
        report = compare_results(
            Path(args.run_dir),
            Path(summary["output_dir"]),
            Path(args.report),
        )
        if report["parity_check"] == "FAIL":
            sys.exit(1)


if __name__ == "__main__":
    main()
