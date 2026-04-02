"""BacktestDataHub — historical CSV data loader for offline backtesting.

Loads pre-collected Binance Futures CSVs, resamples to multiple timeframes,
and serves sliced windows identical to the live DataHub interface.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("backtest.data_loader")

# Resampling rules for OHLCV aggregation
_OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "taker_buy_base_vol": "sum",
    "quote_volume": "sum",
    "trade_count": "sum",
}

TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}


class BacktestDataHub:
    """Drop-in replacement for DataHub that serves historical data by timestamp."""

    def __init__(self, data_dir: Path, coins: list[str]):
        self._coins = coins
        # {coin: {timeframe: DataFrame}}
        self._data: dict[str, dict[str, pd.DataFrame]] = {}
        # OI data: {coin: DataFrame} with timestamp index
        self._oi_data: dict[str, pd.DataFrame] = {}
        # Funding rate: {coin: DataFrame}
        self._funding_data: dict[str, pd.DataFrame] = {}

        for coin in coins:
            sym = coin + "USDT"
            sym_dir = data_dir / sym
            if not sym_dir.exists():
                logger.warning(f"No data dir for {coin} at {sym_dir}")
                continue
            self._data[coin] = self._load_coin(sym_dir)
            self._oi_data[coin] = self._load_oi(sym_dir)
            self._funding_data[coin] = self._load_funding(sym_dir)

        logger.info(
            f"Loaded {len(self._data)} coins: "
            + ", ".join(f"{c}({len(self._data[c].get('1m', []))} 1m bars)" for c in self._data)
        )

    def _load_coin(self, sym_dir: Path) -> dict[str, pd.DataFrame]:
        """Load 1m klines CSV, resample to 5m/15m/1h."""
        result = {}

        # Load 1m
        csv_path = sym_dir / "klines_1m.csv"
        if not csv_path.exists():
            return result

        df = pd.read_csv(csv_path)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df.set_index("open_time", inplace=True)
        df.sort_index(inplace=True)

        # Ensure numeric
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Keep extra columns if available
        for col in ["taker_buy_base_vol", "quote_volume", "trade_count"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        result["1m"] = df

        # Resample to higher timeframes
        agg_cols = {k: v for k, v in _OHLCV_AGG.items() if k in df.columns}
        for tf, minutes in [("5m", 5), ("15m", 15), ("1h", 60)]:
            resampled = df.resample(f"{minutes}min", closed="left", label="left").agg(agg_cols)
            resampled.dropna(subset=["close"], inplace=True)
            result[tf] = resampled

        return result

    def _load_oi(self, sym_dir: Path) -> pd.DataFrame:
        """Load open interest 5m CSV."""
        csv_path = sym_dir / "open_interest_5m.csv"
        if not csv_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(csv_path)
        ts_col = "timestamp" if "timestamp" in df.columns else df.columns[-1]
        df["timestamp"] = pd.to_datetime(df[ts_col], utc=True)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        for col in ["sumOpenInterest", "sumOpenInterestValue"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _load_funding(self, sym_dir: Path) -> pd.DataFrame:
        """Load funding rate CSV."""
        csv_path = sym_dir / "funding_rate.csv"
        if not csv_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(csv_path)
        ts_col = "fundingTime" if "fundingTime" in df.columns else df.columns[1]
        df["timestamp"] = pd.to_datetime(df[ts_col], format="mixed", utc=True)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        if "fundingRate" in df.columns:
            df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        return df

    # ── Data Access (point-in-time safe) ──────────────────

    def get_ohlcv(
        self, coin: str, timeframe: str, limit: int, current_ts: pd.Timestamp
    ) -> pd.DataFrame | None:
        """Return the last `limit` bars up to and including current_ts.

        CRITICAL: No lookahead — only data strictly <= current_ts.
        """
        if coin not in self._data or timeframe not in self._data[coin]:
            return None
        df = self._data[coin][timeframe]
        # Point-in-time: only bars with index <= current_ts
        available = df.loc[:current_ts]
        if len(available) < 2:
            return None
        return available.tail(limit).copy()

    def get_bar(self, coin: str, timeframe: str, ts: pd.Timestamp) -> dict | None:
        """Get single bar at timestamp."""
        if coin not in self._data or timeframe not in self._data[coin]:
            return None
        df = self._data[coin][timeframe]
        if ts in df.index:
            row = df.loc[ts]
            return {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        return None

    def get_open_interest(self, coin: str, ts: pd.Timestamp) -> float | None:
        """Get most recent OI reading at or before ts (forward-fill)."""
        if coin not in self._oi_data or self._oi_data[coin].empty:
            return None
        oi = self._oi_data[coin]
        available = oi.loc[:ts]
        if available.empty:
            return None
        row = available.iloc[-1]
        if "sumOpenInterest" in oi.columns:
            return float(row["sumOpenInterest"])
        return None

    def get_funding_rate(self, coin: str, ts: pd.Timestamp) -> float | None:
        """Get most recent funding rate at or before ts."""
        if coin not in self._funding_data or self._funding_data[coin].empty:
            return None
        fr = self._funding_data[coin]
        available = fr.loc[:ts]
        if available.empty:
            return None
        row = available.iloc[-1]
        if "fundingRate" in fr.columns:
            return float(row["fundingRate"])
        return None

    def get_all_1m_timestamps(self, coins: list[str] | None = None) -> pd.DatetimeIndex:
        """Get sorted union of all 1m timestamps across coins."""
        target_coins = coins or self._coins
        all_idx = pd.DatetimeIndex([])
        for coin in target_coins:
            if coin in self._data and "1m" in self._data[coin]:
                all_idx = all_idx.union(self._data[coin]["1m"].index)
        return all_idx.sort_values()

    # ── Static microstructure helpers (reuse from DataHub) ──

    @staticmethod
    def compute_cvd(df: pd.DataFrame) -> pd.Series:
        """BVC-based Cumulative Volume Delta."""
        if df is None or len(df) < 2:
            return pd.Series(dtype=float)
        c, o, h, l, v = df["close"].values, df["open"].values, df["high"].values, df["low"].values, df["volume"].values
        rng = h - l
        rng[rng == 0] = 1e-10
        buy_ratio = np.where(c >= o, (c - l) / rng, (h - c) / rng)
        buy_vol = v * buy_ratio
        sell_vol = v * (1 - buy_ratio)
        delta = buy_vol - sell_vol
        return pd.Series(np.cumsum(delta), index=df.index, name="cvd")

    @staticmethod
    def compute_ofi(df: pd.DataFrame) -> pd.Series:
        """Order Flow Imbalance proxy."""
        if df is None or len(df) < 2:
            return pd.Series(dtype=float)
        c, o, h, l, v = df["close"].values, df["open"].values, df["high"].values, df["low"].values, df["volume"].values
        rng = h - l
        rng[rng == 0] = 1e-10
        ofi = ((c - o) / rng) * v
        return pd.Series(ofi, index=df.index, name="ofi")

    @staticmethod
    def compute_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Rolling VWAP."""
        if df is None or len(df) < window:
            return pd.Series(dtype=float)
        typical = (df["high"] + df["low"] + df["close"]) / 3
        vol = df["volume"]
        return (typical * vol).rolling(window).sum() / vol.rolling(window).sum()

    @staticmethod
    def compute_vpin(df: pd.DataFrame, window: int = 24) -> float:
        """VPIN calculation."""
        if df is None or len(df) < window:
            return 0.5
        recent = df.iloc[-window:]
        h, l, c, v, o = recent["high"], recent["low"], recent["close"], recent["volume"], recent["open"]
        rng = (h - l).replace(0, 1e-10)
        buy_frac = (c - l).where(c >= o, (h - c)) / rng
        buy_vol = buy_frac * v
        sell_vol = (1 - buy_frac) * v
        abs_imbalance = (buy_vol - sell_vol).abs()
        total_vol = v.replace(0, 1e-10)
        vpin = float(abs_imbalance.mean() / total_vol.mean())
        return min(vpin, 1.0)

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Compute ATR from OHLCV DataFrame. Returns last ATR value."""
        if df is None or len(df) < period + 1:
            return 0.0
        h, l, c = df["high"].values, df["low"].values, df["close"].values
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        if len(tr) < period:
            return float(np.mean(tr)) if len(tr) > 0 else 0.0
        # EMA-style ATR
        atr = np.mean(tr[:period])
        alpha = 1.0 / period
        for i in range(period, len(tr)):
            atr = atr * (1 - alpha) + tr[i] * alpha
        return float(atr)
