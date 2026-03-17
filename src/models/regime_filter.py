"""Regime Filter — 4상태 시장 레짐 분류 + 레짐별 EV 분리.

4상태:
  TREND_UP:   상승 추세 (EMA aligned up + ADX > threshold)
  TREND_DOWN: 하락 추세 (EMA aligned down + ADX > threshold)
  RANGE_LOW:  횡보 저변동 (ADX low + ATR ratio low)
  RANGE_HIGH: 횡보 고변동 (ADX low + ATR ratio high)

기존 RegimeDetector(3상태: TREND/RANGE/VOLATILE)를 확장.
4h bar 기반으로 동작 (tactical tier와 동일).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Regime4(str, Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE_LOW = "RANGE_LOW"
    RANGE_HIGH = "RANGE_HIGH"
    UNKNOWN = "UNKNOWN"


@dataclass
class Regime4Snapshot:
    """4상태 레짐 판별 결과."""
    regime: Regime4
    adx: float = 0.0
    ema_slope: float = 0.0      # EMA 기울기 (양=상승, 음=하락)
    atr_ratio: float = 0.0      # ATR / ATR_MA
    bb_width_pctile: float = 0.0
    vol_ratio: float = 0.0

    def __repr__(self):
        return (f"Regime4({self.regime.value} "
                f"ADX={self.adx:.1f} slope={self.ema_slope:.4f} "
                f"ATR_r={self.atr_ratio:.2f})")


class RegimeFilter:
    """4상태 레짐 필터.

    Usage:
        rf = RegimeFilter()
        regime = rf.classify(df_4h)
        regimes_series = rf.classify_series(df_4h)
    """

    def __init__(
        self,
        adx_trend_threshold: float = 22,
        atr_high_ratio: float = 1.3,
        ema_fast: int = 20,
        ema_slow: int = 50,
        adx_period: int = 14,
        lookback: int = 60,
    ):
        self.adx_trend = adx_trend_threshold
        self.atr_high = atr_high_ratio
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.adx_period = adx_period
        self.lookback = lookback

    def classify(self, df: pd.DataFrame) -> Regime4Snapshot:
        """최신 시점의 4상태 레짐 판별."""
        if len(df) < self.lookback:
            return Regime4Snapshot(regime=Regime4.UNKNOWN)

        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)

        adx, plus_di, minus_di = self._calc_adx_di(high, low, close)
        ema_slope = self._calc_ema_slope(close)
        atr_ratio = self._calc_atr_ratio(high, low, close)
        bb_pctile = self._calc_bb_pctile(close)

        # ── 4상태 분류 ──
        if adx > self.adx_trend:
            # 추세 존재 → 방향 판별
            if ema_slope > 0 and plus_di > minus_di:
                regime = Regime4.TREND_UP
            elif ema_slope < 0 and minus_di > plus_di:
                regime = Regime4.TREND_DOWN
            elif ema_slope > 0:
                regime = Regime4.TREND_UP
            else:
                regime = Regime4.TREND_DOWN
        else:
            # 비추세 → 변동성으로 구분
            if atr_ratio > self.atr_high:
                regime = Regime4.RANGE_HIGH
            else:
                regime = Regime4.RANGE_LOW

        return Regime4Snapshot(
            regime=regime,
            adx=adx,
            ema_slope=ema_slope,
            atr_ratio=atr_ratio,
            bb_width_pctile=bb_pctile,
        )

    def classify_series(self, df: pd.DataFrame) -> pd.Series:
        """전체 시계열에 대해 레짐 Series 반환.

        Returns:
            pd.Series with Regime4 values, index aligned to df.
        """
        n = len(df)
        regimes = pd.Series(Regime4.UNKNOWN, index=df.index, dtype=object)

        if n < self.lookback:
            return regimes

        # Pre-compute indicators for full series
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)

        adx_s, plus_di_s, minus_di_s = self._calc_adx_di_series(high, low, close)
        ema_slope_s = self._calc_ema_slope_series(close)
        atr_ratio_s = self._calc_atr_ratio_series(high, low, close)

        for i in range(self.lookback, n):
            adx_val = adx_s.iloc[i]
            slope_val = ema_slope_s.iloc[i]
            plus_val = plus_di_s.iloc[i]
            minus_val = minus_di_s.iloc[i]
            atr_val = atr_ratio_s.iloc[i]

            if pd.isna(adx_val) or pd.isna(slope_val):
                continue

            if adx_val > self.adx_trend:
                if slope_val > 0 and plus_val > minus_val:
                    regimes.iloc[i] = Regime4.TREND_UP
                elif slope_val < 0 and minus_val > plus_val:
                    regimes.iloc[i] = Regime4.TREND_DOWN
                elif slope_val > 0:
                    regimes.iloc[i] = Regime4.TREND_UP
                else:
                    regimes.iloc[i] = Regime4.TREND_DOWN
            else:
                if atr_val > self.atr_high:
                    regimes.iloc[i] = Regime4.RANGE_HIGH
                else:
                    regimes.iloc[i] = Regime4.RANGE_LOW

        return regimes

    def regime_ev_decomposition(
        self,
        df: pd.DataFrame,
        trade_results: pd.DataFrame,
    ) -> dict:
        """레짐별 EV 분해.

        Args:
            df: OHLCV DataFrame (4h)
            trade_results: DataFrame with columns:
                - entry_time: 진입 시각
                - pnl_eq: equity% 손익
                - side: BUY/SELL

        Returns:
            {regime: {n_trades, mean_ev, std_ev, win_rate, total_pnl}}
        """
        regimes = self.classify_series(df)
        results = {}

        for regime in Regime4:
            if regime == Regime4.UNKNOWN:
                continue

            mask = trade_results["entry_time"].apply(
                lambda t: regimes.asof(t) == regime
                if t in regimes.index or len(regimes.loc[:t]) > 0
                else False
            )
            subset = trade_results[mask]

            if len(subset) == 0:
                results[regime.value] = {
                    "n_trades": 0,
                    "mean_ev": 0.0,
                    "std_ev": 0.0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                }
                continue

            pnl = subset["pnl_eq"]
            results[regime.value] = {
                "n_trades": len(subset),
                "mean_ev": float(pnl.mean()),
                "std_ev": float(pnl.std()) if len(pnl) > 1 else 0.0,
                "win_rate": float((pnl > 0).mean()),
                "total_pnl": float(pnl.sum()),
            }

        return results

    # ── Internal: Full series computations ──

    def _calc_adx_di_series(
        self, high: pd.Series, low: pd.Series, close: pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        p = self.adx_period
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr = tr.rolling(p).mean()
        plus_di = 100 * (plus_dm.rolling(p).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(p).mean() / (atr + 1e-10))

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(p).mean()

        return adx, plus_di, minus_di

    def _calc_adx_di(
        self, high: pd.Series, low: pd.Series, close: pd.Series,
    ) -> tuple[float, float, float]:
        adx_s, plus_s, minus_s = self._calc_adx_di_series(high, low, close)
        return (
            float(adx_s.iloc[-1]) if not pd.isna(adx_s.iloc[-1]) else 0.0,
            float(plus_s.iloc[-1]) if not pd.isna(plus_s.iloc[-1]) else 0.0,
            float(minus_s.iloc[-1]) if not pd.isna(minus_s.iloc[-1]) else 0.0,
        )

    def _calc_ema_slope(self, close: pd.Series, window: int = 5) -> float:
        ema = close.ewm(span=self.ema_fast).mean()
        if len(ema) < window:
            return 0.0
        slope = (ema.iloc[-1] - ema.iloc[-window]) / (ema.iloc[-window] + 1e-10)
        return float(slope) if not pd.isna(slope) else 0.0

    def _calc_ema_slope_series(self, close: pd.Series, window: int = 5) -> pd.Series:
        ema = close.ewm(span=self.ema_fast).mean()
        slope = (ema - ema.shift(window)) / (ema.shift(window) + 1e-10)
        return slope

    def _calc_atr_ratio(
        self, high: pd.Series, low: pd.Series, close: pd.Series,
    ) -> float:
        s = self._calc_atr_ratio_series(high, low, close)
        val = s.iloc[-1]
        return float(val) if not pd.isna(val) else 1.0

    def _calc_atr_ratio_series(
        self, high: pd.Series, low: pd.Series, close: pd.Series,
    ) -> pd.Series:
        p = self.adx_period
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(p).mean()
        atr_ma = atr.rolling(p * 3).mean()
        return atr / (atr_ma + 1e-10)

    def _calc_bb_pctile(self, close: pd.Series, period: int = 20) -> float:
        ma = close.rolling(period).mean()
        std = close.rolling(period).std()
        bb_width = 2 * std / (ma + 1e-10) * 100
        pctile = bb_width.rank(pct=True).iloc[-1]
        return float(pctile) if not pd.isna(pctile) else 0.5
