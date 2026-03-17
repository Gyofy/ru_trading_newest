"""Regime Detector — 규칙 기반 시장 상태 판별.

TREND:    EMA slope aligned + ADX > 25 + BB expanding
RANGE:    ADX < 18 + BB contracting + price oscillating
VOLATILE: ATR surge + volume spike + rapid regime switches

주의: 이 모듈은 **strategic tier (1h bars)** 데이터를 사용합니다.
  파라미터 기본값은 config/settings.yaml → timeframes.strategic.regime 참조.

Usage:
    detector = RegimeDetector()            # config defaults
    detector = RegimeDetector.from_config() # 명시적 config 로드
    regime = detector.detect(df_1h)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from src.signals.contract import Regime


@dataclass
class RegimeSnapshot:
    """특정 시점의 regime 판별 결과."""
    regime: Regime
    adx: float = 0.0
    bb_width: float = 0.0
    bb_width_pctile: float = 0.0   # BB width의 rolling percentile (0~1)
    atr_ratio: float = 0.0        # ATR / ATR_ma (1.0 = 평균)
    ema_aligned: bool = False      # EMA 20/50 방향 일치 여부
    vol_surge: float = 0.0        # volume / volume_ma
    score: dict = None             # 각 regime별 점수

    def __repr__(self):
        return (f"RegimeSnapshot({self.regime.value} "
                f"ADX={self.adx:.1f} BBw%={self.bb_width_pctile:.0%} "
                f"ATR_r={self.atr_ratio:.2f} vol={self.vol_surge:.1f}x)")


class RegimeDetector:
    """규칙 기반 regime 판별.

    기본값은 config/settings.yaml → timeframes.strategic.regime에서 가져옴.
    직접 파라미터를 넘기면 config보다 우선.
    """

    def __init__(
        self,
        adx_trend: float = 25,
        adx_range: float = 18,
        bb_expand_pctile: float = 0.7,
        bb_contract_pctile: float = 0.3,
        atr_surge_ratio: float = 1.5,
        vol_surge_ratio: float = 2.0,
        ema_fast: int = 20,
        ema_slow: int = 50,
        lookback: int = 100,
    ):
        self.adx_trend = adx_trend
        self.adx_range = adx_range
        self.bb_expand_pctile = bb_expand_pctile
        self.bb_contract_pctile = bb_contract_pctile
        self.atr_surge_ratio = atr_surge_ratio
        self.vol_surge_ratio = vol_surge_ratio
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.lookback = lookback

    @classmethod
    def from_config(cls) -> "RegimeDetector":
        """config/settings.yaml에서 strategic tier 파라미터를 로드하여 생성."""
        from src.utils.config import get_strategic
        cfg = get_strategic().get("regime", {})
        return cls(
            ema_fast=cfg.get("ema_fast", 20),
            ema_slow=cfg.get("ema_slow", 50),
            lookback=cfg.get("lookback", 100),
            adx_trend=cfg.get("adx_trend", 25),
            adx_range=cfg.get("adx_range", 18),
        )

    def detect(self, df: pd.DataFrame) -> RegimeSnapshot:
        """DataFrame의 마지막 시점 regime 판별."""
        if len(df) < self.lookback:
            return RegimeSnapshot(regime=Regime.UNKNOWN)

        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        volume = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(0, index=df.index)

        # ── 지표 계산 ──
        adx = self._calc_adx(high, low, close)
        bb_width, bb_pctile = self._calc_bb(close)
        atr_ratio = self._calc_atr_ratio(high, low, close)
        ema_aligned = self._check_ema_alignment(close)
        vol_surge = self._calc_vol_surge(volume)

        # ── 점수 산정 ──
        trend_score = 0.0
        range_score = 0.0
        volatile_score = 0.0

        # TREND signals
        if adx > self.adx_trend:
            trend_score += 2.0
        elif adx > 20:
            trend_score += 1.0
        if ema_aligned:
            trend_score += 1.5
        if bb_pctile > self.bb_expand_pctile:
            trend_score += 1.0

        # RANGE signals
        if adx < self.adx_range:
            range_score += 2.0
        elif adx < 22:
            range_score += 1.0
        if bb_pctile < self.bb_contract_pctile:
            range_score += 1.5
        if not ema_aligned:
            range_score += 0.5

        # VOLATILE signals
        if atr_ratio > self.atr_surge_ratio:
            volatile_score += 2.0
        elif atr_ratio > 1.2:
            volatile_score += 0.5
        if vol_surge > self.vol_surge_ratio:
            volatile_score += 2.0
        elif vol_surge > 1.5:
            volatile_score += 0.5
        if bb_pctile > 0.85:
            volatile_score += 1.0

        scores = {
            "TREND": trend_score,
            "RANGE": range_score,
            "VOLATILE": volatile_score,
        }

        # 최고 점수 regime 선택
        max_regime = max(scores, key=scores.get)
        if scores[max_regime] < 1.5:
            regime = Regime.UNKNOWN
        else:
            regime = Regime(max_regime)

        return RegimeSnapshot(
            regime=regime,
            adx=adx,
            bb_width=bb_width,
            bb_width_pctile=bb_pctile,
            atr_ratio=atr_ratio,
            ema_aligned=ema_aligned,
            vol_surge=vol_surge,
            score=scores,
        )

    def detect_series(
        self, df: pd.DataFrame, step: int = 1,
    ) -> list[tuple[pd.Timestamp, RegimeSnapshot]]:
        """시계열 전체에 대해 regime 판별 (step bars마다)."""
        results = []
        for i in range(self.lookback, len(df), step):
            window = df.iloc[:i+1]
            snapshot = self.detect(window)
            results.append((df.index[i], snapshot))
        return results

    # ── 지표 계산 헬퍼 ──

    def _calc_adx(self, high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 14) -> float:
        """ADX 계산."""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()

        val = adx.iloc[-1]
        return float(val) if not np.isnan(val) else 0.0

    def _calc_bb(self, close: pd.Series, period: int = 20) -> tuple[float, float]:
        """Bollinger Band width + percentile."""
        ma = close.rolling(period).mean()
        std = close.rolling(period).std()
        bb_width = (2 * std / ma * 100)  # %
        current = bb_width.iloc[-1]
        pctile = (bb_width.rank(pct=True)).iloc[-1]
        return (float(current) if not np.isnan(current) else 0.0,
                float(pctile) if not np.isnan(pctile) else 0.5)

    def _calc_atr_ratio(self, high: pd.Series, low: pd.Series,
                        close: pd.Series, period: int = 14) -> float:
        """ATR / ATR_MA 비율."""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        atr_ma = atr.rolling(period * 3).mean()

        ratio = atr.iloc[-1] / (atr_ma.iloc[-1] + 1e-10)
        return float(ratio) if not np.isnan(ratio) else 1.0

    def _check_ema_alignment(self, close: pd.Series) -> bool:
        """EMA fast/slow 방향 일치 여부."""
        ema_f = close.ewm(span=self.ema_fast).mean()
        ema_s = close.ewm(span=self.ema_slow).mean()

        # 최근 5 bars의 EMA 기울기
        f_slope = ema_f.iloc[-1] - ema_f.iloc[-5] if len(ema_f) >= 5 else 0
        s_slope = ema_s.iloc[-1] - ema_s.iloc[-5] if len(ema_s) >= 5 else 0

        return (f_slope > 0 and s_slope > 0) or (f_slope < 0 and s_slope < 0)

    def _calc_vol_surge(self, volume: pd.Series, period: int = 20) -> float:
        """현재 거래량 / 평균 거래량."""
        if volume.sum() == 0:
            return 1.0
        ma = volume.rolling(period).mean()
        ratio = volume.iloc[-1] / (ma.iloc[-1] + 1e-10)
        return float(ratio) if not np.isnan(ratio) else 1.0
