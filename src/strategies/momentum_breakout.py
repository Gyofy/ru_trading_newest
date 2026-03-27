"""Strategy C: Momentum Breakout — 볼륨 확인 돌파 추세 추종.

Trigger: 1h 볼륨 > 3× 평균 + 가격이 24h range 돌파.
Direction: Breakout 방향 추종.
Leverage: CVD가 방향 확인 시 3x, 미확인 2x.
SL: Breakout 시작점.
TP: Trailing stop (2×ATR trail).
Cycle: 5분.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from src.signals.contract import Signal, Action, Regime
from src.strategies.base import StrategyBase

logger = logging.getLogger("strategy.momentum")


class MomentumBreakout(StrategyBase):
    """Trend-following entries on confirmed volume breakouts."""

    async def _eval_one_coin(self, coin: str) -> "Signal | None":
        """Per-coin evaluation — called in parallel by base.evaluate()."""
        cfg = self.config.extra
        vol_mult = cfg.get("volume_mult_threshold", 3.0)
        range_hours = cfg.get("range_lookback_hours", 24)

        result = await self._check_breakout(coin, vol_mult, range_hours)
        if result is None:
            return None

        direction, vol_ratio, breakout_level, cvd_confirmed, range_width_pct, breakout_pct, cvd_1h_delta = result
        lev_base = cfg.get("leverage_base", 2)
        lev_confirmed = cfg.get("leverage_confirmed", 3)
        effective_lev = lev_confirmed if cvd_confirmed else lev_base

        self._log.info(
            f"[{coin}] BREAKOUT → {direction} | "
            f"vol={vol_ratio:.1f}x cvd_conf={'Y' if cvd_confirmed else 'N'} "
            f"lev={effective_lev}x"
        )
        return Signal(
            symbol=coin,
            horizon_min=240,
            regime=Regime.TREND,
            pred_return=vol_ratio * 0.3,
            p_up=0.75 if direction == "LONG" else 0.25,
            confidence=min(vol_ratio / 5.0, 1.0),
            model_name="momentum_breakout",
            strategy_name=self.name,
            action=Action.LONG if direction == "LONG" else Action.SHORT,
            size=1.0,
            ttl_bars=240,
            extra={
                "vol_ratio": vol_ratio,
                "breakout_level": breakout_level,
                "cvd_confirmed": cvd_confirmed,
                "effective_leverage": effective_lev,
                # Algo-dev fields
                "strength": float(vol_ratio),
                "trigger": "vol_breakout",
                "cvd_value": float(cvd_1h_delta),
                "range_width_pct": float(range_width_pct),
                "breakout_pct": float(breakout_pct),
            },
        )

    async def _check_breakout(
        self,
        coin: str,
        vol_mult: float,
        range_hours: int,
    ) -> tuple[str, float, float, bool] | None:
        """Check for volume-confirmed breakout.

        Returns (direction, vol_ratio, breakout_level, cvd_confirmed) or None.
        """
        # Get 1m data for precise volume and CVD analysis
        bars_needed = range_hours * 60 + 60  # 24h + buffer in 1m bars
        df_1m = await self.data_hub.get_ohlcv(coin, "1m", limit=min(bars_needed, 1500))
        if df_1m is None or len(df_1m) < range_hours * 60:
            return None

        close = df_1m["close"].values
        high = df_1m["high"].values
        low = df_1m["low"].values
        vol = df_1m["volume"].values

        # 1h volume aggregation (last 60 bars vs avg of prior hours)
        current_1h_vol = np.sum(vol[-60:])
        prior_hours = range_hours - 1
        prior_vols = []
        for i in range(prior_hours):
            start = -(i + 2) * 60
            end = -(i + 1) * 60
            if abs(start) <= len(vol):
                prior_vols.append(np.sum(vol[start:end]))

        if not prior_vols:
            return None

        avg_1h_vol = np.mean(prior_vols)
        vol_ratio = current_1h_vol / avg_1h_vol if avg_1h_vol > 0 else 0

        if vol_ratio < vol_mult:
            return None

        # 24h range
        lookback = range_hours * 60
        range_high = np.max(high[-lookback:-60])  # exclude current hour
        range_low = np.min(low[-lookback:-60])

        current_close = close[-1]

        # Check breakout
        direction = None
        breakout_level = 0.0

        if current_close > range_high:
            direction = "LONG"
            breakout_level = range_high
        elif current_close < range_low:
            direction = "SHORT"
            breakout_level = range_low
        else:
            return None

        # CVD confirmation
        cvd = self.data_hub.compute_cvd(df_1m)
        cvd_1h_delta = 0.0
        if len(cvd) < 60:
            cvd_confirmed = False
        else:
            cvd_1h_delta = float(cvd.iloc[-1] - cvd.iloc[-60])
            cvd_confirmed = (
                (direction == "LONG" and cvd_1h_delta > 0)
                or (direction == "SHORT" and cvd_1h_delta < 0)
            )

        range_width_pct = (range_high - range_low) / range_low if range_low > 0 else 0.0
        breakout_pct = abs(current_close - breakout_level) / breakout_level if breakout_level > 0 else 0.0
        return (direction, float(vol_ratio), float(breakout_level), cvd_confirmed,
                float(range_width_pct), float(breakout_pct), float(cvd_1h_delta))

    def compute_barriers(
        self, signal: Signal, atr: float, price: float, extra: dict | None = None
    ) -> tuple[float, float]:
        """SL = breakout level, TP = 0 (trailing stop)."""
        breakout_level = signal.extra.get("breakout_level", 0)

        # SL at breakout origin, but at least 1.5×ATR away
        if signal.action == Action.LONG:
            sl_from_breakout = breakout_level
            sl_from_atr = price - atr * 1.5
            sl_price = max(sl_from_breakout, sl_from_atr)  # tighter of the two
            # But not too close
            if price - sl_price < atr:
                sl_price = price - atr
        else:
            sl_from_breakout = breakout_level
            sl_from_atr = price + atr * 1.5
            sl_price = min(sl_from_breakout, sl_from_atr)
            if sl_price - price < atr:
                sl_price = price + atr

        # tp_price=0 → trailing stop
        return (sl_price, 0.0)
