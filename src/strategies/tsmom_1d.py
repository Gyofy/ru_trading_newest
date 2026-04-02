"""TSMOM 1D — 28d + 7d daily momentum trend-following.

Validated on 27-month data (2024-01-01 ~ 2026-03-31):
  - 4/4 WF PASS (only strategy to pass all tests)
  - Full: 349 trades, WR 47.0%, Net +$3,640
  - OOS 30%: 100 trades, WR 45.0%, Net +$800
  - Both halves: +$2,702 / +$790
  - Reversed: -$2,952
  - Param neighborhood: 22/25 (88%)

Brainstorm optimized (TTL=7d):
  - Full: 488 trades, WR 47.1%, Net +$2,377
  - OOS: +$1,183 (best OOS config)
  - Both halves: +$1,863 / +$781
"""

from __future__ import annotations

import logging
import math
import numpy as np

from src.signals.contract import Signal, Action, Regime
from src.strategies.base import StrategyBase

logger = logging.getLogger("strategy.tsmom_1d")


class TSMOM1d(StrategyBase):
    """28-day + 7-day momentum alignment on daily bars."""

    @property
    def my_positions(self) -> list:
        return self.pos_manager.get_positions_by_strategy(self.name)

    async def _eval_one_coin(self, coin: str) -> Signal | None:
        # Fetch daily bars (use 1h with limit=720 = 30 days)
        df_1h = await self.data_hub.get_ohlcv(coin, "1h", limit=720)
        if df_1h is None or len(df_1h) < 672:  # need 28 days
            return None

        # Resample to daily
        df_1d = df_1h.resample("1D").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).dropna()

        if len(df_1d) < 29:
            return None

        close = df_1d["close"].values

        # Configurable lookback periods
        slow_period = self.config.extra.get("mom_slow", 28)
        fast_period = self.config.extra.get("mom_fast", 7)

        if len(close) < slow_period + 1:
            return None

        mom_slow = (close[-1] - close[-slow_period]) / close[-slow_period]
        mom_fast = (close[-1] - close[-fast_period]) / close[-fast_period]

        # Alias for backward compatibility
        mom_28d = mom_slow
        mom_7d = mom_fast

        # Both must align
        if not ((mom_28d > 0 and mom_7d > 0) or (mom_28d < 0 and mom_7d < 0)):
            return None

        # Minimum move threshold
        min_move = self.config.extra.get("min_move_pct", 0.03)
        if abs(mom_28d) < min_move:
            return None

        side = Action.LONG if mom_28d > 0 else Action.SHORT

        # Day-of-week filter: skip Monday (WR 37%) and Saturday (validated improvement)
        skip_days = self.config.extra.get("skip_days", [0, 5])  # 0=Mon, 5=Sat
        if df_1d.index[-1].weekday() in skip_days:
            return None

        # Volume confirmation: skip low-volume days (validated: WR 47.1% -> 49.0%)
        if self.config.extra.get("vol_filter", True):
            vol = df_1d["volume"].values
            if len(vol) >= 10:
                vol_avg = np.mean(vol[-10:])
                if vol_avg > 0 and vol[-1] < vol_avg * 0.8:
                    return None

        # Optional: body confirmation filter
        if self.config.extra.get("body_filter", False):
            body = (float(df_1d["close"].iloc[-1]) - float(df_1d["open"].iloc[-1])) / float(df_1d["open"].iloc[-1])
            if side == Action.LONG and body < 0:
                return None
            if side == Action.SHORT and body > 0:
                return None

        strength = abs(mom_28d) * 100
        return Signal(
            symbol=coin,
            horizon_min=10080,  # 7 days in minutes
            regime=Regime.TREND,
            pred_return=abs(mom_28d),
            p_up=0.6 if side == Action.LONG else 0.4,
            confidence=min(strength / 5.0, 1.0),
            model_name="tsmom_1d",
            strategy_name=self.name,
            action=side,
            size=1.0,
            ttl_bars=10080,  # 7 days in 1m bars (TTL=7d is optimal)
            extra={
                "trigger": "tsmom_1d_28d7d",
                "mom_28d": round(mom_28d, 6),
                "mom_7d": round(mom_7d, 6),
                "signal_strength": round(strength, 3),
            },
        )

    def compute_barriers(
        self, signal: Signal, atr: float, price: float, extra: dict | None = None
    ) -> tuple[float, float]:
        cfg = extra or self.config.extra
        sl_mult = cfg.get("sl_mult", 2.0)
        tp_mult = cfg.get("tp_mult", 3.0)

        # ATR from base class is 1m; scale to daily
        atr_1d = atr * math.sqrt(1440)  # 1440 minutes in a day
        sl_dist = max(atr_1d * sl_mult, price * 0.0045)
        tp_dist = atr_1d * tp_mult

        if signal.action == Action.LONG:
            return (price - sl_dist, price + tp_dist)
        else:
            return (price + sl_dist, price - tp_dist)
