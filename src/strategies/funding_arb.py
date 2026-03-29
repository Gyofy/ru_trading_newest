"""Strategy: MTF Momentum — 다중 시간프레임 추세 정렬 시 진입.

가설: 1분봉 방향 + 15분봉 방향 + 1시간봉 방향이 모두 일치하면,
      다중 시간프레임 참여자들의 합의 → 강한 추세 연장.
      Multi-Timeframe(MTF) alignment는 가장 오래 검증된 추세 추종 edge.
수수료 edge:
  - R:R 3:1 (추세 추종 고R:R). WR 30%면 break-even.
  - 추세 방향 정렬이 맞으면 WR 35~45% 기대.
  - SL = 15분봉 ATR 기반(넓음) → 수수료 비중 낮음.
  - 0.15% 왕복 / 0.6% SL = 수수료가 SL의 25%만 차지.

(funding_arb.py를 재활용 — 파일명 유지, 클래스 교체)
"""

from __future__ import annotations

import logging
import numpy as np

from src.signals.contract import Signal, Action, Regime
from src.strategies.base import StrategyBase

logger = logging.getLogger("strategy.mtf_momentum")


class FundingArb(StrategyBase):
    """Multi-Timeframe Momentum: enter when 1m + 15m + 1h trends align.

    NOTE: Class name kept as FundingArb for backward compatibility with
    STRATEGY_MAP import. strategy_name from config determines the actual tag.
    """

    async def _eval_one_coin(self, coin: str) -> Signal | None:
        cfg = self.config.extra
        fast_period = cfg.get("fast_ema", 8)
        slow_period = cfg.get("slow_ema", 21)
        min_body_pct = cfg.get("min_body_pct", 0.001)  # 0.1% minimum 1m bar body

        # ── 3 Timeframes ──
        df_1m = await self.data_hub.get_ohlcv(coin, "1m", limit=50)
        df_15m = await self.data_hub.get_ohlcv(coin, "15m", limit=30)
        df_1h = await self.data_hub.get_ohlcv(coin, "1h", limit=20)

        if df_1m is None or df_15m is None or df_1h is None:
            return None
        if len(df_1m) < 30 or len(df_15m) < 22 or len(df_1h) < 10:
            return None

        # Direction per timeframe: EMA crossover
        dir_1m = self._ema_direction(df_1m, fast_period, slow_period)
        dir_15m = self._ema_direction(df_15m, fast_period, slow_period)
        dir_1h = self._ema_direction(df_1h, fast_period, slow_period)

        if dir_1m == 0 or dir_15m == 0 or dir_1h == 0:
            return None

        # All three must agree
        if not (dir_1m == dir_15m == dir_1h):
            return None

        # Current 1m bar body confirmation
        body_pct = (float(df_1m["close"].iloc[-1]) - float(df_1m["open"].iloc[-1])) / float(df_1m["open"].iloc[-1])
        if dir_1m == 1 and body_pct < min_body_pct:
            return None  # Bullish alignment but bearish/neutral current bar
        if dir_1m == -1 and body_pct > -min_body_pct:
            return None  # Bearish alignment but bullish/neutral current bar

        direction = "LONG" if dir_1m == 1 else "SHORT"
        action = Action.LONG if dir_1m == 1 else Action.SHORT

        # Strength = how far above/below EMA on each TF
        strength = abs(body_pct) * 100

        self._log.info(
            f"[{coin}] MTF MOMENTUM → {direction} | "
            f"1m={dir_1m} 15m={dir_15m} 1h={dir_1h} body={body_pct:.3%}"
        )
        return Signal(
            symbol=coin, horizon_min=120, regime=Regime.TREND,
            pred_return=abs(body_pct), p_up=0.7 if direction == "LONG" else 0.3,
            confidence=min(strength / 0.5, 1.0),
            model_name="mtf_momentum", strategy_name=self.name,
            action=action, size=1.0, ttl_bars=120,
            extra={"trigger": "mtf_alignment", "dir_1m": dir_1m,
                   "dir_15m": dir_15m, "dir_1h": dir_1h, "body_pct": body_pct},
        )

    @staticmethod
    def _ema_direction(df, fast: int, slow: int) -> int:
        """Return +1 (bullish), -1 (bearish), 0 (indeterminate)."""
        close = df["close"].values.astype(float)
        if len(close) < slow + 1:
            return 0

        ema_fast = _ema(close, fast)
        ema_slow = _ema(close, slow)

        if ema_fast[-1] > ema_slow[-1] and ema_fast[-2] > ema_slow[-2]:
            return 1
        if ema_fast[-1] < ema_slow[-1] and ema_fast[-2] < ema_slow[-2]:
            return -1
        return 0

    def compute_barriers(self, signal, atr, price, extra=None):
        cfg = extra if extra is not None else self.config.extra
        sl_mult = cfg.get("sl_atr_mult", 2.0)
        tp_rr = cfg.get("tp_rr", 3.0)  # 추세추종: 높은 R:R

        atr_15m = atr * (15 ** 0.5)
        sl_dist = max(atr_15m * sl_mult, price * 0.004)
        tp_dist = sl_dist * tp_rr

        if signal.action == Action.SHORT:
            return (price + sl_dist, price - tp_dist)
        return (price - sl_dist, price + tp_dist)


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Simple EMA calculation."""
    alpha = 2.0 / (period + 1)
    ema = np.empty_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema
