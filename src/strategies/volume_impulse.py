"""Strategy: Volume Impulse — 거래량 폭증 + 방향 확인 후 추세 추종.

가설: 평균 대비 5배 이상 거래량 폭발 + 가격 방향이 일치하면,
      큰 참여자(기관/고래)의 방향성 움직임 → 추세 연장.
수수료 edge: 추세 추종은 R:R이 높음 (1:3+). WR 25~30%로도 수수료 커버 가능.
             SL은 impulse bar의 반대쪽에 배치 → 자연스러운 SL 거리.
"""

from __future__ import annotations

import logging
import numpy as np

from src.signals.contract import Signal, Action, Regime
from src.strategies.base import StrategyBase

logger = logging.getLogger("strategy.volume_impulse")


class VolumeImpulse(StrategyBase):
    """Trend-following on extreme volume bars with directional confirmation."""

    async def _eval_one_coin(self, coin: str) -> Signal | None:
        cfg = self.config.extra
        vol_mult = cfg.get("volume_mult", 5.0)
        min_body_pct = cfg.get("min_body_pct", 0.003)  # 0.3% minimum body size
        lookback = cfg.get("lookback", 50)

        df = await self.data_hub.get_ohlcv(coin, "1m", limit=lookback + 10)
        if df is None or len(df) < lookback:
            return None

        vol = df["volume"].values
        close = df["close"].values
        opn = df["open"].values
        high = df["high"].values
        low = df["low"].values

        # Current bar volume vs rolling average
        vol_avg = np.mean(vol[-lookback:-1])
        vol_current = vol[-1]
        if vol_avg <= 0 or vol_current < vol_avg * vol_mult:
            return None

        # Body size (directional strength)
        body_pct = (close[-1] - opn[-1]) / opn[-1] if opn[-1] > 0 else 0
        if abs(body_pct) < min_body_pct:
            return None  # Doji/indecisive bar despite volume

        # Direction: follow the impulse
        if body_pct > 0:
            direction = "LONG"
            action = Action.LONG
            # SL at low of impulse bar
            sl_anchor = float(low[-1])
        else:
            direction = "SHORT"
            action = Action.SHORT
            # SL at high of impulse bar
            sl_anchor = float(high[-1])

        vol_ratio = vol_current / vol_avg

        # CVD direction check (A/B tag — not a filter, just logged for comparison)
        cvd_aligned = False
        try:
            cvd = self.data_hub.compute_cvd(df)
            if len(cvd) >= 2:
                cvd_delta = float(cvd.iloc[-1] - cvd.iloc[-2])
                # CVD increasing = buying pressure, decreasing = selling pressure
                cvd_aligned = (body_pct > 0 and cvd_delta > 0) or (body_pct < 0 and cvd_delta < 0)
        except Exception:
            pass

        self._log.info(
            f"[{coin}] VOLUME IMPULSE → {direction} | "
            f"vol_ratio={vol_ratio:.1f}x body={body_pct:.3%} cvd_aligned={cvd_aligned}"
        )
        return Signal(
            symbol=coin, horizon_min=60, regime=Regime.TREND,
            pred_return=abs(body_pct), p_up=0.7 if direction == "LONG" else 0.3,
            confidence=min(vol_ratio / 10.0, 1.0),
            model_name="volume_impulse", strategy_name=self.name,
            action=action, size=1.0, ttl_bars=120,
            extra={"trigger": "volume_impulse", "vol_ratio": vol_ratio,
                   "body_pct": body_pct, "sl_anchor": sl_anchor,
                   "cvd_aligned": cvd_aligned},
        )

    def compute_barriers(self, signal, atr, price, extra=None):
        cfg = extra if extra is not None else self.config.extra
        tp_rr = cfg.get("tp_rr", 3.0)  # Trend-following: high R:R

        # SL = impulse bar extremity (natural stop)
        sl_anchor = signal.extra.get("sl_anchor", 0) if signal.extra else 0
        if sl_anchor > 0:
            sl_dist = abs(price - sl_anchor)
        else:
            atr_15m = atr * (15 ** 0.5)
            sl_dist = atr_15m * 2.0

        sl_dist = max(sl_dist, price * 0.004)  # 0.40% floor
        tp_dist = sl_dist * tp_rr

        if signal.action == Action.SHORT:
            return (price + sl_dist, price - tp_dist)
        return (price - sl_dist, price + tp_dist)
