"""Strategy: OI Divergence — 가격-OI 괴리에서 방향 예측.

가설: 가격 상승 + OI 감소 = 숏 커버링 (약세, 되돌림 예상) → SHORT
     가격 하락 + OI 감소 = 롱 청산 (반등 예상) → LONG
     가격 상승 + OI 증가 = 신규 롱 유입 (강세 확인) → LONG
     가격 하락 + OI 증가 = 신규 숏 유입 (약세 확인) → SHORT
수수료 edge: OI divergence는 구조적 불균형을 포착 → 방향성 edge가 수수료보다 큼.
             5분봉 기반 장기 보유 → 수수료 비중 낮음.
"""

from __future__ import annotations

import logging

from src.signals.contract import Signal, Action, Regime
from src.strategies.base import StrategyBase

logger = logging.getLogger("strategy.oi_divergence")


class OIDivergence(StrategyBase):
    """Enter on price-OI divergence (structural imbalance)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._oi_history: dict[str, list[float]] = {}  # coin -> recent OI values

    async def _eval_one_coin(self, coin: str) -> Signal | None:
        cfg = self.config.extra
        price_lookback = cfg.get("price_lookback", 12)  # 12 bars of 5m = 1 hour
        min_price_move = cfg.get("min_price_move_pct", 0.003)  # 0.3% minimum

        # Get OI
        oi = await self.data_hub.get_open_interest(coin)
        if oi is None or oi <= 0:
            return None

        # Track OI history
        if coin not in self._oi_history:
            self._oi_history[coin] = []
        self._oi_history[coin].append(oi)
        # Keep last 20 readings
        self._oi_history[coin] = self._oi_history[coin][-20:]

        if len(self._oi_history[coin]) < 3:
            return None  # Need at least 3 readings

        # OI trend (recent 3 readings)
        oi_recent = self._oi_history[coin][-3:]
        oi_change_pct = (oi_recent[-1] - oi_recent[0]) / oi_recent[0]

        # Price trend
        df = await self.data_hub.get_ohlcv(coin, "5m", limit=price_lookback + 5)
        if df is None or len(df) < price_lookback:
            return None

        price_now = float(df["close"].iloc[-1])
        price_prev = float(df["close"].iloc[-price_lookback])
        price_change_pct = (price_now - price_prev) / price_prev

        if abs(price_change_pct) < min_price_move:
            return None

        oi_threshold = cfg.get("oi_change_threshold", 0.005)  # 0.5% OI change
        if abs(oi_change_pct) < oi_threshold:
            return None

        # Divergence logic
        direction = None
        trigger = ""

        if price_change_pct > 0 and oi_change_pct < -oi_threshold:
            # Price up + OI down = short covering → bearish divergence
            direction = "SHORT"
            trigger = "price_up_oi_down"
        elif price_change_pct < 0 and oi_change_pct < -oi_threshold:
            # Price down + OI down = long liquidation → expect bounce
            direction = "LONG"
            trigger = "price_down_oi_down"
        elif price_change_pct > 0 and oi_change_pct > oi_threshold:
            # Price up + OI up = new longs → trend continuation
            direction = "LONG"
            trigger = "price_up_oi_up"
        elif price_change_pct < 0 and oi_change_pct > oi_threshold:
            # Price down + OI up = new shorts → trend continuation
            direction = "SHORT"
            trigger = "price_down_oi_up"

        if direction is None:
            return None

        action = Action.LONG if direction == "LONG" else Action.SHORT

        self._log.info(
            f"[{coin}] OI DIVERGENCE → {direction} | "
            f"price={price_change_pct:.3%} oi={oi_change_pct:.3%} trigger={trigger}"
        )
        return Signal(
            symbol=coin, horizon_min=120, regime=Regime.VOLATILE,
            pred_return=abs(price_change_pct), p_up=0.7 if direction == "LONG" else 0.3,
            confidence=min(abs(oi_change_pct) / 0.02, 1.0),
            model_name="oi_divergence", strategy_name=self.name,
            action=action, size=1.0, ttl_bars=120,
            extra={"trigger": trigger, "price_change_pct": price_change_pct,
                   "oi_change_pct": oi_change_pct, "oi_value": oi},
        )

    def compute_barriers(self, signal, atr, price, extra=None):
        cfg = extra if extra is not None else self.config.extra
        sl_mult = cfg.get("sl_atr_mult", 2.5)
        tp_rr = cfg.get("tp_rr", 2.5)

        atr_5m = atr * (5 ** 0.5)
        sl_dist = max(atr_5m * sl_mult, price * 0.005)  # 0.5% floor (5m strategy)
        tp_dist = sl_dist * tp_rr

        if signal.action == Action.SHORT:
            return (price + sl_dist, price - tp_dist)
        return (price - sl_dist, price + tp_dist)
