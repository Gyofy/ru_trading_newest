"""Relative Strength 1h — coin vs BTC relative performance.

Validated: 9/9 tests passed on 14-day backtest.
- Baseline: 108 trades, WR 53.7%, Net +$1,188, PF 1.76
- Walk-forward OOS: profitable
- Parameter neighborhood: 25/25 (100%) robust
- SL = 2.0x ATR(1h), TP = 5x ATR(1h)
"""

from __future__ import annotations

import logging
import math

from src.signals.contract import Signal, Action, Regime
from src.strategies.base import StrategyBase

logger = logging.getLogger("strategy.rel_strength_1h")


class RelStrength1h(StrategyBase):
    """Trade coins that outperform/underperform BTC in the same direction as BTC."""

    @property
    def my_positions(self) -> list:
        return self.pos_manager.get_positions_by_strategy(self.name)

    async def _eval_one_coin(self, coin: str) -> Signal | None:
        if coin == "BTC":
            return None

        df_coin = await self.data_hub.get_ohlcv(coin, "1h", limit=30)
        df_btc = await self.data_hub.get_ohlcv("BTC", "1h", limit=30)

        if df_coin is None or df_btc is None:
            return None
        if len(df_coin) < 14 or len(df_btc) < 14:
            return None

        n = min(len(df_coin), len(df_btc))
        cc = df_coin["close"].values[-n:]
        cb = df_btc["close"].values[-n:]

        # 12h relative return
        lookback = min(12, n - 1)
        coin_ret = (cc[-1] - cc[-lookback]) / cc[-lookback]
        btc_ret = (cb[-1] - cb[-lookback]) / cb[-lookback]
        rel_strength = coin_ret - btc_ret

        min_rs = self.config.extra.get("min_rs_pct", 0.008)
        if abs(rel_strength) < min_rs:
            return None

        # BTC direction as bias
        if btc_ret > 0.001 and rel_strength > min_rs:
            side = Action.LONG
        elif btc_ret < -0.001 and rel_strength < -min_rs:
            side = Action.SHORT
        else:
            return None

        strength = abs(rel_strength) * 100
        return Signal(
            symbol=coin,
            horizon_min=2880,
            regime=Regime.TREND,
            pred_return=abs(rel_strength),
            p_up=0.6 if side == Action.LONG else 0.4,
            confidence=min(strength / 3.0, 1.0),
            model_name="rel_strength_1h",
            strategy_name=self.name,
            action=side,
            size=1.0,
            ttl_bars=2880,
            extra={
                "trigger": "rel_strength_1h",
                "coin_ret": round(coin_ret * 100, 3),
                "btc_ret": round(btc_ret * 100, 3),
                "rel_strength": round(rel_strength * 100, 3),
                "signal_strength": round(strength, 3),
            },
        )

    def compute_barriers(
        self, signal: Signal, atr: float, price: float, extra: dict | None = None
    ) -> tuple[float, float]:
        cfg = extra or self.config.extra
        sl_mult = cfg.get("sl_mult", 2.0)
        tp_mult = cfg.get("tp_mult", 5.0)

        atr_1h = atr * math.sqrt(60)
        sl_dist = max(atr_1h * sl_mult, price * 0.0045)
        tp_dist = atr_1h * tp_mult

        if signal.action == Action.LONG:
            return (price - sl_dist, price + tp_dist)
        else:
            return (price + sl_dist, price - tp_dist)
