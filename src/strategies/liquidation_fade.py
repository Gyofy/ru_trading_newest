"""Strategy B: Liquidation Fade — 청산 캐스케이드 역행.

Trigger: OI 급감 (>2σ) + Taker Volume 스파이크 (>2x avg) = 강제 청산.
Direction: 청산 반대 (롱 청산 캐스케이드 → LONG 진입).
SL: Recent swing high/low.
TP: VWAP 복귀 (mean reversion).
Cycle: 5분.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from src.signals.contract import Signal, Action, Regime
from src.strategies.base import StrategyBase

logger = logging.getLogger("strategy.liq_fade")


class LiquidationFade(StrategyBase):
    """Fade liquidation cascades — enter counter to forced exits."""

    async def _eval_one_coin(self, coin: str) -> "Signal | None":
        """Per-coin evaluation — called in parallel by base.evaluate()."""
        cfg = self.config.extra
        oi_sigma = cfg.get("oi_sigma_threshold", 2.0)
        taker_mult = cfg.get("taker_spike_mult", 2.0)
        swing_lookback = cfg.get("swing_lookback", 24)

        cascade = await self._detect_cascade(coin, oi_sigma, taker_mult)
        if cascade is None:
            return None

        direction, oi_drop, taker_ratio, recent_return = cascade

        df = await self.data_hub.get_ohlcv(coin, "5m", limit=swing_lookback + 10)
        if df is None or len(df) < swing_lookback:
            return None

        self._log.info(
            f"[{coin}] LIQ CASCADE → {direction} | "
            f"OI_drop={oi_drop:.2f}σ taker={taker_ratio:.1f}x move={recent_return:.3%}"
        )
        return Signal(
            symbol=coin,
            horizon_min=120,
            regime=Regime.VOLATILE,
            pred_return=abs(oi_drop) * 0.5,
            p_up=0.7 if direction == "LONG" else 0.3,
            confidence=min(taker_ratio / 4.0, 1.0),
            model_name="liquidation_fade",
            strategy_name=self.name,
            action=Action.LONG if direction == "LONG" else Action.SHORT,
            size=1.0,
            ttl_bars=120,
            extra={
                "oi_drop_sigma": oi_drop,
                "taker_ratio": taker_ratio,
                "recent_return_pct": float(recent_return),
                "strength": float(oi_drop),
                "trigger": "liq_cascade",
                # Algo-dev fields
                "cvd_value": 0.0,       # liq_fade doesn't use CVD — kept for schema consistency
                "ofi_value": float(taker_ratio),  # volume spike serves as OFI proxy
            },
        )

    async def _detect_cascade(
        self,
        coin: str,
        oi_sigma: float,
        taker_mult: float,
    ) -> tuple[str, float, float, float] | None:
        """Detect liquidation cascade via OI drop + taker volume spike.

        Returns (direction, oi_drop_sigma, taker_ratio) or None.
        """
        df = await self.data_hub.get_ohlcv(coin, "5m", limit=100)
        if df is None or len(df) < 50:
            return None

        vol = df["volume"].values
        close = df["close"].values

        # Volume spike check
        avg_vol = np.mean(vol[-50:-1])
        current_vol = vol[-1]
        taker_ratio = current_vol / avg_vol if avg_vol > 0 else 0

        if taker_ratio < taker_mult:
            return None

        # Price direction in last few bars
        recent_return = (close[-1] / close[-6] - 1) if close[-6] > 0 else 0

        min_move = self.config.extra.get("min_move_pct", 0.005)
        if abs(recent_return) < min_move:
            return None

        # ── Real OI check (preferred) with volume×price proxy fallback ──
        oi_drop_est = 0.0
        try:
            oi = await self.data_hub.get_open_interest(coin)
            if oi is not None and oi > 0:
                if not hasattr(self, '_oi_cache'):
                    self._oi_cache = {}
                prev_oi = self._oi_cache.get(coin, oi)
                oi_change_pct = (oi - prev_oi) / prev_oi if prev_oi > 0 else 0
                self._oi_cache[coin] = oi
                # Real OI declining = actual forced exits
                if oi_change_pct < -0.001:  # OI dropped at least 0.1%
                    oi_drop_est = abs(oi_change_pct) * 100 * taker_ratio  # amplified by volume
                else:
                    oi_drop_est = taker_ratio * abs(recent_return) * 100  # proxy fallback
            else:
                oi_drop_est = taker_ratio * abs(recent_return) * 100  # proxy
        except Exception:
            oi_drop_est = taker_ratio * abs(recent_return) * 100  # proxy on error

        if oi_drop_est < oi_sigma:
            return None

        if recent_return < 0:
            # Price dropped sharply with volume → long liquidations → fade LONG
            return ("LONG", oi_drop_est, taker_ratio, recent_return)
        else:
            # Price spiked sharply with volume → short liquidations → fade SHORT
            return ("SHORT", oi_drop_est, taker_ratio, recent_return)

    def compute_barriers(
        self, signal: Signal, atr: float, price: float, extra: dict | None = None
    ) -> tuple[float, float]:
        """SL/TP based on 5m ATR (scaled from 1m ATR passed by base._try_execute)."""
        cfg = extra if extra is not None else self.config.extra
        sl_mult = cfg.get("sl_atr_mult", 2.5)
        tp_mult = cfg.get("tp_atr_mult", 2.0)

        # Scale 1m ATR → 5m ATR equivalent (base._try_execute passes 1m ATR)
        atr_5m = atr * (5 ** 0.5)  # √5 ≈ 2.24
        sl_dist = max(atr_5m * sl_mult, price * 0.004)  # 0.40% floor
        tp_dist = max(atr_5m * tp_mult, sl_dist * 1.5)  # TP >= 1.5× SL (R:R floor)

        if signal.action == Action.SHORT:
            sl_price = price + sl_dist
            tp_price = price - tp_dist
        else:
            sl_price = price - sl_dist
            tp_price = price + tp_dist

        return (sl_price, tp_price)
