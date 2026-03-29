"""Strategy: VWAP Exhaustion — VWAP 이탈 + 거래량 감소 = 이탈 소진 → 복귀.

가설: 가격이 VWAP에서 크게 이탈하되, 이탈 방향의 거래량이 줄어들면
      이탈의 동력이 소진 → 평균 회귀 압력이 작용.
      단순 VWAP 이탈과 달리 "volume exhaustion" 확인으로 false signal 필터링.
수수료 edge:
  - TP = VWAP (명확한 타겟) → exit point가 물리적으로 정의됨.
  - 0.15% 왕복 / 0.50%+ SL = 수수료가 SL의 30% 이하.
  - Volume 감소 필터가 세션 전환 false signal 제거 → WR 향상.
"""

from __future__ import annotations

import logging
import numpy as np

from src.signals.contract import Signal, Action, Regime
from src.strategies.base import StrategyBase

logger = logging.getLogger("strategy.vwap_reversion")


class VWAPReversion(StrategyBase):
    """Enter counter-trend when VWAP deviation + volume exhaustion detected."""

    async def _eval_one_coin(self, coin: str) -> Signal | None:
        cfg = self.config.extra
        vwap_window = cfg.get("vwap_window", 60)
        sigma_mult = cfg.get("sigma_mult", 2.0)
        vol_decay_threshold = cfg.get("vol_decay_threshold", 0.7)  # current vol < 70% of peak

        df = await self.data_hub.get_ohlcv(coin, "1m", limit=vwap_window + 20)
        if df is None or len(df) < vwap_window:
            return None

        vwap = self.data_hub.compute_vwap(df, window=vwap_window)
        if vwap is None or len(vwap) < 10:
            return None

        price = float(df["close"].iloc[-1])
        current_vwap = float(vwap.iloc[-1])
        if current_vwap <= 0:
            return None

        # Z-score of deviation from VWAP
        dev_series = (df["close"] - vwap) / vwap
        dev_std = float(dev_series.iloc[-vwap_window:].std())
        if dev_std <= 0:
            return None

        current_dev = (price - current_vwap) / current_vwap
        z_score = current_dev / dev_std

        if abs(z_score) < sigma_mult:
            return None

        # ── Volume exhaustion check ──
        # 이탈 방향의 거래량이 최근 피크 대비 감소했는지
        vol = df["volume"].values[-20:]
        if len(vol) < 10:
            return None

        vol_peak = np.max(vol[-10:-1])  # 최근 10bar 중 최대 (마지막 bar 제외)
        vol_current = vol[-1]

        if vol_peak <= 0:
            return None

        vol_ratio = vol_current / vol_peak
        if vol_ratio > vol_decay_threshold:
            return None  # 거래량이 아직 높음 → 이탈 진행 중, 복귀 시기 아님

        # Counter-trend: price above VWAP → SHORT, below → LONG
        if z_score > sigma_mult:
            direction = "SHORT"
            action = Action.SHORT
        else:
            direction = "LONG"
            action = Action.LONG

        self._log.info(
            f"[{coin}] VWAP EXHAUSTION → {direction} | "
            f"z={z_score:.2f} dev={current_dev:.4%} vol_decay={vol_ratio:.2f}"
        )
        return Signal(
            symbol=coin, horizon_min=60, regime=Regime.VOLATILE,
            pred_return=abs(current_dev), p_up=0.3 if direction == "SHORT" else 0.7,
            confidence=min(abs(z_score) / 4.0, 1.0),
            model_name="vwap_exhaustion", strategy_name=self.name,
            action=action, size=1.0, ttl_bars=60,
            extra={"trigger": "vwap_vol_exhaustion", "z_score": z_score,
                   "vwap_dev_pct": current_dev, "vol_decay_ratio": vol_ratio,
                   "vwap_price": current_vwap},
        )

    def compute_barriers(self, signal, atr, price, extra=None):
        cfg = extra if extra is not None else self.config.extra
        sl_mult = cfg.get("sl_atr_mult", 2.0)
        tp_rr = cfg.get("tp_rr", 2.0)

        atr_15m = atr * (15 ** 0.5)
        sl_dist = max(atr_15m * sl_mult, price * 0.004)
        tp_dist = sl_dist * tp_rr

        if signal.action == Action.SHORT:
            return (price + sl_dist, price - tp_dist)
        return (price - sl_dist, price + tp_dist)
