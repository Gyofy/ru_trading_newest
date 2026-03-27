"""Strategy A: CVD Spike Reactor — 극단 orderflow에서 역추세 진입.

Trigger: CVD or OFI가 rolling Q95 돌파 → 방향 소진으로 판단, 역방향 진입.
SL: 3×ATR (close 기반, wick 무시)
TP: Trailing stop (1.5×ATR trail distance)
Cycle: 1분 (가장 공격적)
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from src.signals.contract import Signal, Action, Regime
from src.strategies.base import StrategyBase

logger = logging.getLogger("strategy.cvd_spike")


class CVDSpikeReactor(StrategyBase):
    """Counter-trend entries on extreme CVD/OFI signals."""

    async def _eval_one_coin(self, coin: str) -> "Signal | None":
        """Per-coin evaluation — called in parallel by base.evaluate()."""
        cfg = self.config.extra
        cvd_quantile = cfg.get("cvd_quantile", 0.95)
        ofi_quantile = cfg.get("ofi_quantile", 0.95)
        roll_window = cfg.get("cvd_roll_window", 480)

        df = await self.data_hub.get_ohlcv(coin, "1m", limit=roll_window + 50)
        if df is None or len(df) < roll_window:
            return None

        result = self._check_extreme(df, cvd_quantile, ofi_quantile, roll_window)
        if result is None:
            return None

        direction, strength, trigger_type, diag = result
        self._log.info(
            f"[{coin}] CVD SPIKE → {direction} | "
            f"trigger={trigger_type} strength={strength:.3f} "
            f"cvd_z={diag.get('cvd_z', 0):.2f} ofi_q={diag.get('ofi_quantile_rank', 0):.2f}"
        )
        return Signal(
            symbol=coin,
            horizon_min=60,
            regime=Regime.VOLATILE,
            pred_return=strength,
            p_up=0.3 if direction == "SHORT" else 0.7,
            confidence=min(abs(strength) / 3.0, 1.0),
            model_name="cvd_spike_reactor",
            strategy_name=self.name,
            action=Action.LONG if direction == "LONG" else Action.SHORT,
            size=1.0,
            ttl_bars=150,
            extra={
                "trigger": trigger_type,
                "strength": strength,
                # Diagnostic fields for trade_context and algo development
                "cvd_value": diag.get("cvd_value", 0.0),
                "cvd_z_score": diag.get("cvd_z", 0.0),
                "ofi_value": diag.get("ofi_value", 0.0),
                "cvd_quantile_rank": diag.get("cvd_quantile_rank", 0.0),
                "ofi_quantile_rank": diag.get("ofi_quantile_rank", 0.0),
                "cvd_q_threshold": diag.get("cvd_q_threshold", 0.0),
            },
        )

    def _check_extreme(
        self,
        df: pd.DataFrame,
        cvd_q: float,
        ofi_q: float,
        window: int,
    ) -> tuple[str, float, str, dict] | None:
        """Check if CVD or OFI is at extreme levels.

        Returns (direction, strength, trigger_type, diag_dict) or None.
        Direction is COUNTER to the extreme (extreme buy → SHORT).
        diag_dict contains raw values for trade_context logging.
        """
        # Compute CVD delta (not cumulative — use per-bar delta)
        cvd_series = self.data_hub.compute_cvd(df)
        if len(cvd_series) < window:
            return None

        # Per-bar CVD delta (diff of cumulative)
        cvd_delta = cvd_series.diff().dropna()
        recent = cvd_delta.iloc[-window:]

        # Rolling quantiles
        q_high = recent.quantile(cvd_q)
        q_low = recent.quantile(1 - cvd_q)
        current_cvd = float(cvd_delta.iloc[-1])

        # OFI check
        ofi_series = self.data_hub.compute_ofi(df)
        ofi_recent = ofi_series.iloc[-window:]
        ofi_q_high = ofi_recent.quantile(ofi_q)
        ofi_q_low = ofi_recent.quantile(1 - ofi_q)
        current_ofi = float(ofi_series.iloc[-1])

        # CVD z-score (standardised position within window)
        # NaN is truthy so "or 1.0" does not protect against NaN — use np.isfinite
        _std = float(recent.std())
        cvd_std = _std if np.isfinite(_std) and _std > 0 else 1.0
        cvd_mean = float(recent.mean())
        cvd_z = (current_cvd - cvd_mean) / cvd_std

        # Quantile ranks (0=min, 1=max)
        cvd_q_rank = float((recent < current_cvd).mean())
        ofi_q_rank = float((ofi_recent < current_ofi).mean())

        diag = {
            "cvd_value": current_cvd,
            "cvd_z": cvd_z,
            "ofi_value": current_ofi,
            "cvd_quantile_rank": cvd_q_rank,
            "ofi_quantile_rank": ofi_q_rank,
            "cvd_q_threshold": float(q_high),
        }

        # CVD extreme → counter-trend
        if current_cvd > q_high:
            strength = float((current_cvd - q_high) / (q_high if q_high != 0 else 1))
            return ("SHORT", -strength, "cvd_extreme_buy", diag)

        if current_cvd < q_low:
            strength = float((q_low - current_cvd) / (abs(q_low) if q_low != 0 else 1))
            return ("LONG", strength, "cvd_extreme_sell", diag)

        # OFI extreme → counter-trend
        if current_ofi > ofi_q_high:
            strength = float((current_ofi - ofi_q_high) / (ofi_q_high if ofi_q_high != 0 else 1))
            return ("SHORT", -strength, "ofi_extreme_buy", diag)

        if current_ofi < ofi_q_low:
            strength = float((ofi_q_low - current_ofi) / (abs(ofi_q_low) if ofi_q_low != 0 else 1))
            return ("LONG", strength, "ofi_extreme_sell", diag)

        return None

    def compute_barriers(
        self, signal: Signal, atr: float, price: float, extra: dict | None = None
    ) -> tuple[float, float]:
        """SL = 3×ATR, TP = 0 (trailing stop)."""
        cfg = extra if extra is not None else self.config.extra
        sl_mult = cfg.get("sl_atr_mult", 3.0)
        sl_dist = atr * sl_mult

        if signal.action == Action.SHORT:
            sl_price = price + sl_dist
        else:
            sl_price = price - sl_dist

        # tp_price=0 signals trailing stop mode
        return (sl_price, 0.0)
