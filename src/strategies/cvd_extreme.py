"""Strategy: CVD Extreme — cvd_spike + asymmetric_sniper 통합.

CVD Q95 + Z-score 2.0σ 이상 극단 신호에서 역추세 진입.
고정 TP (SL × 2.5) — trailing stop 제거.
15분봉 ATR 기반 SL로 1분봉 noise 문제 해결.

Replaces:
  - cvd_spike (TP 0건/84거래 → trailing 실패)
  - asymmetric_sniper (SL 0.10% < 수수료 0.18%)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.signals.contract import Signal, Action, Regime
from src.strategies.base import StrategyBase, ROUND_TRIP_FEE_RATE

logger = logging.getLogger("strategy.cvd_extreme")


class CVDExtreme(StrategyBase):
    """Unified CVD extreme counter-trend strategy with fixed TP."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_trade_ts: float = 0.0
        self._daily_trade_count: int = 0
        self._daily_trade_date: str = ""
        # Per-coin cooldown after SL hit
        self._coin_sl_times: dict[str, list[float]] = {}

    # ── Overrides ─────────────────────────────────────────

    async def evaluate(self, coins: list[str]) -> list[Signal]:
        """Global checks: daily cap, cooldown."""
        cfg = self.config.extra
        max_daily = cfg.get("max_daily_trades", 10)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_trade_date:
            self._daily_trade_count = 0
            self._daily_trade_date = today

        if self._daily_trade_count >= max_daily:
            return []

        # Global cooldown (min 3 minutes between any trade)
        cooldown_sec = cfg.get("cooldown_sec", 180)
        if self._last_trade_ts > 0 and (time.time() - self._last_trade_ts) < cooldown_sec:
            return []

        signals = await super().evaluate(coins)
        # Max 1 signal per cycle
        return signals[:1]

    async def _eval_one_coin(self, coin: str) -> Signal | None:
        """CVD extreme detection with dual condition (quantile + z-score)."""
        cfg = self.config.extra
        cvd_quantile = cfg.get("cvd_quantile", 0.95)
        cvd_sigma = cfg.get("cvd_sigma_mult", 2.0)
        roll_window = cfg.get("cvd_roll_window", 240)

        # Per-coin SL cooldown: 2 SL within 1 hour → skip this coin for 1 hour
        now = time.time()
        coin_sl_list = self._coin_sl_times.get(coin, [])
        # Clean old entries (>1h)
        coin_sl_list = [t for t in coin_sl_list if now - t < 3600]
        self._coin_sl_times[coin] = coin_sl_list
        if len(coin_sl_list) >= 2:
            return None

        df = await self.data_hub.get_ohlcv(coin, "1m", limit=roll_window + 50)
        if df is None or len(df) < roll_window:
            return None

        # Spread filter
        ticker = await self.data_hub.get_ticker(coin)
        if ticker.get("spread_bps", 0) > 5.0:
            return None

        result = self._check_cvd_extreme(df, cvd_quantile, cvd_sigma, roll_window)
        if result is None:
            return None

        direction, strength, z_score, cvd_raw, diag = result

        # Volume confirmation
        vol = df["volume"].values
        vol_mean = vol[-min(50, len(vol) - 1):-1].mean() if len(vol) > 2 else 0
        vol_current = vol[-1]
        vol_mult = cfg.get("volume_spike_mult", 1.5)
        if vol_mean > 0 and vol_current < vol_mean * vol_mult:
            return None

        self._log.info(
            f"[{coin}] CVD EXTREME → {direction} | "
            f"z={z_score:.2f} strength={strength:.3f} q_rank={diag.get('cvd_q_rank', 0):.3f}"
        )
        return Signal(
            symbol=coin,
            horizon_min=120,
            regime=Regime.VOLATILE,
            pred_return=strength,
            p_up=0.3 if direction == "SHORT" else 0.7,
            confidence=min(abs(z_score) / 5.0, 1.0),
            model_name="cvd_extreme",
            strategy_name=self.name,
            action=Action.LONG if direction == "LONG" else Action.SHORT,
            size=1.0,
            ttl_bars=120,
            extra={
                "trigger": diag.get("trigger", "cvd_extreme"),
                "strength": strength,
                "z_score": z_score,
                "cvd_value": cvd_raw,
                "cvd_z_score": z_score,
                "cvd_quantile_rank": diag.get("cvd_q_rank", 0),
            },
        )

    def _check_cvd_extreme(
        self,
        df: pd.DataFrame,
        quantile: float,
        sigma_mult: float,
        window: int,
    ) -> tuple[str, float, float, float, dict] | None:
        """Dual condition: quantile breach AND z-score breach.

        Returns (direction, strength, z_score, cvd_raw, diag) or None.
        """
        cvd_series = self.data_hub.compute_cvd(df)
        if len(cvd_series) < window:
            return None

        cvd_delta = cvd_series.diff().dropna()
        if len(cvd_delta) < window:
            return None

        recent = cvd_delta.iloc[-window:]
        current = float(cvd_delta.iloc[-1])

        q_high = float(recent.quantile(quantile))
        q_low = float(recent.quantile(1 - quantile))
        roll_mean = float(recent.mean())
        roll_std = float(recent.std())

        if roll_std == 0 or not np.isfinite(roll_std):
            return None

        z_score = (current - roll_mean) / roll_std
        cvd_q_rank = float((recent < current).mean())

        diag = {"cvd_q_rank": cvd_q_rank, "trigger": ""}

        # DUAL CONDITION: quantile AND z-score
        if current > q_high and z_score > sigma_mult:
            strength = float((current - q_high) / (q_high if q_high != 0 else 1))
            diag["trigger"] = "cvd_extreme_buy"
            return ("SHORT", -strength, z_score, current, diag)

        if current < q_low and z_score < -sigma_mult:
            strength = float((q_low - current) / (abs(q_low) if q_low != 0 else 1))
            diag["trigger"] = "cvd_extreme_sell"
            return ("LONG", strength, z_score, current, diag)

        return None

    # ── Barriers: 15m ATR-based SL + Fixed TP ─────────────

    def compute_barriers(
        self, signal: Signal, atr: float, price: float, extra: dict | None = None
    ) -> tuple[float, float]:
        """SL = ATR_15m × sl_atr_mult, TP = SL × tp_rr (fixed, no trailing).

        Uses 1m ATR scaled to ~15m equivalent: ATR_1m × sqrt(15) ≈ ATR_1m × 3.87.
        """
        cfg = extra if extra is not None else self.config.extra
        sl_mult = cfg.get("sl_atr_mult", 2.0)
        tp_rr = cfg.get("tp_rr", 2.5)

        # Scale 1m ATR to 15m equivalent
        atr_15m = atr * (15 ** 0.5)  # √15 ≈ 3.87
        sl_dist = atr_15m * sl_mult

        # Enforce SL floor (0.40% of price)
        sl_dist = max(sl_dist, price * 0.004)

        # TP = SL × R:R ratio (FIXED, not trailing)
        tp_dist = sl_dist * tp_rr

        if signal.action == Action.SHORT:
            sl_price = price + sl_dist
            tp_price = price - tp_dist
        else:
            sl_price = price - sl_dist
            tp_price = price + tp_dist

        return (sl_price, tp_price)

    # ── Override _try_execute to track cooldown/daily count ──

    async def _try_execute(self, signal) -> dict | None:
        """Wrap base _try_execute with post-trade tracking."""
        result = await super()._try_execute(signal)
        if result:
            self._last_trade_ts = time.time()
            self._daily_trade_count += 1
        return result

    def record_sl_hit(self, coin: str) -> None:
        """Track SL hit for per-coin cooldown."""
        if coin not in self._coin_sl_times:
            self._coin_sl_times[coin] = []
        self._coin_sl_times[coin].append(time.time())
