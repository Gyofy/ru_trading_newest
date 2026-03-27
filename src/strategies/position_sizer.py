"""PositionSizer — dynamic quantity based on coin evaluation.

Instead of fixed risk% per trade, sizes positions based on:
1. ATR-based volatility (high vol → smaller position, low vol → larger)
2. Spread quality (tight spread → larger, wide spread → smaller)
3. Signal confidence (high confidence → larger position)
4. Available allocation budget

Formula:
    base_risk = allocation * risk_pct_per_trade  (e.g., $2400 * 5% = $120)

    vol_factor = median_atr_pct / coin_atr_pct   (normalize by market avg)
    vol_factor = clamp(vol_factor, 0.5, 2.0)      (cap adjustment)

    spread_factor = 1.0 if spread < 3bps else (3.0 / spread_bps)
    spread_factor = clamp(spread_factor, 0.3, 1.0)

    confidence_factor = 0.5 + signal_confidence * 0.5  (0.5x to 1.0x)

    adjusted_risk = base_risk * vol_factor * spread_factor * confidence_factor
    qty = adjusted_risk / sl_distance
    notional = qty * price

    # Cap by max allocation
    notional = min(notional, available_allocation)
    # Floor at Binance minimum
    if notional < 5.0: skip
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

BINANCE_MIN_NOTIONAL = 5.0


class PositionSizer:
    """Compute dynamic position quantity from market state and signal quality."""

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            risk_pct_per_trade    : float  — fraction of allocation risked (default 0.05)
            vol_adjust_enabled    : bool   — enable ATR-based vol scaling
            spread_adjust_enabled : bool   — enable spread-based scaling
            confidence_adjust_enabled : bool — enable signal confidence scaling
            min_factor            : float  — floor for combined adjustment (default 0.3)
            max_factor            : float  — cap for combined adjustment (default 2.0)
        """
        self.risk_pct = config.get("risk_pct_per_trade", 0.05)
        self.vol_adjust = config.get("vol_adjust_enabled", True)
        self.spread_adjust = config.get("spread_adjust_enabled", True)
        self.confidence_adjust = config.get("confidence_adjust_enabled", True)
        self.min_factor = config.get("min_factor", 0.3)
        self.max_factor = config.get("max_factor", 2.0)

    # ------------------------------------------------------------------
    def compute_qty(
        self,
        allocation_usdt: float,
        leverage: int,
        price: float,
        sl_dist: float,
        atr: float,
        spread_bps: float,
        signal_confidence: float,
        median_atr_pct: float = 0.02,
    ) -> tuple[float, float, dict]:
        """Compute position quantity, notional, and sizing breakdown.

        Parameters
        ----------
        allocation_usdt : float
            Strategy-level allocation in USDT.
        leverage : int
            Position leverage.
        price : float
            Current coin price.
        sl_dist : float
            Stop-loss distance in price units.
        atr : float
            Current ATR value (same units as price).
        spread_bps : float
            Current bid-ask spread in basis points.
        signal_confidence : float
            Signal confidence in [0, 1].
        median_atr_pct : float
            Market median ATR as fraction of price (default 2%).

        Returns
        -------
        (qty, notional, sizing_details)
            qty=0.0 and notional=0.0 when position is too small or invalid.
        """
        details: dict = {}

        # --- Validate inputs ------------------------------------------------
        if price <= 0 or sl_dist <= 0 or allocation_usdt <= 0:
            logger.warning(
                "PositionSizer: invalid inputs price=%.6f sl_dist=%.6f alloc=%.2f",
                price, sl_dist, allocation_usdt,
            )
            details.update(skip_reason="invalid_inputs")
            return 0.0, 0.0, details

        # --- Base risk -------------------------------------------------------
        base_risk = allocation_usdt * self.risk_pct
        details["base_risk"] = round(base_risk, 4)

        # --- Volatility factor -----------------------------------------------
        if self.vol_adjust and atr > 0 and price > 0:
            coin_atr_pct = atr / price
            vol_factor = median_atr_pct / coin_atr_pct if coin_atr_pct > 0 else 1.0
            vol_factor = self.clamp(vol_factor, 0.5, 2.0)
        else:
            vol_factor = 1.0
        details["vol_factor"] = round(vol_factor, 4)

        # --- Spread factor ---------------------------------------------------
        if self.spread_adjust and spread_bps > 0:
            if spread_bps <= 3.0:
                spread_factor = 1.0
            else:
                spread_factor = 3.0 / spread_bps
            spread_factor = self.clamp(spread_factor, 0.3, 1.0)
        else:
            spread_factor = 1.0
        details["spread_factor"] = round(spread_factor, 4)

        # --- Confidence factor -----------------------------------------------
        if self.confidence_adjust:
            conf = self.clamp(signal_confidence, 0.0, 1.0)
            confidence_factor = 0.5 + conf * 0.5
        else:
            confidence_factor = 1.0
        details["confidence_factor"] = round(confidence_factor, 4)

        # --- Combined adjustment (clamped) -----------------------------------
        combined = vol_factor * spread_factor * confidence_factor
        combined = self.clamp(combined, self.min_factor, self.max_factor)
        details["combined_factor"] = round(combined, 4)

        # --- Adjusted risk & quantity ----------------------------------------
        adjusted_risk = base_risk * combined
        details["adjusted_risk"] = round(adjusted_risk, 4)

        qty = adjusted_risk / sl_dist
        notional = qty * price
        details["qty_raw"] = round(qty, 8)
        details["notional_raw"] = round(notional, 4)

        # --- Cap by available allocation (leveraged) -------------------------
        max_notional = allocation_usdt * leverage
        if notional > max_notional:
            notional = max_notional
            qty = notional / price
            details["capped_by_allocation"] = True

        # --- Floor at Binance minimum ----------------------------------------
        if notional < BINANCE_MIN_NOTIONAL:
            logger.info(
                "PositionSizer: notional $%.2f < $%.2f minimum — skip",
                notional, BINANCE_MIN_NOTIONAL,
            )
            details["skip_reason"] = "below_min_notional"
            return 0.0, 0.0, details

        details["qty"] = round(qty, 8)
        details["notional"] = round(notional, 4)

        logger.debug(
            "PositionSizer: base=$%.2f combined=%.2fx → risk=$%.2f qty=%.6f notional=$%.2f",
            base_risk, combined, adjusted_risk, qty, notional,
        )

        return qty, notional, details

    # ------------------------------------------------------------------
    @staticmethod
    def clamp(value: float, lo: float, hi: float) -> float:
        """Clamp value between lo and hi."""
        return max(lo, min(hi, value))
