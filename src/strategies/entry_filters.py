"""Entry Filters — rule-based reject gates applied before position entry.

These filters reduce false positive trades based on market microstructure.
Each filter returns (passed: bool, reason: str).

F1: VPIN Filter — reject when VPIN < threshold (noise, no informed flow)
F2: Coin Blacklist — reject coin if SL hit N times in last 24h
F3: ATR Regime — reject when ATR percentile too low (dead market)
F4: Early Exit — flag for 5-min MFE check (post-entry, handled by monitor)
"""

from __future__ import annotations

import time


class EntryFilters:
    """Rule-based pre-entry reject gates. Shared across all strategies."""

    def __init__(self, config: dict):
        self.vpin_enabled = config.get("vpin_filter_enabled", True)
        self.vpin_min = config.get("vpin_min", 0.3)

        self.blacklist_enabled = config.get("blacklist_enabled", True)
        self.blacklist_sl_count = config.get("blacklist_sl_count", 2)
        self.blacklist_window_sec = config.get("blacklist_window_sec", 86400)  # 24h

        self.atr_regime_enabled = config.get("atr_regime_enabled", True)
        self.atr_min_percentile = config.get("atr_min_percentile", 20)

        # SL hit tracking for blacklist
        self._sl_history: dict[str, list[float]] = {}  # coin -> list of unix timestamps

    def check_vpin(self, vpin: float) -> tuple[bool, str]:
        """F1: Reject when VPIN < threshold (no informed flow)."""
        if not self.vpin_enabled:
            return True, ""
        if vpin < self.vpin_min:
            return False, f"vpin_low: {vpin:.3f} < {self.vpin_min}"
        return True, ""

    def check_blacklist(self, coin: str) -> tuple[bool, str]:
        """F2: Reject coin if SL hit N times in recent window."""
        if not self.blacklist_enabled:
            return True, ""
        now = time.time()
        cutoff = now - self.blacklist_window_sec
        history = self._sl_history.get(coin, [])
        recent = [t for t in history if t > cutoff]
        self._sl_history[coin] = recent  # prune old entries
        if len(recent) >= self.blacklist_sl_count:
            return False, f"blacklisted: {len(recent)} SL hits in {self.blacklist_window_sec // 3600}h"
        return True, ""

    def record_sl_hit(self, coin: str) -> None:
        """Called from _on_position_close when reason=SL_HIT."""
        if coin not in self._sl_history:
            self._sl_history[coin] = []
        self._sl_history[coin].append(time.time())

    def check_atr_regime(
        self, atr: float, price: float, atr_history: list[float]
    ) -> tuple[bool, str]:
        """F3: Reject when ATR is below minimum percentile of recent history."""
        if not self.atr_regime_enabled or not atr_history:
            return True, ""
        sorted_atr = sorted(atr_history)
        percentile_idx = int(len(sorted_atr) * self.atr_min_percentile / 100)
        threshold = sorted_atr[min(percentile_idx, len(sorted_atr) - 1)]
        if atr < threshold:
            return False, f"atr_low: {atr:.6f} < P{self.atr_min_percentile}={threshold:.6f}"
        return True, ""

    def check_all(
        self,
        coin: str,
        vpin: float,
        atr: float,
        price: float,
        atr_history: list[float] | None = None,
    ) -> tuple[bool, str]:
        """Run all filters. Returns (passed, first_rejection_reason)."""
        ok, reason = self.check_vpin(vpin)
        if not ok:
            return False, reason
        ok, reason = self.check_blacklist(coin)
        if not ok:
            return False, reason
        ok, reason = self.check_atr_regime(atr, price, atr_history or [])
        if not ok:
            return False, reason
        return True, ""
