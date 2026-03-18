"""Position persistence -- survives process crashes.

Saves open positions to JSON so the bot can recover state
after a restart without losing track of exchange orders.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("live_bot.posstore")


@dataclass
class OpenPosition:
    """An open position tracked by the bot."""
    coin: str
    side: str                # "BUY" or "SELL"
    entry_price: float
    qty: float
    sl_price: float
    tp_price: float
    entry_time: str          # ISO format
    ttl_bars: int
    bars_held: int = 0
    entry_order_id: str = ""
    sl_order_id: str = ""
    tp_order_id: str = ""
    # MFE/MAE tracking (updated every 30s poll by SlTpMonitor)
    price_high: float = 0.0  # highest price seen since entry
    price_low: float = 0.0   # lowest price seen since entry

    def __post_init__(self):
        if self.price_high == 0.0:
            self.price_high = self.entry_price
        if self.price_low == 0.0:
            self.price_low = self.entry_price

    def update_extremes(self, price: float) -> None:
        """Track running high/low for MFE/MAE calculation."""
        if price > self.price_high:
            self.price_high = price
        if price < self.price_low:
            self.price_low = price

    @property
    def mfe_pct(self) -> float:
        """Maximum Favorable Excursion (best unrealized PnL %)."""
        if self.side == "BUY":
            return (self.price_high - self.entry_price) / self.entry_price
        return (self.entry_price - self.price_low) / self.entry_price

    @property
    def mae_pct(self) -> float:
        """Maximum Adverse Excursion (worst unrealized PnL %)."""
        if self.side == "BUY":
            return (self.price_low - self.entry_price) / self.entry_price
        return (self.entry_price - self.price_high) / self.entry_price


class PositionManager:
    """In-memory position manager with JSON persistence."""

    def __init__(self, persist_path: Optional[Path] = None):
        self.positions: dict[str, OpenPosition] = {}
        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load()

    # ── Queries ────────────────────────────────────────────

    def has_position(self, coin: str) -> bool:
        return coin in self.positions

    def get_position(self, coin: str) -> Optional[OpenPosition]:
        return self.positions.get(coin)

    def all_positions(self) -> list[OpenPosition]:
        return list(self.positions.values())

    def count(self) -> int:
        return len(self.positions)

    # ── Mutations (auto-persist) ───────────────────────────

    def add_position(self, pos: OpenPosition) -> None:
        self.positions[pos.coin] = pos
        self._save()

    def remove_position(self, coin: str) -> Optional[OpenPosition]:
        pos = self.positions.pop(coin, None)
        if pos is not None:
            self._save()
        return pos

    # ── Persistence ────────────────────────────────────────

    def _save(self) -> None:
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {coin: asdict(pos) for coin, pos in self.positions.items()}
            self._persist_path.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"[PosStore] Save failed: {e}")

    def _load(self) -> None:
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for coin, d in data.items():
                self.positions[coin] = OpenPosition(**d)
            logger.info(f"[PosStore] Recovered {len(self.positions)} positions")
        except Exception as e:
            logger.warning(f"[PosStore] Load failed: {e}")
