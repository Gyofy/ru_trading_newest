"""Position persistence -- survives process crashes.

Saves open positions to JSON so the bot can recover state
after a restart without losing track of exchange orders.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, fields, asdict
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
    # 3-stage partial TP (0.0 = single TP 하위호환)
    tp1_price: float = 0.0   # ATR×1.0 — 33% 청산, SL→BEP
    tp2_price: float = 0.0   # ATR×2.0 — 33% 청산, SL→TP1
    tp3_price: float = 0.0   # ATR×3.0 — 나머지 청산 (= tp_price)
    tp1_hit: bool = False
    tp2_hit: bool = False
    remaining_qty: float = 0.0  # 0 = 원래 qty 그대로
    # MFE/MAE tracking (updated every 30s poll by SlTpMonitor)
    price_high: float = 0.0  # highest price seen since entry
    price_low: float = 0.0   # lowest price seen since entry

    def __post_init__(self):
        if self.price_high == 0.0:
            self.price_high = self.entry_price
        if self.price_low == 0.0:
            self.price_low = self.entry_price
        if self.remaining_qty == 0.0:
            self.remaining_qty = self.qty

    @property
    def current_qty(self) -> float:
        """현재 남아있는 수량."""
        return self.remaining_qty if self.remaining_qty > 0 else self.qty

    @property
    def partial_tp_enabled(self) -> bool:
        """3단계 분할 익절 활성화 여부."""
        return self.tp1_price > 0 and self.tp2_price > 0 and self.tp3_price > 0

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

    @property
    def _backup_path(self) -> Optional[Path]:
        if self._persist_path is None:
            return None
        return self._persist_path.with_suffix(".bak.json")

    def _save(self) -> None:
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {coin: asdict(pos) for coin, pos in self.positions.items()}
            text = json.dumps(data, indent=2, default=str)
            # 원자적 쓰기: tmp → rename
            tmp = self._persist_path.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(self._persist_path)
            if self._backup_path:
                import shutil
                shutil.copy2(self._persist_path, self._backup_path)
        except Exception as e:
            logger.error(f"[PosStore] Save failed: {e}")

    def _load(self) -> None:
        known = {f.name for f in fields(OpenPosition)}

        def _parse(text: str) -> dict:
            data = json.loads(text)
            result = {}
            for coin, d in data.items():
                filtered = {k: v for k, v in d.items() if k in known}
                result[coin] = OpenPosition(**filtered)
            return result

        # Try primary file first
        try:
            self.positions = _parse(self._persist_path.read_text(encoding="utf-8"))
            logger.info(f"[PosStore] Recovered {len(self.positions)} positions")
            return
        except Exception as e:
            logger.error(f"[PosStore] Primary load FAILED: {e} — trying backup")

        # Fallback to backup file
        if self._backup_path and self._backup_path.exists():
            try:
                self.positions = _parse(self._backup_path.read_text(encoding="utf-8"))
                logger.warning(
                    f"[PosStore] Recovered {len(self.positions)} positions from backup"
                )
                return
            except Exception as e2:
                logger.error(f"[PosStore] Backup load FAILED: {e2} — positions lost")

        logger.error("[PosStore] All recovery attempts failed — starting with empty positions")
