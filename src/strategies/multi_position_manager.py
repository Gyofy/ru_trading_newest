"""MultiPositionManager — strategy-tagged position tracking.

Keys positions by 'strategy:coin'. Enforces same-coin exclusivity
across strategies (only one position per coin globally).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, fields
from pathlib import Path
from typing import Optional

from src.execution.position_store import OpenPosition

logger = logging.getLogger("multi_pos")


class MultiPositionManager:
    """Position manager supporting multiple concurrent strategies."""

    def __init__(self, persist_path: Optional[Path] = None):
        self.positions: dict[str, OpenPosition] = {}  # key = "strategy:coin"
        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load()

    # ── Key helpers ───────────────────────────────────────

    @staticmethod
    def _key(strategy: str, coin: str) -> str:
        return f"{strategy}:{coin}"

    @staticmethod
    def _parse_key(key: str) -> tuple[str, str]:
        parts = key.split(":", 1)
        return (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])

    # ── Queries ───────────────────────────────────────────

    def has_position(self, strategy: str, coin: str) -> bool:
        return self._key(strategy, coin) in self.positions

    def get_position(self, strategy: str, coin: str) -> Optional[OpenPosition]:
        return self.positions.get(self._key(strategy, coin))

    def has_coin_any_strategy(self, coin: str) -> bool:
        """Check if any strategy has a position in this coin."""
        return any(
            self._parse_key(k)[1] == coin for k in self.positions
        )

    def get_positions_by_strategy(self, strategy: str) -> list[OpenPosition]:
        return [
            pos for key, pos in self.positions.items()
            if self._parse_key(key)[0] == strategy
        ]

    def all_positions(self) -> list[OpenPosition]:
        return list(self.positions.values())

    def count(self) -> int:
        return len(self.positions)

    def count_by_strategy(self, strategy: str) -> int:
        return sum(
            1 for k in self.positions if self._parse_key(k)[0] == strategy
        )

    def total_notional(self) -> float:
        """Sum of all open position notionals."""
        return sum(p.entry_price * p.current_qty for p in self.positions.values())

    def strategy_notional(self, strategy: str) -> float:
        """Notional for one strategy."""
        return sum(
            p.entry_price * p.current_qty
            for p in self.get_positions_by_strategy(strategy)
        )

    def direction_counts(self) -> dict[str, int]:
        """Count of BUY and SELL positions."""
        counts = {"BUY": 0, "SELL": 0}
        for pos in self.positions.values():
            counts[pos.side] = counts.get(pos.side, 0) + 1
        return counts

    # ── Mutations ─────────────────────────────────────────

    def add_position(self, strategy: str, pos: OpenPosition) -> None:
        pos.strategy_tag = strategy
        key = self._key(strategy, pos.coin)
        self.positions[key] = pos
        self._save()
        logger.info(f"[MultiPos] Added {key}: {pos.side} {pos.coin} @ {pos.entry_price}")

    def remove_position(self, strategy: str, coin: str) -> Optional[OpenPosition]:
        key = self._key(strategy, coin)
        pos = self.positions.pop(key, None)
        if pos is not None:
            self._save()
            logger.info(f"[MultiPos] Removed {key}")
        return pos

    def update_position(self, strategy: str, coin: str, **kwargs) -> None:
        """Update fields on an existing position."""
        key = self._key(strategy, coin)
        pos = self.positions.get(key)
        if pos is None:
            return
        for k, v in kwargs.items():
            if hasattr(pos, k):
                setattr(pos, k, v)
        self._save()

    # ── Persistence ───────────────────────────────────────

    def _save(self) -> None:
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {key: asdict(pos) for key, pos in self.positions.items()}
            text = json.dumps(data, indent=2, default=str)
            tmp = self._persist_path.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(self._persist_path)
        except Exception as e:
            logger.error(f"[MultiPos] Save failed: {e}")

    def _load(self) -> None:
        known = {f.name for f in fields(OpenPosition)}
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for key, d in data.items():
                filtered = {k: v for k, v in d.items() if k in known}
                self.positions[key] = OpenPosition(**filtered)
            logger.info(f"[MultiPos] Recovered {len(self.positions)} positions")
        except Exception as e:
            logger.error(f"[MultiPos] Load failed: {e} — starting empty")
