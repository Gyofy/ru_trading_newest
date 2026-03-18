"""Signal logger for RL offline learning.

Logs every signal occurrence (accept + reject) with full state vector,
prediction, RL decision, barriers, and downstream execution status.
Results are backfilled when positions close or counterfactuals resolve.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("live_bot.rl.logger")

SIZING_MAP = {0: 0.0, 1: 0.75, 2: 1.0, 3: 1.25}
ACTION_NAMES = {0: "REJECT", 1: "ACCEPT_0.75", 2: "ACCEPT_1.00", 3: "ACCEPT_1.25"}


@dataclass
class SignalRecord:
    """One signal occurrence (accept or reject)."""
    ts: str
    coin: str
    side: str
    regime: str

    # State
    state: list

    # Prediction
    p_trade: float
    p_direction: float
    confidence: float

    # RL decision
    action: int               # 0-3
    action_name: str
    rl_score: float
    sizing_mult: float

    # Barriers
    entry_price: float
    sl_price: float
    tp_price: float

    # Downstream
    risk_gate_passed: bool
    executed: bool

    # Result (filled later)
    result_pnl_pct: Optional[float] = None
    result_exit_reason: Optional[str] = None
    result_bars_held: Optional[int] = None
    result_reward: Optional[float] = None
    result_mfe_pct: Optional[float] = None   # Maximum Favorable Excursion
    result_mae_pct: Optional[float] = None   # Maximum Adverse Excursion
    counterfactual: bool = False
    resolved: bool = False


class SignalLogger:
    """Append-only JSONL logger with result backfill."""

    def __init__(self, log_path: Path):
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # In-memory index for pending results: coin → SignalRecord.ts
        self._pending: dict[str, str] = {}

    def log(
        self,
        coin: str,
        side: str,
        regime: str,
        state: list,
        p_trade: float,
        p_direction: float,
        action: int,
        rl_score: float,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        risk_gate_passed: bool = True,
        executed: bool = True,
    ) -> SignalRecord:
        """Log a signal occurrence."""
        ts = datetime.now(timezone.utc).isoformat()
        rec = SignalRecord(
            ts=ts, coin=coin, side=side, regime=regime,
            state=state,
            p_trade=p_trade, p_direction=p_direction,
            confidence=p_trade * p_direction,
            action=action,
            action_name=ACTION_NAMES.get(action, f"UNKNOWN_{action}"),
            rl_score=rl_score,
            sizing_mult=SIZING_MAP.get(action, 0.0),
            entry_price=entry_price, sl_price=sl_price, tp_price=tp_price,
            risk_gate_passed=risk_gate_passed,
            executed=executed,
        )
        self._append(rec)

        if executed and action > 0:
            self._pending[coin] = ts

        return rec

    def update_result(
        self,
        coin: str,
        pnl_pct: float,
        exit_reason: str,
        bars_held: int,
        mfe_pct: float = 0.0,
        mae_pct: float = 0.0,
    ) -> None:
        """Backfill result for an executed trade.

        mfe_pct: Maximum Favorable Excursion (best unrealized PnL during hold)
        mae_pct: Maximum Adverse Excursion (worst unrealized PnL during hold)
        """
        ts = self._pending.pop(coin, None)
        if ts is None:
            return

        reward = pnl_pct * 100  # symmetric scale
        result = {
            "update_ts": datetime.now(timezone.utc).isoformat(),
            "original_ts": ts,
            "coin": coin,
            "result_pnl_pct": round(pnl_pct, 6),
            "result_exit_reason": exit_reason,
            "result_bars_held": bars_held,
            "result_reward": round(reward, 4),
            "result_mfe_pct": round(mfe_pct, 6),
            "result_mae_pct": round(mae_pct, 6),
            "counterfactual": False,
            "resolved": True,
        }
        self._append_raw(result)
        logger.info(
            f"[RL Log] {coin} result: pnl={pnl_pct:+.2%} mfe={mfe_pct:+.2%} "
            f"mae={mae_pct:+.2%} reward={reward:+.2f}"
        )

    def log_counterfactual(
        self,
        coin: str,
        original_ts: str,
        cf_pnl_pct: float,
    ) -> None:
        """Log counterfactual result for a rejected signal."""
        reward = -cf_pnl_pct * 100  # symmetric: missed profit = negative reward
        result = {
            "update_ts": datetime.now(timezone.utc).isoformat(),
            "original_ts": original_ts,
            "coin": coin,
            "result_pnl_pct": round(cf_pnl_pct, 6),
            "result_reward": round(reward, 4),
            "counterfactual": True,
            "resolved": True,
        }
        self._append_raw(result)
        logger.info(
            f"[RL Log] {coin} counterfactual: cf_pnl={cf_pnl_pct:+.2%} reward={reward:+.2f}"
        )

    def _append(self, rec: SignalRecord) -> None:
        self._append_raw(asdict(rec))

    def _append_raw(self, data: dict) -> None:
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, default=str) + "\n")
        except Exception as e:
            logger.error(f"[RL Log] Write failed: {e}")
