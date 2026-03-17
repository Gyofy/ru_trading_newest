"""Position State Machine — 심볼별 포지션 생명주기 관리.

순수 상태 전이 로직. I/O는 Command를 반환하여 LiveEngine이 실행.
Idempotent: 중복 이벤트 무시.

States:
    IDLE → ENTRY_PENDING → FILLED → PROTECTED → (TP_HIT | SL_HIT | TIME_STOP) → IDLE
                 ↘ ENTRY_TIMEOUT → IDLE
    FILLED/PROTECTED → EMERGENCY_CLOSE → IDLE
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from src.execution.order_ledger import OrderLedger

logger = logging.getLogger(__name__)


class State(str, Enum):
    IDLE = "IDLE"
    ENTRY_PENDING = "ENTRY_PENDING"
    FILLED = "FILLED"
    PROTECTED = "PROTECTED"
    TP_HIT = "TP_HIT"
    SL_HIT = "SL_HIT"
    TIME_STOP = "TIME_STOP"
    ENTRY_TIMEOUT = "ENTRY_TIMEOUT"
    EMERGENCY_CLOSE = "EMERGENCY_CLOSE"


class CmdType(str, Enum):
    PLACE_ENTRY = "PLACE_ENTRY"
    PLACE_SL = "PLACE_SL"
    PLACE_TP = "PLACE_TP"
    CANCEL_ORDER = "CANCEL_ORDER"
    MARKET_CLOSE = "MARKET_CLOSE"
    CANCEL_PROTECTIVES = "CANCEL_PROTECTIVES"
    LOG_TRADE = "LOG_TRADE"


@dataclass
class Command:
    """ExchangeAdapter가 실행할 명령."""
    type: CmdType
    symbol: str
    params: dict = field(default_factory=dict)


@dataclass
class PositionContext:
    """현재 포지션의 실행 정보."""
    entry_order_id: str = ""
    entry_price: float = 0.0
    entry_qty: float = 0.0
    side: str = ""  # BUY or SELL
    sl_order_id: str = ""
    tp_order_id: str = ""
    sl_price: float = 0.0
    tp_price: float = 0.0
    bars_held: int = 0
    ttl_bars: int = 6


# Valid state transitions
_VALID_TRANSITIONS: dict[State, set[State]] = {
    State.IDLE: {State.ENTRY_PENDING},
    State.ENTRY_PENDING: {State.FILLED, State.ENTRY_TIMEOUT},
    State.FILLED: {State.PROTECTED, State.EMERGENCY_CLOSE},
    State.PROTECTED: {State.TP_HIT, State.SL_HIT, State.TIME_STOP, State.EMERGENCY_CLOSE},
    State.TP_HIT: {State.IDLE},
    State.SL_HIT: {State.IDLE},
    State.TIME_STOP: {State.IDLE},
    State.ENTRY_TIMEOUT: {State.IDLE},
    State.EMERGENCY_CLOSE: {State.IDLE},
}


class PositionStateMachine:
    """심볼 하나의 포지션 상태 머신.

    on_event() → Command 리스트 반환 (side-effect 없음).
    LiveEngine이 Command를 실행하고 결과를 다시 이벤트로 전달.
    """

    def __init__(self, symbol: str, ledger: OrderLedger):
        self.symbol = symbol
        self.ledger = ledger
        self.state = State.IDLE
        self.ctx = PositionContext()
        self._event_log: list[str] = []

    def restore(self) -> None:
        """Ledger에서 마지막 상태 복구."""
        last = self.ledger.get_last_state(self.symbol)
        try:
            self.state = State(last)
        except ValueError:
            self.state = State.IDLE

        if self.state in (State.FILLED, State.PROTECTED):
            positions = self.ledger.get_open_positions()
            for pos in positions:
                if pos["symbol"] == self.symbol:
                    entry = pos["entry_order"]
                    self.ctx.entry_order_id = entry["order_link_id"]
                    self.ctx.entry_price = entry.get("fill_price", 0)
                    self.ctx.entry_qty = entry.get("fill_qty", 0)
                    self.ctx.side = entry["side"]

                    # Recover SL/TP order IDs
                    children = self.ledger.get_child_orders(self.ctx.entry_order_id)
                    for child in children:
                        if child["purpose"] == "stop_loss":
                            self.ctx.sl_order_id = child["order_link_id"]
                            self.ctx.sl_price = child.get("stop_trigger", 0)
                        elif child["purpose"] == "take_profit":
                            self.ctx.tp_order_id = child["order_link_id"]
                            self.ctx.tp_price = child.get("stop_trigger", 0)
                    break

        logger.info(f"[SM:{self.symbol}] Restored to {self.state.value}")

    def on_event(self, event: str, data: dict | None = None) -> list[Command]:
        """이벤트 처리 → Command 리스트 반환.

        Events:
            new_signal      — 신규 시그널 (data: side, qty, price, sl, tp, ttl, order_id)
            entry_filled    — 진입 체결 (data: fill_price, fill_qty, fee)
            entry_timeout   — 진입 미체결 타임아웃
            sl_confirmed    — SL 주문 접수 확인
            sl_filled       — SL 체결
            tp_filled       — TP 체결
            time_stop       — TTL 만료
            bar_tick        — 바 경과 (bars_held 증가)
            kill_switch     — 긴급 청산
        """
        data = data or {}
        commands: list[Command] = []

        if event == "new_signal":
            commands = self._on_new_signal(data)
        elif event == "entry_filled":
            commands = self._on_entry_filled(data)
        elif event == "entry_timeout":
            commands = self._on_entry_timeout(data)
        elif event == "sl_confirmed":
            commands = self._on_sl_confirmed(data)
        elif event == "sl_filled":
            commands = self._on_sl_filled(data)
        elif event == "tp_filled":
            commands = self._on_tp_filled(data)
        elif event == "time_stop":
            commands = self._on_time_stop(data)
        elif event == "bar_tick":
            commands = self._on_bar_tick(data)
        elif event == "kill_switch":
            commands = self._on_kill_switch(data)
        else:
            logger.warning(f"[SM:{self.symbol}] Unknown event: {event}")

        return commands

    # ── Event Handlers ──────────────────────────────────────

    def _on_new_signal(self, data: dict) -> list[Command]:
        if self.state != State.IDLE:
            logger.debug(f"[SM:{self.symbol}] Ignoring new_signal in {self.state}")
            return []

        self._transition(State.ENTRY_PENDING, "new_signal", data.get("order_id"))

        self.ctx = PositionContext(
            entry_order_id=data["order_id"],
            side=data["side"],
            entry_qty=data["qty"],
            sl_price=data["sl"],
            tp_price=data["tp"],
            ttl_bars=data.get("ttl", 6),
        )

        return [Command(
            type=CmdType.PLACE_ENTRY,
            symbol=self.symbol,
            params={
                "side": data["side"],
                "qty": data["qty"],
                "price": data["price"],
                "order_id": data["order_id"],
            },
        )]

    def _on_entry_filled(self, data: dict) -> list[Command]:
        if self.state != State.ENTRY_PENDING:
            logger.debug(f"[SM:{self.symbol}] Ignoring entry_filled in {self.state}")
            return []

        # Idempotency: check if already have fill
        if self.ledger.has_fill(self.ctx.entry_order_id):
            logger.info(f"[SM:{self.symbol}] Duplicate entry_filled, ignoring")
            return []

        self.ctx.entry_price = data["fill_price"]
        self.ctx.entry_qty = data.get("fill_qty", self.ctx.entry_qty)

        self._transition(State.FILLED, "entry_filled", self.ctx.entry_order_id)

        # Immediately place protective orders
        sl_id = self.ctx.entry_order_id + "-sl"
        tp_id = self.ctx.entry_order_id + "-tp"
        self.ctx.sl_order_id = sl_id
        self.ctx.tp_order_id = tp_id

        exit_side = "SELL" if self.ctx.side == "BUY" else "BUY"

        return [
            Command(
                type=CmdType.PLACE_SL,
                symbol=self.symbol,
                params={
                    "side": exit_side,
                    "qty": self.ctx.entry_qty,
                    "stop_price": self.ctx.sl_price,
                    "order_id": sl_id,
                    "parent_id": self.ctx.entry_order_id,
                },
            ),
            Command(
                type=CmdType.PLACE_TP,
                symbol=self.symbol,
                params={
                    "side": exit_side,
                    "qty": self.ctx.entry_qty,
                    "tp_price": self.ctx.tp_price,
                    "order_id": tp_id,
                    "parent_id": self.ctx.entry_order_id,
                },
            ),
        ]

    def _on_entry_timeout(self, data: dict) -> list[Command]:
        if self.state != State.ENTRY_PENDING:
            return []

        self._transition(State.ENTRY_TIMEOUT, "entry_timeout", self.ctx.entry_order_id)
        cmds = [Command(
            type=CmdType.CANCEL_ORDER,
            symbol=self.symbol,
            params={"order_id": self.ctx.entry_order_id},
        )]
        # Auto-transition to IDLE
        self._transition(State.IDLE, "auto_reset")
        self.ctx = PositionContext()
        return cmds

    def _on_sl_confirmed(self, data: dict) -> list[Command]:
        if self.state != State.FILLED:
            return []
        self._transition(State.PROTECTED, "sl_order_confirmed", self.ctx.sl_order_id)
        return []

    def _on_sl_filled(self, data: dict) -> list[Command]:
        if self.state != State.PROTECTED:
            logger.debug(f"[SM:{self.symbol}] Ignoring sl_filled in {self.state}")
            return []

        self._transition(State.SL_HIT, "sl_filled", self.ctx.sl_order_id)

        cmds = [
            # Cancel the TP order (OCO behavior)
            Command(
                type=CmdType.CANCEL_ORDER,
                symbol=self.symbol,
                params={"order_id": self.ctx.tp_order_id},
            ),
            Command(
                type=CmdType.LOG_TRADE,
                symbol=self.symbol,
                params={
                    "exit_reason": "SL",
                    "entry_price": self.ctx.entry_price,
                    "exit_price": data.get("fill_price", self.ctx.sl_price),
                    "qty": self.ctx.entry_qty,
                    "side": self.ctx.side,
                    "fee": data.get("fee", 0),
                },
            ),
        ]

        # Reset to IDLE
        self._transition(State.IDLE, "auto_reset")
        self.ctx = PositionContext()
        return cmds

    def _on_tp_filled(self, data: dict) -> list[Command]:
        if self.state != State.PROTECTED:
            logger.debug(f"[SM:{self.symbol}] Ignoring tp_filled in {self.state}")
            return []

        self._transition(State.TP_HIT, "tp_filled", self.ctx.tp_order_id)

        cmds = [
            # Cancel the SL order (OCO behavior)
            Command(
                type=CmdType.CANCEL_ORDER,
                symbol=self.symbol,
                params={"order_id": self.ctx.sl_order_id},
            ),
            Command(
                type=CmdType.LOG_TRADE,
                symbol=self.symbol,
                params={
                    "exit_reason": "TP",
                    "entry_price": self.ctx.entry_price,
                    "exit_price": data.get("fill_price", self.ctx.tp_price),
                    "qty": self.ctx.entry_qty,
                    "side": self.ctx.side,
                    "fee": data.get("fee", 0),
                },
            ),
        ]

        self._transition(State.IDLE, "auto_reset")
        self.ctx = PositionContext()
        return cmds

    def _on_time_stop(self, data: dict) -> list[Command]:
        if self.state != State.PROTECTED:
            return []

        self._transition(State.TIME_STOP, "ttl_expired", self.ctx.entry_order_id)

        exit_side = "SELL" if self.ctx.side == "BUY" else "BUY"
        cmds = [
            # Cancel existing SL and TP
            Command(
                type=CmdType.CANCEL_PROTECTIVES,
                symbol=self.symbol,
                params={
                    "sl_order_id": self.ctx.sl_order_id,
                    "tp_order_id": self.ctx.tp_order_id,
                },
            ),
            # Market close
            Command(
                type=CmdType.MARKET_CLOSE,
                symbol=self.symbol,
                params={
                    "side": exit_side,
                    "qty": self.ctx.entry_qty,
                    "order_id": self.ctx.entry_order_id + "-ts",
                    "reason": "TIME_STOP",
                },
            ),
        ]

        self._transition(State.IDLE, "auto_reset")
        self.ctx = PositionContext()
        return cmds

    def _on_bar_tick(self, data: dict) -> list[Command]:
        if self.state not in (State.FILLED, State.PROTECTED):
            return []

        self.ctx.bars_held += 1
        if self.ctx.bars_held >= self.ctx.ttl_bars:
            return self._on_time_stop(data)
        return []

    def _on_kill_switch(self, data: dict) -> list[Command]:
        if self.state not in (State.FILLED, State.PROTECTED, State.ENTRY_PENDING):
            return []

        if self.state == State.ENTRY_PENDING:
            return self._on_entry_timeout(data)

        self._transition(State.EMERGENCY_CLOSE, "kill_switch", self.ctx.entry_order_id)

        exit_side = "SELL" if self.ctx.side == "BUY" else "BUY"
        cmds = [
            Command(
                type=CmdType.CANCEL_PROTECTIVES,
                symbol=self.symbol,
                params={
                    "sl_order_id": self.ctx.sl_order_id,
                    "tp_order_id": self.ctx.tp_order_id,
                },
            ),
            Command(
                type=CmdType.MARKET_CLOSE,
                symbol=self.symbol,
                params={
                    "side": exit_side,
                    "qty": self.ctx.entry_qty,
                    "order_id": self.ctx.entry_order_id + "-em",
                    "reason": "EMERGENCY",
                },
            ),
        ]

        self._transition(State.IDLE, "auto_reset")
        self.ctx = PositionContext()
        return cmds

    # ── Internal ────────────────────────────────────────────

    def _transition(
        self,
        to: State,
        trigger: str,
        order_link_id: str | None = None,
    ) -> None:
        from_state = self.state

        # Auto-reset transitions bypass validation
        if trigger != "auto_reset":
            valid = _VALID_TRANSITIONS.get(from_state, set())
            if to not in valid:
                logger.error(
                    f"[SM:{self.symbol}] Invalid transition "
                    f"{from_state.value} → {to.value} (trigger={trigger})"
                )
                return

        self.state = to
        self._event_log.append(f"{from_state.value}→{to.value}:{trigger}")

        # Persist to ledger
        self.ledger.log_transition(
            symbol=self.symbol,
            from_state=from_state.value,
            to_state=to.value,
            trigger=trigger,
            order_link_id=order_link_id,
        )

        logger.info(
            f"[SM:{self.symbol}] {from_state.value} → {to.value} "
            f"(trigger={trigger})"
        )

    @property
    def is_idle(self) -> bool:
        return self.state == State.IDLE

    @property
    def has_position(self) -> bool:
        return self.state in (State.FILLED, State.PROTECTED)
