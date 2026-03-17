"""Live Engine — 메인 오케스트레이터.

on_bar_close() 루프:
  FeatureStore → Model → RiskEngine → ExchangeAdapter → StateMachine

상태 머신 기반, polling 아닌 이벤트 드리븐.
프로세스 재시작 시 SQLite에서 복구.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Optional

import numpy as np

from src.execution.order_ledger import OrderLedger
from src.execution.state_machine import PositionStateMachine, State, CmdType, Command
from src.execution.risk_engine import RiskEngine, RiskConfig, PreTradeCheck
from src.execution.exchange_adapter import ExchangeAdapter
from src.execution.feature_store import FeatureStore
from src.models.model_store import load_artifact

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("C:/Users/RJ/Desktop/CLAUDE_CRYPTO_AGENT")
HEARTBEAT_PATH = PROJECT_ROOT / "data" / "heartbeat.json"


class LiveEngineConfig:
    """LiveEngine 설정."""

    def __init__(self, settings: dict | None = None):
        s = settings or {}
        self.mode: str = s.get("mode", "demo")
        self.symbols: list[str] = s.get("symbols", ["DOT", "DOGE", "BTC", "XRP", "ADA"])

        # Timing
        self.bar_minutes: int = s.get("bar_minutes", 240)
        self.ttl_bars: int = s.get("ttl_bars", 6)
        self.entry_timeout_sec: float = s.get("entry_timeout_seconds", 20)
        self.poll_interval_sec: float = s.get("poll_interval_seconds", 30)
        self.no_trade_after_bar_sec: int = s.get("no_trade_after_bar_seconds", 300)

        # Barriers (학습과 동일해야 함 — model artifact에서 override)
        self.default_k_upper: float = s.get("k_upper", 1.5)
        self.default_k_lower: float = s.get("k_lower", 1.5)
        self.min_barrier_pct: float = s.get("min_barrier_pct", 0.002)
        self.atr_period: int = s.get("atr_period", 14)

        # Paths
        self.model_dir: str = s.get("model_artifact_dir", "data/models/production")
        self.ledger_db: str = s.get("ledger_db", "data/ledger.db")

        # API
        self.api_key: str = s.get("api_key", "")
        self.api_secret: str = s.get("api_secret", "")


class LiveEngine:
    """메인 트레이딩 엔진.

    Usage:
        engine = LiveEngine(config)
        await engine.start()
        await engine.run_forever()  # blocks
    """

    def __init__(self, config: LiveEngineConfig | None = None):
        self.config = config or LiveEngineConfig()

        # Components
        self.ledger = OrderLedger(
            PROJECT_ROOT / self.config.ledger_db,
        )
        self.risk_engine = RiskEngine(
            config=RiskConfig(),
            ledger=self.ledger,
        )
        self.exchange = ExchangeAdapter(
            mode=self.config.mode,
            api_key=self.config.api_key,
            secret=self.config.api_secret,
        )
        self.feature_store = FeatureStore()

        # Per-symbol state machines
        self.machines: dict[str, PositionStateMachine] = {}

        # Model artifacts per symbol
        self._models: dict[str, dict] = {}

        # Engine state
        self._running = False
        self._last_bar_check: dict[str, datetime] = {}
        self._last_daily_reset: date | None = None

    async def start(self) -> None:
        """Initialize all components and restore state."""
        logger.info(f"LiveEngine starting (mode={self.config.mode})")

        # 1. Exchange initialization
        await self.exchange.initialize()

        # 2. Load models
        self._load_models()

        # 3. Get initial equity
        balance = await self.exchange.fetch_balance()
        equity = balance.get("total", 0)
        self.risk_engine.set_initial_equity(equity)
        logger.info(f"Initial equity: {equity:.2f} USDT")

        # 4. Create state machines + restore from ledger
        for symbol in self.config.symbols:
            sm = PositionStateMachine(symbol, self.ledger)
            sm.restore()
            self.machines[symbol] = sm
            logger.info(f"[{symbol}] State: {sm.state.value}")

        self._running = True
        logger.info("LiveEngine started successfully")

    def _load_models(self) -> None:
        """Load model artifacts for active symbols."""
        model_dir = PROJECT_ROOT / self.config.model_dir
        for symbol in self.config.symbols:
            try:
                artifact = load_artifact(symbol, model_dir)
                self._models[symbol] = artifact

                # Set feature_store columns from first loaded model
                if not self.feature_store.feature_cols:
                    self.feature_store.feature_cols = artifact["feature_cols"]

                logger.info(
                    f"[{symbol}] Model loaded: "
                    f"{len(artifact['feature_cols'])} features, "
                    f"k={artifact['params'].get('k_upper')}/{artifact['params'].get('k_lower')}"
                )
            except FileNotFoundError:
                logger.warning(f"[{symbol}] No model artifact, skipping")
            except Exception as e:
                logger.error(f"[{symbol}] Model load failed: {e}")

    # ── Main Loop ───────────────────────────────────────────

    async def run_forever(self) -> None:
        """메인 루프: bar completion 체크 + 주문 상태 폴링."""
        logger.info("Entering main loop...")
        while self._running:
            try:
                now = datetime.now(timezone.utc)

                # Daily reset
                self._check_daily_reset(now)

                # Write heartbeat
                self._write_heartbeat(now)

                # Check each symbol
                for symbol in self.config.symbols:
                    if symbol not in self._models:
                        continue

                    sm = self.machines[symbol]

                    # Bar completion check
                    if sm.is_idle and self.feature_store.is_bar_complete(symbol, now):
                        await self._on_bar_close(symbol)

                    # Bar tick for positions
                    if sm.has_position and self.feature_store.is_bar_complete(symbol, now):
                        commands = sm.on_event("bar_tick")
                        await self._execute_commands(commands)

                    # Poll pending entries
                    if sm.state == State.ENTRY_PENDING:
                        await self._poll_entry(symbol)

                # Sleep
                await asyncio.sleep(self.config.poll_interval_sec)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt — shutting down")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)
                await asyncio.sleep(10)

        await self.shutdown()

    async def _on_bar_close(self, symbol: str) -> None:
        """4h 바 종가 확정 → 시그널 생성 → 진입 시도."""
        sm = self.machines[symbol]
        artifact = self._models[symbol]
        params = artifact["params"]

        try:
            # 1. Feature computation (학습과 동일)
            X, atr, last_close = self.feature_store.update_and_compute(symbol)

            if atr <= 0 or last_close <= 0:
                logger.warning(f"[{symbol}] Invalid ATR={atr} or close={last_close}")
                return

            # 2. Model prediction — 2-stage binary
            stage1 = artifact["stage1_ensemble"]
            stage2 = artifact["stage2_ensemble"]

            # Stage 1: Trade/NoTrade
            X_2d = X.reshape(1, -1)
            s1_probs = stage1.predict_proba(X_2d)
            p_trade = float(s1_probs[0, 1])  # P(trade)

            # Threshold filter (NO probability-based sizing)
            threshold = params.get("stage1_threshold", 0.40)
            if p_trade < threshold:
                logger.debug(f"[{symbol}] p_trade={p_trade:.3f} < {threshold}, skip")
                return

            # Stage 2: Long/Short
            s2_probs = stage2.predict_proba(X_2d)
            p_long = float(s2_probs[0, 1])   # P(UP)
            side = "BUY" if p_long > 0.5 else "SELL"

            # 3. Barrier computation (학습과 동일한 k, ATR)
            k_upper = params.get("k_upper", self.config.default_k_upper)
            k_lower = params.get("k_lower", self.config.default_k_lower)

            sl_price, tp_price = RiskEngine.compute_barriers(
                entry_price=last_close,
                atr=atr,
                side=side,
                k_upper=k_upper,
                k_lower=k_lower,
                min_barrier_pct=self.config.min_barrier_pct,
            )

            # 4. Pre-trade gate
            balance = await self.exchange.fetch_balance()
            equity = balance.get("total", 0)

            ticker = await self.exchange.fetch_ticker(symbol)
            spread_bps = ticker.get("spread_bps", 0)

            funding = await self.exchange.fetch_funding_rate(symbol)

            check = self.risk_engine.pre_trade_gate(
                symbol=symbol,
                side=side,
                entry_price=last_close,
                sl_price=sl_price,
                equity_usdt=equity,
                p_trade=p_trade,
                atr=atr,
                funding_rate=funding,
                spread_bps=spread_bps,
            )

            if not check.approved:
                logger.info(f"[{symbol}] Gate rejected: {check.reason}")
                return

            if check.warnings:
                for w in check.warnings:
                    logger.warning(f"[{symbol}] Warning: {w}")

            # 5. Precision rounding
            qty = check.sizing.qty
            entry_price = await self.exchange.get_maker_entry_price(symbol, side)

            qty, entry_price, sl_price, tp_price = self.exchange.round_all(
                symbol, qty, entry_price, sl_price, tp_price,
            )

            # Min qty check
            min_qty = self.exchange.get_min_qty(symbol)
            if qty < min_qty:
                logger.info(f"[{symbol}] qty {qty} < min {min_qty}, skip")
                return

            # 6. Generate order ID
            order_id = ExchangeAdapter.make_order_id(symbol, side)

            # 7. State machine → entry command
            commands = sm.on_event("new_signal", {
                "side": side,
                "qty": qty,
                "price": entry_price,
                "sl": sl_price,
                "tp": tp_price,
                "ttl": self.config.ttl_bars,
                "order_id": order_id,
            })

            # 8. Execute commands
            await self._execute_commands(commands)

            logger.info(
                f"[{symbol}] Signal: {side} p_trade={p_trade:.3f} p_long={p_long:.3f} "
                f"entry={entry_price} SL={sl_price} TP={tp_price} "
                f"qty={qty} risk=${check.sizing.risk_usdt:.2f}"
            )

        except Exception as e:
            logger.error(f"[{symbol}] on_bar_close error: {e}", exc_info=True)

    # ── Command Execution ───────────────────────────────────

    async def _execute_commands(self, commands: list[Command]) -> None:
        """State machine commands → exchange operations."""
        for cmd in commands:
            try:
                if cmd.type == CmdType.PLACE_ENTRY:
                    await self._exec_place_entry(cmd)
                elif cmd.type == CmdType.PLACE_SL:
                    await self._exec_place_sl(cmd)
                elif cmd.type == CmdType.PLACE_TP:
                    await self._exec_place_tp(cmd)
                elif cmd.type == CmdType.CANCEL_ORDER:
                    await self._exec_cancel(cmd)
                elif cmd.type == CmdType.CANCEL_PROTECTIVES:
                    await self._exec_cancel_protectives(cmd)
                elif cmd.type == CmdType.MARKET_CLOSE:
                    await self._exec_market_close(cmd)
                elif cmd.type == CmdType.LOG_TRADE:
                    self._exec_log_trade(cmd)
                else:
                    logger.warning(f"Unknown command: {cmd.type}")
            except Exception as e:
                logger.error(f"Command execution failed: {cmd.type} {cmd.symbol}: {e}")

    async def _exec_place_entry(self, cmd: Command) -> None:
        p = cmd.params
        sym = cmd.symbol

        # Record in ledger
        self.ledger.insert_order(
            order_link_id=p["order_id"],
            symbol=sym,
            side=p["side"],
            order_type="LIMIT",
            purpose="entry",
            qty=p["qty"],
            price=p["price"],
        )

        # Place post-only entry
        result = await self.exchange.place_post_only_entry(
            symbol=sym,
            side=p["side"],
            qty=p["qty"],
            price=p["price"],
            order_link_id=p["order_id"],
        )

        if result["success"]:
            self.ledger.update_order_status(
                p["order_id"], "OPEN",
                exchange_order_id=result.get("exchange_order_id"),
            )

            # Wait for fill or cancel
            fill = await self.exchange.wait_fill_or_cancel(
                symbol=sym,
                order_link_id=p["order_id"],
                ttl_sec=self.config.entry_timeout_sec,
                max_reprices=1,
            )

            sm = self.machines[sym]
            if fill:
                # Record fill
                self.ledger.insert_fill(
                    order_link_id=p["order_id"],
                    fill_price=fill["fill_price"],
                    fill_qty=fill["fill_qty"],
                    fee=fill.get("fee", 0),
                )
                self.ledger.update_order_status(p["order_id"], "FILLED")

                # Trigger state machine
                commands = sm.on_event("entry_filled", fill)
                await self._execute_commands(commands)
            else:
                self.ledger.update_order_status(p["order_id"], "CANCELLED")
                sm.on_event("entry_timeout")
        else:
            self.ledger.update_order_status(p["order_id"], "REJECTED")
            self.machines[sym].on_event("entry_timeout")

    async def _exec_place_sl(self, cmd: Command) -> None:
        p = cmd.params

        self.ledger.insert_order(
            order_link_id=p["order_id"],
            symbol=cmd.symbol,
            side=p["side"],
            order_type="CONDITIONAL",
            purpose="stop_loss",
            qty=p["qty"],
            stop_trigger=p["stop_price"],
            parent_id=p.get("parent_id"),
        )

        result = await self.exchange.place_protective_stop(
            symbol=cmd.symbol,
            side=p["side"],
            qty=p["qty"],
            stop_price=p["stop_price"],
            order_link_id=p["order_id"],
        )

        if result["success"]:
            self.ledger.update_order_status(
                p["order_id"], "CONDITIONAL",
                exchange_order_id=result.get("exchange_order_id"),
            )
            # Confirm SL placement → transition to PROTECTED
            sm = self.machines[cmd.symbol]
            sm.on_event("sl_confirmed")
        else:
            logger.error(f"[{cmd.symbol}] SL placement FAILED: {result.get('error')}")

    async def _exec_place_tp(self, cmd: Command) -> None:
        p = cmd.params

        self.ledger.insert_order(
            order_link_id=p["order_id"],
            symbol=cmd.symbol,
            side=p["side"],
            order_type="CONDITIONAL",
            purpose="take_profit",
            qty=p["qty"],
            stop_trigger=p["tp_price"],
            parent_id=p.get("parent_id"),
        )

        result = await self.exchange.place_take_profit(
            symbol=cmd.symbol,
            side=p["side"],
            qty=p["qty"],
            tp_price=p["tp_price"],
            order_link_id=p["order_id"],
        )

        if result["success"]:
            self.ledger.update_order_status(
                p["order_id"], "CONDITIONAL",
                exchange_order_id=result.get("exchange_order_id"),
            )

    async def _exec_cancel(self, cmd: Command) -> None:
        p = cmd.params
        oid = p.get("order_id", "")
        await self.exchange.cancel_order(cmd.symbol, order_link_id=oid)
        self.ledger.update_order_status(oid, "CANCELLED")

    async def _exec_cancel_protectives(self, cmd: Command) -> None:
        p = cmd.params
        for oid in [p.get("sl_order_id"), p.get("tp_order_id")]:
            if oid:
                await self.exchange.cancel_order(cmd.symbol, order_link_id=oid)
                self.ledger.update_order_status(oid, "CANCELLED")

    async def _exec_market_close(self, cmd: Command) -> None:
        p = cmd.params
        close_id = p.get("order_id", f"{cmd.symbol}-mktclose")

        self.ledger.insert_order(
            order_link_id=close_id,
            symbol=cmd.symbol,
            side=p["side"],
            order_type="MARKET",
            purpose=p.get("reason", "market_close").lower(),
            qty=p["qty"],
        )

        result = await self.exchange.market_close(
            symbol=cmd.symbol,
            side=p["side"],
            qty=p["qty"],
            order_link_id=close_id,
        )

        if result["success"]:
            self.ledger.update_order_status(close_id, "FILLED")
            self.ledger.insert_fill(
                order_link_id=close_id,
                fill_price=result.get("fill_price", 0),
                fill_qty=p["qty"],
            )

    def _exec_log_trade(self, cmd: Command) -> None:
        """P&L 기록."""
        p = cmd.params
        entry = p.get("entry_price", 0)
        exit_ = p.get("exit_price", 0)
        qty = p.get("qty", 0)
        side = p.get("side", "BUY")
        fee = p.get("fee", 0)

        if side == "BUY":
            pnl = (exit_ - entry) * qty - fee
        else:
            pnl = (entry - exit_) * qty - fee

        self.ledger.record_pnl(
            symbol=cmd.symbol,
            realized_pnl=pnl,
            fees=fee,
        )

        logger.info(
            f"[TRADE] {cmd.symbol} {p.get('exit_reason', '?')} "
            f"entry={entry:.4f} exit={exit_:.4f} "
            f"pnl={pnl:+.4f} USDT"
        )

    # ── Polling ─────────────────────────────────────────────

    async def _poll_entry(self, symbol: str) -> None:
        """Check if pending entry has been filled or timed out."""
        sm = self.machines[symbol]
        if sm.state != State.ENTRY_PENDING:
            return

        # Entry timeout is handled by wait_fill_or_cancel in _exec_place_entry
        # This is a safety net for edge cases
        pass

    # ── Housekeeping ────────────────────────────────────────

    def _check_daily_reset(self, now: datetime) -> None:
        today = now.date()
        if self._last_daily_reset != today:
            self._last_daily_reset = today
            # Reset daily drawdown tracking
            asyncio.ensure_future(self._reset_daily_equity())

    async def _reset_daily_equity(self) -> None:
        try:
            balance = await self.exchange.fetch_balance()
            equity = balance.get("total", 0)
            self.risk_engine.reset_daily(equity)
            logger.info(f"Daily equity reset: {equity:.2f} USDT")
        except Exception as e:
            logger.error(f"Daily reset failed: {e}")

    def _write_heartbeat(self, now: datetime) -> None:
        """Write heartbeat for watchdog."""
        try:
            HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "ts": now.isoformat(),
                "mode": self.config.mode,
                "symbols": self.config.symbols,
                "states": {
                    sym: sm.state.value
                    for sym, sm in self.machines.items()
                },
                "kill_switch": self.risk_engine.is_killed,
            }
            with open(HEARTBEAT_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Non-critical

    # ── Emergency ───────────────────────────────────────────

    async def emergency_close_all(self, reason: str = "manual") -> None:
        """모든 포지션 긴급 청산."""
        logger.warning(f"EMERGENCY CLOSE ALL: {reason}")
        self.risk_engine.activate_kill_switch(reason)

        for symbol, sm in self.machines.items():
            if sm.has_position:
                commands = sm.on_event("kill_switch", {"reason": reason})
                await self._execute_commands(commands)

    # ── Shutdown ────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("LiveEngine shutting down...")
        self._running = False
        await self.exchange.close()
        self.ledger.close()
        logger.info("LiveEngine stopped")
