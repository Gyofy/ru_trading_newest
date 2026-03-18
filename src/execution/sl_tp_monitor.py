"""Real-time SL/TP/TTL monitoring via background polling.

Solves: the v4.1 bot only checked SL/TP every 2 hours.
This module polls every 30s (configurable) and closes positions
when barriers are breached.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

logger = logging.getLogger("live_bot.monitor")


class SlTpMonitor:
    """Background task that polls SL/TP/TTL at high frequency."""

    def __init__(
        self,
        exchange,
        pos_manager,
        close_callback,
        partial_tp_callback=None,
        poll_seconds: float = 30.0,
        bar_minutes: int = 240,
    ):
        self.exchange = exchange
        self.pos_manager = pos_manager
        self._close = close_callback          # async def close(coin, reason, price)
        self._partial_tp = partial_tp_callback  # async def partial_tp(coin, stage, price)
        self.poll_seconds = poll_seconds
        self.bar_minutes = bar_minutes
        self._task: asyncio.Task | None = None
        self._running = False

    def start(self) -> None:
        """Start the background monitor."""
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.ensure_future(self._loop())
        logger.info(f"[Monitor] Started (poll every {self.poll_seconds}s)")

    async def stop(self) -> None:
        """Stop the background monitor."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("[Monitor] Stopped")

    async def _loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                await self._check_all_positions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Monitor] Error: {e}")

            try:
                await asyncio.sleep(self.poll_seconds)
            except asyncio.CancelledError:
                break

    async def _check_all_positions(self) -> None:
        """Check SL/TP/TTL for all open positions."""
        positions = self.pos_manager.all_positions()
        if not positions:
            return

        for pos in positions:
            # Re-check before async yield — position may have been closed already
            if not self.pos_manager.has_position(pos.coin):
                continue
            try:
                ticker = await self.exchange.fetch_ticker(pos.coin)
                current_price = ticker["last"]
            except Exception as e:
                logger.warning(f"[Monitor] {pos.coin}: price fetch failed: {e}")
                continue

            # Re-check after await — close callback may have run during fetch
            if not self.pos_manager.has_position(pos.coin):
                continue

            # Track running high/low for MFE/MAE
            pos.update_extremes(current_price)

            # ── Check Stop Loss ────────────────────────────
            if pos.side == "BUY" and current_price <= pos.sl_price:
                logger.info(
                    f"[Monitor] {pos.coin} SL HIT: {current_price:.4f} <= {pos.sl_price:.4f}"
                )
                await self._close(pos.coin, "SL_HIT", current_price)
                continue

            if pos.side == "SELL" and current_price >= pos.sl_price:
                logger.info(
                    f"[Monitor] {pos.coin} SL HIT: {current_price:.4f} >= {pos.sl_price:.4f}"
                )
                await self._close(pos.coin, "SL_HIT", current_price)
                continue

            # ── Check Take Profit (3단계 분할 or 단일) ───────
            if pos.partial_tp_enabled and self._partial_tp:
                await self._check_partial_tp(pos, current_price)
                # TP3 후 포지션 제거됐으면 continue
                if not self.pos_manager.has_position(pos.coin):
                    continue
            else:
                # 하위호환: 단일 TP
                tp_hit = (pos.side == "BUY" and current_price >= pos.tp_price) or \
                         (pos.side == "SELL" and current_price <= pos.tp_price)
                if tp_hit:
                    logger.info(
                        f"[Monitor] {pos.coin} TP HIT: {current_price:.4f}"
                    )
                    await self._close(pos.coin, "TP_HIT", current_price)
                    continue

            # ── Check TTL (time-based) ─────────────────────
            if pos.ttl_bars > 0 and pos.bars_held >= pos.ttl_bars:
                logger.info(
                    f"[Monitor] {pos.coin} TTL EXPIRED: {pos.bars_held}/{pos.ttl_bars} bars"
                )
                await self._close(pos.coin, "TIME_STOP", current_price)
                continue

    async def _check_partial_tp(self, pos, current_price: float) -> None:
        """3단계 분할 익절 체크."""
        is_buy = pos.side == "BUY"

        # TP1 체크
        if not pos.tp1_hit:
            tp1_triggered = (is_buy and current_price >= pos.tp1_price) or \
                            (not is_buy and current_price <= pos.tp1_price)
            if tp1_triggered:
                logger.info(
                    f"[Monitor] {pos.coin} TP1 HIT @ {current_price:.4f} "
                    f"(tp1={pos.tp1_price:.4f}) — 33% 청산, SL→BEP"
                )
                await self._partial_tp(pos.coin, 1, current_price)
                return

        # TP2 체크
        if pos.tp1_hit and not pos.tp2_hit:
            tp2_triggered = (is_buy and current_price >= pos.tp2_price) or \
                            (not is_buy and current_price <= pos.tp2_price)
            if tp2_triggered:
                logger.info(
                    f"[Monitor] {pos.coin} TP2 HIT @ {current_price:.4f} "
                    f"(tp2={pos.tp2_price:.4f}) — 33% 청산, SL→TP1"
                )
                await self._partial_tp(pos.coin, 2, current_price)
                return

        # TP3 체크 (전량 청산)
        if pos.tp1_hit and pos.tp2_hit:
            tp3_triggered = (is_buy and current_price >= pos.tp3_price) or \
                            (not is_buy and current_price <= pos.tp3_price)
            if tp3_triggered:
                logger.info(
                    f"[Monitor] {pos.coin} TP3 HIT @ {current_price:.4f} "
                    f"(tp3={pos.tp3_price:.4f}) — 잔여 전량 청산"
                )
                await self._close(pos.coin, "TP3_HIT", current_price)

    def increment_all_bars(self) -> None:
        """Called once per 2h cycle to increment bars_held."""
        for pos in self.pos_manager.all_positions():
            pos.bars_held += 1
