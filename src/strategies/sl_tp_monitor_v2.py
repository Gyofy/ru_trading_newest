"""SlTpMonitorV2 — trailing stop + strategy-aware position monitoring.

Extends the base SlTpMonitor with:
  - Trailing stop support (SL follows favorable price movement)
  - Strategy-tagged close callbacks
  - Close-based SL (uses ticker 'last', not intrabar high/low)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from src.strategies.multi_position_manager import MultiPositionManager

logger = logging.getLogger("monitor_v2")


class SlTpMonitorV2:
    """Background monitor with trailing stop for multi-strategy bot."""

    def __init__(
        self,
        exchange,
        pos_manager: MultiPositionManager,
        close_callback,
        poll_seconds: float = 15.0,
        paper_mode: bool = True,
    ):
        self.exchange = exchange
        self.pos_manager = pos_manager
        self._close = close_callback  # async def close(strategy, coin, reason, price)
        self.poll_seconds = poll_seconds
        self.paper_mode = paper_mode
        self._task: asyncio.Task | None = None
        self._running = False

    def start(self) -> None:
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.ensure_future(self._loop())
        logger.info(f"[MonitorV2] Started (poll every {self.poll_seconds}s)")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("[MonitorV2] Stopped")

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._check_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[MonitorV2] Error: {e}", exc_info=True)

            try:
                await asyncio.sleep(self.poll_seconds)
            except asyncio.CancelledError:
                break

    async def _check_all(self) -> None:
        """Check all positions for SL/TP/TTL/trailing."""
        # Snapshot both the list and the key mapping before any awaits
        # to avoid RuntimeError from dict mutation during async callbacks.
        pos_snapshot = list(self.pos_manager.positions.items())
        if not pos_snapshot:
            return

        for matched_key, pos in pos_snapshot:
            # Verify position still exists after any prior await in this loop
            if matched_key not in self.pos_manager.positions:
                continue
            strategy, coin = self.pos_manager._parse_key(matched_key)

            try:
                ticker = await self.exchange.fetch_ticker(coin)
                price = ticker["last"]
            except Exception as e:
                logger.warning(f"[MonitorV2] {coin}: ticker failed: {e}")
                continue

            # Position may have been closed during await
            if not self.pos_manager.has_position(strategy, coin):
                continue

            # Track extremes
            pos.update_extremes(price)

            # ── Trailing stop update ──────────────────────
            if pos.trailing_sl and pos.trail_distance > 0:
                await self._update_trailing_sl(pos, coin, price)

            # ── SL check (close-based: uses 'last' price) ──
            sl_hit = (
                (pos.side == "BUY" and price <= pos.sl_price)
                or (pos.side == "SELL" and price >= pos.sl_price)
            )
            if sl_hit:
                logger.info(
                    f"[MonitorV2] {coin} SL HIT: {price:.4f} vs {pos.sl_price:.4f}"
                )
                # Cancel both exchange SL and TP orders before closing
                # (prevents double-close race: exchange SL + software market_close)
                await self._cancel_exchange_order(
                    pos, coin, "sl_exchange_id", "sl_order_id", "SL",
                )
                await self._cancel_exchange_order(
                    pos, coin, "tp_exchange_id", "tp_order_id", "TP",
                )
                await self._close(strategy, coin, "SL_HIT", price)
                continue

            # ── TP check ──────────────────────────────────
            if pos.tp_price > 0 and not pos.trailing_sl:
                tp_hit = (
                    (pos.side == "BUY" and price >= pos.tp_price)
                    or (pos.side == "SELL" and price <= pos.tp_price)
                )
                if tp_hit:
                    logger.info(
                        f"[MonitorV2] {coin} TP HIT: {price:.4f} vs {pos.tp_price:.4f}"
                    )
                    # Cancel exchange SL order (TP handled by exchange or already hit)
                    await self._cancel_exchange_order(
                        pos, coin, "sl_exchange_id", "sl_order_id", "SL",
                    )
                    await self._close(strategy, coin, "TP_HIT", price)
                    continue

            # ── TTL check ─────────────────────────────────
            if pos.ttl_bars > 0 and pos.bars_held >= pos.ttl_bars:
                logger.info(
                    f"[MonitorV2] {coin} TTL: {pos.bars_held}/{pos.ttl_bars}"
                )
                # Cancel both exchange SL and TP orders
                await self._cancel_exchange_order(
                    pos, coin, "sl_exchange_id", "sl_order_id", "SL",
                )
                await self._cancel_exchange_order(
                    pos, coin, "tp_exchange_id", "tp_order_id", "TP",
                )
                await self._close(strategy, coin, "TIME_STOP", price)
                continue

    # Minimum SL move (% of price) before syncing to exchange.
    # Prevents excessive API calls on tiny price fluctuations.
    _MIN_SL_MOVE_PCT = 0.002  # 0.2% — ~2 API calls saved per position per minute

    async def _update_trailing_sl(self, pos, coin: str, price: float) -> None:
        """Move SL in favorable direction only. Updates exchange SL too.

        Exchange SL sync is throttled: only fires when the accumulated
        SL movement exceeds _MIN_SL_MOVE_PCT of entry price, preventing
        excessive cancel+place API calls during small price oscillations.
        """
        moved = False
        if pos.side == "BUY":
            new_sl = price - pos.trail_distance
            if new_sl > pos.sl_price:
                pos.sl_price = new_sl
                moved = True
        else:
            new_sl = price + pos.trail_distance
            if new_sl < pos.sl_price:
                pos.sl_price = new_sl
                moved = True

        if moved:
            pos.sl_tighten_count = getattr(pos, "sl_tighten_count", 0) + 1

        if not moved or self.paper_mode:
            return

        # Throttle exchange SL sync: only update when SL moved significantly
        last_synced = getattr(pos, "_last_exchange_sl", pos.entry_price)
        sl_move_pct = abs(pos.sl_price - last_synced) / pos.entry_price if pos.entry_price > 0 else 0

        if sl_move_pct >= self._MIN_SL_MOVE_PCT:
            await self._update_exchange_sl(pos, coin, pos.sl_price)
            pos._last_exchange_sl = pos.sl_price
            logger.debug(
                f"[Trail] {coin} exchange SL synced: {last_synced:.4f} → {pos.sl_price:.4f} "
                f"(move={sl_move_pct:.3%})"
            )

    async def _update_exchange_sl(self, pos, coin: str, new_sl: float) -> None:
        """Cancel old exchange SL and place new one at updated trailing level."""
        sl_side = "SELL" if pos.side == "BUY" else "BUY"

        # Cancel old exchange SL
        if hasattr(pos, "sl_exchange_id") and pos.sl_exchange_id:
            try:
                await self.exchange.cancel_order(
                    coin,
                    exchange_order_id=pos.sl_exchange_id,
                    order_link_id=pos.sl_order_id if pos.sl_order_id else None,
                )
            except Exception as e:
                logger.warning(f"[Trail] {coin} cancel old SL failed: {e}")

        # Place new exchange SL at updated price
        rounded_sl = self.exchange.round_price(coin, new_sl)
        new_oid = self.exchange.make_order_id(coin, sl_side, prefix="v8tsl")
        try:
            result = await self.exchange.place_protective_stop(
                symbol=coin, side=sl_side, qty=pos.current_qty,
                stop_price=rounded_sl, order_link_id=new_oid,
            )
            if result.get("success"):
                pos.sl_exchange_id = result.get("exchange_order_id", "")
                pos.sl_order_id = new_oid
                logger.debug(f"[Trail] {coin} exchange SL updated → {rounded_sl}")
            else:
                err_msg = str(result.get("error", ""))
                # -4509: no exchange position / -4130: duplicate GTE stop exists → software-only SL
                if "-4509" in err_msg or "-4130" in err_msg or "GTE" in err_msg:
                    logger.warning(
                        f"[Trail] {coin} no exchange position (-4509) — "
                        f"software SL only @ {new_sl:.4f}"
                    )
                    pos.sl_exchange_id = ""
                    pos.sl_order_id = ""
                else:
                    logger.warning(
                        f"[Trail] {coin} exchange SL FAILED: {err_msg} "
                        f"(software SL still active @ {new_sl:.4f})"
                    )
        except Exception as e:
            logger.warning(f"[Trail] {coin} exchange SL place failed: {e}")

    async def _cancel_exchange_order(
        self, pos, coin: str,
        exchange_id_attr: str, order_id_attr: str, label: str,
    ) -> None:
        """Cancel an exchange-side order (SL or TP) if it exists."""
        if self.paper_mode:
            return
        exchange_id = getattr(pos, exchange_id_attr, "")
        order_id = getattr(pos, order_id_attr, "")
        if not exchange_id and not order_id:
            return
        try:
            await self.exchange.cancel_order(
                coin,
                exchange_order_id=exchange_id if exchange_id else None,
                order_link_id=order_id if order_id else None,
            )
            logger.info(f"[MonitorV2] {coin} exchange {label} order cancelled")
        except Exception as e:
            logger.warning(f"[MonitorV2] {coin} cancel exchange {label} failed: {e}")

    def increment_bars(self) -> None:
        """Increment bars_held for all positions. Call once per smallest cycle."""
        for pos in self.pos_manager.all_positions():
            pos.bars_held += 1
