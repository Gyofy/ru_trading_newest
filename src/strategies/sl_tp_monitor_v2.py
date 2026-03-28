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

            # Track MFE timing (which bar the best price occurred)
            if pos.side == "BUY":
                if price >= getattr(pos, 'mfe_price', 0):
                    pos.mfe_price = price
                    pos.mfe_bar = pos.bars_held
            else:
                if pos.mfe_price == 0 or price <= pos.mfe_price:
                    pos.mfe_price = price
                    pos.mfe_bar = pos.bars_held

            # ── Ensure exchange SL/TP exist (register if missing) ─────────
            if not self.paper_mode:
                await self._ensure_exchange_orders(pos, coin)
            # Re-check position still alive after potential await
            if not self.pos_manager.has_position(strategy, coin):
                continue

            # ── Trailing stop update ──────────────────────
            if pos.trailing_sl and pos.trail_distance > 0:
                await self._update_trailing_sl(pos, coin, price)

            # ── F4: Early exit if zero MFE after 5 bars (direction completely wrong) ──
            # One-time check: fires on first poll cycle where bars_held >= 5
            if pos.bars_held >= 5 and not getattr(pos, '_zero_mfe_checked', False):
                setattr(pos, '_zero_mfe_checked', True)
                if pos.side == "BUY":
                    favorable = (pos.price_high - pos.entry_price) / pos.entry_price
                else:
                    favorable = (pos.entry_price - pos.price_low) / pos.entry_price
                if favorable < 0.0005:  # less than 0.05% favorable move in 5 minutes
                    logger.info(
                        f"[MonitorV2] {coin} EARLY EXIT: zero MFE after {pos.bars_held} bars "
                        f"(favorable={favorable:.4%})"
                    )
                    await self._cancel_exchange_order(
                        pos, coin, "sl_exchange_id", "sl_order_id", "SL",
                    )
                    await self._cancel_exchange_order(
                        pos, coin, "tp_exchange_id", "tp_order_id", "TP",
                    )
                    await self._close(strategy, coin, "EARLY_EXIT_NO_MFE", price)
                    continue

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

            # ── TP check (applies to ALL strategies, including trailing) ──
            if pos.tp_price > 0:
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

    async def _ensure_exchange_orders(self, pos, coin: str) -> None:
        """Register missing exchange SL/TP orders for a position.

        Called every poll cycle. Only places orders when sl/tp_exchange_id is
        empty, so it is effectively a no-op for healthy positions.
        Throttled by _ensure_attempt_count to avoid hammering on repeated failures.
        """
        sl_missing = not getattr(pos, "sl_exchange_id", "")
        tp_missing = not getattr(pos, "tp_exchange_id", "") and pos.tp_price > 0

        if not sl_missing and not tp_missing:
            return

        # Throttle: only retry every 3 poll cycles to avoid 429
        attempt_key = f"_ensure_fail_{coin}"
        fail_count = getattr(pos, attempt_key, 0)
        if fail_count > 0 and fail_count % 3 != 0:
            setattr(pos, attempt_key, fail_count + 1)
            return

        sl_side = "SELL" if pos.side == "BUY" else "BUY"
        qty = getattr(pos, "current_qty", pos.qty)

        if sl_missing and pos.sl_price > 0:
            rounded_sl = self.exchange.round_price(coin, pos.sl_price)
            sl_oid = self.exchange.make_order_id(coin, sl_side, prefix="v8mon_sl")
            try:
                result = await self.exchange.place_protective_stop(
                    symbol=coin, side=sl_side, qty=qty,
                    stop_price=rounded_sl, order_link_id=sl_oid,
                )
                if result.get("success"):
                    pos.sl_exchange_id = result.get("exchange_order_id", "")
                    pos.sl_order_id = sl_oid
                    logger.info(f"[MonitorV2] {coin} SL re-registered @ {rounded_sl}")
                    setattr(pos, attempt_key, 0)
                else:
                    err_str = str(result.get("error", ""))
                    # -4509: 거래소에 포지션 없음 → 고스트 포지션 → close callback으로 정리
                    if "-4509" in err_str or "GTE" in err_str:
                        strategy = self._pos_strategy(coin)
                        logger.warning(
                            f"[MonitorV2] {coin} no exchange position (-4509) "
                            f"— closing ghost position from tracker"
                        )
                        await self._close(strategy, coin, "GHOST_CLEANUP", pos.entry_price)
                        return
                    setattr(pos, attempt_key, fail_count + 1)
                    logger.warning(f"[MonitorV2] {coin} SL re-register failed: {result.get('error')}")
            except Exception as e:
                err_str = str(e)
                if "-4509" in err_str or "GTE" in err_str:
                    strategy = self._pos_strategy(coin)
                    logger.warning(
                        f"[MonitorV2] {coin} no exchange position (-4509) "
                        f"— closing ghost position from tracker"
                    )
                    await self._close(strategy, coin, "GHOST_CLEANUP", pos.entry_price)
                    return
                setattr(pos, attempt_key, fail_count + 1)
                logger.warning(f"[MonitorV2] {coin} SL re-register error: {e}")

        if tp_missing and pos.tp_price > 0:
            rounded_tp = self.exchange.round_price(coin, pos.tp_price)
            tp_oid = self.exchange.make_order_id(coin, sl_side, prefix="v8mon_tp")
            try:
                result = await self.exchange.place_take_profit(
                    symbol=coin, side=sl_side, qty=qty,
                    tp_price=rounded_tp, order_link_id=tp_oid,
                )
                if result.get("success"):
                    pos.tp_exchange_id = result.get("exchange_order_id", "")
                    pos.tp_order_id = tp_oid
                    logger.info(f"[MonitorV2] {coin} TP re-registered @ {rounded_tp}")
                else:
                    logger.warning(f"[MonitorV2] {coin} TP re-register failed: {result.get('error')}")
            except Exception as e:
                logger.warning(f"[MonitorV2] {coin} TP re-register error: {e}")

        # Persist updated order IDs
        self.pos_manager._save()

    # Minimum SL move (% of price) before syncing to exchange.
    # Prevents excessive API calls on tiny price fluctuations.
    _MIN_SL_MOVE_PCT = 0.002  # 0.2% — ~2 API calls saved per position per minute

    # Phase thresholds
    # _BREAKEVEN_TRIGGER_PCT is now computed dynamically per position (fee + trail_dist)
    _TRAIL_START_ATR_MULT = 1.5     # MFE > 1.5×trail_distance → start tightening

    # Round-trip fee rate: taker×2 + slippage_entry + slippage_exit
    # 0.00055*2 + 0.0003 + 0.0005 = 0.0019 (0.19%)
    _ROUND_TRIP_FEE_RATE = 0.0019

    async def _update_trailing_sl(self, pos, coin: str, price: float) -> None:
        """3-phase trailing SL:

        Phase 1 (Initial): Keep original SL — survive initial noise.
        Phase 2 (Breakeven): Once price moves favorably by (fee + trail_dist),
                             move SL to entry + fee so exit is net-zero, not net-negative.
        Phase 3 (Trail): Once MFE exceeds trail_distance * _TRAIL_START_ATR_MULT,
                         start trailing with tight distance.
        """
        entry = pos.entry_price
        if entry <= 0:
            return

        old_sl = pos.sl_price

        # Dynamic breakeven trigger: price must move far enough that even if stopped
        # at breakeven SL + one trail distance back, we've covered the round-trip fee.
        trail_dist_pct = pos.trail_distance / entry if pos.trail_distance > 0 else 0.003
        breakeven_trigger = max(self._ROUND_TRIP_FEE_RATE + trail_dist_pct, 0.002)

        if pos.side == "BUY":
            favorable_pct = (price - entry) / entry
            # Phase 2: Breakeven — SL moves to entry + fee (true net-zero exit)
            if favorable_pct >= breakeven_trigger:
                breakeven_sl = entry * (1.0 + self._ROUND_TRIP_FEE_RATE)
                if breakeven_sl > pos.sl_price:
                    pos.sl_price = breakeven_sl

            # Phase 3: Trailing (only after significant move)
            if pos.trail_distance > 0:
                mfe_dist = price - entry
                if mfe_dist > pos.trail_distance * self._TRAIL_START_ATR_MULT:
                    new_sl = price - pos.trail_distance
                    if new_sl > pos.sl_price:
                        pos.sl_price = new_sl
        else:
            favorable_pct = (entry - price) / entry
            # Phase 2: Breakeven — SL moves to entry - fee (true net-zero exit for short)
            if favorable_pct >= breakeven_trigger:
                breakeven_sl = entry * (1.0 - self._ROUND_TRIP_FEE_RATE)
                if breakeven_sl < pos.sl_price:
                    pos.sl_price = breakeven_sl

            # Phase 3: Trailing
            if pos.trail_distance > 0:
                mfe_dist = entry - price
                if mfe_dist > pos.trail_distance * self._TRAIL_START_ATR_MULT:
                    new_sl = price + pos.trail_distance
                    if new_sl < pos.sl_price:
                        pos.sl_price = new_sl

        # Track tighten count (only when SL actually moved)
        if pos.sl_price != old_sl:
            pos.sl_tighten_count = getattr(pos, "sl_tighten_count", 0) + 1

        # Exchange SL sync (live mode only, throttled)
        if not self.paper_mode and pos.entry_price > 0:
            last_synced = getattr(pos, "_last_exchange_sl", pos.entry_price)
            sl_move_pct = abs(pos.sl_price - last_synced) / pos.entry_price
            if sl_move_pct >= self._MIN_SL_MOVE_PCT:
                await self._update_exchange_sl(pos, coin, pos.sl_price)
                pos._last_exchange_sl = pos.sl_price

    async def _update_exchange_sl(self, pos, coin: str, new_sl: float) -> None:
        """Place new exchange SL first, then cancel old one only after success."""
        sl_side = "SELL" if pos.side == "BUY" else "BUY"

        # 1. Place new exchange SL FIRST — retry up to 3 times
        rounded_sl = self.exchange.round_price(coin, new_sl)
        result = {"success": False}
        new_oid = None
        for attempt in range(3):
            if attempt > 0:
                await asyncio.sleep(0.5)
            new_oid = self.exchange.make_order_id(coin, sl_side, prefix=f"v8tsl{attempt}")
            try:
                result = await self.exchange.place_protective_stop(
                    symbol=coin, side=sl_side, qty=pos.current_qty,
                    stop_price=rounded_sl, order_link_id=new_oid,
                )
            except Exception as e:
                result = {"success": False, "error": str(e)}
            if result.get("success"):
                logger.debug(
                    f"[Trail] {coin} new exchange SL placed → {rounded_sl}"
                    + (f" (retry {attempt})" if attempt > 0 else "")
                )
                break

        if not result.get("success"):
            err_msg = str(result.get("error", ""))
            if "-4130" in err_msg or "GTE" in err_msg:
                # Duplicate SL — old one still active, no action needed
                logger.warning(
                    f"[Trail] {coin} duplicate SL order (-4130), old SL still active — skipping update"
                )
            elif "-4509" in err_msg:
                # Position doesn't exist on exchange — clear tracking
                logger.warning(
                    f"[Trail] {coin} position not on exchange (-4509), clearing SL tracking"
                )
                pos.sl_exchange_id = ""
                pos.sl_order_id = ""
            else:
                logger.error(
                    f"[SL_FAIL] [Trail] {coin} new SL placement failed 3x: {err_msg} "
                    f"— keeping OLD SL, software SL @ {new_sl:.4f}"
                )
            # Do NOT cancel old SL — position stays protected
            return

        # 2. New SL confirmed — NOW cancel the old one
        old_exchange_id = getattr(pos, "sl_exchange_id", "")
        old_order_id = getattr(pos, "sl_order_id", "")
        if old_exchange_id:
            try:
                await self.exchange.cancel_order(
                    coin,
                    exchange_order_id=old_exchange_id,
                    order_link_id=old_order_id if old_order_id else None,
                )
            except Exception as e:
                logger.warning(f"[Trail] {coin} cancel old SL failed (new SL already active): {e}")

        # 3. Update position tracking with new SL
        pos.sl_exchange_id = result.get("exchange_order_id", "")
        pos.sl_order_id = new_oid

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

    def _pos_strategy(self, coin: str) -> str:
        """coin에 해당하는 strategy 이름 반환. 매칭 안되면 빈 문자열."""
        for key in self.pos_manager.positions:
            strat, c = self.pos_manager._parse_key(key)
            if c == coin:
                return strat
        return ""

    def increment_bars(self) -> None:
        """Increment bars_held for all positions. Call once per smallest cycle."""
        for pos in self.pos_manager.all_positions():
            pos.bars_held += 1
