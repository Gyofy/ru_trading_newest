"""StrategyBase — ABC for all trading strategies.

Each strategy independently evaluates triggers and generates signals.
Execution, risk checks, and position tracking are handled by the shared layer.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.execution.exchange_adapter import ExchangeAdapter
    from src.execution.order_ledger import OrderLedger
    from src.strategies.data_hub import DataHub
    from src.strategies.multi_position_manager import MultiPositionManager
    from src.strategies.portfolio_risk import PortfolioRiskManager

from src.signals.contract import Signal, Action

logger = logging.getLogger("strategy")

# ── Fee constants (Binance USDT-M Futures, taker-only round-trip) ───────────
# Entry: market/IOC taker 0.055%  |  Exit: STOP_MARKET / TP_MARKET taker 0.055%
# Slippage: entry 0.03% + exit 0.05%
# Total round-trip rate applied to notional (= qty × price)
_TAKER_FEE     = 0.00055   # 0.055%
_SLIP_ENTRY    = 0.0003    # 0.030%
_SLIP_EXIT     = 0.0005    # 0.050%
ROUND_TRIP_FEE_RATE = _TAKER_FEE * 2 + _SLIP_ENTRY + _SLIP_EXIT  # ≈ 0.190%


@dataclass
class StrategyConfig:
    """Per-strategy parameters loaded from YAML."""
    name: str
    enabled: bool = True
    allocation_usdt: float = 20.0
    leverage: int = 2
    cycle_seconds: int = 60
    max_positions: int = 2
    sl_atr_mult: float = 3.0
    extra: dict = None
    paper_mode: bool = True  # True = simulate orders, False = real orders
    bot_version: str = ""    # 봇 버전 (config YAML version 필드)

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


class StrategyBase(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(
        self,
        config: StrategyConfig,
        exchange: ExchangeAdapter,
        portfolio_risk: PortfolioRiskManager,
        pos_manager: MultiPositionManager,
        ledger: OrderLedger,
        data_hub: DataHub,
        portfolio_lock: asyncio.Lock,
        trade_logger=None,
        coin_profiles=None,
        position_sizer=None,
        entry_filters=None,
    ):
        self.config = config
        self.name = config.name
        self.exchange = exchange
        self.portfolio_risk = portfolio_risk
        self.pos_manager = pos_manager
        self.ledger = ledger
        self.data_hub = data_hub
        self._portfolio_lock = portfolio_lock
        self.trade_logger = trade_logger      # TradeLogger instance (shared)
        self.coin_profiles = coin_profiles    # CoinProfileStore (shared)
        self.position_sizer = position_sizer  # PositionSizer (shared)
        self.entry_filters = entry_filters    # EntryFilters (shared, optional)
        self._log = logging.getLogger(f"strategy.{self.name}")
        self._paused = False
        self._trade_count = 0
        self._leverage_initialized: set[str] = set()  # coins initialized in this session

    # ── Abstract methods ──────────────────────────────────

    @abstractmethod
    async def _eval_one_coin(self, coin: str) -> "Signal | None":
        """Evaluate a single coin. Returns Signal or None.

        Called in parallel by evaluate(). Must be stateless across coins
        (no mutations to shared strategy state).
        """

    @abstractmethod
    def compute_barriers(
        self, signal: Signal, atr: float, price: float, extra: dict | None = None
    ) -> tuple[float, float]:
        """Compute (sl_price, tp_price) for this strategy.

        Args:
            extra: Per-coin adaptive params override for self.config.extra.
                   If None, falls back to self.config.extra. Pass this explicitly
                   instead of mutating self.config.extra directly.

        Returns:
            (sl_price, tp_price) — tp_price=0 means trailing stop only.
        """

    # ── Parallel evaluation ───────────────────────────────

    async def evaluate(self, coins: list[str]) -> list[Signal]:
        """Parallel evaluation: all coins launched simultaneously via asyncio.gather.

        Subclasses implement _eval_one_coin(coin) for per-coin logic.
        Override this method only for strategies with global-state side effects
        (e.g. daily trade caps, cooldowns) that must be checked before dispatch.
        """
        available = self.config.max_positions - len(self.my_positions)
        if available <= 0:
            return []

        candidates = [c for c in coins if not self.pos_manager.has_coin_any_strategy(c)]
        if not candidates:
            return []

        async def _safe(coin: str) -> "Signal | None":
            try:
                return await self._eval_one_coin(coin)
            except Exception as e:
                self._log.warning(f"[{coin}] eval error: {e}")
                return None

        # Launch ALL candidates simultaneously — no sequential waiting
        results = await asyncio.gather(*[_safe(c) for c in candidates])
        signals = [r for r in results if r is not None]
        return signals[:available]

    # ── Common execution flow ─────────────────────────────

    async def tick(self, coins: list[str]) -> list[dict]:
        """One evaluation cycle. Called by orchestrator."""
        if self._paused:
            return []

        results = []
        try:
            signals = await self.evaluate(coins)
            for sig in signals:
                result = await self._try_execute(sig)
                if result:
                    results.append(result)
        except Exception as e:
            self._log.error(f"[{self.name}] tick error: {e}", exc_info=True)

        return results

    async def _try_execute(self, signal: Signal) -> dict | None:
        """Attempt to execute a signal through portfolio risk gates."""
        coin = signal.symbol
        side = "BUY" if signal.action == Action.LONG else "SELL"

        # Fetch current price + ATR
        ticker = await self.data_hub.get_ticker(coin)
        price = ticker["last"]
        df = await self.data_hub.get_ohlcv(coin, "1m", limit=100)
        atr = self._compute_atr(df, period=14)

        if atr <= 0:
            self._log.warning(f"[{coin}] ATR=0, skip")
            return None

        # Entry filters (rule-based reject gates)
        if self.entry_filters:
            _vpin = self.data_hub.compute_vpin(df, window=24)
            # ATR history for percentile check
            _atr_vals = []
            if df is not None and len(df) >= 100:
                for i in range(14, min(100, len(df))):
                    _atr_vals.append(self._compute_atr(df.iloc[:i + 1], period=14))

            passed, reject_reason = self.entry_filters.check_all(
                coin=coin, vpin=_vpin, atr=atr, price=price, atr_history=_atr_vals,
            )
            if not passed:
                self._log.info(f"[{coin}] Filter rejected: {reject_reason}")
                return None

        # Apply per-coin adaptive params — pass as argument, never mutate self.config.extra.
        effective_extra = (
            self.coin_profiles.get_params(coin, self.config.extra)
            if self.coin_profiles else self.config.extra
        )
        sl_price, tp_price = self.compute_barriers(signal, atr, price, extra=effective_extra)

        # Stop distance for sizing
        sl_dist = abs(price - sl_price)
        # SL must be at least 1.5x round-trip cost to have any profit potential
        min_sl_dist = price * ROUND_TRIP_FEE_RATE * 1.5
        if sl_dist < min_sl_dist:
            self._log.warning(
                f"[{coin}] SL too tight: {sl_dist:.6f} < fee_threshold {min_sl_dist:.6f} "
                f"({ROUND_TRIP_FEE_RATE:.3%} of price) — skip"
            )
            return None
        # Cap sl_dist at 8% of price — prevents testnet ATR spikes from collapsing notional
        sl_dist = min(sl_dist, price * 0.08)

        # Dynamic position sizing — use effective_leverage from signal if provided
        effective_leverage = (
            signal.extra.get("effective_leverage", self.config.leverage)
            if signal.extra else self.config.leverage
        )
        risk_usdt = 0.0  # default; overwritten in both branches below
        if self.position_sizer:
            qty_raw, notional, sizing_info = self.position_sizer.compute_qty(
                allocation_usdt=self.config.allocation_usdt,
                leverage=effective_leverage,
                price=price,
                sl_dist=sl_dist,
                atr=atr,
                spread_bps=ticker.get("spread_bps", 0),
                signal_confidence=signal.confidence,
            )
            if qty_raw <= 0:
                self._log.info(f"[{coin}] PositionSizer skip: {sizing_info}")
                return None
            risk_usdt = sl_dist * qty_raw  # approximate risk
        else:
            # Fallback: fixed 10% risk
            risk_usdt = self.config.allocation_usdt * 0.10
            qty_raw = risk_usdt / sl_dist
            notional = qty_raw * price

        # Cap by remaining allocation
        max_notional = self.config.allocation_usdt * effective_leverage
        used = self.pos_manager.strategy_notional(self.name)
        available = max_notional - used
        if notional > available:
            notional = available
            qty_raw = notional / price

        # Per-position cap: single position cannot exceed 1/max_positions of total budget
        max_pos = max(1, self.config.max_positions)
        per_pos_cap = max_notional / max_pos
        if notional > per_pos_cap:
            notional = per_pos_cap
            qty_raw = notional / price

        if notional < 5.0:  # Binance minimum
            self._log.info(f"[{coin}] notional {notional:.2f} < $5, skip")
            return None

        # Fetch funding rate for gate check
        try:
            funding_rate = await self.data_hub.get_funding_rate(coin)
            if funding_rate is None:
                funding_rate = 0.0
        except Exception:
            funding_rate = 0.0

        # Portfolio-level approval (atomic with position add)
        async with self._portfolio_lock:
            # Double-checked locking: re-verify coin not taken by another strategy
            # (first check was in evaluate() before gather, but race condition exists)
            if self.pos_manager.has_coin_any_strategy(coin):
                self._log.info(f"[{coin}] rejected: coin taken during gather (race)")
                return None

            approved, reason = self.portfolio_risk.approve_entry(
                strategy_name=self.name,
                coin=coin,
                side=side,
                notional=notional,
                funding_rate=funding_rate,
            )
            if not approved:
                self._log.info(f"[{coin}] rejected: {reason}")
                return None

            # Round qty for exchange
            qty = self.exchange.round_qty(coin, qty_raw)
            if qty <= 0:
                return None

            # Place order
            _is_new_coin = coin not in self._leverage_initialized
            try:
                result = await self._place_entry(coin, side, qty, price)
            except Exception as e:
                self._log.error(f"[{coin}] order failed: {e}")
                return None

            if not result or not result.get("success"):
                return None

            fill_price = result.get("fill_price", price)

            # Record entry to OrderLedger
            if self.ledger:
                try:
                    _order_id = result.get("order_id", "")
                    self.ledger.insert_order(
                        order_link_id=_order_id,
                        symbol=coin,
                        side=side,
                        order_type="limit" if not self.config.paper_mode else "paper",
                        qty=qty,
                        price=fill_price,
                        purpose="entry",
                    )
                    self.ledger.insert_fill(
                        order_link_id=_order_id,
                        fill_price=fill_price,
                        fill_qty=result.get("fill_qty", qty),
                        fee=result.get("fee", 0),
                    )
                except Exception as e:
                    self._log.debug(f"[{coin}] Ledger write failed: {e}")

            # Recalculate barriers from fill price using per-coin params
            sl_price, tp_price = self.compute_barriers(signal, atr, fill_price, extra=effective_extra)

            # Determine trailing
            use_trailing = tp_price == 0
            # Use trailing_atr_mult if provided (e.g. 0.7×ATR for tight trail),
            # fallback to sl_atr_mult so trail distance ≠ SL distance by accident.
            _trail_mult = effective_extra.get("trailing_atr_mult", self.config.sl_atr_mult)
            trail_dist = atr * _trail_mult if use_trailing else 0.0

            # For trailing strategies: compute initial TP for exchange display
            # Uses tp_rr_ratio config (default 2.0x SL distance)
            # Fee offset ensures net RR = intended RR after paying round-trip fees
            if use_trailing:
                sl_dist = abs(fill_price - sl_price)
                tp_rr = effective_extra.get("tp_rr_ratio", 2.0)
                fee_offset = fill_price * ROUND_TRIP_FEE_RATE  # price units to cover fees
                if side == "BUY":
                    tp_price = fill_price + sl_dist * tp_rr + fee_offset
                else:
                    tp_price = fill_price - sl_dist * tp_rr - fee_offset

            # Register position
            from src.execution.position_store import OpenPosition  # noqa: PLC0415
            pos = OpenPosition(
                coin=coin,
                side=side,
                entry_price=fill_price,
                qty=qty,
                sl_price=sl_price,
                tp_price=tp_price if tp_price > 0 else sl_price,  # placeholder
                entry_time=signal.ts.isoformat(),
                ttl_bars=signal.ttl_bars or 96,
                strategy_tag=self.name,
                trailing_sl=use_trailing,
                trail_distance=trail_dist,
                leverage=self.config.leverage,
            )
            self.pos_manager.add_position(self.name, pos)
            self._trade_count += 1

            # ── Register exchange-side SL/TP (live/demo mode) ──────
            # RULE: every position MUST have both SL and TP on exchange.
            #       If SL fails after retries → force-close position.
            #       If TP fails after retries → force-close position.
            if not self.config.paper_mode:
                sl_side = "SELL" if side == "BUY" else "BUY"
                rounded_sl = self.exchange.round_price(coin, sl_price)
                rounded_tp = self.exchange.round_price(coin, tp_price) if tp_price > 0 else 0

                # Binance: closePosition=True (FOR POSITION) requires the position to exist
                # in Binance's DB before GTE TIF is accepted. Fill confirmation (order=FILLED)
                # propagates to the position engine with up to ~1s delay on testnet.
                # New coins (first trade this session): margin mode change (CROSS→ISOLATED)
                # also needs to propagate — takes up to 5-10s on testnet.
                _sl_init_wait = 3.0 if _is_new_coin else 1.0
                if _is_new_coin:
                    self._log.info(
                        f"[{coin}] 신규 종목 — SL/TP 등록 전 {_sl_init_wait}s 대기 "
                        f"(margin mode 전파 대기)"
                    )
                await asyncio.sleep(_sl_init_wait)

                # ── SL: 5-retry with 1s delay ──
                sl_result = {"success": False}
                for attempt in range(5):
                    if attempt > 0:
                        await asyncio.sleep(1.0)
                    sl_oid = self.exchange.make_order_id(
                        coin, sl_side, self._trade_count, prefix=f"v8sl{attempt}",
                    )
                    sl_result = await self.exchange.place_protective_stop(
                        symbol=coin, side=sl_side, qty=qty,
                        stop_price=rounded_sl, order_link_id=sl_oid,
                    )
                    if sl_result.get("success"):
                        pos.sl_order_id = sl_oid
                        pos.sl_exchange_id = sl_result.get("exchange_order_id", "")
                        self._log.info(
                            f"[{coin}] Exchange SL registered @ {sl_price:.4f}"
                            + (f" (retry {attempt})" if attempt > 0 else "")
                        )
                        break

                if not sl_result.get("success"):
                    # SL 등록 완전 실패 → 포지션 즉시 강제 청산 (SL 없는 포지션 금지)
                    self._log.error(
                        f"[SL_FAIL_CRITICAL] ⚠️ {coin} SL 5회 실패 → 포지션 강제 청산 "
                        f"(err: {sl_result.get('error')})"
                    )
                    close_side = "SELL" if side == "BUY" else "BUY"
                    close_ok = False
                    try:
                        await self.exchange.market_close(
                            coin, close_side, qty,
                            order_link_id=self.exchange.make_order_id(coin, close_side, prefix="v8emg"),
                        )
                        close_ok = True
                    except Exception as ce:
                        self._log.error(
                            f"[SL_FAIL_CRITICAL] {coin} 강제청산도 실패: {ce} "
                            f"— NAKED POSITION 위험! 수동 개입 필요"
                        )
                    if close_ok:
                        self.pos_manager.remove_position(self.name, coin)
                    # If close failed: keep position in tracker so monitor keeps watching
                    return None

                # ── TP: 5-retry with 1s delay ──
                if rounded_tp > 0:
                    tp_result = {"success": False}
                    for attempt in range(5):
                        if attempt > 0:
                            await asyncio.sleep(1.0)
                        tp_oid = self.exchange.make_order_id(
                            coin, sl_side, self._trade_count, prefix=f"v8tp{attempt}",
                        )
                        tp_result = await self.exchange.place_take_profit(
                            symbol=coin, side=sl_side, qty=qty,
                            tp_price=rounded_tp, order_link_id=tp_oid,
                        )
                        if tp_result.get("success"):
                            pos.tp_order_id = tp_oid
                            pos.tp_exchange_id = tp_result.get("exchange_order_id", "")
                            self._log.info(
                                f"[{coin}] Exchange TP registered @ {tp_price:.4f}"
                                + (f" (retry {attempt})" if attempt > 0 else "")
                            )
                            break

                    if not tp_result.get("success"):
                        # TP 등록 완전 실패 → 포지션 강제 청산 (TP 없는 포지션 금지)
                        self._log.error(
                            f"[TP_FAIL_CRITICAL] ⚠️ {coin} TP 5회 실패 → 포지션 강제 청산 "
                            f"(err: {tp_result.get('error')})"
                        )
                        # Cancel SL first (best-effort), then force-close
                        try:
                            if pos.sl_exchange_id:
                                await self.exchange.cancel_order(
                                    coin, exchange_order_id=pos.sl_exchange_id,
                                    order_link_id=pos.sl_order_id,
                                )
                        except Exception as _cancel_e:
                            self._log.debug(
                                f"[{coin}] SL cancel during TP_FAIL_CRITICAL ignored: {_cancel_e}"
                            )
                        try:
                            close_side = "SELL" if side == "BUY" else "BUY"
                            await self.exchange.market_close(
                                coin, close_side, qty,
                                order_link_id=self.exchange.make_order_id(coin, close_side, prefix="v8emg"),
                            )
                        except Exception as ce:
                            self._log.error(
                                f"[TP_FAIL_CRITICAL] {coin} 강제청산도 실패: {ce} "
                                f"— NAKED POSITION 위험! 수동 개입 필요"
                            )
                            # Keep position in tracker — monitor continues watching
                            return None
                        self.pos_manager.remove_position(self.name, coin)
                        return None

            # Persist position
            self.pos_manager._save()

            # ── Trade context logging ──
            if self.trade_logger:
                try:
                    _sl_dist_log = abs(fill_price - sl_price)
                    _rr_est_log = (
                        abs(tp_price - fill_price) / _sl_dist_log
                        if (_sl_dist_log > 0 and tp_price > 0) else 0.0
                    )
                    _concurrent = len(self.pos_manager.positions)
                    _portfolio_notional = self.pos_manager.total_notional() if hasattr(self.pos_manager, "total_notional") else 0.0
                    await self.trade_logger.capture_entry_context(
                        data_hub=self.data_hub,
                        coin=coin,
                        strategy_name=self.name,
                        side=side,
                        signal_extra=signal.extra,
                        fill_price=fill_price,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        qty=qty,
                        notional=notional,
                        leverage=self.config.leverage,
                        atr=atr,
                        trailing_sl=use_trailing,
                        trail_distance=trail_dist,
                        risk_usdt=risk_usdt,
                        rr_estimate=_rr_est_log,
                        strategy_params={
                            "bot_version": self.config.bot_version,
                            "sl_atr_mult": self.config.sl_atr_mult,
                            "leverage": self.config.leverage,
                            "allocation_usdt": self.config.allocation_usdt,
                            "max_positions": self.config.max_positions,
                            **self.config.extra,
                        },
                        paper_mode=self.config.paper_mode,
                        concurrent_positions=_concurrent,
                        portfolio_notional=_portfolio_notional,
                    )
                except Exception as e:
                    self._log.warning(f"[{coin}] Trade context capture failed: {e}")

            fee_usdt = notional * ROUND_TRIP_FEE_RATE
            sl_pct = sl_dist / fill_price * 100
            tp_pct = abs(tp_price - fill_price) / fill_price * 100 if tp_price > 0 else 0
            self._log.info(
                f"[{coin}] ENTRY {side} @ {fill_price:.4f} | "
                f"qty={qty} | SL={sl_price:.4f}(-{sl_pct:.2f}%) | "
                f"TP={tp_price:.4f}(+{tp_pct:.2f}%) | "
                f"trailing={'Y' if use_trailing else 'N'} | "
                f"notional=${notional:.2f} | fee≈${fee_usdt:.3f}({ROUND_TRIP_FEE_RATE:.3%})"
            )

            return {
                "strategy": self.name,
                "coin": coin,
                "side": side,
                "price": fill_price,
                "qty": qty,
                "sl": sl_price,
                "tp": tp_price,
                "trailing": use_trailing,
            }

    async def _place_entry(
        self, coin: str, side: str, qty: float, price: float
    ) -> dict:
        """Post-Only maker entry with fill-or-cancel.

        Paper mode: simulate instant fill at maker price (no real orders).
        """
        # Paper mode: simulate fill at current bid/ask price (via cached ticker)
        if self.config.paper_mode:
            ticker = await self.data_hub.get_ticker(coin)
            maker_price = ticker["bid"] if side.upper() == "BUY" else ticker["ask"]
            if maker_price <= 0:
                maker_price = ticker["last"]
            if maker_price <= 0:
                self._log.warning(
                    f"[{coin}] Paper fill skipped — ticker all zeros (exchange unavailable?)"
                )
                return {"success": False, "error": "ticker_unavailable"}
            order_id = self.exchange.make_order_id(
                coin, side, self._trade_count + 1, prefix="v8p",
            )
            self._log.info(f"[{coin}] PAPER fill {side} @ {maker_price:.4f} qty={qty}")
            return {
                "success": True,
                "fill_price": float(maker_price),
                "fill_qty": float(qty),
                "order_id": order_id,
            }

        # Live mode: real Post-Only maker entry
        try:
            await self.exchange.set_leverage_async(coin, self.config.leverage)
            self._leverage_initialized.add(coin)
        except Exception as _lev_e:
            self._log.warning(
                f"[{coin}] set_leverage({self.config.leverage}x) FAILED: {_lev_e} "
                f"— proceeding with current exchange leverage"
            )

        maker_price = await self.exchange.get_maker_entry_price(coin, side)
        order_id = self.exchange.make_order_id(
            coin, side, self._trade_count + 1, prefix="v8",
        )

        result = await self.exchange.place_post_only_entry(
            symbol=coin, side=side, qty=qty,
            price=maker_price, order_link_id=order_id,
        )
        if not result.get("success"):
            return {"success": False, "error": result.get("error", "post-only failed")}

        fill = await self.exchange.wait_fill_or_cancel(
            symbol=coin, order_link_id=order_id,
            ttl_sec=20.0, poll_interval=2.0,
        )
        if not fill or not fill.get("filled"):
            return {"success": False, "error": "not filled within 20s"}

        return {
            "success": True,
            "fill_price": float(fill["fill_price"]),
            "fill_qty": float(fill.get("fill_qty", qty)),
            "order_id": order_id,
        }

    # ── Utilities ─────────────────────────────────────────

    @staticmethod
    def _compute_atr(df, period: int = 14) -> float:
        """Compute ATR from OHLCV DataFrame."""
        if df is None or len(df) < period + 1:
            return 0.0
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        tr = []
        for i in range(1, len(high)):
            tr.append(max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            ))
        if len(tr) < period:
            return 0.0
        return float(sum(tr[-period:]) / period)

    @property
    def my_positions(self) -> list:
        """Positions owned by this strategy."""
        return self.pos_manager.get_positions_by_strategy(self.name)

    def pause(self, reason: str = ""):
        self._paused = True
        self._log.warning(f"[{self.name}] PAUSED: {reason}")

    def resume(self):
        self._paused = False
        self._log.info(f"[{self.name}] RESUMED")
