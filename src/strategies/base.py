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
        self._log = logging.getLogger(f"strategy.{self.name}")
        self._paused = False
        self._trade_count = 0

    # ── Abstract methods ──────────────────────────────────

    @abstractmethod
    async def _eval_one_coin(self, coin: str) -> "Signal | None":
        """Evaluate a single coin. Returns Signal or None.

        Called in parallel by evaluate(). Must be stateless across coins
        (no mutations to shared strategy state).
        """

    @abstractmethod
    def compute_barriers(
        self, signal: Signal, atr: float, price: float
    ) -> tuple[float, float]:
        """Compute (sl_price, tp_price) for this strategy.

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

        # Apply per-coin adaptive params.
        # NOTE: No await between mutation and restore — safe in asyncio single-thread model.
        # _try_execute is always called sequentially from tick() so no true concurrency here.
        original_extra = self.config.extra
        effective_extra = (
            self.coin_profiles.get_params(coin, original_extra)
            if self.coin_profiles else original_extra
        )
        self.config.extra = effective_extra
        try:
            sl_price, tp_price = self.compute_barriers(signal, atr, price)
        finally:
            self.config.extra = original_extra

        # Stop distance for sizing
        sl_dist = abs(price - sl_price)
        if sl_dist < price * 0.001:  # min 0.1% SL distance
            self._log.warning(f"[{coin}] SL too tight: {sl_dist:.6f}")
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
            try:
                result = await self._place_entry(coin, side, qty, price)
            except Exception as e:
                self._log.error(f"[{coin}] order failed: {e}")
                return None

            if not result or not result.get("success"):
                return None

            fill_price = result.get("fill_price", price)

            # Recalculate barriers from fill price using per-coin params
            self.config.extra = effective_extra
            try:
                sl_price, tp_price = self.compute_barriers(signal, atr, fill_price)
            finally:
                self.config.extra = original_extra

            # Determine trailing
            use_trailing = tp_price == 0
            trail_dist = atr * self.config.sl_atr_mult if use_trailing else 0.0

            # For trailing strategies: compute initial TP for exchange display
            # Uses tp_rr_ratio config (default 2.0x SL distance)
            if use_trailing and tp_price == 0:
                sl_dist = abs(fill_price - sl_price)
                tp_rr = self.config.extra.get("tp_rr_ratio", 2.0)
                if side == "BUY":
                    tp_price = fill_price + sl_dist * tp_rr
                else:
                    tp_price = fill_price - sl_dist * tp_rr

            # Register position
            from src.execution.position_store import OpenPosition
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
            )
            self.pos_manager.add_position(self.name, pos)
            self._trade_count += 1

            # ── Register exchange-side SL/TP (live mode only) ──────
            if not self.config.paper_mode:
                sl_side = "SELL" if side == "BUY" else "BUY"
                sl_oid = self.exchange.make_order_id(
                    coin, sl_side, self._trade_count, prefix="v8sl",
                )
                rounded_sl = self.exchange.round_price(coin, sl_price)
                sl_result = await self.exchange.place_protective_stop(
                    symbol=coin, side=sl_side, qty=qty,
                    stop_price=rounded_sl, order_link_id=sl_oid,
                )
                if sl_result.get("success"):
                    pos.sl_order_id = sl_oid
                    pos.sl_exchange_id = sl_result.get("exchange_order_id", "")
                    self._log.info(f"[{coin}] Exchange SL registered @ {sl_price:.4f}")
                else:
                    self._log.error(
                        f"[SL_FAIL] ⚠️ {coin} SL 거래소 등록 실패: {sl_result.get('error')} "
                        f"— 소프트웨어 SL 유지 중 @ {sl_price:.4f} | 봇 크래시 시 포지션 미보호"
                    )

                if tp_price > 0:
                    tp_side = sl_side
                    tp_oid = self.exchange.make_order_id(
                        coin, tp_side, self._trade_count, prefix="v8tp",
                    )
                    rounded_tp = self.exchange.round_price(coin, tp_price)
                    tp_result = await self.exchange.place_take_profit(
                        symbol=coin, side=tp_side, qty=qty,
                        tp_price=rounded_tp, order_link_id=tp_oid,
                    )
                    if tp_result.get("success"):
                        pos.tp_order_id = tp_oid
                        pos.tp_exchange_id = tp_result.get("exchange_order_id", "")
                        self._log.info(f"[{coin}] Exchange TP registered @ {tp_price:.4f}")
                    else:
                        self._log.warning(
                            f"[{coin}] Exchange TP FAILED: {tp_result.get('error')} "
                            f"(software TP still active)"
                        )

            # Persist position
            self.pos_manager._save()

            # ── Trade context logging ──
            if self.trade_logger:
                try:
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
                        rr_estimate=0.0,
                        strategy_params=self.config.extra,
                        paper_mode=self.config.paper_mode,
                    )
                except Exception as e:
                    self._log.warning(f"[{coin}] Trade context capture failed: {e}")

            self._log.info(
                f"[{coin}] ENTRY {side} @ {fill_price:.4f} | "
                f"qty={qty} | SL={sl_price:.4f} | "
                f"trailing={'Y' if use_trailing else 'N'} | "
                f"notional=${notional:.2f}"
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
            self.exchange.set_leverage(coin, self.config.leverage)
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
