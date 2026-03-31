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

# ── Fee constants (Binance USDT-M Futures, VIP 0 기준) ───────────────────────
# Binance USDM Futures VIP 0: Maker 0.0200%, Taker 0.0500%
# BNB 결제 시 10% 추가 할인: Taker 0.0450%, Maker 0.0180%
# 수수료 = Notional × Rate (레버리지 포함 금액에 부과, 마진이 아님!)
#   예: $100 마진 × 5x = $500 notional → fee = $500 × rate
# v9.1 Data-Gathering Mode: 전면 시장가(Taker) 진입
#   → 시그널 즉시 체결, 놓치는 알파 방지, MFE/MAE 정확한 타점 로깅
# Entry: Market taker 0.0500%  |  Exit: STOP_MARKET taker 0.0500%
# Slippage: entry 0.05% + exit 0.05% (시장가 양방향 동일)
_ENTRY_FEE     = 0.0005    # 0.0500% — 시장가 진입 (Taker)
_EXIT_FEE      = 0.0005    # 0.0500% — 시장가/SL/TP 청산 (Taker)
_SLIP_ENTRY    = 0.0005    # 0.050%  — 시장가 진입 슬리피지
_SLIP_EXIT     = 0.0005    # 0.050%  — 청산 슬리피지 추정
# 양방향 taker 기준 왕복 수수료
ROUND_TRIP_FEE_RATE = _ENTRY_FEE + _EXIT_FEE + _SLIP_ENTRY + _SLIP_EXIT  # = 0.0020 (0.20%)

# 강제 청산 수수료 — 일반 거래 수수료와 별도, 포지션 노셔널의 0.5%
LIQUIDATION_FEE_RATE = 0.005  # 0.50% — 강제 청산 시에만 적용


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
                if self.trade_logger:
                    self.trade_logger.record_rejection(
                        strategy=self.name, coin=coin, reject_reason=reject_reason,
                        signal_strength=signal.pred_return, signal_confidence=signal.confidence,
                        atr=atr, price=price, vpin=_vpin, trigger_type=signal.trigger_type if hasattr(signal, "trigger_type") else "",
                    )
                return None

        # Spread filter: block entry when bid-ask spread is too wide.
        # Wide spread means immediate adverse fill + higher effective taker cost.
        # Threshold: 8bps — futures spreads should be well below this for liquid pairs.
        # Note: testnet spreads are often 0 (bid/ask not provided) → only filter when > 0.
        _MAX_ENTRY_SPREAD_BPS = 8.0  # 8bps = 0.08%
        _entry_spread_bps = ticker.get("spread_bps", 0) if ticker else 0
        if _entry_spread_bps is None:
            _entry_spread_bps = 0
        _entry_spread_bps = float(_entry_spread_bps)
        if _entry_spread_bps > _MAX_ENTRY_SPREAD_BPS and _entry_spread_bps > 0:
            self._log.warning(
                f"[SpreadFilter] {coin} spread {_entry_spread_bps:.1f}bps "
                f"> {_MAX_ENTRY_SPREAD_BPS}bps — 진입 거부"
            )
            if self.trade_logger:
                self.trade_logger.record_rejection(
                    strategy=self.name,
                    coin=coin,
                    reject_reason=f"spread_too_wide:{_entry_spread_bps:.1f}bps",
                    signal_strength=signal.pred_return,
                    signal_confidence=signal.confidence,
                    atr=atr,
                    price=price,
                )
            return None

        # ── v10.0: ClusterTracker — 시그널 발생 기록 ──
        if hasattr(self, "cluster_tracker") and self.cluster_tracker:
            _strength = abs(signal.pred_return) if signal.pred_return else signal.confidence
            self.cluster_tracker.publish(
                strategy=self.name, coin=coin, side=side,
                strength=min(_strength, 1.0),
            )

        # ── v10.0: DrawdownThrottle — 사이즈 배율 ──
        _dd_factor = 1.0
        if hasattr(self, "drawdown_throttle") and self.drawdown_throttle:
            _dd_factor = self.drawdown_throttle.get_size_factor(self.name)
            if _dd_factor <= 0:
                self._log.info(f"[{coin}] DrawdownThrottle PAUSED — skip")
                return None

        # Apply per-coin adaptive params — pass as argument, never mutate self.config.extra.
        effective_extra = (
            self.coin_profiles.get_params(coin, self.config.extra)
            if self.coin_profiles else self.config.extra
        )
        sl_price, tp_price = self.compute_barriers(signal, atr, price, extra=effective_extra)

        # Stop distance for sizing
        sl_dist = abs(price - sl_price)

        # ── SL floor: Data-Gathering Mode (v9.1) ──
        # 데모에서는 SL을 넉넉하게 열어 MAE/MFE 한계치까지 데이터 수집
        # 실전 복귀 시: SL_FLOOR_PCT = 0.004, SL_FEE_MULT = 2.5 로 되돌릴 것
        SL_FLOOR_PCT = 0.015  # 1.50% absolute minimum (데이터 수집용)
        SL_FEE_MULT = 2.5     # SL must be at least 2.5× round-trip cost
        min_sl_dist = max(price * ROUND_TRIP_FEE_RATE * SL_FEE_MULT, price * SL_FLOOR_PCT)
        if sl_dist < min_sl_dist:
            # 데이터 수집 모드: SL을 floor까지 확장 (거부 대신 조정)
            self._log.info(
                f"[{coin}] SL widened for data gathering: "
                f"{sl_dist/price:.4%} → {min_sl_dist/price:.4%}"
            )
            sl_dist = min_sl_dist
            if side == "BUY":
                sl_price = price - sl_dist
            else:
                sl_price = price + sl_dist
        # Cap sl_dist at 15% of price (데이터 수집 모드: 확장)
        max_sl_dist = price * 0.15
        if sl_dist > max_sl_dist:
            sl_dist = max_sl_dist
            # Recompute sl_price to match capped distance (prevent sizing/SL mismatch)
            if side == "BUY":
                sl_price = price - sl_dist
            else:
                sl_price = price + sl_dist

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

        # ── v10.0: DrawdownThrottle size factor 적용 ──
        if _dd_factor < 1.0:
            notional *= _dd_factor
            qty_raw = notional / price
            self._log.info(f"[{coin}] DrawdownThrottle: size ×{_dd_factor:.1f}")

        if notional < 5.0:  # Binance minimum
            self._log.info(f"[{coin}] notional {notional:.2f} < $5, skip")
            return None

        # ── v10.0: FeeEVGate — 수수료 포함 기대값 검증 ──
        if hasattr(self, "fee_ev_gate") and self.fee_ev_gate:
            _sl_pct = sl_dist / price if price > 0 else 0
            _tp_pct = abs(tp_price - price) / price if (price > 0 and tp_price > 0) else _sl_pct * 2.0
            _ev_ok, _ev_val, _ev_reason = self.fee_ev_gate.check(
                strategy=self.name,
                sl_dist_pct=_sl_pct,
                tp_dist_pct=_tp_pct,
                notional=notional,
            )
            if not _ev_ok:
                self._log.info(f"[{coin}] FeeEV REJECTED: {_ev_reason}")
                # ── EV 그리드 서치: EV > 0 달성 파라미터 탐색 + 적용 ──
                if hasattr(self, "ev_grid_search") and self.ev_grid_search and atr > 0:
                    _atr_pct = atr / price if price > 0 else 0.01
                    _gs_applied, _gs_summary = self.ev_grid_search.search_and_apply(
                        strategy_obj=self,
                        fee_ev_gate=self.fee_ev_gate,
                        atr_pct=_atr_pct,
                        sl_floor_pct=SL_FLOOR_PCT,
                    )
                    if _gs_applied:
                        self._log.warning(
                            f"[{coin}] EV GridSearch 파라미터 조정: "
                            + ", ".join(
                                f"{k}={v[0]}→{v[1]}" for k, v in _gs_applied.items()
                            )
                        )
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
                if self.trade_logger:
                    self.trade_logger.record_rejection(
                        strategy=self.name, coin=coin, reject_reason=f"portfolio:{reason}",
                        signal_strength=signal.pred_return, signal_confidence=signal.confidence,
                        atr=atr, price=price,
                    )
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
                # -4130 (duplicate closePosition order) → 기존 같은 방향 SL 취소 후 재시도
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
                        pos._last_exchange_sl = sl_price  # 모니터 중복 시도 방지
                        self._log.info(
                            f"[{coin}] Exchange SL registered @ {sl_price:.4f}"
                            + (f" (retry {attempt})" if attempt > 0 else "")
                        )
                        break
                    # -4130: 기존 SL이 이미 있음 → 취소 후 재시도
                    _err = str(sl_result.get("error", ""))
                    if "-4130" in _err or "GTE" in _err:
                        self._log.warning(
                            f"[{coin}] SL -4130 (duplicate) — 기존 SL 주문 취소 후 재시도"
                        )
                        try:
                            _ccxt_sym = self.exchange._ccxt_symbol(coin)
                            _open_orders = await self.exchange._exchange.fetch_open_orders(_ccxt_sym)
                            for _ord in _open_orders:
                                _ord_type = (_ord.get("type") or "").lower()
                                _ord_side = (_ord.get("side") or "").lower()
                                if "stop" in _ord_type and _ord_side == sl_side.lower():
                                    try:
                                        await self.exchange._exchange.cancel_order(
                                            _ord["id"], _ccxt_sym
                                        )
                                        self._log.info(f"[{coin}] 기존 SL 취소: {_ord['id']}")
                                    except Exception:
                                        pass
                        except Exception as _ce:
                            self._log.warning(f"[{coin}] 기존 SL 조회/취소 실패: {_ce}")

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
                    # Merge signal.confidence + is_maker into extra for trade_logger
                    _is_maker = result.get("is_maker", False)  # v9.1: 시장가 = taker
                    # v10.0: 클러스터 피처 주입
                    _cluster_features = {}
                    if hasattr(self, "cluster_tracker") and self.cluster_tracker:
                        _cluster_features = self.cluster_tracker.get_trade_context_features(coin, side)
                    _signal_extra_with_conf = {
                        **(signal.extra or {}),
                        "confidence": signal.confidence,
                        "fill_taker": not _is_maker,
                        **_cluster_features,  # v10.0: cluster signal data
                    }
                    await self.trade_logger.capture_entry_context(
                        data_hub=self.data_hub,
                        coin=coin,
                        strategy_name=self.name,
                        side=side,
                        signal_extra=_signal_extra_with_conf,
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
        """Market (taker) entry — 즉시 체결.

        v9.1 Data-Gathering Mode: 시장가로 즉시 체결하여
        시그널 발생 시점의 정확한 가격 궤적(MFE/MAE)을 로깅.
        Post-Only로 놓치는 "가장 완벽했던 타점" 데이터를 수집.

        실전 복귀 시: place_post_only_entry()로 되돌릴 것.
        """
        # Paper mode: simulate instant fill at last price
        if self.config.paper_mode:
            ticker = await self.data_hub.get_ticker(coin)
            # 시장가 시뮬레이션: BUY→ask(worst), SELL→bid(worst) for realistic fill
            fill_price = ticker["ask"] if side.upper() == "BUY" else ticker["bid"]
            if fill_price <= 0:
                fill_price = ticker["last"]
            if fill_price <= 0:
                self._log.warning(
                    f"[{coin}] Paper fill skipped — ticker all zeros (exchange unavailable?)"
                )
                return {"success": False, "error": "ticker_unavailable"}
            order_id = self.exchange.make_order_id(
                coin, side, self._trade_count + 1, prefix="v91p",
            )
            self._log.info(f"[{coin}] PAPER MARKET fill {side} @ {fill_price:.4f} qty={qty}")
            return {
                "success": True,
                "fill_price": float(fill_price),
                "fill_qty": float(qty),
                "order_id": order_id,
                "is_maker": False,  # 시장가 = taker
            }

        # Live/Demo mode: Market order — 즉시 체결
        try:
            await self.exchange.set_leverage_async(coin, self.config.leverage)
            self._leverage_initialized.add(coin)
        except Exception as _lev_e:
            self._log.warning(
                f"[{coin}] set_leverage({self.config.leverage}x) FAILED: {_lev_e} "
                f"— proceeding with current exchange leverage"
            )

        order_id = self.exchange.make_order_id(
            coin, side, self._trade_count + 1, prefix="v91",
        )
        ccxt_sym = self.exchange._ccxt_symbol(coin)
        rounded_qty = self.exchange.round_qty(coin, qty)

        try:
            order = await self.exchange._exchange.create_order(
                symbol=ccxt_sym,
                type="market",
                side=side.lower(),
                amount=rounded_qty,
                params={self.exchange._client_id_key(): order_id},
            )
        except Exception as e:
            _err = str(e)
            if "-1021" in _err or "ahead of the server" in _err:
                self._log.warning(f"[Entry] {coin} -1021 time drift — resyncing clock")
                try:
                    await self.exchange._exchange.load_time_difference()
                except Exception:
                    pass
            return {"success": False, "error": _err}

        fill_price = float(order.get("average", order.get("price", 0)))
        fill_qty = float(order.get("filled", rounded_qty))
        status = order.get("status", "")

        if status != "closed" or fill_price <= 0:
            self._log.warning(
                f"[Entry] {coin} Market order status={status} price={fill_price} — unexpected"
            )
            return {"success": False, "error": f"market_order_status_{status}"}

        # Fee extraction from exchange response
        api_fee = (order.get("fee") or {}).get("cost")
        fee = float(api_fee) if api_fee else fill_price * fill_qty * 0.0005  # taker fallback

        self._log.info(
            f"[Entry] {coin} MARKET {side} @ {fill_price:.4f} qty={fill_qty} "
            f"fee=${fee:.4f}"
        )
        return {
            "success": True,
            "fill_price": fill_price,
            "fill_qty": fill_qty,
            "order_id": order_id,
            "is_maker": False,  # 시장가 = taker
            "fee": fee,
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
