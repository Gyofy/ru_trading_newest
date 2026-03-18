"""Risk Engine — Stop-distance 기반 포지션 사이징 + 리스크 게이트.

핵심 원칙:
  - 사이즈 = risk_usdt / stop_distance (증거금이 아닌 손실 한도 기반)
  - threshold는 진입 필터일 뿐, 확률 기반 사이징 절대 금지
  - 레버리지는 증거금 절감 수단이지 수익 증폭기가 아님
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

from src.execution.order_ledger import OrderLedger

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """실행 리스크 설정."""
    risk_frac: float = 0.005            # 0.5% of equity per trade
    max_notional_usdt: float = 500.0    # per-position notional cap
    daily_drawdown_pct: float = 0.02    # -2% daily halt
    weekly_drawdown_pct: float = 0.05   # -5% weekly halt
    consecutive_loss_limit: int = 3     # kill switch
    alt_bucket_loss_pct: float = 0.01   # alt total loss 1%
    funding_rate_block: float = 0.001   # 0.1% per 8h = block
    funding_rate_warn: float = 0.0005   # 0.05% per 8h = warn
    entry_spread_max_bps: float = 50    # skip if spread > 50bps
    min_prob_threshold: float = 0.40    # entry filter (not sizing!)
    leverage: float = 1.0               # start with 1x (no leverage)
    min_stop_distance_pct: float = 0.003  # 0.3% min stop distance


@dataclass
class SizingResult:
    """포지션 사이징 결과."""
    approved: bool
    qty: float = 0.0
    notional: float = 0.0
    risk_usdt: float = 0.0
    margin_required: float = 0.0
    reason: str = ""


@dataclass
class PreTradeCheck:
    """진입 전 종합 검사 결과."""
    approved: bool
    sizing: SizingResult | None = None
    reason: str = ""
    warnings: list[str] | None = None


class RiskEngine:
    """리스크 관리 엔진.

    Usage:
        engine = RiskEngine(config, ledger)
        check = engine.pre_trade_gate(
            symbol="XRP", side="BUY",
            entry_price=0.55, sl_price=0.53,
            equity_usdt=10000, atr=0.015,
        )
        if check.approved:
            qty = check.sizing.qty
    """

    # BTC는 별도 버킷, 나머지는 alt 버킷
    BTC_BUCKET = {"BTC"}

    def __init__(self, config: RiskConfig | None = None, ledger: OrderLedger | None = None):
        self.config = config or RiskConfig()
        self.ledger = ledger
        self._daily_start_equity: float = 0.0
        self._weekly_start_equity: float = 0.0
        self._kill_switch: bool = False
        self._kill_reason: str = ""

    def set_initial_equity(self, equity: float) -> None:
        self._daily_start_equity = equity
        self._weekly_start_equity = equity

    # ── Core Sizing ─────────────────────────────────────────

    def compute_qty(
        self,
        entry_price: float,
        sl_price: float,
        equity_usdt: float,
    ) -> SizingResult:
        """Stop-distance 기반 포지션 사이즈 계산.

        qty = risk_usdt / stop_dist
        notional = qty * entry_price
        qty = min(qty, max_notional / entry_price)

        확률은 사이징에 영향을 주지 않음.
        """
        cfg = self.config

        # Stop distance
        stop_dist = abs(entry_price - sl_price)
        if stop_dist < entry_price * cfg.min_stop_distance_pct:
            return SizingResult(
                approved=False,
                reason=f"Stop distance too small: {stop_dist:.6f} "
                       f"(min {entry_price * cfg.min_stop_distance_pct:.6f})",
            )

        # Risk amount
        risk_usdt = equity_usdt * cfg.risk_frac

        # Raw qty from stop distance
        raw_qty = risk_usdt / stop_dist

        # Notional check
        raw_notional = raw_qty * entry_price
        if raw_notional > cfg.max_notional_usdt:
            raw_qty = cfg.max_notional_usdt / entry_price
            raw_notional = cfg.max_notional_usdt

        # Margin required
        margin = raw_notional / cfg.leverage

        return SizingResult(
            approved=True,
            qty=raw_qty,
            notional=raw_notional,
            risk_usdt=risk_usdt,
            margin_required=margin,
            reason="ok",
        )

    # ── Barrier Computation ─────────────────────────────────

    @staticmethod
    def compute_barriers(
        entry_price: float,
        atr: float,
        side: str,
        k_upper: float = 1.5,
        k_lower: float = 1.5,
        min_barrier_pct: float = 0.002,
    ) -> tuple[float, float]:
        """학습과 동일한 배리어 계산.

        Returns: (sl_price, tp_price)
        """
        tp_dist = max(k_upper * atr, entry_price * min_barrier_pct)  # wide TP
        sl_dist = max(k_lower * atr, entry_price * min_barrier_pct)  # tight SL

        if side == "BUY":  # LONG: TP above, SL below
            tp_price = entry_price + tp_dist
            sl_price = entry_price - sl_dist
        else:  # SHORT: TP below, SL above (same R:R as LONG)
            tp_price = entry_price - tp_dist
            sl_price = entry_price + sl_dist

        return sl_price, tp_price

    # ── Pre-Trade Gate ──────────────────────────────────────

    def pre_trade_gate(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        sl_price: float,
        equity_usdt: float,
        p_trade: float,
        atr: float,
        funding_rate: float = 0.0,
        spread_bps: float = 0.0,
    ) -> PreTradeCheck:
        """진입 전 종합 검사. 모든 게이트를 순서대로 통과해야 approved."""
        cfg = self.config
        warnings: list[str] = []

        # 1. Kill switch
        if self._kill_switch:
            return PreTradeCheck(
                approved=False,
                reason=f"Kill switch active: {self._kill_reason}",
            )

        # 2. Probability threshold (entry filter ONLY, not sizing)
        if p_trade < cfg.min_prob_threshold:
            return PreTradeCheck(
                approved=False,
                reason=f"p_trade {p_trade:.3f} < threshold {cfg.min_prob_threshold}",
            )

        # 3. Spread filter
        if spread_bps > cfg.entry_spread_max_bps:
            return PreTradeCheck(
                approved=False,
                reason=f"Spread {spread_bps:.1f}bps > max {cfg.entry_spread_max_bps}bps",
            )

        # 4. Funding rate
        funding_status = self._check_funding(funding_rate)
        if funding_status == "block":
            return PreTradeCheck(
                approved=False,
                reason=f"Funding rate {funding_rate:.4f} too high",
            )
        elif funding_status == "warn":
            warnings.append(f"High funding rate: {funding_rate:.4f}")

        # 5. Daily drawdown
        dd_ok, dd_pct = self.check_daily_drawdown(equity_usdt)
        if not dd_ok:
            self._kill_switch = True
            self._kill_reason = f"Daily drawdown {dd_pct:.2%}"
            return PreTradeCheck(
                approved=False,
                reason=f"Daily drawdown {dd_pct:.2%} > {cfg.daily_drawdown_pct:.0%}",
            )

        # 6. Consecutive losses
        if self.ledger and self.ledger.get_consecutive_losses(symbol, cfg.consecutive_loss_limit):
            return PreTradeCheck(
                approved=False,
                reason=f"{symbol}: {cfg.consecutive_loss_limit} consecutive losses",
                warnings=warnings,
            )

        # 7. Alt bucket correlation cap
        if self.ledger and symbol not in self.BTC_BUCKET:
            alt_ok = self._check_alt_bucket(equity_usdt)
            if not alt_ok:
                return PreTradeCheck(
                    approved=False,
                    reason=f"Alt bucket loss exceeds {cfg.alt_bucket_loss_pct:.1%}",
                    warnings=warnings,
                )

        # 8. Liquidation pre-check
        liq_ok = self._check_liquidation(entry_price, sl_price, side)
        if not liq_ok:
            return PreTradeCheck(
                approved=False,
                reason="Stop loss beyond estimated liquidation price",
                warnings=warnings,
            )

        # 9. Sizing
        sizing = self.compute_qty(entry_price, sl_price, equity_usdt)
        if not sizing.approved:
            return PreTradeCheck(
                approved=False,
                reason=sizing.reason,
                warnings=warnings,
            )

        # 10. Fee-adjusted EV check
        # 예상 수수료(진입+청산) 합산 후 손익분기 확인
        # fee = taker_fee × 2 (진입+청산 각 1회), notional 기준
        taker_fee = 0.00055   # Binance Futures taker
        round_trip_fee_pct = taker_fee * 2  # 0.11%
        fee_usdt = sizing.notional * round_trip_fee_pct
        sl_loss  = sizing.qty * abs(entry_price - sl_price)  # SL 도달 시 손실
        # SL 손실 + 수수료가 risk_usdt의 150%를 초과하면 수수료 비중이 너무 큼
        if sl_loss > 1e-10 and fee_usdt / sl_loss > 0.5:
            warnings.append(
                f"High fee ratio: fee={fee_usdt:.4f} USDT = "
                f"{fee_usdt/sl_loss*100:.0f}% of SL loss"
            )
        # 최소 노셔널 체크: 수수료를 감당하려면 notional이 fee의 10배 이상
        if sizing.notional < fee_usdt * 10:
            return PreTradeCheck(
                approved=False,
                reason=(
                    f"Notional {sizing.notional:.2f} USDT too small vs fee "
                    f"{fee_usdt:.4f} USDT — not fee-efficient"
                ),
                warnings=warnings,
            )

        return PreTradeCheck(
            approved=True,
            sizing=sizing,
            reason="all gates passed",
            warnings=warnings if warnings else None,
        )

    # ── Drawdown Tracking ───────────────────────────────────

    def check_daily_drawdown(self, current_equity: float) -> tuple[bool, float]:
        """Returns (ok, drawdown_pct). ok=False if limit breached."""
        if self._daily_start_equity <= 0:
            return True, 0.0
        dd = (self._daily_start_equity - current_equity) / self._daily_start_equity
        return dd <= self.config.daily_drawdown_pct, dd

    def reset_daily(self, equity: float) -> None:
        self._daily_start_equity = equity
        # Lift daily kill switch if it was daily
        if self._kill_switch and "Daily" in self._kill_reason:
            self._kill_switch = False
            self._kill_reason = ""
            logger.info("Daily kill switch lifted")

    def reset_weekly(self, equity: float) -> None:
        self._weekly_start_equity = equity
        if self._kill_switch and "Weekly" in self._kill_reason:
            self._kill_switch = False
            self._kill_reason = ""

    # ── Internal Checks ─────────────────────────────────────

    def _check_funding(self, rate: float) -> str:
        """ok / warn / block"""
        abs_rate = abs(rate)
        if abs_rate >= self.config.funding_rate_block:
            return "block"
        elif abs_rate >= self.config.funding_rate_warn:
            return "warn"
        return "ok"

    def _check_alt_bucket(self, equity: float) -> bool:
        """Alt 코인 총 일일 손실 체크."""
        if not self.ledger:
            return True
        daily = self.ledger.get_daily_pnl()
        alt_loss = sum(
            v["realized_pnl"] for sym, v in daily.items()
            if sym not in self.BTC_BUCKET and v["realized_pnl"] < 0
        )
        limit = equity * self.config.alt_bucket_loss_pct
        return abs(alt_loss) <= limit

    def _check_liquidation(
        self, entry: float, sl: float, side: str,
    ) -> bool:
        """1x 레버리지에서는 청산 불가. 레버리지 사용 시 검증."""
        if self.config.leverage <= 1.0:
            return True

        # Estimated liquidation price (simplified)
        # For LONG: liq = entry * (1 - 1/leverage + maintenance_margin)
        # maintenance_margin ~= 0.5% for major pairs
        mm = 0.005
        if side == "BUY":
            liq = entry * (1 - 1 / self.config.leverage + mm)
            return sl > liq  # SL must trigger before liquidation
        else:
            liq = entry * (1 + 1 / self.config.leverage - mm)
            return sl < liq

    # ── Kill Switch ─────────────────────────────────────────

    def activate_kill_switch(self, reason: str) -> None:
        self._kill_switch = True
        self._kill_reason = reason
        logger.warning(f"KILL SWITCH ACTIVATED: {reason}")

    def deactivate_kill_switch(self) -> None:
        self._kill_switch = False
        self._kill_reason = ""
        logger.info("Kill switch deactivated")

    @property
    def is_killed(self) -> bool:
        return self._kill_switch

    @property
    def kill_reason(self) -> str:
        return self._kill_reason
