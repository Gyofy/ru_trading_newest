"""Paper Trader — 시뮬레이션 매매 실행 엔진.

하나의 Signal을 받아 가상 포지션을 관리한다.
실제 주문 없이 equity curve, trade log, risk limit을 검증.

핵심 규칙:
- one-position-per-coin (코인당 1 포지션만)
- TP/SL/TTL 자동 청산
- 수수료 + 슬리피지 반영
- 일손실 2% / 주간손실 5% 중단
- 총 노출 80% 강제
- 고정 1~2% 사이징 (Kelly 미사용)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

from src.signals.contract import Signal, Action

logger = logging.getLogger(__name__)


# ── 설정 상수 ──────────────────────────────────────────────

INITIAL_CAPITAL = 10_000_000  # 1천만 원 (시뮬레이션 기준)
COST_BPS_ONE_WAY = 20         # 편도 20bps = 0.2%
FIXED_SIZE_PCT = 0.015        # 고정 1.5% (1~2% 범위 중간)
MAX_PER_COIN_PCT = 0.05       # 코인당 최대 5%
MAX_TOTAL_EXPOSURE = 0.80     # 총 노출 80%
DAILY_LOSS_LIMIT = 0.02       # 일 손실 2%
WEEKLY_LOSS_LIMIT = 0.05      # 주간 손실 5%
BAR_MINUTES = 240             # 4h bar

ACTIVE_UNIVERSE = ["DOT", "DOGE", "BTC", "XRP", "ADA"]


class PositionState(str, Enum):
    FLAT = "FLAT"
    OPEN = "OPEN"


@dataclass
class Position:
    """단일 코인 포지션."""
    symbol: str
    action: Action                # LONG or SHORT
    entry_price: float
    entry_time: datetime
    size_pct: float               # 계좌 대비 비중 (0~1)
    size_qty: float               # 수량 (= capital * size_pct / entry_price)
    take_profit_price: float
    stop_loss_price: float
    ttl_bars: int
    bars_held: int = 0
    signal_id: str = ""
    state: PositionState = PositionState.OPEN

    @property
    def is_long(self) -> bool:
        return self.action == Action.LONG


@dataclass
class TradeRecord:
    """완료된 거래 기록."""
    symbol: str
    action: str
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    size_pct: float
    size_qty: float
    pnl_pct: float              # 비용 차감 후 수익률
    pnl_amount: float           # 금액
    exit_reason: str            # TP / SL / TTL / MANUAL
    bars_held: int
    signal_id: str = ""
    cost_total: float = 0.0     # 진입+청산 비용 합


@dataclass
class EquitySnapshot:
    """시점별 자산 스냅샷."""
    ts: str
    equity: float
    cash: float
    exposure_pct: float
    open_positions: int
    daily_pnl_pct: float


class PaperTrader:
    """Paper Trading 엔진.

    Usage:
        trader = PaperTrader()
        trader.on_signal(signal, current_prices)   # 진입
        trader.on_bar(current_prices)               # 바마다 TP/SL/TTL 체크
        trader.snapshot()                            # equity 기록
        trader.save_report(path)                     # JSON 저장
    """

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        fixed_size_pct: float = FIXED_SIZE_PCT,
        cost_bps: float = COST_BPS_ONE_WAY,
        active_universe: list[str] | None = None,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.fixed_size_pct = fixed_size_pct
        self.cost_bps = cost_bps
        self.active_universe = active_universe or ACTIVE_UNIVERSE

        # 코인별 포지션 (one-position-per-coin)
        self.positions: dict[str, Position] = {}

        # 기록
        self.trade_log: list[TradeRecord] = []
        self.equity_curve: list[EquitySnapshot] = []

        # 리스크 상태
        self._daily_start_equity: float = initial_capital
        self._weekly_start_equity: float = initial_capital
        self._last_daily_reset: datetime | None = None
        self._last_weekly_reset: datetime | None = None
        self._halted: bool = False
        self._halt_reason: str = ""

        # 바 카운터
        self._bar_count: int = 0

    # ── 핵심 메서드 ─────────────────────────────────────────

    def on_signal(
        self,
        signal: Signal,
        current_prices: dict[str, float],
        now: datetime | None = None,
    ) -> bool:
        """시그널 수신 → 진입 시도. 성공 시 True."""
        now = now or datetime.now(timezone.utc)
        symbol = signal.symbol

        # 기본 검증
        if not signal.is_actionable:
            logger.debug(f"[{symbol}] Signal not actionable, skip")
            return False

        if symbol not in self.active_universe:
            logger.debug(f"[{symbol}] Not in active universe, skip")
            return False

        if self._halted:
            logger.warning(f"Trading halted: {self._halt_reason}")
            return False

        if symbol in self.positions:
            logger.debug(f"[{symbol}] Already has open position, skip")
            return False

        if symbol not in current_prices:
            logger.warning(f"[{symbol}] No price data, skip")
            return False

        # 노출 한도 체크
        current_exposure = self._total_exposure_pct()
        if current_exposure + self.fixed_size_pct > MAX_TOTAL_EXPOSURE:
            logger.warning(
                f"[{symbol}] Total exposure {current_exposure:.1%} + "
                f"{self.fixed_size_pct:.1%} > {MAX_TOTAL_EXPOSURE:.0%}, skip"
            )
            return False

        # 진입
        entry_price = current_prices[symbol]
        size_pct = min(self.fixed_size_pct, MAX_PER_COIN_PCT)
        alloc_amount = self.cash * size_pct  # 현금 기준 배분
        entry_cost = alloc_amount * (self.cost_bps / 10000)
        size_qty = (alloc_amount - entry_cost) / entry_price

        # TP/SL 가격 계산
        if signal.action == Action.LONG:
            tp_price = entry_price * (1 + signal.take_profit / 100)
            sl_price = entry_price * (1 - signal.stop_loss / 100)
        else:  # SHORT
            tp_price = entry_price * (1 - signal.take_profit / 100)
            sl_price = entry_price * (1 + signal.stop_loss / 100)

        pos = Position(
            symbol=symbol,
            action=signal.action,
            entry_price=entry_price,
            entry_time=now,
            size_pct=size_pct,
            size_qty=size_qty,
            take_profit_price=tp_price,
            stop_loss_price=sl_price,
            ttl_bars=signal.ttl_bars if signal.ttl_bars > 0 else 6,
            signal_id=signal.signal_id,
        )

        self.positions[symbol] = pos
        self.cash -= (alloc_amount)
        logger.info(
            f"[ENTRY] {symbol} {signal.action.value} @ {entry_price:.4f} "
            f"size={size_pct:.1%} TP={tp_price:.4f} SL={sl_price:.4f} "
            f"TTL={pos.ttl_bars} bars"
        )
        return True

    def on_bar(
        self,
        current_prices: dict[str, float],
        now: datetime | None = None,
    ) -> list[TradeRecord]:
        """매 바마다 호출. TP/SL/TTL 체크 후 청산된 거래 반환."""
        now = now or datetime.now(timezone.utc)
        self._bar_count += 1
        closed_trades: list[TradeRecord] = []

        # 일/주 리셋 체크
        self._check_period_reset(now)

        symbols_to_close = []

        for symbol, pos in self.positions.items():
            if symbol not in current_prices:
                continue

            price = current_prices[symbol]
            pos.bars_held += 1
            exit_reason = None

            # TP/SL 체크
            if pos.is_long:
                if price >= pos.take_profit_price:
                    exit_reason = "TP"
                elif price <= pos.stop_loss_price:
                    exit_reason = "SL"
            else:  # SHORT
                if price <= pos.take_profit_price:
                    exit_reason = "TP"
                elif price >= pos.stop_loss_price:
                    exit_reason = "SL"

            # TTL 체크
            if exit_reason is None and pos.bars_held >= pos.ttl_bars:
                exit_reason = "TTL"

            if exit_reason:
                trade = self._close_position(pos, price, now, exit_reason)
                closed_trades.append(trade)
                symbols_to_close.append(symbol)

        # 포지션 제거
        for sym in symbols_to_close:
            del self.positions[sym]

        # 일손실/주간손실 체크
        equity = self._calc_equity(current_prices)
        self._check_loss_limits(equity)

        return closed_trades

    def snapshot(
        self,
        current_prices: dict[str, float],
        now: datetime | None = None,
    ) -> EquitySnapshot:
        """현재 상태 스냅샷 생성 + equity curve에 추가."""
        now = now or datetime.now(timezone.utc)
        equity = self._calc_equity(current_prices)
        daily_pnl = (equity - self._daily_start_equity) / self._daily_start_equity

        snap = EquitySnapshot(
            ts=now.isoformat(),
            equity=round(equity, 2),
            cash=round(self.cash, 2),
            exposure_pct=round(self._total_exposure_pct(), 4),
            open_positions=len(self.positions),
            daily_pnl_pct=round(daily_pnl, 6),
        )
        self.equity_curve.append(snap)
        return snap

    # ── 내부 메서드 ─────────────────────────────────────────

    def _close_position(
        self, pos: Position, exit_price: float,
        now: datetime, reason: str,
    ) -> TradeRecord:
        """포지션 청산 → TradeRecord 생성."""
        # PnL 계산
        if pos.is_long:
            raw_pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            raw_pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        # 비용: 진입 + 청산 각각 cost_bps
        round_trip_cost_pct = 2 * self.cost_bps / 10000
        net_pnl_pct = raw_pnl_pct - round_trip_cost_pct

        # 금액 계산
        position_value = pos.size_qty * pos.entry_price
        pnl_amount = position_value * net_pnl_pct
        cost_total = position_value * round_trip_cost_pct

        # 현금 복원
        exit_value = pos.size_qty * exit_price
        exit_cost = exit_value * (self.cost_bps / 10000)
        self.cash += (exit_value - exit_cost)

        trade = TradeRecord(
            symbol=pos.symbol,
            action=pos.action.value,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time.isoformat(),
            exit_time=now.isoformat(),
            size_pct=pos.size_pct,
            size_qty=pos.size_qty,
            pnl_pct=round(net_pnl_pct, 6),
            pnl_amount=round(pnl_amount, 2),
            exit_reason=reason,
            bars_held=pos.bars_held,
            signal_id=pos.signal_id,
            cost_total=round(cost_total, 2),
        )

        emoji = "+" if net_pnl_pct >= 0 else ""
        logger.info(
            f"[EXIT:{reason}] {pos.symbol} {pos.action.value} "
            f"@ {exit_price:.4f} PnL={emoji}{net_pnl_pct:.2%} "
            f"({emoji}{pnl_amount:.0f}) bars={pos.bars_held}"
        )

        self.trade_log.append(trade)
        return trade

    def _calc_equity(self, current_prices: dict[str, float]) -> float:
        """현재 총 자산 (현금 + 미실현 포지션 가치)."""
        equity = self.cash
        for symbol, pos in self.positions.items():
            if symbol not in current_prices:
                equity += pos.size_qty * pos.entry_price  # fallback: 진입가
                continue
            price = current_prices[symbol]
            if pos.is_long:
                unrealized = pos.size_qty * price
            else:
                # SHORT: 2 * entry - current (숏 수익 반영)
                unrealized = pos.size_qty * (2 * pos.entry_price - price)
            equity += unrealized
        return equity

    def _total_exposure_pct(self) -> float:
        """현재 총 노출 비율."""
        if not self.positions:
            return 0.0
        return sum(p.size_pct for p in self.positions.values())

    def _check_period_reset(self, now: datetime) -> None:
        """일/주 시작 시 기준 equity 리셋."""
        if self._last_daily_reset is None or now.date() != self._last_daily_reset.date():
            self._daily_start_equity = self.cash  # 대략적 (포지션 미반영)
            self._last_daily_reset = now
            if self._halted and self._halt_reason.startswith("daily"):
                self._halted = False
                self._halt_reason = ""
                logger.info("Daily halt lifted (new day)")

        if self._last_weekly_reset is None:
            self._last_weekly_reset = now
            self._weekly_start_equity = self.cash
        elif (now - self._last_weekly_reset).days >= 7:
            self._weekly_start_equity = self.cash
            self._last_weekly_reset = now
            if self._halted and self._halt_reason.startswith("weekly"):
                self._halted = False
                self._halt_reason = ""
                logger.info("Weekly halt lifted (new week)")

    def _check_loss_limits(self, equity: float) -> None:
        """일/주간 손실 한도 체크."""
        if self._halted:
            return

        daily_loss = (self._daily_start_equity - equity) / self._daily_start_equity
        if daily_loss > DAILY_LOSS_LIMIT:
            self._halted = True
            self._halt_reason = f"daily loss {daily_loss:.2%} > {DAILY_LOSS_LIMIT:.0%}"
            logger.warning(f"HALT: {self._halt_reason}")
            return

        weekly_loss = (self._weekly_start_equity - equity) / self._weekly_start_equity
        if weekly_loss > WEEKLY_LOSS_LIMIT:
            self._halted = True
            self._halt_reason = f"weekly loss {weekly_loss:.2%} > {WEEKLY_LOSS_LIMIT:.0%}"
            logger.warning(f"HALT: {self._halt_reason}")

    # ── 리포트 / 저장 ──────────────────────────────────────

    def stats(self) -> dict:
        """거래 통계 요약."""
        if not self.trade_log:
            return {"total_trades": 0}

        trades = self.trade_log
        wins = [t for t in trades if t.pnl_pct > 0]
        losses = [t for t in trades if t.pnl_pct <= 0]

        total_pnl = sum(t.pnl_amount for t in trades)
        total_cost = sum(t.cost_total for t in trades)

        # equity curve 기반 MDD
        mdd = self._calc_mdd()

        # Sharpe (일별 수익률 기반)
        sharpe = self._calc_sharpe()

        # coin별 통계
        coin_stats = {}
        for sym in self.active_universe:
            sym_trades = [t for t in trades if t.symbol == sym]
            if sym_trades:
                sym_wins = [t for t in sym_trades if t.pnl_pct > 0]
                coin_stats[sym] = {
                    "trades": len(sym_trades),
                    "win_rate": len(sym_wins) / len(sym_trades),
                    "total_pnl": round(sum(t.pnl_amount for t in sym_trades), 2),
                    "avg_pnl_pct": round(
                        sum(t.pnl_pct for t in sym_trades) / len(sym_trades), 6
                    ),
                }

        # exit reason 분포
        exit_reasons = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        return {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(trades), 4) if trades else 0,
            "total_pnl": round(total_pnl, 2),
            "total_cost": round(total_cost, 2),
            "avg_pnl_pct": round(
                sum(t.pnl_pct for t in trades) / len(trades), 6
            ),
            "avg_win_pct": round(
                sum(t.pnl_pct for t in wins) / len(wins), 6
            ) if wins else 0,
            "avg_loss_pct": round(
                sum(t.pnl_pct for t in losses) / len(losses), 6
            ) if losses else 0,
            "max_drawdown": round(mdd, 4),
            "sharpe": round(sharpe, 4),
            "avg_bars_held": round(
                sum(t.bars_held for t in trades) / len(trades), 1
            ),
            "exit_reasons": exit_reasons,
            "coin_stats": coin_stats,
            "halted": self._halted,
            "halt_reason": self._halt_reason,
        }

    def _calc_mdd(self) -> float:
        """Equity curve 기반 Maximum Drawdown."""
        if len(self.equity_curve) < 2:
            return 0.0
        peak = self.equity_curve[0].equity
        max_dd = 0.0
        for snap in self.equity_curve:
            if snap.equity > peak:
                peak = snap.equity
            dd = (peak - snap.equity) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _calc_sharpe(self, annual_factor: float = 365.25 * 6) -> float:
        """일별 수익률 기반 Sharpe ratio (연환산, 4h bar 기준)."""
        if len(self.equity_curve) < 3:
            return 0.0
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev = self.equity_curve[i - 1].equity
            curr = self.equity_curve[i].equity
            if prev > 0:
                returns.append((curr - prev) / prev)
        if not returns:
            return 0.0
        import statistics
        mean_r = statistics.mean(returns)
        std_r = statistics.stdev(returns) if len(returns) > 1 else 1e-9
        if std_r < 1e-12:
            return 0.0
        return mean_r / std_r * (annual_factor ** 0.5)

    def save_report(self, path: str | Path) -> Path:
        """전체 리포트를 JSON으로 저장."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "initial_capital": self.initial_capital,
            "final_equity": self.equity_curve[-1].equity if self.equity_curve else self.cash,
            "bars_processed": self._bar_count,
            "active_universe": self.active_universe,
            "config": {
                "fixed_size_pct": self.fixed_size_pct,
                "cost_bps": self.cost_bps,
                "max_per_coin_pct": MAX_PER_COIN_PCT,
                "max_total_exposure": MAX_TOTAL_EXPOSURE,
                "daily_loss_limit": DAILY_LOSS_LIMIT,
                "weekly_loss_limit": WEEKLY_LOSS_LIMIT,
            },
            "stats": self.stats(),
            "trade_log": [asdict(t) for t in self.trade_log],
            "equity_curve": [asdict(s) for s in self.equity_curve],
            "open_positions": {
                sym: {
                    "action": pos.action.value,
                    "entry_price": pos.entry_price,
                    "entry_time": pos.entry_time.isoformat(),
                    "size_pct": pos.size_pct,
                    "bars_held": pos.bars_held,
                    "tp": pos.take_profit_price,
                    "sl": pos.stop_loss_price,
                    "ttl_remaining": pos.ttl_bars - pos.bars_held,
                }
                for sym, pos in self.positions.items()
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Paper trader report saved: {path}")
        return path

    def force_close_all(
        self,
        current_prices: dict[str, float],
        now: datetime | None = None,
    ) -> list[TradeRecord]:
        """모든 포지션 강제 청산."""
        now = now or datetime.now(timezone.utc)
        closed = []
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            price = current_prices.get(symbol, pos.entry_price)
            trade = self._close_position(pos, price, now, "MANUAL")
            closed.append(trade)
        self.positions.clear()
        return closed

    @property
    def is_halted(self) -> bool:
        return self._halted

    def status_summary(self, current_prices: dict[str, float]) -> str:
        """현재 상태 요약 문자열."""
        equity = self._calc_equity(current_prices)
        pnl_pct = (equity - self.initial_capital) / self.initial_capital
        n_open = len(self.positions)
        n_trades = len(self.trade_log)
        exposure = self._total_exposure_pct()

        lines = [
            f"=== Paper Trader Status ===",
            f"Equity: {equity:,.0f} ({pnl_pct:+.2%})",
            f"Cash: {self.cash:,.0f}",
            f"Open: {n_open} positions ({exposure:.1%} exposure)",
            f"Trades: {n_trades} completed",
            f"Halted: {self._halted} ({self._halt_reason})" if self._halted else f"Halted: No",
        ]

        if self.positions:
            lines.append("--- Open Positions ---")
            for sym, pos in self.positions.items():
                price = current_prices.get(sym, pos.entry_price)
                if pos.is_long:
                    unr = (price - pos.entry_price) / pos.entry_price
                else:
                    unr = (pos.entry_price - price) / pos.entry_price
                lines.append(
                    f"  {sym} {pos.action.value} @ {pos.entry_price:.4f} "
                    f"now={price:.4f} unr={unr:+.2%} "
                    f"bars={pos.bars_held}/{pos.ttl_bars}"
                )

        return "\n".join(lines)
