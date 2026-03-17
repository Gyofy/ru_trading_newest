"""Cost Model — 트레이드 비용 정밀 산출.

모든 비용을 equity% 단위로 통일.
stop-distance sizing에서 SL 폭에 따라 노셔널이 달라지므로
비용이 코인/파라미터별로 상이함을 반영.

비용 구성:
  1. entry_fee: maker(Post-Only) or taker(IOC/market)
  2. exit_fee:  SL=taker(시장가), TP=maker(지정가), time_stop=taker
  3. slippage:  entry+exit 양쪽
  4. funding:   보유시간 × funding_rate × 방향
  5. miss_fill_penalty: Post-Only reject 확률 × 놓친 시그널 기대값
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ExitType(str, Enum):
    TAKE_PROFIT = "take_profit"   # limit (maker)
    STOP_LOSS = "stop_loss"       # market (taker)
    TIME_STOP = "time_stop"       # market (taker)
    EMERGENCY = "emergency"       # market (taker)


@dataclass
class FeeSchedule:
    """Binance USDT-M Futures 기본 수수료 (Regular user).

    실제 운영 시 My Fee Rate에서 확인하여 override.
    maker 0.02% / taker 0.05% (BNB 할인 미적용 기준)
    """
    maker_fee: float = 0.0002     # 0.02%
    taker_fee: float = 0.0005     # 0.05%
    slippage_entry: float = 0.0003  # 0.03% (Post-Only → 낮음)
    slippage_exit_limit: float = 0.0001  # TP → 미미
    slippage_exit_market: float = 0.0005  # SL market → 높음


@dataclass
class FundingConfig:
    """Funding rate 설정."""
    interval_hours: float = 8.0   # Binance 기본 8h
    default_rate: float = 0.0001  # 0.01% per interval (시장 평균)


@dataclass
class MissFillConfig:
    """Post-Only 미체결 비용 모델."""
    reject_prob: float = 0.15     # Post-Only reject 확률 (보수적 추정)
    missed_ev_pct: float = 0.0015 # 놓친 시그널의 추정 EV (equity% 소수, 0.15%)


@dataclass
class CostBreakdown:
    """트레이드 비용 분해 — 모든 단위 equity%."""
    entry_fee_eq: float = 0.0     # 진입 수수료 (equity%)
    exit_fee_eq: float = 0.0      # 청산 수수료 (equity%)
    slippage_eq: float = 0.0      # 슬리피지 (equity%)
    funding_eq: float = 0.0       # funding rate (equity%)
    miss_fill_eq: float = 0.0     # 미체결 비용 (equity%)
    total_eq: float = 0.0         # 합계 (equity%)

    # 참조값
    notional_ratio: float = 0.0   # notional / equity
    entry_method: str = "maker"
    exit_method: str = "taker"
    holding_hours: float = 0.0

    def __repr__(self):
        return (f"Cost(total={self.total_eq:.4%} | "
                f"entry={self.entry_fee_eq:.4%} exit={self.exit_fee_eq:.4%} "
                f"slip={self.slippage_eq:.4%} fund={self.funding_eq:.4%} "
                f"miss={self.miss_fill_eq:.4%} | "
                f"not/eq={self.notional_ratio:.3f}x)")


class CostModel:
    """트레이드 비용 엔진.

    Usage:
        model = CostModel()
        cost = model.estimate_trade_cost(
            entry_price=2.30, sl_price=2.27, tp_price=2.35,
            risk_frac=0.005, atr_pct=0.0098,
            side="BUY", holding_bars=3,
        )
        print(cost.total_eq)  # 0.00102 → 0.102% of equity
    """

    def __init__(
        self,
        fee_schedule: FeeSchedule | None = None,
        funding_config: FundingConfig | None = None,
        miss_fill_config: MissFillConfig | None = None,
    ):
        self.fees = fee_schedule or FeeSchedule()
        self.funding = funding_config or FundingConfig()
        self.miss_fill = miss_fill_config or MissFillConfig()

    def estimate_trade_cost(
        self,
        entry_price: float,
        sl_price: float,
        tp_price: float | None = None,
        risk_frac: float = 0.005,
        atr_pct: float = 0.01,
        side: str = "BUY",
        exit_type: ExitType = ExitType.STOP_LOSS,
        holding_bars: float = 3.0,
        bar_minutes: int = 240,
        funding_rate: float | None = None,
        entry_is_maker: bool = True,
    ) -> CostBreakdown:
        """트레이드 비용 산출 (모든 결과 equity%).

        Args:
            entry_price: 진입 가격
            sl_price: 손절 가격
            tp_price: 익절 가격 (EV 가중 시 사용)
            risk_frac: equity 대비 리스크 비율 (default 0.5%)
            atr_pct: ATR / price (소수, e.g. 0.0098)
            side: "BUY" or "SELL"
            exit_type: 청산 방법
            holding_bars: 예상 보유 bar 수
            bar_minutes: bar 길이 (분)
            funding_rate: 심볼별 funding rate (None → default)
            entry_is_maker: Post-Only 성공 시 True
        """
        # ── 1. Position sizing → notional/equity ratio ──
        stop_dist = abs(entry_price - sl_price)
        stop_dist_pct = stop_dist / entry_price
        if stop_dist_pct < 1e-6:
            stop_dist_pct = 0.003  # fallback

        notional_ratio = risk_frac / stop_dist_pct

        # ── 2. Entry fee (equity%) ──
        if entry_is_maker:
            entry_fee_rate = self.fees.maker_fee
            slip_entry = self.fees.slippage_entry
        else:
            entry_fee_rate = self.fees.taker_fee
            slip_entry = self.fees.slippage_exit_market

        entry_fee_eq = entry_fee_rate * notional_ratio

        # ── 3. Exit fee — 방법에 따라 다름 ──
        if exit_type == ExitType.TAKE_PROFIT:
            exit_fee_rate = self.fees.maker_fee    # limit order
            slip_exit = self.fees.slippage_exit_limit
        else:
            # SL, time_stop, emergency → 전부 taker
            exit_fee_rate = self.fees.taker_fee
            slip_exit = self.fees.slippage_exit_market

        exit_fee_eq = exit_fee_rate * notional_ratio

        # ── 4. Slippage (equity%) ──
        slippage_eq = (slip_entry + slip_exit) * notional_ratio

        # ── 5. Funding (equity%) ──
        holding_hours = holding_bars * bar_minutes / 60
        fr = funding_rate if funding_rate is not None else self.funding.default_rate
        n_funding_periods = holding_hours / self.funding.interval_hours
        funding_eq = abs(fr) * n_funding_periods * notional_ratio

        # ── 6. Miss-fill penalty (equity%) ──
        # Post-Only가 reject되면 시그널을 놓침.
        # 비용 = reject 확률 × 놓친 EV / (1 - reject 확률)
        # 이는 실제 체결된 트레이드에 배분되는 기회비용
        if entry_is_maker and self.miss_fill.reject_prob > 0:
            miss_fill_eq = (
                self.miss_fill.reject_prob
                * self.miss_fill.missed_ev_pct
                / (1 - self.miss_fill.reject_prob + 1e-10)
            )
        else:
            miss_fill_eq = 0.0

        total = entry_fee_eq + exit_fee_eq + slippage_eq + funding_eq + miss_fill_eq

        return CostBreakdown(
            entry_fee_eq=entry_fee_eq,
            exit_fee_eq=exit_fee_eq,
            slippage_eq=slippage_eq,
            funding_eq=funding_eq,
            miss_fill_eq=miss_fill_eq,
            total_eq=total,
            notional_ratio=notional_ratio,
            entry_method="maker" if entry_is_maker else "taker",
            exit_method="maker" if exit_type == ExitType.TAKE_PROFIT else "taker",
            holding_hours=holding_hours,
        )

    def estimate_weighted_cost(
        self,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        risk_frac: float = 0.005,
        win_rate: float = 0.60,
        holding_bars_win: float = 2.5,
        holding_bars_loss: float = 1.5,
        bar_minutes: int = 240,
        funding_rate: float | None = None,
    ) -> CostBreakdown:
        """승/패 가중 평균 비용.

        승리 시 TP(maker), 패배 시 SL(taker) — exit 방법이 다르므로
        비용을 가중평균.
        """
        cost_win = self.estimate_trade_cost(
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            risk_frac=risk_frac,
            exit_type=ExitType.TAKE_PROFIT,
            holding_bars=holding_bars_win,
            bar_minutes=bar_minutes,
            funding_rate=funding_rate,
        )
        cost_loss = self.estimate_trade_cost(
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            risk_frac=risk_frac,
            exit_type=ExitType.STOP_LOSS,
            holding_bars=holding_bars_loss,
            bar_minutes=bar_minutes,
            funding_rate=funding_rate,
        )

        # Weighted average
        w_entry = win_rate * cost_win.entry_fee_eq + (1 - win_rate) * cost_loss.entry_fee_eq
        w_exit = win_rate * cost_win.exit_fee_eq + (1 - win_rate) * cost_loss.exit_fee_eq
        w_slip = win_rate * cost_win.slippage_eq + (1 - win_rate) * cost_loss.slippage_eq
        w_fund = win_rate * cost_win.funding_eq + (1 - win_rate) * cost_loss.funding_eq
        w_miss = cost_win.miss_fill_eq  # miss-fill은 동일

        return CostBreakdown(
            entry_fee_eq=w_entry,
            exit_fee_eq=w_exit,
            slippage_eq=w_slip,
            funding_eq=w_fund,
            miss_fill_eq=w_miss,
            total_eq=w_entry + w_exit + w_slip + w_fund + w_miss,
            notional_ratio=cost_win.notional_ratio,
            entry_method="maker",
            exit_method="weighted",
            holding_hours=win_rate * cost_win.holding_hours + (1 - win_rate) * cost_loss.holding_hours,
        )

    def compute_net_ev(
        self,
        s2_accuracy: float,
        k_upper: float,
        k_lower: float,
        atr_pct: float,
        entry_price: float,
        risk_frac: float = 0.005,
        holding_bars_win: float = 2.5,
        holding_bars_loss: float = 1.5,
        bar_minutes: int = 240,
        funding_rate: float | None = None,
    ) -> dict:
        """net EV 산출 — 비용 엔진 통합.

        Returns dict with:
            ev_eq: net expected value (equity% per trade)
            ev_gross: cost 전 EV
            cost: CostBreakdown
            bep: breakeven win rate
            margin_pct: S2 - BEP (percentage points)
            rr: risk-reward ratio
        """
        rr = k_upper / k_lower
        win_eq = risk_frac * rr
        loss_eq = risk_frac

        # Barrier prices (for cost calc)
        sl_dist = k_lower * atr_pct * entry_price
        tp_dist = k_upper * atr_pct * entry_price
        sl_price = entry_price - sl_dist  # assume BUY side
        tp_price = entry_price + tp_dist

        # Weighted cost
        cost = self.estimate_weighted_cost(
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            risk_frac=risk_frac,
            win_rate=s2_accuracy,
            holding_bars_win=holding_bars_win,
            holding_bars_loss=holding_bars_loss,
            bar_minutes=bar_minutes,
            funding_rate=funding_rate,
        )

        ev_gross = s2_accuracy * win_eq - (1 - s2_accuracy) * loss_eq
        ev_net = ev_gross - cost.total_eq

        # Breakeven
        # BEP × win_eq - (1-BEP) × loss_eq - cost = 0
        # BEP × (win_eq + loss_eq) = loss_eq + cost
        bep = (loss_eq + cost.total_eq) / (win_eq + loss_eq) if (win_eq + loss_eq) > 0 else 1.0

        return {
            "ev_net_eq": ev_net,
            "ev_gross_eq": ev_gross,
            "cost": cost,
            "bep": bep,
            "margin_pct": s2_accuracy - bep,
            "rr": rr,
            "win_eq": win_eq,
            "loss_eq": loss_eq,
        }


def compute_fee_threshold(
    cost_model: CostModel,
    atr_pct: float = 0.01,
    entry_price: float = 1.0,
    risk_frac: float = 0.005,
    k_lower: float = 1.5,
) -> float:
    """라벨링용 fee threshold 산출.

    기존 static 0.2% 대신 비용 모델에서 동적으로 계산.
    노셔널 비례 비용을 가격 변동률로 역변환.
    """
    stop_dist_pct = k_lower * atr_pct
    notional_ratio = risk_frac / stop_dist_pct

    # 라운드트립 비용 (가격 대비)
    rt_cost_pct = (
        (cost_model.fees.maker_fee + cost_model.fees.taker_fee)  # entry maker + exit taker
        + cost_model.fees.slippage_entry
        + cost_model.fees.slippage_exit_market
    )

    return rt_cost_pct
