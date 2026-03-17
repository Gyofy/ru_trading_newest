"""Trade Audit -- 트레이드 단위 감사표.

평가 우선순위 (accuracy/MCC는 보조):
  1. net EV (equity%/trade) -- funding/cost 포함
  2. n_trades (거래 횟수)
  3. 95% CI (신뢰구간)
  4. max drawdown
  5. fill rate / maker ratio
  6. funding 포함 후 EV
  7. accuracy / MCC (보조)

DOT 분해 검증: 예측력 기여 vs 비용절감 기여 분리.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import scipy.stats as stats

from src.execution.cost_model import CostModel, CostBreakdown, ExitType
from src.models.regime_filter import RegimeFilter, Regime4


@dataclass
class CoinAuditResult:
    """코인별 감사 결과."""
    symbol: str

    # ── 1순위: net EV ──
    n_trades: int = 0
    ev_gross_eq: float = 0.0      # 비용 전 EV (equity%/trade)
    ev_net_eq: float = 0.0        # 비용 후 EV
    ev_net_ci_lo: float = 0.0     # 95% CI lower
    ev_net_ci_hi: float = 0.0     # 95% CI upper

    # ── 비용 분해 ──
    cost_total_eq: float = 0.0
    cost_entry_fee: float = 0.0
    cost_exit_fee: float = 0.0
    cost_slippage: float = 0.0
    cost_funding: float = 0.0
    cost_miss_fill: float = 0.0

    # ── 성과 ──
    win_rate: float = 0.0
    rr_ratio: float = 0.0
    bep_win_rate: float = 0.0
    margin_over_bep: float = 0.0  # S2 - BEP
    max_drawdown_eq: float = 0.0
    max_consecutive_loss: int = 0
    profit_factor: float = 0.0

    # ── 체결 ──
    fill_rate: float = 0.0        # 시그널 → 실체결 비율
    maker_ratio: float = 0.0      # maker 체결 비율

    # ── 보유/회전 ──
    avg_holding_bars: float = 0.0
    avg_holding_hours: float = 0.0
    monthly_turnover: float = 0.0  # 월간 거래 횟수

    # ── 보조 지표 ──
    s2_accuracy: float = 0.0
    mcc: float = 0.0

    # ── EV 분해: 예측력 vs 비용절감 ──
    ev_from_prediction: float = 0.0   # 예측력이 만든 EV
    ev_from_cost_saving: float = 0.0  # 비용구조가 만든 EV 개선

    # ── 레짐별 ──
    regime_ev: dict = field(default_factory=dict)

    @property
    def grade(self) -> str:
        """종합 판정."""
        if self.ev_net_ci_lo > 0 and self.n_trades >= 20:
            return "A -- 실전 투입 가능"
        elif self.ev_net_eq > 0 and self.n_trades >= 10:
            return "B -- Demo 검증 필요"
        elif self.ev_net_eq > 0:
            return "C -- 데이터 부족"
        else:
            return "D -- 제외 권장"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "grade": self.grade,
            "n_trades": self.n_trades,
            "ev_net_eq_pct": f"{self.ev_net_eq:.4%}",
            "ev_95ci": f"[{self.ev_net_ci_lo:.4%}, {self.ev_net_ci_hi:.4%}]",
            "cost_total_eq_pct": f"{self.cost_total_eq:.4%}",
            "win_rate": f"{self.win_rate:.1%}",
            "rr": f"{self.rr_ratio:.3f}",
            "bep": f"{self.bep_win_rate:.1%}",
            "margin": f"{self.margin_over_bep:+.1%}p",
            "max_dd": f"{self.max_drawdown_eq:.2%}",
            "fill_rate": f"{self.fill_rate:.1%}",
            "maker_ratio": f"{self.maker_ratio:.1%}",
            "avg_hold_h": f"{self.avg_holding_hours:.1f}h",
            "monthly_turns": f"{self.monthly_turnover:.0f}",
            "s2_acc": f"{self.s2_accuracy:.1%}",
            "mcc": f"{self.mcc:.3f}",
            "ev_pred": f"{self.ev_from_prediction:.4%}",
            "ev_cost": f"{self.ev_from_cost_saving:.4%}",
            "regime_ev": self.regime_ev,
        }


class TradeAuditor:
    """트레이드 단위 감사 엔진.

    Usage:
        auditor = TradeAuditor(cost_model, regime_filter)
        result = auditor.audit_coin(
            symbol="XRP",
            s2_accuracy=0.70, mcc=0.243,
            k_upper=2.0, k_lower=1.5,
            atr_pct=0.0098, entry_price=2.30,
            ...
        )
        print(result.grade)
    """

    def __init__(
        self,
        cost_model: CostModel | None = None,
        regime_filter: RegimeFilter | None = None,
        risk_frac: float = 0.005,
        bar_minutes: int = 240,
    ):
        self.cost_model = cost_model or CostModel()
        self.regime_filter = regime_filter or RegimeFilter()
        self.risk_frac = risk_frac
        self.bar_minutes = bar_minutes

    def audit_coin(
        self,
        symbol: str,
        s2_accuracy: float,
        mcc: float,
        k_upper: float,
        k_lower: float,
        atr_pct: float,
        entry_price: float,
        s1_accuracy: float = 0.50,
        n_trades_estimate: int = 30,
        holding_bars_win: float = 2.5,
        holding_bars_loss: float = 1.5,
        funding_rate: float | None = None,
        fill_rate: float = 0.85,
        maker_ratio: float = 0.90,
        ohlcv_df: pd.DataFrame | None = None,
    ) -> CoinAuditResult:
        """코인별 종합 감사."""
        result = CoinAuditResult(symbol=symbol)

        # ── net EV from cost model ──
        ev_data = self.cost_model.compute_net_ev(
            s2_accuracy=s2_accuracy,
            k_upper=k_upper,
            k_lower=k_lower,
            atr_pct=atr_pct,
            entry_price=entry_price,
            risk_frac=self.risk_frac,
            holding_bars_win=holding_bars_win,
            holding_bars_loss=holding_bars_loss,
            bar_minutes=self.bar_minutes,
            funding_rate=funding_rate,
        )

        cost: CostBreakdown = ev_data["cost"]

        result.ev_gross_eq = ev_data["ev_gross_eq"]
        result.ev_net_eq = ev_data["ev_net_eq"]
        result.cost_total_eq = cost.total_eq
        result.cost_entry_fee = cost.entry_fee_eq
        result.cost_exit_fee = cost.exit_fee_eq
        result.cost_slippage = cost.slippage_eq
        result.cost_funding = cost.funding_eq
        result.cost_miss_fill = cost.miss_fill_eq

        result.win_rate = s2_accuracy
        result.rr_ratio = ev_data["rr"]
        result.bep_win_rate = ev_data["bep"]
        result.margin_over_bep = ev_data["margin_pct"]

        result.s2_accuracy = s2_accuracy
        result.mcc = mcc
        result.fill_rate = fill_rate
        result.maker_ratio = maker_ratio
        result.n_trades = n_trades_estimate

        # ── 보유시간 ──
        avg_bars = s2_accuracy * holding_bars_win + (1 - s2_accuracy) * holding_bars_loss
        result.avg_holding_bars = avg_bars
        result.avg_holding_hours = avg_bars * self.bar_minutes / 60

        # ── 월간 turnover ──
        # 6 bars/day × s1_pass_rate × fill_rate × 30 days
        s1_pass_rate = max(0.3, s1_accuracy)  # conservative
        signals_per_day = 6 * s1_pass_rate
        result.monthly_turnover = signals_per_day * fill_rate * 30

        # ── 95% CI ──
        result.ev_net_ci_lo, result.ev_net_ci_hi = self._compute_ci(
            ev=ev_data["ev_net_eq"],
            win_rate=s2_accuracy,
            win_eq=ev_data["win_eq"],
            loss_eq=ev_data["loss_eq"],
            cost_eq=cost.total_eq,
            n=n_trades_estimate,
        )

        # ── Max drawdown estimate ──
        result.max_drawdown_eq = self._estimate_max_dd(
            win_rate=s2_accuracy,
            loss_eq=ev_data["loss_eq"] + cost.total_eq,
            n_trades=n_trades_estimate,
        )

        # ── Max consecutive losses ──
        result.max_consecutive_loss = self._estimate_max_streak(
            loss_rate=1 - s2_accuracy,
            n_trades=n_trades_estimate,
        )

        # ── Profit factor ──
        avg_win = ev_data["win_eq"] - cost.total_eq
        avg_loss = ev_data["loss_eq"] + cost.total_eq
        if avg_loss > 0:
            result.profit_factor = (s2_accuracy * avg_win) / ((1 - s2_accuracy) * avg_loss)
        else:
            result.profit_factor = float("inf")

        # ── EV decomposition: prediction vs cost savings ──
        result.ev_from_prediction, result.ev_from_cost_saving = (
            self._decompose_ev(
                s2_accuracy=s2_accuracy,
                k_upper=k_upper,
                k_lower=k_lower,
                atr_pct=atr_pct,
                entry_price=entry_price,
                cost=cost,
            )
        )

        # ── Regime EV (if OHLCV provided) ──
        if ohlcv_df is not None and len(ohlcv_df) > 0:
            result.regime_ev = self._regime_ev_estimate(
                df=ohlcv_df,
                s2_accuracy=s2_accuracy,
                ev_net=ev_data["ev_net_eq"],
            )

        return result

    def audit_portfolio(
        self,
        coins: dict,
        atr_data: dict[str, float],
        price_data: dict[str, float],
        ohlcv_data: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, CoinAuditResult]:
        """포트폴리오 전체 감사.

        Args:
            coins: {symbol: {s2, mcc, k_upper, k_lower, ...}}
            atr_data: {symbol: atr_pct}
            price_data: {symbol: entry_price}
        """
        results = {}
        for symbol, params in coins.items():
            ohlcv = ohlcv_data.get(symbol) if ohlcv_data else None
            results[symbol] = self.audit_coin(
                symbol=symbol,
                s2_accuracy=params["s2"],
                mcc=params["mcc"],
                k_upper=params["k_upper"],
                k_lower=params["k_lower"],
                atr_pct=atr_data.get(symbol, 0.01),
                entry_price=price_data.get(symbol, 1.0),
                s1_accuracy=params.get("s1", 0.50),
                funding_rate=params.get("funding_rate"),
                ohlcv_df=ohlcv,
            )
        return results

    def print_audit_report(self, results: dict[str, CoinAuditResult]) -> str:
        """감사 리포트 출력."""
        lines = []
        lines.append("=" * 80)
        lines.append("  TRADE AUDIT REPORT -- net EV 기반 종합 감사")
        lines.append("=" * 80)
        lines.append("")

        # Sort by ev_net_eq descending
        sorted_coins = sorted(
            results.values(),
            key=lambda r: r.ev_net_eq,
            reverse=True,
        )

        for r in sorted_coins:
            lines.append(f"{'─' * 60}")
            lines.append(f"  {r.symbol}  |  Grade: {r.grade}")
            lines.append(f"{'─' * 60}")
            lines.append(f"  net EV:        {r.ev_net_eq:+.4%} /trade (eq%)")
            lines.append(f"  95% CI:        [{r.ev_net_ci_lo:+.4%}, {r.ev_net_ci_hi:+.4%}]")
            lines.append(f"  gross EV:      {r.ev_gross_eq:+.4%}")
            lines.append(f"  total cost:    {r.cost_total_eq:.4%}")
            lines.append(f"    entry fee:   {r.cost_entry_fee:.4%}")
            lines.append(f"    exit fee:    {r.cost_exit_fee:.4%}")
            lines.append(f"    slippage:    {r.cost_slippage:.4%}")
            lines.append(f"    funding:     {r.cost_funding:.4%}")
            lines.append(f"    miss-fill:   {r.cost_miss_fill:.4%}")
            lines.append(f"")
            lines.append(f"  win rate:      {r.win_rate:.1%}  |  BEP: {r.bep_win_rate:.1%}  |  margin: {r.margin_over_bep:+.1%}p")
            lines.append(f"  R:R:           {r.rr_ratio:.3f}")
            lines.append(f"  profit factor: {r.profit_factor:.2f}")
            lines.append(f"  max DD (est):  {r.max_drawdown_eq:.2%}")
            lines.append(f"  max streak:    {r.max_consecutive_loss} losses")
            lines.append(f"")
            lines.append(f"  fill rate:     {r.fill_rate:.0%}  |  maker: {r.maker_ratio:.0%}")
            lines.append(f"  avg hold:      {r.avg_holding_hours:.1f}h ({r.avg_holding_bars:.1f} bars)")
            lines.append(f"  monthly turns: ~{r.monthly_turnover:.0f}")
            lines.append(f"")
            lines.append(f"  S2 accuracy:   {r.s2_accuracy:.1%}  |  MCC: {r.mcc:.3f}")
            lines.append(f"  EV from pred:  {r.ev_from_prediction:+.4%}")
            lines.append(f"  EV from cost:  {r.ev_from_cost_saving:+.4%}")

            if r.regime_ev:
                lines.append(f"")
                lines.append(f"  Regime breakdown:")
                for regime, ev_info in r.regime_ev.items():
                    lines.append(f"    {regime:12s}: pct={ev_info['pct']:.0%}  ev_adj={ev_info['ev_adj']:+.4%}")

            lines.append("")

        # ── Portfolio summary ──
        lines.append("=" * 80)
        lines.append("  PORTFOLIO SUMMARY")
        lines.append("=" * 80)
        total_monthly = sum(
            r.ev_net_eq * r.monthly_turnover
            for r in sorted_coins if r.ev_net_eq > 0
        )
        lines.append(f"  Est. monthly return (positive EV coins only): {total_monthly:+.2%}")
        lines.append(f"  Coins with grade A/B: {sum(1 for r in sorted_coins if r.grade.startswith(('A', 'B')))}")
        lines.append(f"  Recommended: {', '.join(r.symbol for r in sorted_coins if r.grade.startswith(('A', 'B')))}")
        lines.append("")

        report = "\n".join(lines)
        return report

    # ── Internal helpers ──

    def _compute_ci(
        self,
        ev: float,
        win_rate: float,
        win_eq: float,
        loss_eq: float,
        cost_eq: float,
        n: int,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """트레이드 결과의 95% CI 추정.

        Bernoulli 분포 기반: 각 trade는 +win_eq-cost 또는 -loss_eq-cost.
        """
        if n < 2:
            return (ev - 0.01, ev + 0.01)

        # Per-trade outcomes
        win_outcome = win_eq - cost_eq
        loss_outcome = -(loss_eq + cost_eq)

        # Variance of per-trade returns
        var = win_rate * (win_outcome - ev) ** 2 + (1 - win_rate) * (loss_outcome - ev) ** 2
        se = np.sqrt(var / n)

        z = stats.norm.ppf((1 + confidence) / 2)
        return (ev - z * se, ev + z * se)

    def _estimate_max_dd(
        self,
        win_rate: float,
        loss_eq: float,
        n_trades: int,
    ) -> float:
        """최대 드로다운 추정 (Monte Carlo simplified).

        보수적 추정: 연속 패배 * 패배 크기.
        """
        max_streak = self._estimate_max_streak(1 - win_rate, n_trades)
        return max_streak * loss_eq

    def _estimate_max_streak(
        self,
        loss_rate: float,
        n_trades: int,
    ) -> int:
        """최대 연속 패배 추정.

        E[max streak] ≈ log(n) / log(1/p) for Bernoulli trials.
        """
        if loss_rate <= 0 or loss_rate >= 1 or n_trades <= 0:
            return 0
        expected = np.log(n_trades) / np.log(1 / loss_rate)
        return int(np.ceil(expected))

    def _decompose_ev(
        self,
        s2_accuracy: float,
        k_upper: float,
        k_lower: float,
        atr_pct: float,
        entry_price: float,
        cost: CostBreakdown,
    ) -> tuple[float, float]:
        """EV를 예측력 기여 vs 비용구조 기여로 분해.

        baseline: 50% 승률 (동전 던지기)
        - ev_from_prediction = 실제 EV - baseline EV
        - ev_from_cost_saving = baseline EV - worst_cost EV

        여기서 baseline EV는 현재 비용구조에서 50% 승률일 때의 EV.
        """
        rr = k_upper / k_lower
        risk = self.risk_frac

        # 현재 비용구조에서 50% 승률 EV
        ev_baseline = 0.50 * risk * rr - 0.50 * risk - cost.total_eq

        # 실제 EV
        ev_actual = s2_accuracy * risk * rr - (1 - s2_accuracy) * risk - cost.total_eq

        # flat 비용(0.30%)에서 실제 승률 EV
        flat_cost = 0.0030 * (risk / (k_lower * atr_pct))  # 0.30% × notional ratio
        ev_flat_cost = s2_accuracy * risk * rr - (1 - s2_accuracy) * risk - flat_cost

        ev_from_prediction = ev_actual - ev_baseline
        ev_from_cost_saving = ev_actual - ev_flat_cost  # negative = cost helped

        return ev_from_prediction, -ev_from_cost_saving

    def _regime_ev_estimate(
        self,
        df: pd.DataFrame,
        s2_accuracy: float,
        ev_net: float,
    ) -> dict:
        """레짐별 EV 추정 (시계열 비중 × EV 조정).

        정확한 레짐별 EV는 개별 거래 기록이 필요하지만,
        여기서는 레짐 비중 + 레짐 특성 기반 조정.
        """
        regimes = self.regime_filter.classify_series(df)
        valid = regimes[regimes != Regime4.UNKNOWN]

        if len(valid) == 0:
            return {}

        counts = valid.value_counts()
        total = len(valid)

        # 레짐별 EV 조정 계수 (경험적)
        # 추세장에서는 방향 예측이 쉬워 EV 상승
        # 횡보 고변동에서는 whipsaw로 EV 하락
        ev_adjustments = {
            Regime4.TREND_UP: 1.2,
            Regime4.TREND_DOWN: 1.1,
            Regime4.RANGE_LOW: 0.7,
            Regime4.RANGE_HIGH: 0.5,
        }

        result = {}
        for regime in Regime4:
            if regime == Regime4.UNKNOWN:
                continue
            cnt = counts.get(regime, 0)
            pct = cnt / total if total > 0 else 0
            adj = ev_adjustments.get(regime, 1.0)
            result[regime.value] = {
                "pct": pct,
                "ev_adj": ev_net * adj,
                "n_bars": cnt,
            }

        return result
