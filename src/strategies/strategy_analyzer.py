"""StrategyAnalyzer — rule-based strategy improvement data pipeline.

Reads trade_context.jsonl and computes:
  1. Per-strategy performance metrics (WR, PnL, Sharpe, MDD, Profit Factor)
  2. Per-coin breakdown (which coins are most profitable per strategy)
  3. Signal quality analysis (z-score/CVD buckets vs outcomes)
  4. Time-of-day / session analysis (Asia/London/NY win rates)
  5. Volatility regime analysis (HIGH/MED/LOW performance)
  6. SL/TP optimization hints (optimal ATR multiples from MFE/MAE distribution)
  7. Exit reason analysis (SL_HIT vs TP_HIT ratio, hold time distribution)

Output:
  - data/reports/multi_strategy/strategy_analysis.json  (machine-readable)
  - data/reports/multi_strategy/strategy_analysis.md    (human-readable)

Usage:
    analyzer = StrategyAnalyzer(Path("data/reports/multi_strategy"))
    report = analyzer.run()
    print(report.to_markdown())
    analyzer.post_to_discord(report)
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("strategy_analyzer")


# ── Data structures ────────────────────────────────────────────────────────


@dataclass
class StrategyStats:
    name: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_usdt: float = 0.0
    total_pnl_pct: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    avg_win_usdt: float = 0.0
    avg_loss_usdt: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    avg_bars_held: float = 0.0
    avg_mfe_pct: float = 0.0
    avg_mae_pct: float = 0.0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    mdd_pct: float = 0.0
    win_rate: float = 0.0
    # Exit reason counts
    exit_sl: int = 0
    exit_tp: int = 0
    exit_time: int = 0
    exit_trailing: int = 0
    # Suggestions
    suggestions: list[str] = field(default_factory=list)


@dataclass
class CoinStats:
    coin: str
    strategy: str
    total_trades: int = 0
    win_rate: float = 0.0
    total_pnl_usdt: float = 0.0
    avg_pnl_pct: float = 0.0
    avg_mfe_pct: float = 0.0
    avg_mae_pct: float = 0.0
    avg_hold_bars: float = 0.0


@dataclass
class AnalysisReport:
    generated_at: str = ""
    total_trades: int = 0
    total_pnl_usdt: float = 0.0
    overall_win_rate: float = 0.0
    strategy_stats: list[StrategyStats] = field(default_factory=list)
    coin_stats: list[CoinStats] = field(default_factory=list)
    session_stats: dict[str, dict] = field(default_factory=dict)
    regime_stats: dict[str, dict] = field(default_factory=dict)
    zscore_buckets: dict[str, dict] = field(default_factory=dict)
    sl_tp_hints: dict[str, dict] = field(default_factory=dict)
    top_recommendations: list[str] = field(default_factory=list)


# ── Main Analyzer ──────────────────────────────────────────────────────────


class StrategyAnalyzer:
    """Analyze trade_context.jsonl and produce strategy improvement insights."""

    def __init__(self, report_dir: Path):
        self._dir = report_dir
        self._input = report_dir / "trade_context.jsonl"
        self._output_json = report_dir / "strategy_analysis.json"
        self._output_md = report_dir / "strategy_analysis.md"

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self, min_trades: int = 5) -> AnalysisReport:
        """Run full analysis pipeline. Returns AnalysisReport."""
        trades = self._load_trades()
        if not trades:
            logger.info("[Analyzer] No trades found")
            return AnalysisReport(generated_at=self._now_str())

        report = AnalysisReport(
            generated_at=self._now_str(),
            total_trades=len(trades),
            total_pnl_usdt=sum(t.get("pnl_usdt", 0) for t in trades),
        )
        wins = sum(1 for t in trades if t.get("pnl_usdt", 0) > 0)
        report.overall_win_rate = wins / len(trades) if trades else 0.0

        # 1. Per-strategy stats
        report.strategy_stats = self._compute_strategy_stats(trades)

        # 2. Per-coin stats
        report.coin_stats = self._compute_coin_stats(trades)

        # 3. Session analysis
        report.session_stats = self._compute_session_stats(trades)

        # 4. Volatility regime
        report.regime_stats = self._compute_regime_stats(trades)

        # 5. Z-score buckets
        report.zscore_buckets = self._compute_zscore_buckets(trades)

        # 6. SL/TP optimization hints
        report.sl_tp_hints = self._compute_sl_tp_hints(trades)

        # 7. Top recommendations
        report.top_recommendations = self._generate_recommendations(report)

        # Save outputs
        self._save_json(report)
        self._save_markdown(report)

        logger.info(
            f"[Analyzer] Analysis complete: {len(trades)} trades | "
            f"PnL={report.total_pnl_usdt:+.2f} USDT | WR={report.overall_win_rate:.1%}"
        )
        return report

    def discord_summary(self, report: AnalysisReport) -> str:
        """Generate compact Discord message from report."""
        if report.total_trades == 0:
            return "**📊 전략 분석**: 거래 없음"

        lines = [
            f"**📊 전략 분석 리포트** | {report.total_trades}건",
            f"총 PnL: `{report.total_pnl_usdt:+.2f} USDT` | 승률: `{report.overall_win_rate:.1%}`",
            "",
        ]

        for s in report.strategy_stats:
            pf_str = f"{s.profit_factor:.2f}" if 0 < s.profit_factor < 999 else ("INF" if s.profit_factor >= 999 else "N/A")
            lines.append(
                f"**[{s.name}]** {s.total_trades}건 | WR:`{s.win_rate:.0%}` "
                f"PnL:`{s.total_pnl_usdt:+.2f}` PF:`{pf_str}`"
            )

        if report.session_stats:
            best_session = max(
                report.session_stats.items(),
                key=lambda x: x[1].get("win_rate", 0) if x[1].get("trades", 0) >= 3 else -1,
            )
            lines.append(f"\n최고 세션: `{best_session[0]}` (WR {best_session[1].get('win_rate', 0):.0%})")

        if report.top_recommendations:
            lines.append("\n**💡 개선 제안:**")
            for r in report.top_recommendations[:3]:
                lines.append(f"• {r}")

        return "\n".join(lines)

    # ── Analysis methods ───────────────────────────────────────────────────

    def _compute_strategy_stats(self, trades: list[dict]) -> list[StrategyStats]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for t in trades:
            grouped[t.get("strategy", "unknown")].append(t)

        results = []
        for name, strat_trades in grouped.items():
            stats = StrategyStats(name=name, total_trades=len(strat_trades))
            pnls = [t.get("pnl_usdt", 0.0) for t in strat_trades]

            # Basic metrics
            for pnl in pnls:
                if pnl > 0:
                    stats.wins += 1
                    stats.gross_profit += pnl
                    stats.max_win = max(stats.max_win, pnl)
                else:
                    stats.losses += 1
                    stats.gross_loss += abs(pnl)
                    stats.max_loss = max(stats.max_loss, abs(pnl))

            stats.total_pnl_usdt = sum(pnls)
            stats.total_pnl_pct = sum(t.get("pnl_pct", 0) for t in strat_trades)
            stats.win_rate = stats.wins / stats.total_trades if stats.total_trades > 0 else 0
            stats.avg_win_usdt = stats.gross_profit / stats.wins if stats.wins > 0 else 0
            stats.avg_loss_usdt = stats.gross_loss / stats.losses if stats.losses > 0 else 0
            stats.avg_bars_held = sum(t.get("bars_held", 0) for t in strat_trades) / len(strat_trades)
            stats.avg_mfe_pct = sum(t.get("mfe_pct", 0) for t in strat_trades) / len(strat_trades)
            stats.avg_mae_pct = sum(t.get("mae_pct", 0) for t in strat_trades) / len(strat_trades)
            stats.profit_factor = stats.gross_profit / stats.gross_loss if stats.gross_loss > 0 else float("inf")

            # Sharpe (simplified: mean/std of pnl_pct returns — not absolute USDT)
            pct_returns = [t.get("pnl_pct", 0.0) for t in strat_trades]
            if len(pct_returns) >= 2:
                mean_r = sum(pct_returns) / len(pct_returns)
                std_r = math.sqrt(sum((p - mean_r) ** 2 for p in pct_returns) / len(pct_returns))
                stats.sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 1e-10 else 0.0

            # Max drawdown (sequential equity curve)
            equity = 0.0
            peak = 0.0
            max_dd = 0.0
            for pnl in pnls:
                equity += pnl
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            stats.mdd_pct = max_dd

            # Exit reasons
            for t in strat_trades:
                reason = t.get("exit_reason", "").upper()
                if "SL" in reason:
                    stats.exit_sl += 1
                elif "TP" in reason:
                    stats.exit_tp += 1
                elif "TIME" in reason or "TTL" in reason:
                    stats.exit_time += 1
                elif "TRAIL" in reason:
                    stats.exit_trailing += 1

            # Per-strategy suggestions
            if stats.win_rate < 0.40 and stats.total_trades >= 5:
                stats.suggestions.append(f"승률 {stats.win_rate:.0%} < 40% — 신호 필터 강화 필요")
            if stats.profit_factor < 1.2 and stats.total_trades >= 5:
                stats.suggestions.append(f"PF {stats.profit_factor:.2f} < 1.2 — TP 확대 또는 SL 축소 검토")
            if stats.exit_sl > stats.exit_tp * 2 and stats.total_trades >= 5:
                stats.suggestions.append(
                    f"SL히트({stats.exit_sl}) >> TP히트({stats.exit_tp}) — "
                    f"SL 거리 ATR×{max(2.0, stats.avg_mae_pct / max(stats.avg_mfe_pct, 0.001)):.1f} 검토"
                )
            if stats.avg_bars_held < 5 and stats.total_trades >= 5:
                stats.suggestions.append(f"평균 보유 {stats.avg_bars_held:.1f}바 — 너무 빠른 청산 (노이즈 SL?)")

            results.append(stats)

        return sorted(results, key=lambda s: s.total_pnl_usdt, reverse=True)

    def _compute_coin_stats(self, trades: list[dict]) -> list[CoinStats]:
        grouped: dict[tuple, list[dict]] = defaultdict(list)
        for t in trades:
            key = (t.get("coin", "?"), t.get("strategy", "?"))
            grouped[key].append(t)

        results = []
        for (coin, strategy), coin_trades in grouped.items():
            pnls = [t.get("pnl_usdt", 0) for t in coin_trades]
            wins = sum(1 for p in pnls if p > 0)
            results.append(CoinStats(
                coin=coin,
                strategy=strategy,
                total_trades=len(coin_trades),
                win_rate=wins / len(coin_trades) if coin_trades else 0,
                total_pnl_usdt=sum(pnls),
                avg_pnl_pct=sum(t.get("pnl_pct", 0) for t in coin_trades) / len(coin_trades),
                avg_mfe_pct=sum(t.get("mfe_pct", 0) for t in coin_trades) / len(coin_trades),
                avg_mae_pct=sum(t.get("mae_pct", 0) for t in coin_trades) / len(coin_trades),
                avg_hold_bars=sum(t.get("bars_held", 0) for t in coin_trades) / len(coin_trades),
            ))

        return sorted(results, key=lambda c: c.total_pnl_usdt, reverse=True)

    def _compute_session_stats(self, trades: list[dict]) -> dict[str, dict]:
        sessions = defaultdict(list)
        for t in trades:
            s = t.get("session_utc") or "Unknown"
            sessions[s].append(t.get("pnl_usdt", 0))

        result = {}
        for session, pnls in sessions.items():
            wins = sum(1 for p in pnls if p > 0)
            result[session] = {
                "trades": len(pnls),
                "win_rate": wins / len(pnls) if pnls else 0,
                "total_pnl": round(sum(pnls), 4),
                "avg_pnl": round(sum(pnls) / len(pnls), 4) if pnls else 0,
            }
        return result

    def _compute_regime_stats(self, trades: list[dict]) -> dict[str, dict]:
        regimes = defaultdict(list)
        for t in trades:
            r = t.get("volatility_regime") or "Unknown"
            regimes[r].append(t.get("pnl_usdt", 0))

        result = {}
        for regime, pnls in regimes.items():
            wins = sum(1 for p in pnls if p > 0)
            result[regime] = {
                "trades": len(pnls),
                "win_rate": wins / len(pnls) if pnls else 0,
                "total_pnl": round(sum(pnls), 4),
                "avg_pnl": round(sum(pnls) / len(pnls), 4) if pnls else 0,
            }
        return result

    def _compute_zscore_buckets(self, trades: list[dict]) -> dict[str, dict]:
        """CVD z-score bucket analysis: which strength leads to best outcomes."""
        buckets: dict[str, list[float]] = defaultdict(list)
        for t in trades:
            z = abs(t.get("cvd_z_score", 0))
            if z < 1.0:
                bucket = "z<1"
            elif z < 2.0:
                bucket = "z1-2"
            elif z < 3.0:
                bucket = "z2-3"
            elif z < 4.0:
                bucket = "z3-4"
            else:
                bucket = "z>4"
            buckets[bucket].append(t.get("pnl_usdt", 0))

        result = {}
        for bucket, pnls in sorted(buckets.items()):
            wins = sum(1 for p in pnls if p > 0)
            result[bucket] = {
                "trades": len(pnls),
                "win_rate": round(wins / len(pnls), 3) if pnls else 0,
                "avg_pnl": round(sum(pnls) / len(pnls), 4) if pnls else 0,
                "total_pnl": round(sum(pnls), 4),
            }
        return result

    def _compute_sl_tp_hints(self, trades: list[dict]) -> dict[str, dict]:
        """
        Compute optimal ATR multiples for SL and TP per strategy based on
        actual MFE/MAE distributions.

        Key insight:
          - Optimal SL = P75 of MAE distribution (allows most trades to breathe)
          - Optimal TP = P75 of MFE distribution for winning trades (capture actual moves)
          - Current SL distance vs MAE: if avg_mae > sl_dist, SL is being hit by noise
        """
        by_strategy: dict[str, list[dict]] = defaultdict(list)
        for t in trades:
            by_strategy[t.get("strategy", "unknown")].append(t)

        hints = {}
        for strategy, strat_trades in by_strategy.items():
            if len(strat_trades) < 5:
                continue

            # ATR-normalized MAE/MFE
            maes = []
            mfes_win = []
            for t in strat_trades:
                atr = t.get("entry_atr", 0)
                entry = t.get("entry_price", 1)
                mae = abs(t.get("mae_pct", 0))
                mfe = t.get("mfe_pct", 0)
                atr_pct = atr / entry if entry > 0 else 0
                if atr_pct > 0:
                    maes.append(mae / atr_pct)  # MAE in ATR units
                    if t.get("pnl_usdt", 0) > 0:
                        mfes_win.append(mfe / atr_pct)  # MFE (wins only)

            if not maes:
                continue

            maes_sorted = sorted(maes)
            mfes_sorted = sorted(mfes_win) if mfes_win else [2.0]

            def _pct(arr: list, p: float) -> float:
                """Safe percentile: clamp index to valid range."""
                if not arr:
                    return 0.0
                idx = min(int(len(arr) * p), len(arr) - 1)
                return arr[idx]

            p50_mae = _pct(maes_sorted, 0.50)
            p75_mae = _pct(maes_sorted, 0.75)
            p75_mfe = _pct(mfes_sorted, 0.75) if mfes_win else 2.0
            p50_mfe = _pct(mfes_sorted, 0.50) if mfes_win else 1.5

            # Current SL mult (from strategy params, if available)
            sample = strat_trades[0]
            current_sl_mult = sample.get("strategy_params", {}).get("sl_atr_mult", 3.0)

            hints[strategy] = {
                "sample_size": len(strat_trades),
                "mae_p50_atr": round(p50_mae, 2),
                "mae_p75_atr": round(p75_mae, 2),
                "mfe_p50_atr_wins": round(p50_mfe, 2),
                "mfe_p75_atr_wins": round(p75_mfe, 2),
                "current_sl_atr_mult": current_sl_mult,
                "suggested_sl_atr_mult": round(max(p75_mae * 1.2, 1.0), 1),
                "suggested_tp_atr_mult": round(max(p50_mfe * 0.9, 1.5), 1),
                "note": (
                    "SL이 너무 좁음 — MAE P75보다 작음" if current_sl_mult < p75_mae
                    else "SL이 너무 넓음 — 리스크 과다" if current_sl_mult > p75_mae * 2.0
                    else "SL 적절"
                ),
            }

        return hints

    def _generate_recommendations(self, report: AnalysisReport) -> list[str]:
        """Generate top-level actionable recommendations."""
        recs = []

        # Strategy-level
        for s in report.strategy_stats:
            if s.win_rate < 0.35 and s.total_trades >= 10:
                recs.append(f"[{s.name}] 승률 {s.win_rate:.0%} — 비활성화 또는 신호 필터 강화")
            if s.profit_factor > 2.0 and s.total_trades >= 10:
                recs.append(f"[{s.name}] PF {s.profit_factor:.1f} — 우수. 할당 증가 검토")

        # Session-level
        for session, stats in report.session_stats.items():
            if stats["trades"] >= 5 and stats["win_rate"] < 0.30:
                recs.append(f"{session} 세션 승률 {stats['win_rate']:.0%} — 해당 세션 거래 중단 고려")

        # Regime-level
        for regime, stats in report.regime_stats.items():
            if stats["trades"] >= 5 and regime == "HIGH" and stats["win_rate"] > 0.60:
                recs.append(f"고변동성 레짐에서 승률 {stats['win_rate']:.0%} — ATR 필터 추가 고려")
            if stats["trades"] >= 5 and regime == "LOW" and stats["win_rate"] < 0.35:
                recs.append(f"저변동성 레짐 승률 {stats['win_rate']:.0%} — 저변동성 시 진입 억제")

        # Z-score insights
        for bucket, stats in report.zscore_buckets.items():
            if stats["trades"] >= 5 and stats["win_rate"] > 0.65:
                recs.append(f"CVD {bucket} 구간에서 승률 {stats['win_rate']:.0%} — 이 구간 진입 우선화")

        # SL/TP hints
        for strategy, hint in report.sl_tp_hints.items():
            if hint["note"] != "SL 적절" and hint["sample_size"] >= 10:
                recs.append(
                    f"[{strategy}] SL 조정: 현재 ATR×{hint['current_sl_atr_mult']} "
                    f"→ 권장 ATR×{hint['suggested_sl_atr_mult']}"
                )

        return recs[:10]  # top 10

    # ── I/O ───────────────────────────────────────────────────────────────

    def _load_trades(self) -> list[dict]:
        if not self._input.exists():
            return []
        trades = []
        try:
            with open(self._input, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            trades.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"[Analyzer] Load failed: {e}")
        return trades

    def _save_json(self, report: AnalysisReport) -> None:
        try:
            data = asdict(report)
            self._output_json.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )
        except Exception as e:
            logger.error(f"[Analyzer] JSON save failed: {e}")

    def _save_markdown(self, report: AnalysisReport) -> None:
        try:
            md = self._to_markdown(report)
            self._output_md.write_text(md, encoding="utf-8")
        except Exception as e:
            logger.error(f"[Analyzer] MD save failed: {e}")

    def _to_markdown(self, report: AnalysisReport) -> str:
        lines = [
            f"# Strategy Analysis Report",
            f"Generated: {report.generated_at}",
            f"",
            f"## Summary",
            f"- Total trades: {report.total_trades}",
            f"- Total PnL: {report.total_pnl_usdt:+.4f} USDT",
            f"- Overall win rate: {report.overall_win_rate:.1%}",
            f"",
            f"## Per-Strategy",
            f"| Strategy | Trades | WR | PnL (USDT) | PF | Sharpe | MDD |",
            f"|----------|--------|----|-----------|----|----|-----|",
        ]
        for s in report.strategy_stats:
            pf = f"{s.profit_factor:.2f}" if s.profit_factor < 999 else "∞"
            lines.append(
                f"| {s.name} | {s.total_trades} | {s.win_rate:.0%} | "
                f"{s.total_pnl_usdt:+.2f} | {pf} | {s.sharpe:.2f} | {s.mdd_pct:.1%} |"
            )

        lines += ["", "## Per-Coin Top 10",
                  "| Coin | Strategy | Trades | WR | PnL |",
                  "|------|----------|--------|----|-----|"]
        for c in report.coin_stats[:10]:
            lines.append(
                f"| {c.coin} | {c.strategy} | {c.total_trades} | "
                f"{c.win_rate:.0%} | {c.total_pnl_usdt:+.2f} |"
            )

        lines += ["", "## Session Analysis",
                  "| Session | Trades | WR | Avg PnL |",
                  "|---------|--------|----|---------|"]
        for session, st in sorted(report.session_stats.items()):
            lines.append(
                f"| {session} | {st['trades']} | {st['win_rate']:.0%} | {st['avg_pnl']:+.4f} |"
            )

        lines += ["", "## Volatility Regime",
                  "| Regime | Trades | WR | Avg PnL |",
                  "|--------|--------|----|---------|"]
        for regime, st in sorted(report.regime_stats.items()):
            lines.append(
                f"| {regime} | {st['trades']} | {st['win_rate']:.0%} | {st['avg_pnl']:+.4f} |"
            )

        lines += ["", "## CVD Z-Score Buckets",
                  "| Z-Score | Trades | WR | Avg PnL |",
                  "|---------|--------|----|---------|"]
        for bucket, st in sorted(report.zscore_buckets.items()):
            lines.append(
                f"| {bucket} | {st['trades']} | {st['win_rate']:.0%} | {st['avg_pnl']:+.4f} |"
            )

        if report.sl_tp_hints:
            lines += ["", "## SL/TP Optimization Hints",
                      "| Strategy | Current SL ATR× | Suggested SL | Suggested TP | Note |",
                      "|----------|----------------|-------------|-------------|------|"]
            for strategy, h in report.sl_tp_hints.items():
                lines.append(
                    f"| {strategy} | ATR×{h['current_sl_atr_mult']} | "
                    f"ATR×{h['suggested_sl_atr_mult']} | "
                    f"ATR×{h['suggested_tp_atr_mult']} | {h['note']} |"
                )

        if report.top_recommendations:
            lines += ["", "## Top Recommendations"]
            for i, rec in enumerate(report.top_recommendations, 1):
                lines.append(f"{i}. {rec}")

        return "\n".join(lines)

    @staticmethod
    def _now_str() -> str:
        return datetime.now(timezone.utc).isoformat()
