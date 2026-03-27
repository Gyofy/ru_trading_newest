"""거래 종료 시점 분석 리포트 생성.

trades.jsonl + trade_analysis.jsonl + trajectories.jsonl 을 읽어
data/reports/tsmom_paper/analysis_report.md 에 저장.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


STATE_DIR = Path("data/reports/tsmom_paper")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def generate_report() -> str:
    """분석 리포트 생성 후 파일 저장. 리포트 문자열 반환."""
    trades = _load_jsonl(STATE_DIR / "trades.jsonl")
    analysis = _load_jsonl(STATE_DIR / "trade_analysis.jsonl")
    trajectories = _load_jsonl(STATE_DIR / "trajectories.jsonl")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# v6.1 Trade Analysis Report",
        f"Generated: {now}",
        f"",
    ]

    if not trades:
        lines.append("_No trades recorded yet._")
        report = "\n".join(lines)
        (STATE_DIR / "analysis_report.md").write_text(report, encoding="utf-8")
        return report

    # ── 전체 통계 ──────────────────────────────────────────────────────────────
    n = len(trades)
    pnls = [t.get("pnl_net_pct", t.get("pnl_pct", 0)) for t in trades]
    pnls_usd = [t.get("pnl_usd", 0) for t in trades]
    wins = sum(1 for p in pnls_usd if p > 0)
    losses = n - wins
    wr = wins / n * 100 if n else 0
    avg_pnl = float(np.mean(pnls)) * 100
    total_pnl_usd = sum(pnls_usd)
    sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(n)) if np.std(pnls) > 0 else 0
    gross_w = sum(p for p in pnls_usd if p > 0)
    gross_l = abs(sum(p for p in pnls_usd if p < 0))
    pf = gross_w / gross_l if gross_l > 0 else float("inf")

    eq_curve = list(np.cumsum(pnls))
    peak = 0.0
    mdd = 0.0
    for v in eq_curve:
        peak = max(peak, v)
        mdd = min(mdd, v - peak)

    bars_held = [t.get("bars_held", 0) for t in trades]
    avg_hold = float(np.mean(bars_held)) if bars_held else 0

    lines += [
        "## 전체 통계",
        "",
        f"| 지표 | 값 |",
        f"|------|-----|",
        f"| 총 거래 수 | {n}건 |",
        f"| 승률 | {wr:.1f}% ({wins}W / {losses}L) |",
        f"| 평균 PnL | {avg_pnl:+.3f}% |",
        f"| 누적 PnL | {total_pnl_usd:+.4f} USDT |",
        f"| per-trade Sharpe | {sharpe:.3f} |",
        f"| Profit Factor | {pf:.2f} |",
        f"| MDD (누적) | {mdd*100:.2f}% |",
        f"| 평균 보유 시간 | {avg_hold:.1f}봉 ({avg_hold:.0f}h) |",
        "",
    ]

    # ── 청산 사유별 ────────────────────────────────────────────────────────────
    by_reason: dict[str, list[float]] = {}
    for t in trades:
        r = t.get("exit_reason", "UNKNOWN")
        by_reason.setdefault(r, []).append(t.get("pnl_usd", 0))

    lines += ["## 청산 사유별", "", "| 사유 | 횟수 | WR | 합계 PnL |", "|------|------|-----|---------|"]
    reason_display = {
        "TP_HIT": "🎯 TP",
        "SL_HIT": "🛑 SL",
        "DIR_FLIP": "🔄 DIR_FLIP",
        "TIME_STOP": "⏱ TTL",
        "SHUTDOWN": "🖐 수동",
    }
    for reason, ps in sorted(by_reason.items()):
        r_n = len(ps)
        r_wr = sum(1 for p in ps if p > 0) / r_n * 100
        r_total = sum(ps)
        label = reason_display.get(reason, reason)
        lines.append(f"| {label} | {r_n} | {r_wr:.0f}% | {r_total:+.4f} USDT |")
    lines.append("")

    # ── 코인별 ────────────────────────────────────────────────────────────────
    by_coin: dict[str, list[float]] = {}
    for t in trades:
        c = t.get("coin", "?")
        by_coin.setdefault(c, []).append(t.get("pnl_usd", 0))

    lines += ["## 코인별", "", "| 코인 | 거래 | WR | 평균 PnL | 합계 PnL |",
              "|------|------|-----|---------|---------|"]
    for coin, ps in sorted(by_coin.items()):
        c_n = len(ps)
        c_wr = sum(1 for p in ps if p > 0) / c_n * 100
        c_avg = float(np.mean(ps))
        c_total = sum(ps)
        lines.append(f"| {coin} | {c_n} | {c_wr:.0f}% | {c_avg:+.4f} | {c_total:+.4f} USDT |")
    lines.append("")

    # ── 진입 조건 분석 (trade_analysis.jsonl ENTRY 레코드) ────────────────────
    entries = [r for r in analysis if r.get("event") == "ENTRY"]
    if entries:
        rsi_vals = [r.get("rsi", 50) for r in entries]
        cvd_vals = [r.get("cvd", 0) for r in entries]
        rel_vals = [r.get("relative_strength", 0) for r in entries]
        btc_vals = [r.get("btc_7d_return", 0) for r in entries]

        lines += [
            "## 진입 조건 통계 (평균값)",
            "",
            f"| 지표 | 평균 | 최소 | 최대 |",
            f"|------|------|------|------|",
            f"| RSI | {np.mean(rsi_vals):.1f} | {np.min(rsi_vals):.1f} | {np.max(rsi_vals):.1f} |",
            f"| CVD | {np.mean(cvd_vals):.4f} | {np.min(cvd_vals):.4f} | {np.max(cvd_vals):.4f} |",
            f"| 상대강도 | {np.mean(rel_vals)*100:+.2f}% | {np.min(rel_vals)*100:+.2f}% | {np.max(rel_vals)*100:+.2f}% |",
            f"| BTC 7d 수익률 | {np.mean(btc_vals)*100:+.2f}% | {np.min(btc_vals)*100:+.2f}% | {np.max(btc_vals)*100:+.2f}% |",
            "",
        ]

    # ── 청산 원인 진단 분포 ────────────────────────────────────────────────────
    exits_analysis = [r for r in analysis if r.get("event") == "EXIT"]
    if exits_analysis:
        cause_counts: dict[str, int] = {}
        for r in exits_analysis:
            c = r.get("cause", "unknown")
            cause_counts[c] = cause_counts.get(c, 0) + 1

        lines += ["## 청산 원인 진단", ""]
        for cause, cnt in sorted(cause_counts.items(), key=lambda x: -x[1]):
            pct = cnt / len(exits_analysis) * 100
            lines.append(f"- **{cause}**: {cnt}건 ({pct:.0f}%)")
        lines.append("")

    # ── 궤적 통계 (RL 데이터) ────────────────────────────────────────────────
    if trajectories:
        lines += [
            "## 궤적 데이터 (RL 학습용)",
            "",
            f"- 총 스텝 수: {len(trajectories)}개",
        ]
        unr = [r.get("unrealized_pnl", 0) for r in trajectories]
        if unr:
            lines.append(f"- 미실현 PnL 평균: {np.mean(unr)*100:+.3f}%")
            lines.append(f"- 미실현 PnL 분포: min={min(unr)*100:+.2f}% max={max(unr)*100:+.2f}%")
        lines.append("")

    # ── 개별 거래 기록 ────────────────────────────────────────────────────────
    lines += ["## 개별 거래 기록", ""]
    lines.append("| # | 코인 | 방향 | 진입가 | 청산가 | 사유 | 보유 | PnL(USDT) |")
    lines.append("|---|------|------|--------|--------|------|------|-----------|")
    for i, t in enumerate(trades, 1):
        side = "L" if t.get("side") == "BUY" else "S"
        ep = t.get("entry_price", 0)
        xp = t.get("exit_price", 0)
        reason = reason_display.get(t.get("exit_reason", "?"), t.get("exit_reason", "?"))
        bh = t.get("bars_held", 0)
        pnl_u = t.get("pnl_usd", 0)
        lines.append(
            f"| {i} | {t.get('coin','?')} | {side} | ${ep:.4f} | ${xp:.4f} | {reason} | {bh}h | {pnl_u:+.4f} |"
        )
    lines.append("")

    report = "\n".join(lines)
    (STATE_DIR / "analysis_report.md").write_text(report, encoding="utf-8")
    return report
