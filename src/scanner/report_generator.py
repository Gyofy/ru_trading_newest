"""Report Generator - Top 15 단기수익 가능성 리포트.

거래량+변동성 기반으로 단기 트레이딩 기회가 높은 코인을 보여준다.
"""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

from src.scanner.tradability_scorer import CoinScore


def _bar(value: float, max_val: float = 100, width: int = 10) -> str:
    filled = int(value / max_val * width)
    return "#" * filled + "-" * (width - filled)


def _fmt_vol(usd: float) -> str:
    if usd >= 1e9:
        return f"${usd/1e9:.1f}B"
    elif usd >= 1e6:
        return f"${usd/1e6:.0f}M"
    elif usd >= 1e3:
        return f"${usd/1e3:.0f}K"
    return f"${usd:.0f}"


def _dir_label(s: CoinScore) -> str:
    if s.direction == "neutral":
        return "---"
    pct = f"{s.direction_strength:.0%}"
    return f"{s.direction} ({pct})"


def _risk_label(s: CoinScore) -> str:
    if not s.risk_flags:
        return ""
    return "[!]" if s.risk_deduction < 20 else "[!!]"


def generate_report(
    scores: list[CoinScore],
    btc_dominance: float = None,
    fear_greed: dict = None,
    top_n: int = 15,
) -> str:
    """Top N 단기수익 리포트 생성."""
    now = datetime.now()
    top = scores[:top_n]
    L = []  # lines

    L.append("# Short-term Profit Opportunity Report")
    L.append(f"**Generated**: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"**Scanned**: {len(scores)} coins -> Top {len(top)}")
    if fear_greed:
        L.append(f"**Fear & Greed**: {fear_greed['value']} ({fear_greed['classification']})")
    if btc_dominance:
        L.append(f"**BTC Dominance**: {btc_dominance:.1f}%")
    L.append("")

    # ── 1. Top 15 Ranking ──
    L.append("## 1. Profit Score Ranking")
    L.append("")
    L.append("| # | Coin | Score | VolPwr | VolQty | Profit | Mom | Dir | 24h Vol | 24h% | Risk |")
    L.append("|---|------|-------|--------|--------|--------|-----|-----|---------|------|------|")

    for i, s in enumerate(top):
        L.append(
            f"| {i+1} | **{s.coin}** | **{s.profit_score:.1f}** | "
            f"{s.volume_power:.0f} | {s.volatility_quality:.0f} | "
            f"{s.profit_ratio:.0f} | {s.momentum_clarity:.0f} | "
            f"{_dir_label(s)} | {_fmt_vol(s.daily_volume_usd)} | "
            f"{s.ret_24h:+.1f}% | {_risk_label(s)} |"
        )
    L.append("")

    # ── 2. Top 5 상세 ──
    L.append("## 2. Top 5 Detail")
    L.append("")

    for s in top[:5]:
        L.append(f"### #{top.index(s)+1} {s.coin} - ${s.price:.4f}")
        L.append("")
        L.append(f"**Profit Score: {s.profit_score:.1f}/100** | {_dir_label(s)}")
        L.append("")

        L.append("| Factor | Score | Key Data |")
        L.append("|--------|-------|----------|")
        L.append(f"| Volume Power | {s.volume_power:.0f} {_bar(s.volume_power)} | "
                 f"Daily: {_fmt_vol(s.daily_volume_usd)}, Surge: {s.volume_surge:.1f}x |")
        L.append(f"| Volatility Quality | {s.volatility_quality:.0f} {_bar(s.volatility_quality)} | "
                 f"24h: {s.vol_24h_pct:.2f}%, 7d: {s.vol_7d_pct:.2f}% |")
        L.append(f"| Profit Ratio | {s.profit_ratio:.0f} {_bar(s.profit_ratio)} | "
                 f"Range: {s.avg_bar_range_pct:.3f}%, Cost ratio: {s.cost_ratio:.3f} |")
        L.append(f"| Momentum | {s.momentum_clarity:.0f} {_bar(s.momentum_clarity)} | "
                 f"4h: {s.ret_4h:+.2f}%, 24h: {s.ret_24h:+.2f}%, ADX: {s.adx:.0f} |")
        L.append("")

        if s.risk_flags:
            L.append(f"**Risk**: {', '.join(s.risk_flags)} (deduction: -{s.risk_deduction:.0f})")
            L.append("")

        if s.vol_5m_burst > 0:
            L.append(f"**5m Volume Burst**: {s.vol_5m_burst:.1f}x (top 10% vs avg)")
            L.append("")

    # ── 3. 거래량 급등 알림 ──
    surging = [s for s in top if s.volume_surge >= 1.5]
    if surging:
        L.append("## 3. Volume Surge Alert")
        L.append("")
        L.append("Coins with 6h volume >= 1.5x average:")
        L.append("")
        for s in sorted(surging, key=lambda x: x.volume_surge, reverse=True):
            L.append(f"- **{s.coin}**: {s.volume_surge:.1f}x surge "
                     f"({_fmt_vol(s.daily_volume_usd)} daily, {s.ret_24h:+.1f}%)")
        L.append("")

    # ── 4. 방향성 요약 ──
    L.append("## 4. Direction Summary")
    L.append("")
    longs = [s for s in top if s.direction == "LONG"]
    shorts = [s for s in top if s.direction == "SHORT"]
    neutrals = [s for s in top if s.direction == "neutral"]

    if longs:
        coins_str = ", ".join(f"{s.coin}({s.direction_strength:.0%})" for s in longs)
        L.append(f"- **LONG**: {coins_str}")
    if shorts:
        coins_str = ", ".join(f"{s.coin}({s.direction_strength:.0%})" for s in shorts)
        L.append(f"- **SHORT**: {coins_str}")
    if neutrals:
        L.append(f"- **Neutral**: {', '.join(s.coin for s in neutrals)}")
    L.append("")

    # ── 5. 위험 종목 ──
    risky = [s for s in top if s.risk_deduction >= 15]
    if risky:
        L.append("## 5. Risk Warning")
        L.append("")
        for s in risky:
            L.append(f"- **{s.coin}**: {', '.join(s.risk_flags)} (penalty: -{s.risk_deduction:.0f})")
        L.append("")

    # ── 6. 핵심 요약 ──
    L.append("## 6. Action Summary")
    L.append("")

    # 상위 3개 = Best, 4~8 = Watchlist (상대 기준)
    best = top[:3]
    watch = top[3:8]

    L.append("**Best Opportunities (Top 3):**")
    for s in best:
        L.append(f"- {s.coin}: Score {s.profit_score:.1f}, {_dir_label(s)}, "
                 f"Vol {_fmt_vol(s.daily_volume_usd)}, 24h {s.ret_24h:+.1f}%")
    L.append("")

    L.append("**Watchlist Candidates (4th~8th):**")
    for s in watch:
        L.append(f"- {s.coin}: Score {s.profit_score:.1f}, {_dir_label(s)}, "
                 f"Vol {_fmt_vol(s.daily_volume_usd)}")
    L.append("")

    # 전체 시장 분위기
    avg_score = sum(s.profit_score for s in top) / len(top) if top else 0
    if avg_score >= 60:
        market_mood = "Active - plenty of short-term opportunities"
    elif avg_score >= 45:
        market_mood = "Normal - selective entry possible"
    elif avg_score >= 30:
        market_mood = "Quiet - wait or scalp only"
    else:
        market_mood = "Dead - not suitable for trading"
    L.append(f"**Market Mood**: {market_mood} (avg score: {avg_score:.1f})")
    L.append("")

    return "\n".join(L)


def save_report(
    scores: list[CoinScore],
    btc_dominance: float = None,
    fear_greed: dict = None,
    output_dir: str = None,
    top_n: int = 15,
) -> Path:
    """리포트를 MD + JSON으로 저장."""
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")

    if output_dir:
        rdir = Path(output_dir)
    else:
        rdir = Path("data/reports") / date_str
    rdir.mkdir(parents=True, exist_ok=True)

    # Markdown
    md = generate_report(scores, btc_dominance, fear_greed, top_n)
    md_path = rdir / f"profit_scan_{time_str}.md"
    md_path.write_text(md, encoding="utf-8")

    # JSON (전체 스코어 저장)
    data = {
        "generated_at": datetime.now().isoformat(),
        "btc_dominance": btc_dominance,
        "fear_greed": fear_greed,
        "top_n": top_n,
        "total_scanned": len(scores),
        "scores": [asdict(s) for s in scores],
    }
    json_path = rdir / f"profit_scan_{time_str}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)

    # latest 링크
    (rdir / "profit_scan_latest.md").write_text(md, encoding="utf-8")

    print(f"\n[Report] Saved: {md_path}")
    return md_path
