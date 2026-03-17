"""Hot Coin Report Generator - 핫 코인 리포트 생성.

"지금 어디서 돈이 되는가"를 한 눈에 보여주는 리포트.
"""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

from src.scanner.hot_scanner import HotCoin


def _fmt_vol(usd: float) -> str:
    if usd >= 1e9:
        return f"${usd/1e9:.1f}B"
    elif usd >= 1e6:
        return f"${usd/1e6:.0f}M"
    elif usd >= 1e3:
        return f"${usd/1e3:.0f}K"
    return f"${usd:.0f}"


def _bar(val: float, mx: float = 100, w: int = 10) -> str:
    f = int(val / mx * w)
    return "#" * f + "-" * (w - f)


def _dir_str(hc: HotCoin) -> str:
    if hc.direction == "neutral":
        return "---"
    return f"{hc.direction} ({hc.direction_confidence:.0%})"


def generate_hot_report(
    hot_coins: list[HotCoin],
    fear_greed: dict = None,
    btc_dominance: float = None,
    total_scanned: int = 0,
    top_n: int = 15,
) -> str:
    """핫 코인 리포트 생성."""
    now = datetime.now()
    top = hot_coins[:top_n]
    L = []

    L.append("# Hot Coin Discovery Report")
    L.append(f"**Generated**: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"**Scanned**: {total_scanned} coins -> {len(top)} hot")
    if fear_greed:
        L.append(f"**Fear & Greed**: {fear_greed['value']} ({fear_greed['classification']})")
    if btc_dominance:
        L.append(f"**BTC Dominance**: {btc_dominance:.1f}%")
    L.append("")

    # ── 1. Hot Ranking ──
    L.append("## 1. Opportunity Ranking")
    L.append("")
    L.append("| # | Coin | Opp | Heat | Chart | Media | Dir | 24h Vol | 4h% | 24h% | Catalyst |")
    L.append("|---|------|-----|------|-------|-------|-----|---------|-----|------|----------|")

    for i, hc in enumerate(top):
        catalyst = hc.catalyst[:30] + "..." if len(hc.catalyst) > 30 else hc.catalyst
        L.append(
            f"| {i+1} | **{hc.coin}** | **{hc.opportunity_score:.0f}** | "
            f"{hc.heat_score:.0f} | {hc.chart_score:.0f} | {hc.media_validation_score:.0f} | "
            f"{_dir_str(hc)} | {_fmt_vol(hc.daily_volume_usd)} | "
            f"{hc.ret_4h:+.1f}% | {hc.ret_24h:+.1f}% | {catalyst} |"
        )
    L.append("")

    # ── 2. Top 5 Deep Analysis ──
    L.append("## 2. Top Opportunities")
    L.append("")

    for hc in top[:5]:
        L.append(f"### {hc.coin} - ${hc.price:.4f} | {_dir_str(hc)}")
        L.append("")

        L.append(f"**Opportunity Score: {hc.opportunity_score:.0f}/100**")
        L.append("")

        # Anomaly reasons
        if hc.anomaly_reasons:
            L.append(f"**Why hot**: {', '.join(hc.anomaly_reasons)}")
            L.append("")

        # Score breakdown
        L.append("| Factor | Score | Detail |")
        L.append("|--------|-------|--------|")
        L.append(f"| Heat (anomaly) | {hc.heat_score:.0f} {_bar(hc.heat_score)} | "
                 f"Vol surge: {hc.volume_surge:.1f}x, Vol 24h: {hc.vol_24h_pct:.1f}% |")
        L.append(f"| Chart | {hc.chart_score:.0f} {_bar(hc.chart_score)} | "
                 f"RSI: {hc.rsi:.0f}, MACD: {hc.macd_signal}, ADX: {hc.adx:.0f}, "
                 f"BB: {hc.bb_position} |")
        L.append(f"| Media | {hc.media_validation_score:.0f} {_bar(hc.media_validation_score)} | "
                 f"Sentiment: {hc.media_sentiment}({hc.media_score:+.2f}), "
                 f"Buzz: {hc.media_buzz_count} items |")
        L.append(f"| Vol-Price | {'Aligned' if hc.vp_aligned else 'Divergent'} | "
                 f"Cost ratio: {hc.cost_ratio:.3f}, R/R: {hc.risk_reward:.1f}x |")
        L.append("")

        # Key levels
        if hc.support > 0 and hc.resistance > 0:
            L.append(f"**Key Levels**: Support ${hc.support:.4f} | "
                     f"Current ${hc.price:.4f} | Resistance ${hc.resistance:.4f}")
            L.append(f"**Expected Range (4h)**: +/-{hc.expected_range_pct:.1f}%")
            L.append("")

        # Headlines
        if hc.top_headlines:
            L.append("**Top Headlines**:")
            for hl in hc.top_headlines[:3]:
                sent_icon = "+" if hl["score"] > 0 else ("-" if hl["score"] < 0 else "=")
                L.append(f"- [{sent_icon}] {hl['title']} ({hl['source']})")
            L.append("")

    # ── 3. Signal Clusters ──
    L.append("## 3. Signal Clusters")
    L.append("")

    # 3a. 거래량 폭발
    vol_surge = [hc for hc in top if hc.volume_surge >= 1.5]
    if vol_surge:
        L.append("**Volume Explosion:**")
        for hc in sorted(vol_surge, key=lambda x: x.volume_surge, reverse=True):
            L.append(f"- {hc.coin}: {hc.volume_surge:.1f}x surge, "
                     f"{_fmt_vol(hc.daily_volume_usd)} daily")
        L.append("")

    # 3b. 강한 모멘텀
    strong_mom = [hc for hc in top if abs(hc.ret_4h) >= 2.0]
    if strong_mom:
        L.append("**Strong Momentum (4h >= 2%):**")
        for hc in sorted(strong_mom, key=lambda x: abs(x.ret_4h), reverse=True):
            L.append(f"- {hc.coin}: {hc.ret_4h:+.1f}% (4h), {hc.ret_24h:+.1f}% (24h)")
        L.append("")

    # 3c. 차트 신호
    chart_signals = [hc for hc in top if "cross" in hc.macd_signal]
    if chart_signals:
        L.append("**MACD Cross Signals:**")
        for hc in chart_signals:
            L.append(f"- {hc.coin}: {hc.macd_signal}, RSI {hc.rsi:.0f}")
        L.append("")

    # ── 4. Direction Map ──
    L.append("## 4. Direction Map")
    L.append("")
    longs = [hc for hc in top if hc.direction == "LONG"]
    shorts = [hc for hc in top if hc.direction == "SHORT"]
    neutrals = [hc for hc in top if hc.direction == "neutral"]

    if longs:
        L.append(f"**LONG**: {', '.join(f'{hc.coin}({hc.direction_confidence:.0%})' for hc in longs)}")
    if shorts:
        L.append(f"**SHORT**: {', '.join(f'{hc.coin}({hc.direction_confidence:.0%})' for hc in shorts)}")
    if neutrals:
        L.append(f"**Neutral**: {', '.join(hc.coin for hc in neutrals)}")
    L.append("")

    # ── 5. Action Items ──
    L.append("## 5. Action Items")
    L.append("")

    # Best opp
    best = [hc for hc in top[:5] if hc.direction != "neutral" and hc.direction_confidence >= 0.3]
    if best:
        L.append("**High-Conviction Entries:**")
        for hc in best:
            L.append(f"- **{hc.coin} {hc.direction}** (conf: {hc.direction_confidence:.0%})")
            L.append(f"  - Entry: ~${hc.price:.4f}, Range: +/-{hc.expected_range_pct:.1f}%")
            L.append(f"  - Support: ${hc.support:.4f}, Resistance: ${hc.resistance:.4f}")
            L.append(f"  - R/R: {hc.risk_reward:.1f}x, Chart: {hc.chart_score:.0f}, Heat: {hc.heat_score:.0f}")
    else:
        L.append("**No high-conviction entries found** - wait for clearer signals")
    L.append("")

    # Watch
    watch = [hc for hc in top[:10] if hc.direction_confidence < 0.3 and hc.heat_score >= 30]
    if watch:
        L.append("**Watch for setup:**")
        for hc in watch:
            L.append(f"- {hc.coin}: Heat {hc.heat_score:.0f}, needs direction clarity")
        L.append("")

    # Market mood
    avg_opp = sum(hc.opportunity_score for hc in top) / len(top) if top else 0
    avg_heat = sum(hc.heat_score for hc in top) / len(top) if top else 0
    if avg_heat >= 40:
        mood = "HOT - many coins moving, active market"
    elif avg_heat >= 25:
        mood = "WARM - selective opportunities exist"
    elif avg_heat >= 15:
        mood = "COOL - few signals, be patient"
    else:
        mood = "COLD - wait for market to wake up"
    L.append(f"**Market Temperature**: {mood} (avg heat: {avg_heat:.0f}, avg opp: {avg_opp:.0f})")
    L.append("")

    return "\n".join(L)


def save_hot_report(
    hot_coins: list[HotCoin],
    fear_greed: dict = None,
    btc_dominance: float = None,
    total_scanned: int = 0,
    top_n: int = 15,
) -> Path:
    """리포트 MD + JSON 저장."""
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    rdir = Path("data/reports") / date_str
    rdir.mkdir(parents=True, exist_ok=True)

    md = generate_hot_report(hot_coins, fear_greed, btc_dominance, total_scanned, top_n)
    md_path = rdir / f"hot_scan_{time_str}.md"
    md_path.write_text(md, encoding="utf-8")

    # JSON
    data = {
        "generated_at": datetime.now().isoformat(),
        "btc_dominance": btc_dominance,
        "fear_greed": fear_greed,
        "total_scanned": total_scanned,
        "top_n": top_n,
        "coins": [],
    }
    for hc in hot_coins[:top_n]:
        coin_data = {
            "coin": hc.coin, "price": hc.price,
            "opportunity_score": hc.opportunity_score,
            "heat_score": hc.heat_score,
            "chart_score": hc.chart_score,
            "media_validation_score": hc.media_validation_score,
            "direction": hc.direction,
            "direction_confidence": hc.direction_confidence,
            "volume_surge": hc.volume_surge,
            "ret_4h": hc.ret_4h, "ret_24h": hc.ret_24h,
            "daily_volume_usd": hc.daily_volume_usd,
            "rsi": hc.rsi, "macd_signal": hc.macd_signal,
            "adx": hc.adx, "support": hc.support, "resistance": hc.resistance,
            "media_sentiment": hc.media_sentiment,
            "media_score": hc.media_score,
            "media_buzz_count": hc.media_buzz_count,
            "anomaly_reasons": hc.anomaly_reasons,
            "top_headlines": hc.top_headlines,
            "catalyst": hc.catalyst,
        }
        data["coins"].append(coin_data)

    json_path = rdir / f"hot_scan_{time_str}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)

    (rdir / "hot_scan_latest.md").write_text(md, encoding="utf-8")

    print(f"\n[Report] Saved: {md_path}")
    return md_path
