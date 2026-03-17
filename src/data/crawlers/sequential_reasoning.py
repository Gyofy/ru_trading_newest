"""순차적 사고 엔진 (sequential-thinking 스킬).

수집된 모든 데이터를 구조화된 논리 체인으로 연결하여
최종 투자 결론을 도출합니다.
"""

from datetime import datetime
from typing import Any


def run_sequential_analysis(
    ticker: str,
    price_data: dict,
    technical: dict,
    sentiment_agg: dict,
    strategies: list[dict],
    top_news: list[dict],
) -> str:
    """모든 데이터를 종합하여 구조화된 투자 리포트를 생성합니다."""

    report = []
    report.append(f"{'='*70}")
    report.append(f"  [{ticker}] 종합 투자 분석 리포트")
    report.append(f"  생성: {datetime.now().strftime('%Y-%m-%d %H:%M KST')}")
    report.append(f"{'='*70}")

    # === STEP 1: 시장 데이터 ===
    report.append(f"\n## STEP 1: 시장 데이터 (CoinGecko)")
    if price_data:
        report.append(f"  현재가: ${price_data.get('price_usd', 0):,.2f}")
        report.append(f"  시총 순위: #{price_data.get('market_cap_rank', '?')}")
        report.append(f"  24h: {price_data.get('change_24h_pct', 0):+.2f}%")
        report.append(f"  7일: {price_data.get('change_7d_pct', 0):+.2f}%")
        report.append(f"  30일: {price_data.get('change_30d_pct', 0):+.2f}%")
        report.append(f"  ATH 대비: {price_data.get('ath_change_pct', 0):+.1f}%")
        report.append(f"  24h 거래량: ${price_data.get('total_volume_usd', 0):,.0f}")

        # 추세 판단
        chg_30d = price_data.get("change_30d_pct", 0)
        chg_7d = price_data.get("change_7d_pct", 0)
        if chg_30d < -20:
            trend = "강한 하락 추세"
        elif chg_30d < -5:
            trend = "약한 하락 추세"
        elif chg_30d > 20:
            trend = "강한 상승 추세"
        elif chg_30d > 5:
            trend = "약한 상승 추세"
        else:
            trend = "횡보/보합"

        report.append(f"  >> 추세 판단: {trend}")
    else:
        report.append(f"  [데이터 없음]")

    # === STEP 2: 기술적 분석 ===
    report.append(f"\n## STEP 2: 기술적 분석")
    if technical and "error" not in technical:
        report.append(f"  종합 판정: {technical['overall']} (점수: {technical['signal_score']:+.1f})")
        for sig in technical.get("signals", []):
            icon = {"bullish": "+", "bearish": "-", "neutral": "o"}[sig["direction"]]
            report.append(f"  [{icon}] {sig['indicator']}: {sig['value']} -> {sig['signal']}")

        ind = technical.get("indicators", {})
        rsi = ind.get("rsi_14")
        if rsi is not None:
            if rsi < 30:
                report.append(f"  >> RSI {rsi:.1f} = 과매도 구간 -> 반등 가능성")
            elif rsi > 70:
                report.append(f"  >> RSI {rsi:.1f} = 과매수 구간 -> 조정 가능성")
    else:
        report.append(f"  [분석 불가]")

    # === STEP 3: 뉴스 감성 분석 ===
    report.append(f"\n## STEP 3: 뉴스/소셜 감성")
    report.append(f"  전체 심리: {sentiment_agg.get('overall', '?').upper()} ({sentiment_agg.get('avg_score', 0):+.3f})")
    report.append(f"  데이터: {sentiment_agg.get('count', 0)}건")
    report.append(f"  강세 {sentiment_agg.get('bullish_count', 0)} / 약세 {sentiment_agg.get('bearish_count', 0)} / 중립 {sentiment_agg.get('neutral_count', 0)}")

    if top_news:
        report.append(f"\n  주요 뉴스:")
        for i, news in enumerate(top_news[:5], 1):
            s = news.get("sentiment", {})
            score = s.get("score", 0)
            icon = "+" if score > 0 else "-" if score < 0 else "o"
            report.append(f"  {i}. [{icon}] {news.get('title', '')[:80]}")

    # === STEP 4: 전략 평가 ===
    report.append(f"\n## STEP 4: 전략 평가 (8개 전략 검토)")
    viable = [s for s in strategies if s.get("viable")]
    for i, s in enumerate(strategies[:5], 1):
        mark = ">>>" if i <= 3 and s.get("viable") else "   "
        report.append(f"  {mark} {i}. {s['name']} [{s['direction']}] 점수: {s['score']:+.1f}")
        if s.get("rationale"):
            for r in s["rationale"][:2]:
                report.append(f"       - {r}")

    # === STEP 5: 논리적 결론 도출 ===
    report.append(f"\n{'='*70}")
    report.append(f"## STEP 5: 최종 결론 (Sequential Reasoning)")
    report.append(f"{'='*70}")

    conclusion = _derive_conclusion(ticker, price_data, technical, sentiment_agg, strategies)
    for line in conclusion:
        report.append(line)

    return "\n".join(report)


def _derive_conclusion(
    ticker: str,
    price_data: dict,
    technical: dict,
    sentiment_agg: dict,
    strategies: list[dict],
) -> list[str]:
    """데이터를 논리적으로 연결하여 최종 결론을 도출합니다."""
    lines = []

    # 데이터 수집
    chg_30d = price_data.get("change_30d_pct", 0) if price_data else 0
    chg_7d = price_data.get("change_7d_pct", 0) if price_data else 0
    chg_24h = price_data.get("change_24h_pct", 0) if price_data else 0
    ath_drop = price_data.get("ath_change_pct", 0) if price_data else 0
    ta_overall = technical.get("overall", "NEUTRAL") if technical else "NEUTRAL"
    ta_score = technical.get("signal_score", 0) if technical else 0
    sent = sentiment_agg.get("overall", "neutral")
    sent_score = sentiment_agg.get("avg_score", 0)
    top_strategy = strategies[0] if strategies else {}

    # 전제 (Premises)
    lines.append(f"\n### 전제 (Premises)")
    lines.append(f"  P1: {ticker} 30일 변동률 = {chg_30d:+.1f}%")
    lines.append(f"  P2: 기술적 분석 판정 = {ta_overall} ({ta_score:+.1f})")
    lines.append(f"  P3: 뉴스 감성 = {sent.upper()} ({sent_score:+.3f})")
    lines.append(f"  P4: ATH 대비 = {ath_drop:+.1f}%")
    lines.append(f"  P5: 최고 전략 = {top_strategy.get('name', '?')} ({top_strategy.get('score', 0):+.1f})")

    # 추론 (Inference)
    lines.append(f"\n### 추론 (Inference)")

    bull_points = 0
    bear_points = 0
    reasons_bull = []
    reasons_bear = []

    # 가격 추세
    if chg_30d < -15:
        bear_points += 2
        reasons_bear.append(f"30일 {chg_30d:+.1f}% 하락 → 하락 추세 확인")
    elif chg_30d > 15:
        bull_points += 2
        reasons_bull.append(f"30일 {chg_30d:+.1f}% 상승 → 상승 추세 확인")

    # 과매도/과매수 역발상
    if ath_drop < -60:
        bull_points += 1
        reasons_bull.append(f"ATH 대비 {ath_drop:.0f}% → 역사적 저평가 구간")

    # 기술적 분석
    if ta_score > 2:
        bull_points += 2
        reasons_bull.append(f"기술 지표 강세 ({ta_overall})")
    elif ta_score < -2:
        bear_points += 2
        reasons_bear.append(f"기술 지표 약세 ({ta_overall})")

    # 감성
    if sent_score > 0.2:
        bull_points += 1
        reasons_bull.append(f"시장 심리 강세 ({sent_score:+.3f})")
    elif sent_score < -0.2:
        bear_points += 1
        reasons_bear.append(f"시장 심리 약세 ({sent_score:+.3f})")

    # 극단적 공포 = 역발상 기회
    if sent_score < -0.4:
        bull_points += 1
        reasons_bull.append(f"극단적 공포 → 역발상 매수 기회 가능")

    for r in reasons_bull:
        lines.append(f"  [강세] {r}")
    for r in reasons_bear:
        lines.append(f"  [약세] {r}")

    lines.append(f"\n  강세 포인트: {bull_points} / 약세 포인트: {bear_points}")

    # 결론 (Conclusion)
    lines.append(f"\n### 결론 (Conclusion)")
    net = bull_points - bear_points

    if net >= 3:
        action = "적극 매수"
        confidence = min(9, 5 + net)
        position = "현물 매수 또는 선물 롱 (저레버리지)"
    elif net >= 1:
        action = "소량 매수 / DCA"
        confidence = min(7, 4 + net)
        position = "분할 매수 (DCA) 권고"
    elif net >= -1:
        action = "관망"
        confidence = 5
        position = "신규 진입 보류, 기존 포지션 유지"
    elif net >= -3:
        action = "숏 검토 / 포지션 축소"
        confidence = min(7, 4 + abs(net))
        position = "선물 숏 (저레버리지) 또는 포지션 축소"
    else:
        action = "적극 숏 / 전량 매도"
        confidence = min(9, 5 + abs(net))
        position = "선물 숏 또는 현물 전량 매도"

    lines.append(f"  행동: {action}")
    lines.append(f"  확신도: {confidence}/10")
    lines.append(f"  포지션: {position}")

    # 리스크 경고
    lines.append(f"\n### 리스크 경고")
    if bear_points > 0:
        lines.append(f"  - 약세 요인 {bear_points}개 존재 → 손절 라인 필수 설정")
    if abs(chg_24h) > 5:
        lines.append(f"  - 24h 변동 {chg_24h:+.1f}% → 높은 변동성, 포지션 사이즈 축소 권고")
    if ath_drop < -70:
        lines.append(f"  - ATH 대비 {ath_drop:.0f}% → 장기 침체 가능성 고려")
    lines.append(f"  - 이 분석은 AI 기반 참고자료이며, 투자 판단의 최종 책임은 투자자에게 있습니다")

    return lines
