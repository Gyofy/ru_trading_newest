"""멀티 포지션 전략 분석기.

롱/숏/선물/헤지/차익거래 등 모든 방향의 투자전략을 검토하고
현 시점에서 최대 수익 가능한 전략을 도출합니다.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.crawlers.sentiment_analyzer import (
    analyze_batch,
    aggregate_sentiment,
    filter_by_ticker,
    analyze_sentiment_rule,
)


# === 전략 템플릿 ===
STRATEGY_TEMPLATES = {
    "spot_long": {
        "name": "현물 매수 (Spot Long)",
        "direction": "long",
        "leverage": 1,
        "description": "현물 시장에서 직접 매수 후 상승 시 매도",
        "best_when": "강한 상승 추세, 과매도 반등, 호재 모멘텀",
        "risk": "하락 시 원금 손실, 유동성 리스크",
    },
    "futures_long": {
        "name": "선물 롱 (Futures Long)",
        "direction": "long",
        "leverage_range": "2x-10x",
        "description": "선물 계약으로 레버리지 매수",
        "best_when": "확실한 상승 시그널, 변동성 축소 구간",
        "risk": "레버리지 청산, 펀딩비 부담, 급락 시 원금 초과 손실",
    },
    "futures_short": {
        "name": "선물 숏 (Futures Short)",
        "direction": "short",
        "leverage_range": "2x-10x",
        "description": "선물 계약으로 레버리지 공매도",
        "best_when": "확실한 하락 시그널, 과매수 구간, 악재 모멘텀",
        "risk": "숏 스퀴즈, 무한 손실 가능성, 급등 시 청산",
    },
    "hedge_long_short": {
        "name": "헤지 (Long + Short)",
        "direction": "neutral",
        "description": "강세 종목 롱 + 약세 종목 숏으로 시장 중립 포지션",
        "best_when": "방향성 불확실, 종목 간 상대적 강약 차이 존재",
        "risk": "양방향 동시 손실(양쪽 포지션 모두 불리하게 움직일 때)",
    },
    "dca_accumulation": {
        "name": "분할 매수 (DCA)",
        "direction": "long",
        "leverage": 1,
        "description": "일정 금액을 주기적으로 분할 매수하여 평단가 낮추기",
        "best_when": "장기 상승 확신, 단기 변동성 높은 구간, 바닥 확인 어려울 때",
        "risk": "지속적 하락 시 기회비용, 추세 전환 실패",
    },
    "scalp_short": {
        "name": "단기 숏 스캘핑",
        "direction": "short",
        "leverage_range": "3x-5x",
        "description": "단기 과매수 구간에서 빠른 숏 진입/청산",
        "best_when": "급등 후 과매수, RSI 70+ 구간, 저항선 부근",
        "risk": "타이밍 실패, 추세 지속 시 손실 확대",
    },
    "pair_trade": {
        "name": "페어 트레이딩 (BTC vs ALT)",
        "direction": "neutral",
        "description": "상관관계 높은 두 종목 간 괴리 발생 시 수렴 베팅",
        "best_when": "BTC 도미넌스 변화, 알트 시즌/비트 시즌 전환",
        "risk": "상관관계 붕괴, 괴리 확대",
    },
    "volatility_play": {
        "name": "변동성 매매 (Straddle/Strangle)",
        "direction": "neutral",
        "description": "옵션 또는 양방향 포지션으로 큰 변동성 자체에 베팅",
        "best_when": "중대 이벤트 앞두고 방향 불확실, 변동성 급증 예상",
        "risk": "횡보 시 시간가치 소모, 프리미엄 손실",
    },
}


def evaluate_strategies(
    ticker: str,
    items: list[dict],
    sentiment_agg: dict,
) -> list[dict[str, Any]]:
    """모든 전략을 현재 데이터 기반으로 평가합니다."""
    score_map = _compute_market_signals(items, sentiment_agg)
    evaluated = []

    for key, template in STRATEGY_TEMPLATES.items():
        strategy = dict(template)
        strategy["id"] = key
        score, rationale = _score_strategy(key, score_map, sentiment_agg)
        strategy["score"] = score          # -10 ~ +10
        strategy["rationale"] = rationale
        strategy["viable"] = score > 2.0
        evaluated.append(strategy)

    # 점수순 정렬
    evaluated.sort(key=lambda x: x["score"], reverse=True)
    return evaluated


def _compute_market_signals(items: list[dict], agg: dict) -> dict:
    """뉴스/소셜 데이터에서 시장 시그널을 추출합니다."""
    signals = {
        "sentiment_score": agg.get("avg_score", 0),
        "sentiment_label": agg.get("overall", "neutral"),
        "total_mentions": agg.get("count", 0),
        "bearish_ratio": 0,
        "bullish_ratio": 0,
        "fear_keywords": 0,
        "greed_keywords": 0,
        "crash_mentions": 0,
        "rally_mentions": 0,
        "regulation_mentions": 0,
        "whale_mentions": 0,
        "liquidation_mentions": 0,
        "institutional_mentions": 0,
        "high_engagement": 0,
    }

    total = agg.get("count", 1) or 1
    signals["bearish_ratio"] = agg.get("bearish_count", 0) / total
    signals["bullish_ratio"] = agg.get("bullish_count", 0) / total

    for item in items:
        text = f"{item.get('title', '')} {item.get('selftext', '')}".lower()

        if any(w in text for w in ["crash", "collapse", "plunge", "capitulation"]):
            signals["crash_mentions"] += 1
        if any(w in text for w in ["rally", "surge", "breakout", "moon", "pump"]):
            signals["rally_mentions"] += 1
        if any(w in text for w in ["sec", "regulation", "ban", "lawsuit"]):
            signals["regulation_mentions"] += 1
        if any(w in text for w in ["whale", "large holder", "accumulation"]):
            signals["whale_mentions"] += 1
        if any(w in text for w in ["liquidat", "margin call", "forced sell"]):
            signals["liquidation_mentions"] += 1
        if any(w in text for w in ["institutional", "etf", "fund", "blackrock"]):
            signals["institutional_mentions"] += 1
        if any(w in text for w in ["fear", "panic", "scared"]):
            signals["fear_keywords"] += 1
        if any(w in text for w in ["greed", "fomo", "euphori"]):
            signals["greed_keywords"] += 1

        score = item.get("score", 0)
        comments = item.get("num_comments", 0)
        if score > 100 or comments > 50:
            signals["high_engagement"] += 1

    return signals


def _score_strategy(
    strategy_id: str,
    signals: dict,
    agg: dict,
) -> tuple[float, list[str]]:
    """개별 전략의 적합도를 점수화합니다."""
    score = 0.0
    rationale = []
    sent = signals["sentiment_score"]
    bear_r = signals["bearish_ratio"]
    bull_r = signals["bullish_ratio"]

    if strategy_id == "spot_long":
        # 과매도 반등 노림
        if sent < -0.3:
            score += 3.0
            rationale.append("극단적 약세 심리 → 역발상 매수 기회")
        elif sent > 0.2:
            score += 2.0
            rationale.append("긍정적 모멘텀 유지")
        else:
            score += 0.5

        if signals["institutional_mentions"] > 2:
            score += 2.0
            rationale.append(f"기관/ETF 관련 뉴스 {signals['institutional_mentions']}건")

        if signals["fear_keywords"] > 3:
            score += 1.5
            rationale.append("공포 심리 과다 → 바닥 신호 가능")

        score -= signals["crash_mentions"] * 0.3

    elif strategy_id == "futures_long":
        if sent > 0.3:
            score += 4.0
            rationale.append("강한 강세 심리 → 레버리지 롱 적합")
        elif sent < -0.3:
            score -= 1.0
            rationale.append("약세 심리에서 레버리지 롱은 고위험")

        if signals["rally_mentions"] > 3:
            score += 2.0
            rationale.append(f"랠리/돌파 언급 {signals['rally_mentions']}건")

        if signals["liquidation_mentions"] > 1:
            score -= 2.0
            rationale.append("청산 뉴스 존재 → 레버리지 위험 증가")

    elif strategy_id == "futures_short":
        if sent < -0.2:
            score += 4.0
            rationale.append("약세 심리 → 숏 포지션 유리")
        elif sent > 0.3:
            score -= 2.0
            rationale.append("강세 심리에서 숏은 위험")

        if signals["crash_mentions"] > 2:
            score += 2.5
            rationale.append(f"급락/붕괴 언급 {signals['crash_mentions']}건 → 추가 하락 모멘텀")

        if bear_r > 0.4:
            score += 2.0
            rationale.append(f"약세 뉴스 비율 {bear_r:.0%}")

        if signals["whale_mentions"] > 1:
            score += 1.0
            rationale.append("고래 움직임 감지 → 변동성 확대 가능")

        # 과도한 약세에서는 숏 스퀴즈 위험
        if sent < -0.5 and signals["fear_keywords"] > 5:
            score -= 1.5
            rationale.append("극단적 공포 → 숏 스퀴즈 역풍 경계")

    elif strategy_id == "hedge_long_short":
        # 방향성 불확실할수록 유리
        if -0.15 < sent < 0.15:
            score += 4.0
            rationale.append("시장 방향성 불확실 → 헤지 전략 적합")
        else:
            score += 1.0

        if signals["regulation_mentions"] > 2:
            score += 1.5
            rationale.append("규제 이슈 → 방향 예측 어려움, 헤지 유리")

    elif strategy_id == "dca_accumulation":
        if sent < -0.2:
            score += 5.0
            rationale.append("하락장에서 DCA는 장기적 최적 전략")
        if signals["fear_keywords"] > 2:
            score += 2.0
            rationale.append("공포 구간 분할 매수 = 역사적 고수익")
        if signals["institutional_mentions"] > 1:
            score += 1.5
            rationale.append("기관 관심 유지 → 장기 펀더멘탈 건재")

    elif strategy_id == "scalp_short":
        if 0 < sent < 0.3 and signals["rally_mentions"] > 2:
            score += 3.5
            rationale.append("과열 조짐 → 단기 숏 스캘핑 기회")
        elif sent < -0.3:
            score += 2.0
            rationale.append("하락 추세 지속 → 반등 시 숏 진입")

        if signals["liquidation_mentions"] > 0:
            score += 1.5
            rationale.append("청산 연쇄 가능성 → 숏 스캘핑 유리")

    elif strategy_id == "pair_trade":
        score += 2.5  # 시장 중립 전략은 항상 일정 점수
        rationale.append("시장 중립 전략 — BTC 도미넌스 변화 모니터링 필요")
        if signals["high_engagement"] > 3:
            score += 1.0
            rationale.append("높은 관심 → 종목 간 차별화 기회")

    elif strategy_id == "volatility_play":
        extreme = abs(sent) > 0.3
        high_activity = signals["high_engagement"] > 5
        if extreme or high_activity:
            score += 4.0
            rationale.append("높은 변동성 → 양방향 베팅 유리")
        if signals["regulation_mentions"] > 2:
            score += 2.0
            rationale.append("규제 이벤트 → 큰 가격 움직임 예상")

    # 공통 보정
    if signals["total_mentions"] < 10:
        score *= 0.7
        rationale.append("데이터 부족 — 신뢰도 할인 적용")

    return round(score, 2), rationale


def format_strategy_report(
    ticker: str,
    strategies: list[dict],
    sentiment_agg: dict,
    items: list[dict],
) -> str:
    """전략 평가 결과를 리포트로 포맷합니다."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"  [{ticker.upper()}] 멀티 포지션 전략 분석 리포트")
    lines.append(f"  생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"{'='*60}")
    lines.append(f"\n## 시장 심리: {sentiment_agg['overall'].upper()} ({sentiment_agg['avg_score']:+.3f})")
    lines.append(f"데이터: {sentiment_agg['count']}건 (강세 {sentiment_agg['bullish_count']} / 약세 {sentiment_agg['bearish_count']} / 중립 {sentiment_agg['neutral_count']})")

    lines.append(f"\n## 전략 랭킹 (점수순)")
    lines.append(f"{'─'*60}")

    for rank, s in enumerate(strategies, 1):
        viable = "OK" if s["viable"] else "--"
        lines.append(f"\n### {rank}. {s['name']}  [{viable}]  점수: {s['score']:+.1f}/10")
        lines.append(f"   방향: {s['direction']} | {s['description']}")
        lines.append(f"   적합 조건: {s['best_when']}")
        lines.append(f"   위험: {s['risk']}")
        if s["rationale"]:
            lines.append(f"   근거:")
            for r in s["rationale"]:
                lines.append(f"     - {r}")

    # 최종 추천
    top = [s for s in strategies if s["viable"]]
    lines.append(f"\n{'='*60}")
    lines.append(f"## 최종 추천 전략")
    lines.append(f"{'='*60}")

    if top:
        best = top[0]
        lines.append(f"\n1순위: {best['name']} (점수: {best['score']:+.1f})")
        for r in best["rationale"]:
            lines.append(f"  - {r}")

        if len(top) > 1:
            second = top[1]
            lines.append(f"\n2순위: {second['name']} (점수: {second['score']:+.1f})")
            for r in second["rationale"]:
                lines.append(f"  - {r}")

        if len(top) > 2:
            third = top[2]
            lines.append(f"\n3순위: {third['name']} (점수: {third['score']:+.1f})")
    else:
        lines.append("\n현재 데이터로는 확신 있는 전략 없음 — 관망 권고")

    return "\n".join(lines)
