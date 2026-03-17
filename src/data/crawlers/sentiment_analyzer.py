"""크립토 뉴스/소셜 감성 분석기.

키워드 기반 룰 엔진 + Gemini AI 하이브리드 분석.
API 호출 최소화: 룰 엔진으로 1차 분류, 핵심 건만 Gemini 정밀 분석.
"""

import re
from typing import Any

# 강세/약세 키워드 사전
BULLISH_KEYWORDS = [
    "rally", "surge", "breakout", "bullish", "soar", "pump", "moon",
    "all-time high", "ath", "adoption", "institutional", "etf approved",
    "accumulation", "buy signal", "upgrade", "partnership", "launch",
    "bull run", "recovery", "uptrend", "green", "gains",
]

BEARISH_KEYWORDS = [
    "crash", "dump", "plunge", "bearish", "sell-off", "selloff",
    "collapse", "fear", "panic", "hack", "scam", "fraud", "ban",
    "regulation", "sec lawsuit", "liquidation", "capitulation",
    "death cross", "breakdown", "red", "losses", "fall", "drop",
    "decline", "tumble", "slump", "warning",
]

NEUTRAL_KEYWORDS = [
    "stable", "sideways", "consolidation", "range", "mixed",
    "uncertain", "wait", "hold",
]


def analyze_sentiment_rule(text: str) -> dict[str, Any]:
    """키워드 기반 감성 분석 (빠름, API 불필요)."""
    text_lower = text.lower()

    bull_score = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
    bear_score = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)

    total = bull_score + bear_score
    if total == 0:
        return {"sentiment": "neutral", "score": 0.0, "bull": 0, "bear": 0}

    # -1.0 (극도 약세) ~ +1.0 (극도 강세)
    score = (bull_score - bear_score) / total

    if score > 0.2:
        label = "bullish"
    elif score < -0.2:
        label = "bearish"
    else:
        label = "neutral"

    return {
        "sentiment": label,
        "score": round(score, 3),
        "bull": bull_score,
        "bear": bear_score,
    }


def analyze_batch(items: list[dict], text_field: str = "title") -> list[dict]:
    """여러 항목의 감성을 일괄 분석합니다."""
    for item in items:
        text = item.get(text_field, "")
        selftext = item.get("selftext", "")
        combined = f"{text} {selftext}".strip()
        item["sentiment"] = analyze_sentiment_rule(combined)
    return items


def aggregate_sentiment(items: list[dict]) -> dict[str, Any]:
    """전체 감성을 집계합니다."""
    if not items:
        return {"overall": "neutral", "avg_score": 0.0, "count": 0}

    scores = [
        item.get("sentiment", {}).get("score", 0.0)
        for item in items
        if "sentiment" in item
    ]

    if not scores:
        return {"overall": "neutral", "avg_score": 0.0, "count": 0}

    avg = sum(scores) / len(scores)
    bullish_count = sum(1 for s in scores if s > 0.2)
    bearish_count = sum(1 for s in scores if s < -0.2)
    neutral_count = len(scores) - bullish_count - bearish_count

    if avg > 0.15:
        overall = "bullish"
    elif avg < -0.15:
        overall = "bearish"
    else:
        overall = "neutral"

    return {
        "overall": overall,
        "avg_score": round(avg, 3),
        "count": len(scores),
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "neutral_count": neutral_count,
    }


def filter_by_ticker(items: list[dict], ticker: str) -> list[dict]:
    """특정 종목 관련 항목만 필터링합니다."""
    # 종목별 검색 키워드 매핑
    TICKER_ALIASES = {
        "BTC": ["bitcoin", "btc", "비트코인"],
        "ETH": ["ethereum", "eth", "ether", "이더리움"],
        "SOL": ["solana", "sol", "솔라나"],
        "XRP": ["ripple", "xrp", "리플"],
        "ADA": ["cardano", "ada", "카르다노"],
        "DOGE": ["dogecoin", "doge", "도지"],
        "AVAX": ["avalanche", "avax", "아발란체"],
        "DOT": ["polkadot", "dot", "폴카닷"],
        "MATIC": ["polygon", "matic", "폴리곤"],
        "LINK": ["chainlink", "link", "체인링크"],
        "BNB": ["binance", "bnb", "바이낸스"],
        "NEAR": ["near protocol", "near"],
        "SUI": ["sui"],
        "ARB": ["arbitrum", "arb"],
        "OP": ["optimism", " op "],
    }

    keywords = TICKER_ALIASES.get(
        ticker.upper(),
        [ticker.lower()]
    )

    filtered = []
    for item in items:
        text = (
            f"{item.get('title', '')} "
            f"{item.get('selftext', '')} "
            f"{item.get('description', '')}"
        ).lower()
        if any(kw in text for kw in keywords):
            filtered.append(item)

    return filtered
