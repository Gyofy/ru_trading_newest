"""특정 크립토 종목에 대한 종합 의견 생성기.

크롤링 → 필터링 → 감성분석 → Gemini 정밀분석 파이프라인.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import requests
from src.data.crawlers.reddit_crawler import crawl_all_reddit
from src.data.crawlers.google_news_crawler import fetch_google_news
from src.data.crawlers.x_crawler import fetch_x_trends_via_google
from src.data.crawlers.sentiment_analyzer import (
    analyze_batch,
    aggregate_sentiment,
    filter_by_ticker,
)


def collect_ticker_data(ticker: str) -> dict:
    """특정 종목 관련 데이터를 모든 소스에서 수집합니다."""
    TICKER_NAMES = {
        "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
        "XRP": "ripple", "ADA": "cardano", "DOGE": "dogecoin",
        "AVAX": "avalanche", "DOT": "polkadot", "LINK": "chainlink",
        "BNB": "binance coin", "NEAR": "near protocol", "SUI": "sui",
        "ARB": "arbitrum", "OP": "optimism",
    }
    name = TICKER_NAMES.get(ticker.upper(), ticker.lower())
    ticker_upper = ticker.upper()

    print(f"\n{'='*60}")
    print(f"[{ticker_upper}] 종목 데이터 수집 시작")
    print(f"{'='*60}")

    all_items = []

    # 1. Google News — 종목 특화 검색
    print(f"\n[Google News] {name} 뉴스 수집...")
    queries = [
        f"{name} crypto news",
        f"{name} price prediction",
        f"{name} analysis today",
    ]
    for q in queries:
        articles = fetch_google_news(q, num=8)
        for a in articles:
            a["text_field"] = a.get("title", "")
        all_items.extend(articles)
        print(f"  '{q}': {len(articles)}건")

    # 2. Reddit — 전체 크롤링 후 종목 필터
    print(f"\n[Reddit] 크립토 서브레딧에서 {ticker_upper} 필터링...")
    from src.data.crawlers.reddit_crawler import fetch_subreddit_posts
    import time
    reddit_items = []
    for sub in ["cryptocurrency", "Bitcoin", "ethereum", "CryptoMarkets", "altcoin"]:
        posts = fetch_subreddit_posts(sub, sort="hot", limit=20)
        reddit_items.extend(posts)
        time.sleep(1.2)

    ticker_reddit = filter_by_ticker(reddit_items, ticker_upper)
    all_items.extend(ticker_reddit)
    print(f"  전체 {len(reddit_items)}건 중 {ticker_upper} 관련: {len(ticker_reddit)}건")

    # 3. X/Twitter
    print(f"\n[X/Twitter] {name} 트렌드 수집...")
    x_items = fetch_x_trends_via_google(f"{name} crypto", num=5)
    all_items.extend(x_items)
    print(f"  {len(x_items)}건")

    return {
        "ticker": ticker_upper,
        "name": name,
        "total_items": len(all_items),
        "items": all_items,
        "collected_at": datetime.now().isoformat(),
    }


def analyze_ticker(ticker: str, use_gemini: bool = True) -> str:
    """특정 종목의 종합 의견을 생성합니다."""

    # === 1. 데이터 수집 ===
    data = collect_ticker_data(ticker)
    items = data["items"]

    if not items:
        return f"[{ticker.upper()}] 관련 데이터를 찾을 수 없습니다."

    # === 2. 감성 분석 ===
    print(f"\n[감성분석] {len(items)}건 분석 중...")
    analyzed = analyze_batch(items, text_field="title")
    agg = aggregate_sentiment(analyzed)
    print(f"  결과: {agg['overall']} (평균: {agg['avg_score']:.3f})")
    print(f"  강세: {agg['bullish_count']}건 / 약세: {agg['bearish_count']}건 / 중립: {agg['neutral_count']}건")

    # === 3. 핵심 뉴스 추출 (감성 점수 절대값 기준 정렬) ===
    sorted_items = sorted(
        [i for i in analyzed if i.get("sentiment")],
        key=lambda x: abs(x["sentiment"].get("score", 0)),
        reverse=True,
    )
    top_items = sorted_items[:15]

    # === 4. 종합 요약 생성 ===
    summary_lines = []
    summary_lines.append(f"## [{ticker.upper()}] 데이터 수집 요약")
    summary_lines.append(f"- 수집 건수: {data['total_items']}건")
    summary_lines.append(f"- 시장 심리: **{agg['overall'].upper()}** (점수: {agg['avg_score']:.3f})")
    summary_lines.append(f"- 강세 {agg['bullish_count']} / 약세 {agg['bearish_count']} / 중립 {agg['neutral_count']}")
    summary_lines.append(f"\n## 주요 뉴스/게시글 (감성 강도순)")
    for i, item in enumerate(top_items, 1):
        sent = item["sentiment"]
        icon = "+" if sent["score"] > 0 else "-" if sent["score"] < 0 else "o"
        source = item.get("source", item.get("media_source", "?"))
        summary_lines.append(
            f"{i}. [{icon}{sent['score']:+.2f}] [{source}] {item.get('title', 'N/A')}"
        )

    local_summary = "\n".join(summary_lines)

    # === 5. Gemini 정밀 분석 (옵션) ===
    if use_gemini:
        gemini_opinion = _gemini_opinion(ticker, local_summary, agg)
        full_report = f"{local_summary}\n\n---\n\n{gemini_opinion}"
    else:
        full_report = local_summary

    # === 6. 저장 ===
    date_str = datetime.now().strftime("%Y%m%d")
    save_dir = PROJECT_ROOT / "data" / "reports" / date_str
    save_dir.mkdir(parents=True, exist_ok=True)
    report_path = save_dir / f"{ticker.upper()}_opinion.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# {ticker.upper()} 종합 의견 리포트\n")
        f.write(f"**생성 시각**: {datetime.now().isoformat()}\n\n")
        f.write(full_report)

    print(f"\n리포트 저장: {report_path}")
    return full_report


def _gemini_opinion(ticker: str, summary: str, sentiment_agg: dict) -> str:
    """Gemini로 종합 의견을 생성합니다."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    model = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

    if not api_key:
        return "[Gemini 미설정 — 로컬 분석만 제공]"

    prompt = f"""당신은 암호화폐 시장 전문 애널리스트입니다.

<collected_data>
{summary}
</collected_data>

<sentiment_aggregate>
전체 심리: {sentiment_agg['overall']}
평균 점수: {sentiment_agg['avg_score']}
강세/약세/중립: {sentiment_agg['bullish_count']}/{sentiment_agg['bearish_count']}/{sentiment_agg['neutral_count']}
</sentiment_aggregate>

<task>
{ticker.upper()} 에 대해 다음을 분석해주세요:

1. **현재 상황 요약** (2-3줄)
2. **단기 전망** (1-7일): 방향성, 주요 지지/저항선
3. **리스크 요인**: 하방 리스크 TOP 3
4. **기회 요인**: 상방 촉매 TOP 3
5. **최종 의견**: 매수/매도/관망 + 확신도(1-10) + 근거

수집된 뉴스 데이터에 근거하여 분석하세요.
한국어로 작성하되 간결하게 핵심만 담아주세요.
</task>"""

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )

    print(f"\n[Gemini] {ticker.upper()} 정밀 분석 중...")
    try:
        resp = requests.post(
            url,
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 4096},
            },
            timeout=90,
        )
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return f"## Gemini AI 정밀 분석\n\n{text}"
    except Exception as e:
        return f"[Gemini 분석 실패: {e}]"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", help="분석할 크립토 종목 (예: BTC, ETH, SOL)")
    parser.add_argument("--no-gemini", action="store_true", help="Gemini 분석 생략")
    args = parser.parse_args()

    result = analyze_ticker(args.ticker, use_gemini=not args.no_gemini)
    print("\n" + "=" * 60)
    print(result)
