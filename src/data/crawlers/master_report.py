"""마스터 리포트 생성기.

CoinGecko(시세) + 기술적분석 + 뉴스감성 + 전략평가 + 순차추론
→ 하나의 종합 투자 리포트를 생성합니다.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.data.crawlers.coingecko_client import (
    get_price_data,
    get_price_history,
    format_price_summary,
    enrich_with_history,
)
from src.data.crawlers.technical_analysis import (
    run_full_technical_analysis,
    format_ta_report,
)
from src.data.crawlers.crypto_opinion import collect_ticker_data
from src.data.crawlers.sentiment_analyzer import (
    analyze_batch,
    aggregate_sentiment,
)
from src.data.crawlers.strategy_analyzer import (
    evaluate_strategies,
    format_strategy_report,
)
from src.data.crawlers.sequential_reasoning import run_sequential_analysis


def generate_master_report(ticker: str) -> str:
    """특정 종목의 마스터 리포트를 생성합니다."""

    print(f"\n{'#'*70}")
    print(f"  [{ticker.upper()}] 마스터 리포트 생성 시작")
    print(f"{'#'*70}")

    # === 1. CoinGecko 실시간 시세 ===
    print(f"\n[1/5] CoinGecko 시세 조회...")
    price_data = get_price_data(ticker)
    if price_data:
        print(f"  ${price_data['price_usd']:,.2f} | 24h: {price_data['change_24h_pct']:+.2f}% | 7d: {price_data['change_7d_pct']:+.2f}%")
    else:
        print(f"  시세 조회 실패")
    time.sleep(1)

    # === 2. 가격 히스토리 + 기술적 분석 ===
    print(f"\n[2/5] 기술적 분석 (30일 히스토리)...")
    history = get_price_history(ticker, days=30)
    if history:
        # 히스토리로 7d/30d 변동률 보강
        price_data = enrich_with_history(price_data, history)
        ta_result = run_full_technical_analysis(history)
        print(f"  판정: {ta_result.get('overall', '?')} (점수: {ta_result.get('signal_score', 0):+.1f})")
    else:
        ta_result = {"error": "히스토리 데이터 없음"}
        print(f"  히스토리 조회 실패")
    time.sleep(1)

    # === 3. 뉴스/소셜 감성 분석 ===
    print(f"\n[3/5] 뉴스/소셜 데이터 수집 + 감성 분석...")
    crawl_data = collect_ticker_data(ticker)
    items = crawl_data["items"]
    analyzed = analyze_batch(items, text_field="title")
    sentiment_agg = aggregate_sentiment(analyzed)
    print(f"  {len(items)}건 수집 → 심리: {sentiment_agg['overall']} ({sentiment_agg['avg_score']:+.3f})")

    # 핵심 뉴스 추출
    sorted_items = sorted(
        [i for i in analyzed if i.get("sentiment")],
        key=lambda x: abs(x["sentiment"].get("score", 0)),
        reverse=True,
    )
    top_news = sorted_items[:10]

    # === 4. 멀티 포지션 전략 평가 ===
    print(f"\n[4/5] 8개 전략 평가...")
    strategies = evaluate_strategies(ticker, analyzed, sentiment_agg)
    top3 = [s for s in strategies if s.get("viable")][:3]
    for s in top3:
        print(f"  {s['name']}: {s['score']:+.1f}")

    # === 5. 순차 추론 → 최종 결론 ===
    print(f"\n[5/5] 순차 추론 (Sequential Reasoning)...")
    final_report = run_sequential_analysis(
        ticker=ticker.upper(),
        price_data=price_data,
        technical=ta_result,
        sentiment_agg=sentiment_agg,
        strategies=strategies,
        top_news=top_news,
    )

    # === 저장 ===
    date_str = datetime.now().strftime("%Y%m%d")
    save_dir = PROJECT_ROOT / "data" / "reports" / date_str
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{ticker.upper()}_MASTER.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(final_report)

    print(f"\n{'#'*70}")
    print(f"  마스터 리포트 저장: {path}")
    print(f"{'#'*70}")

    return final_report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["BTC"]

    for ticker in tickers:
        report = generate_master_report(ticker)
        print(report)
        print("\n\n")
