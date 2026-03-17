"""종합 투자 분석 파이프라인.

크롤링 → 감성분석 → 전략평가 → 리포트 생성까지 원스톱 실행.
Gemini API 사용하지 않음 — 로컬 분석만 수행.
"""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.crawlers.crypto_opinion import collect_ticker_data
from src.data.crawlers.sentiment_analyzer import (
    analyze_batch,
    aggregate_sentiment,
)
from src.data.crawlers.strategy_analyzer import (
    evaluate_strategies,
    format_strategy_report,
)


def full_analysis(ticker: str) -> str:
    """특정 종목에 대한 완전한 분석을 수행합니다."""

    # 1. 데이터 수집
    data = collect_ticker_data(ticker)
    items = data["items"]

    if not items:
        return f"[{ticker}] 데이터 없음"

    # 2. 감성 분석
    print(f"\n[감성분석] {len(items)}건 처리 중...")
    analyzed = analyze_batch(items, text_field="title")
    agg = aggregate_sentiment(analyzed)
    print(f"  결과: {agg['overall']} ({agg['avg_score']:+.3f})")

    # 3. 전략 평가
    print(f"\n[전략분석] 8개 전략 평가 중...")
    strategies = evaluate_strategies(ticker, analyzed, agg)

    # 4. 리포트 생성
    report = format_strategy_report(ticker, strategies, agg, analyzed)

    # 5. 주요 뉴스 부록
    sorted_items = sorted(
        [i for i in analyzed if i.get("sentiment")],
        key=lambda x: abs(x["sentiment"].get("score", 0)),
        reverse=True,
    )[:10]

    report += f"\n\n{'='*60}"
    report += f"\n## 핵심 뉴스/게시글 TOP 10"
    report += f"\n{'─'*60}"
    for i, item in enumerate(sorted_items, 1):
        s = item["sentiment"]
        icon = "UP" if s["score"] > 0 else "DN" if s["score"] < 0 else "--"
        src = item.get("source", item.get("media_source", ""))
        report += f"\n{i}. [{icon}] [{src}] {item.get('title', '')}"

    # 6. 저장
    date_str = datetime.now().strftime("%Y%m%d")
    save_dir = PROJECT_ROOT / "data" / "reports" / date_str
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{ticker.upper()}_strategy.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n리포트 저장: {path}")

    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["BTC"]

    for ticker in tickers:
        result = full_analysis(ticker)
        print(result)
        print("\n")
