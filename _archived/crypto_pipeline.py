"""크립토 데이터 크롤링 통합 파이프라인.

Reddit + Google News + X/Twitter 데이터를 수집하고,
Gemini AI로 분석하여 종목 추천까지 수행합니다.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import requests
from src.data.crawlers.reddit_crawler import crawl_all_reddit
from src.data.crawlers.google_news_crawler import crawl_all_google_news
from src.data.crawlers.x_crawler import crawl_all_x


def run_full_crawl() -> dict:
    """전체 크롤링 파이프라인을 실행합니다."""
    print("=" * 60)
    print(f"크립토 데이터 크롤링 시작: {datetime.now().isoformat()}")
    print("=" * 60)

    # === 1. Reddit 크롤링 ===
    reddit_data = crawl_all_reddit(limit_per_sub=15)

    # === 2. Google News 크롤링 ===
    google_data = crawl_all_google_news(num_per_query=8)

    # === 3. X/Twitter 크롤링 ===
    x_data = crawl_all_x(num_per_query=5)

    # === 통합 ===
    all_data = {
        "crawl_timestamp": datetime.now().isoformat(),
        "reddit": reddit_data,
        "google_news": google_data,
        "x_twitter": x_data,
        "stats": {
            "reddit_crypto": len(reddit_data.get("crypto", [])),
            "reddit_macro": len(reddit_data.get("macro", [])),
            "google_crypto": len(google_data.get("crypto", [])),
            "google_macro": len(google_data.get("macro", [])),
            "x_twitter": len(x_data),
        },
    }

    # === 저장 ===
    date_str = datetime.now().strftime("%Y%m%d")
    save_dir = PROJECT_ROOT / "data" / "raw" / date_str
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "crypto_crawl.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"크롤링 완료! 저장: {save_path}")
    print(f"통계: {json.dumps(all_data['stats'], indent=2)}")
    print("=" * 60)

    return all_data


def analyze_with_gemini(crawled_data: dict) -> str:
    """Gemini AI로 수집된 데이터를 분석하여 종목 추천을 받습니다."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    model = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

    if not api_key:
        return "[ERROR] GEMINI_API_KEY 미설정"

    # 데이터 요약 (토큰 절약을 위해 핵심만 추출)
    summary = _build_data_summary(crawled_data)

    prompt = f"""당신은 암호화폐 시장 전문 애널리스트입니다.

아래는 Reddit, Google News, X/Twitter에서 실시간 수집된 크립토 및 거시경제 데이터입니다.

<crawled_data>
{summary}
</crawled_data>

<task>
위 데이터를 기반으로 다음을 분석해주세요:

1. **시장 심리 분석**: 현재 크립토 시장의 전반적 심리 (극도의 공포~극도의 탐욕)
2. **핫 토픽**: 가장 많이 언급되고 주목받는 이슈 TOP 5
3. **종목 추천**: 현재 데이터 기반으로 주목할 암호화폐 종목 5~10개
   - 각 종목에 대해: 이유, 리스크, 확신도(1-10), 추천 포지션(롱/숏/관망)
4. **거시경제 영향**: 거시경제 뉴스가 크립토에 미칠 영향
5. **위험 경고**: 현재 시점에서 주의해야 할 리스크

반드시 수집된 데이터의 근거를 인용하며 분석해주세요.
JSON이 아닌, 읽기 쉬운 마크다운 형식으로 작성해주세요.
</task>"""

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )

    print("\n[Gemini] 수집 데이터 분석 중...")

    try:
        resp = requests.post(
            url,
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.4,
                    "maxOutputTokens": 8192,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]

        # 분석 결과 저장
        date_str = datetime.now().strftime("%Y%m%d")
        save_dir = PROJECT_ROOT / "data" / "reports" / date_str
        save_dir.mkdir(parents=True, exist_ok=True)
        report_path = save_dir / "crypto_analysis.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# 크립토 시장 분석 리포트\n")
            f.write(f"**생성 시각**: {datetime.now().isoformat()}\n")
            f.write(f"**분석 모델**: {model}\n\n---\n\n")
            f.write(text)

        print(f"[Gemini] 분석 완료! 저장: {report_path}")
        return text

    except Exception as e:
        return f"[Gemini 분석 실패: {e}]"


def _build_data_summary(data: dict) -> str:
    """크롤링 데이터를 Gemini에 보낼 요약으로 변환합니다."""
    parts = []

    # Reddit 크립토 - 상위 스코어 게시글
    reddit_crypto = data.get("reddit", {}).get("crypto", [])
    if reddit_crypto:
        sorted_posts = sorted(reddit_crypto, key=lambda x: x.get("score", 0), reverse=True)[:20]
        parts.append("## Reddit 크립토 (인기순 상위 20)")
        for p in sorted_posts:
            parts.append(
                f"- [r/{p['subreddit']}] (score:{p['score']}, comments:{p['num_comments']}) "
                f"{p['title']}"
            )
            if p.get("selftext"):
                parts.append(f"  > {p['selftext'][:150]}")

    # Reddit 거시경제
    reddit_macro = data.get("reddit", {}).get("macro", [])
    if reddit_macro:
        sorted_posts = sorted(reddit_macro, key=lambda x: x.get("score", 0), reverse=True)[:15]
        parts.append("\n## Reddit 거시경제 (인기순 상위 15)")
        for p in sorted_posts:
            parts.append(
                f"- [r/{p['subreddit']}] (score:{p['score']}) {p['title']}"
            )

    # Google News 크립토
    google_crypto = data.get("google_news", {}).get("crypto", [])
    if google_crypto:
        parts.append("\n## Google News 크립토")
        seen_titles = set()
        for a in google_crypto[:30]:
            title = a["title"]
            if title not in seen_titles:
                seen_titles.add(title)
                parts.append(f"- [{a.get('media_source','')}] {title}")

    # Google News 거시경제
    google_macro = data.get("google_news", {}).get("macro", [])
    if google_macro:
        parts.append("\n## Google News 거시경제")
        seen_titles = set()
        for a in google_macro[:20]:
            title = a["title"]
            if title not in seen_titles:
                seen_titles.add(title)
                parts.append(f"- [{a.get('media_source','')}] {title}")

    # X/Twitter
    x_data = data.get("x_twitter", [])
    if x_data:
        parts.append("\n## X/Twitter 크립토 트렌드")
        seen = set()
        for t in x_data[:20]:
            title = t["title"]
            if title not in seen:
                seen.add(title)
                parts.append(f"- {title}")

    return "\n".join(parts)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    data = run_full_crawl()
    analysis = analyze_with_gemini(data)
    print("\n" + "=" * 60)
    print("GEMINI 분석 결과")
    print("=" * 60)
    print(analysis)
