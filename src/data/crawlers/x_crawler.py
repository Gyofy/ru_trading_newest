"""X(Twitter) 크립토 인플루언서 & 트렌드 크롤러.

X API 대신 Nitter(오픈소스 프론트엔드) RSS 또는
웹 검색을 통해 X의 크립토 관련 트렌드를 수집합니다.
"""

import requests
import re
from typing import Any

HEADERS = {
    "User-Agent": "CryptoAgent/1.0 (research bot)"
}

# 크립토 핵심 인플루언서/계정 (검색 키워드로 활용)
CRYPTO_INFLUENCERS = [
    "CryptoQuant CEO",
    "Willy Woo bitcoin",
    "PlanB bitcoin",
    "Cobie crypto",
    "ZachXBT crypto",
    "Arthur Hayes crypto",
]

# X에서 추적할 크립토 트렌드 키워드
CRYPTO_TREND_KEYWORDS = [
    "crypto bullish site:x.com OR site:twitter.com",
    "bitcoin breakout site:x.com OR site:twitter.com",
    "ethereum upgrade site:x.com OR site:twitter.com",
    "altcoin season site:x.com OR site:twitter.com",
    "crypto crash warning site:x.com OR site:twitter.com",
    "whale alert crypto site:x.com OR site:twitter.com",
    "SEC crypto ruling site:x.com OR site:twitter.com",
]


def fetch_x_trends_via_google(
    query: str, num: int = 5
) -> list[dict[str, Any]]:
    """Google 검색으로 X/Twitter의 크립토 게시글을 수집합니다."""
    from urllib.parse import quote
    import xml.etree.ElementTree as ET

    encoded = quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en&gl=US&ceid=US:en"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as e:
        print(f"  [X/Twitter] '{query}' 수집 실패: {e}")
        return []

    results = []
    for item in root.findall(".//item")[:num]:
        title = item.findtext("title", "")
        source = ""
        if " - " in title:
            parts = title.rsplit(" - ", 1)
            title = parts[0]
            source = parts[1]

        results.append({
            "source": "x_twitter",
            "query": query,
            "title": title,
            "media_source": source,
            "url": item.findtext("link", ""),
            "published_at": item.findtext("pubDate", ""),
        })

    return results


def crawl_all_x(num_per_query: int = 5) -> list[dict[str, Any]]:
    """X/Twitter 크립토 트렌드를 수집합니다."""
    all_results = []

    print("[X/Twitter] 크립토 트렌드 수집 시작...")
    for query in CRYPTO_TREND_KEYWORDS:
        results = fetch_x_trends_via_google(query, num=num_per_query)
        all_results.extend(results)
        short_q = query.split("site:")[0].strip()
        print(f"  '{short_q}': {len(results)}건")

    print(f"[X/Twitter] 완료: 총 {len(all_results)}건")
    return all_results
