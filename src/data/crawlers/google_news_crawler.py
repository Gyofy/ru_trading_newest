"""Google News RSS 크립토/거시경제 뉴스 크롤러.

Google News RSS 피드를 사용하여 API 키 없이 뉴스를 수집합니다.
"""

import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any
from urllib.parse import quote

HEADERS = {
    "User-Agent": "CryptoAgent/1.0 (research bot)"
}

CRYPTO_QUERIES = [
    "cryptocurrency market today",
    "bitcoin price prediction",
    "ethereum news",
    "solana crypto",
    "altcoin rally",
    "crypto regulation",
    "SEC crypto",
    "bitcoin ETF",
    "DeFi news",
    "crypto whale movement",
]

MACRO_QUERIES = [
    "Federal Reserve interest rate",
    "US inflation CPI",
    "global economy outlook",
    "US dollar index",
    "treasury yield",
    "recession risk",
    "oil price impact",
    "China economy",
    "geopolitical risk market",
]


def fetch_google_news(query: str, num: int = 10) -> list[dict[str, Any]]:
    """Google News RSS에서 뉴스를 수집합니다."""
    encoded_query = quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en&gl=US&ceid=US:en"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [GoogleNews] '{query}' 수집 실패: {e}")
        return []

    articles = []
    try:
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")[:num]

        for item in items:
            title = item.findtext("title", "")
            # 출처 분리 (Google News 형식: "제목 - 출처")
            source = ""
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title = parts[0]
                source = parts[1] if len(parts) > 1 else ""

            pub_date = item.findtext("pubDate", "")

            articles.append({
                "source": "google_news",
                "query": query,
                "title": title,
                "media_source": source,
                "url": item.findtext("link", ""),
                "published_at": pub_date,
                "description": _clean_html(item.findtext("description", "")),
            })
    except ET.ParseError as e:
        print(f"  [GoogleNews] XML 파싱 실패: {e}")

    return articles


def _clean_html(text: str) -> str:
    """HTML 태그 제거."""
    clean = re.sub(r"<[^>]+>", "", text)
    return clean[:300]


def crawl_all_google_news(num_per_query: int = 8) -> dict[str, list[dict]]:
    """모든 크립토/거시 뉴스를 수집합니다."""
    results = {"crypto": [], "macro": []}

    print("[GoogleNews] 크립토 뉴스 수집 시작...")
    for query in CRYPTO_QUERIES:
        articles = fetch_google_news(query, num=num_per_query)
        results["crypto"].extend(articles)
        print(f"  '{query}': {len(articles)}건")

    print("[GoogleNews] 거시경제 뉴스 수집 시작...")
    for query in MACRO_QUERIES:
        articles = fetch_google_news(query, num=num_per_query)
        results["macro"].extend(articles)
        print(f"  '{query}': {len(articles)}건")

    print(
        f"[GoogleNews] 완료: 크립토 {len(results['crypto'])}건, "
        f"거시경제 {len(results['macro'])}건"
    )
    return results
