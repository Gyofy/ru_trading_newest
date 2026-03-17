"""전문 크립토 미디어 크롤러.

코인데스크, 코인텔레그래프, 더블록, 디크립트, 코인니스 등
주요 크립토 전문 미디어에서 RSS/Google News 프록시로 수집합니다.
"""

import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any
from urllib.parse import quote

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CryptoResearch/2.0"
}

# ==================== RSS 피드 직접 수집 ====================

RSS_FEEDS = {
    "coindesk": {
        "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "name": "CoinDesk",
        "reliability": 0.9,
    },
    "cointelegraph": {
        "url": "https://cointelegraph.com/rss",
        "name": "CoinTelegraph",
        "reliability": 0.85,
    },
    "theblock": {
        "url": "https://www.theblock.co/rss.xml",
        "name": "The Block",
        "reliability": 0.9,
    },
    "decrypt": {
        "url": "https://decrypt.co/feed",
        "name": "Decrypt",
        "reliability": 0.8,
    },
}

# Google News 프록시로 수집하는 소스 (RSS 없거나 차단 시)
GOOGLE_NEWS_PROXY_SOURCES = {
    "coindesk": "coindesk cryptocurrency",
    "cointelegraph": "cointelegraph crypto news",
    "theblock": "the block crypto research",
    "decrypt": "decrypt crypto web3",
    "coinness": "coinness crypto korea",
    "glassnode": "glassnode on-chain bitcoin",
    "messari": "messari crypto research report",
    "tiger_research": "tiger research crypto asia web3",
}

# 온체인/분석 전용 쿼리
ONCHAIN_QUERIES = [
    "bitcoin whale movement on-chain",
    "ethereum gas fees defi",
    "crypto exchange inflow outflow",
    "bitcoin miner capitulation",
    "stablecoin supply crypto",
    "bitcoin ETF flow institutional",
    "crypto liquidation data",
    "bitcoin fear greed index",
]


def fetch_rss_feed(source_id: str, feed_info: dict, max_items: int = 15) -> list[dict]:
    """RSS 피드에서 직접 수집합니다."""
    try:
        resp = requests.get(feed_info["url"], headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception:
        return []

    articles = []
    try:
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")[:max_items]

        for item in items:
            title = item.findtext("title", "")
            pub_date = item.findtext("pubDate", "")
            desc = item.findtext("description", "")
            desc = re.sub(r"<[^>]+>", "", desc)[:300]

            articles.append({
                "source": source_id,
                "media_name": feed_info["name"],
                "title": title,
                "description": desc,
                "url": item.findtext("link", ""),
                "published_at": pub_date,
                "reliability": feed_info["reliability"],
            })
    except ET.ParseError:
        pass

    return articles


def fetch_google_news_proxy(source_id: str, query: str, max_items: int = 8) -> list[dict]:
    """Google News RSS를 프록시로 사용하여 특정 소스 뉴스를 수집합니다."""
    encoded = quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en&gl=US&ceid=US:en"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception:
        return []

    articles = []
    try:
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")[:max_items]

        for item in items:
            title = item.findtext("title", "")
            media_source = ""
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title = parts[0]
                media_source = parts[1] if len(parts) > 1 else ""

            articles.append({
                "source": source_id,
                "media_name": media_source or source_id,
                "title": title,
                "description": re.sub(r"<[^>]+>", "", item.findtext("description", ""))[:300],
                "url": item.findtext("link", ""),
                "published_at": item.findtext("pubDate", ""),
                "reliability": 0.7,
            })
    except ET.ParseError:
        pass

    return articles


def crawl_all_crypto_media() -> dict[str, Any]:
    """모든 크립토 전문 미디어를 수집합니다."""
    print("[CryptoMedia] 전문 미디어 크롤링 시작...")

    all_items = {}
    total = 0

    # 1. RSS 직접 수집
    for src_id, feed_info in RSS_FEEDS.items():
        items = fetch_rss_feed(src_id, feed_info)
        if items:
            all_items[src_id] = items
            total += len(items)
            print(f"  [RSS] {feed_info['name']}: {len(items)}건")
        else:
            # RSS 실패 시 Google News 프록시 폴백
            query = GOOGLE_NEWS_PROXY_SOURCES.get(src_id, f"{src_id} crypto")
            items = fetch_google_news_proxy(src_id, query)
            all_items[src_id] = items
            total += len(items)
            print(f"  [Proxy] {feed_info['name']}: {len(items)}건 (RSS fallback)")

    # 2. RSS 없는 소스는 프록시로
    proxy_only = {k: v for k, v in GOOGLE_NEWS_PROXY_SOURCES.items() if k not in RSS_FEEDS}
    for src_id, query in proxy_only.items():
        items = fetch_google_news_proxy(src_id, query)
        all_items[src_id] = items
        total += len(items)
        print(f"  [Proxy] {src_id}: {len(items)}건")

    # 3. 온체인 데이터 뉴스
    onchain_items = []
    for query in ONCHAIN_QUERIES:
        items = fetch_google_news_proxy("onchain", query, max_items=5)
        onchain_items.extend(items)
    all_items["onchain"] = onchain_items
    total += len(onchain_items)
    print(f"  [Proxy] onchain: {len(onchain_items)}건")

    # 감성 분석
    from src.data.crawlers.sentiment_analyzer import analyze_sentiment_rule
    for src_id, items in all_items.items():
        for item in items:
            sent = analyze_sentiment_rule(item.get("title", "") + " " + item.get("description", ""))
            item["sentiment"] = sent

    # 소스별 평균 감성
    source_sentiments = {}
    for src_id, items in all_items.items():
        if items:
            scores = [i["sentiment"]["score"] for i in items if isinstance(i.get("sentiment"), dict)]
            source_sentiments[src_id] = {
                "avg_sentiment": sum(scores) / len(scores) if scores else 0,
                "count": len(items),
                "items": items,
            }

    print(f"[CryptoMedia] 완료: 총 {total}건, {len(source_sentiments)}개 소스")

    return source_sentiments
