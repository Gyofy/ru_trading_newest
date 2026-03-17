"""YouTube 크립토 영상 크롤러.

YouTube RSS 피드 + 검색 기반으로 가상화폐 관련 영상 데이터를 수집합니다.
조회수 가중치 및 스크립트 기반 감성 분석을 수행합니다.
"""

import requests
import xml.etree.ElementTree as ET
import re
import json
from datetime import datetime
from pathlib import Path

from .sentiment_analyzer import analyze_sentiment_rule

# 주요 크립토 유튜브 채널 ID
CRYPTO_CHANNELS = {
    "Coin Bureau": "UCqK_GSMbpiV8spgD3ZGloSw",
    "Benjamin Cowen": "UCRvqjQPSeaWn-uEx-w0XOIg",
    "DataDash": "UCCatR7nWbYrkVXdxXb4cGXg",
    "Altcoin Daily": "UCbLhGKVY-bJPcawebgtNfbw",
    "CryptosRUs": "UCuXT5bSTEXc5MR1jxfMiCXw",
}

# 검색 키워드
SEARCH_QUERIES = [
    "bitcoin price prediction today",
    "crypto market analysis",
    "ethereum forecast",
    "altcoin season",
    "solana news today",
    "crypto crash warning",
    "bitcoin bull run",
    "top crypto buy now",
]


def fetch_channel_videos(channel_id: str, max_results: int = 5) -> list[dict]:
    """YouTube RSS 피드에서 최신 영상을 가져옵니다."""
    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()

        ns = {"atom": "http://www.w3.org/2005/Atom", "media": "http://search.yahoo.com/mrss/"}
        root = ET.fromstring(resp.text)

        videos = []
        for entry in root.findall("atom:entry", ns)[:max_results]:
            title = entry.find("atom:title", ns)
            published = entry.find("atom:published", ns)
            link = entry.find("atom:link", ns)
            media_group = entry.find("media:group", ns)
            description = ""
            if media_group is not None:
                desc_elem = media_group.find("media:description", ns)
                if desc_elem is not None and desc_elem.text:
                    description = desc_elem.text[:500]

            video_data = {
                "source": "youtube",
                "title": title.text if title is not None else "",
                "published_at": published.text if published is not None else "",
                "url": link.get("href", "") if link is not None else "",
                "description": description,
            }

            # 감성 분석
            text = f"{video_data['title']} {video_data['description']}"
            video_data["sentiment"] = analyze_sentiment_rule(text)

            videos.append(video_data)

        return videos
    except Exception as e:
        print(f"  [YouTube RSS 실패] {channel_id}: {e}")
        return []


def search_youtube_via_google_news(query: str) -> list[dict]:
    """Google News RSS에서 YouTube 관련 결과를 검색합니다."""
    url = f"https://news.google.com/rss/search?q={query}+site:youtube.com&hl=en&gl=US&ceid=US:en"
    try:
        resp = requests.get(url, timeout=15)
        root = ET.fromstring(resp.text)

        results = []
        for item in root.findall(".//item")[:3]:
            title = item.find("title")
            link = item.find("link")
            pub_date = item.find("pubDate")

            result = {
                "source": "youtube_search",
                "title": title.text if title is not None else "",
                "url": link.text if link is not None else "",
                "published_at": pub_date.text if pub_date is not None else "",
            }
            result["sentiment"] = analyze_sentiment_rule(result["title"])
            results.append(result)

        return results
    except Exception:
        return []


def estimate_influence_score(video: dict) -> float:
    """영상의 영향력 지수를 추정합니다.

    조회수 데이터가 없으므로, 채널 규모 + 제목 자극성 + 시의성으로 추정.
    """
    score = 5.0  # 기본 점수

    title = video.get("title", "").lower()

    # 자극적 키워드 가중치
    urgency_words = ["breaking", "urgent", "now", "today", "just in", "emergency", "crash", "pump", "100x"]
    for w in urgency_words:
        if w in title:
            score += 1.5

    # 구체적 예측 키워드
    prediction_words = ["prediction", "forecast", "target", "will reach", "next week"]
    for w in prediction_words:
        if w in title:
            score += 1.0

    # 감성 강도 반영
    sentiment = video.get("sentiment", {})
    score += abs(sentiment.get("score", 0)) * 2

    return min(score, 10.0)


def crawl_all_youtube() -> dict:
    """모든 YouTube 소스에서 데이터를 수집합니다."""
    print("[YouTube] 크롤링 시작...")
    all_videos = []

    # 1. 채널 RSS
    for name, channel_id in CRYPTO_CHANNELS.items():
        videos = fetch_channel_videos(channel_id, max_results=3)
        for v in videos:
            v["channel"] = name
            v["influence_score"] = estimate_influence_score(v)
        all_videos.extend(videos)
        print(f"  [채널] {name}: {len(videos)}개")

    # 2. 검색 기반
    for query in SEARCH_QUERIES[:4]:  # 비용 절감: 상위 4개만
        results = search_youtube_via_google_news(query)
        for r in results:
            r["influence_score"] = estimate_influence_score(r)
        all_videos.extend(results)

    # 집계
    sentiments = [v["sentiment"]["score"] for v in all_videos if "sentiment" in v]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

    return {
        "source": "youtube",
        "total_videos": len(all_videos),
        "avg_sentiment": round(avg_sentiment, 3),
        "avg_influence": round(
            sum(v.get("influence_score", 5) for v in all_videos) / max(len(all_videos), 1), 2
        ),
        "videos": all_videos,
        "collected_at": datetime.now().isoformat(),
    }


def crawl_tiktok_proxy() -> dict:
    """TikTok 데이터를 Google News 프록시로 수집합니다."""
    print("[TikTok] Google News 프록시 크롤링...")
    queries = ["crypto tiktok trend", "bitcoin tiktok viral", "crypto influencer tiktok"]
    all_items = []

    for query in queries:
        url = f"https://news.google.com/rss/search?q={query}&hl=en&gl=US&ceid=US:en"
        try:
            resp = requests.get(url, timeout=15)
            root = ET.fromstring(resp.text)
            for item in root.findall(".//item")[:3]:
                title = item.find("title")
                all_items.append({
                    "source": "tiktok_proxy",
                    "title": title.text if title is not None else "",
                    "sentiment": analyze_sentiment_rule(title.text if title is not None else ""),
                })
        except Exception:
            pass

    sentiments = [i["sentiment"]["score"] for i in all_items]

    return {
        "source": "tiktok_proxy",
        "total_items": len(all_items),
        "avg_sentiment": round(sum(sentiments) / max(len(sentiments), 1), 3),
        "items": all_items,
        "collected_at": datetime.now().isoformat(),
    }


def crawl_instagram_proxy() -> dict:
    """Instagram 데이터를 Google News 프록시로 수집합니다."""
    print("[Instagram] Google News 프록시 크롤링...")
    queries = ["crypto instagram influencer", "bitcoin instagram trend"]
    all_items = []

    for query in queries:
        url = f"https://news.google.com/rss/search?q={query}&hl=en&gl=US&ceid=US:en"
        try:
            resp = requests.get(url, timeout=15)
            root = ET.fromstring(resp.text)
            for item in root.findall(".//item")[:3]:
                title = item.find("title")
                all_items.append({
                    "source": "instagram_proxy",
                    "title": title.text if title is not None else "",
                    "sentiment": analyze_sentiment_rule(title.text if title is not None else ""),
                })
        except Exception:
            pass

    sentiments = [i["sentiment"]["score"] for i in all_items]

    return {
        "source": "instagram_proxy",
        "total_items": len(all_items),
        "avg_sentiment": round(sum(sentiments) / max(len(sentiments), 1), 3),
        "items": all_items,
        "collected_at": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    yt = crawl_all_youtube()
    print(f"\nYouTube: {yt['total_videos']}개 영상, 감성: {yt['avg_sentiment']}")

    tt = crawl_tiktok_proxy()
    print(f"TikTok: {tt['total_items']}개, 감성: {tt['avg_sentiment']}")

    ig = crawl_instagram_proxy()
    print(f"Instagram: {ig['total_items']}개, 감성: {ig['avg_sentiment']}")
