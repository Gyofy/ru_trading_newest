"""Reddit 크립토/거시경제 서브레딧 크롤러.

Reddit Public JSON API를 사용하여 인증 없이 데이터를 수집합니다.
"""

import time
import requests
from datetime import datetime, timezone
from typing import Any

CRYPTO_SUBREDDITS = [
    "cryptocurrency",
    "Bitcoin",
    "ethereum",
    "altcoin",
    "CryptoMarkets",
    "defi",
    "solana",
]

MACRO_SUBREDDITS = [
    "economics",
    "wallstreetbets",
    "investing",
    "StockMarket",
    "finance",
]

HEADERS = {
    "User-Agent": "CryptoAgent/1.0 (research bot)"
}


def fetch_subreddit_posts(
    subreddit: str,
    sort: str = "hot",
    limit: int = 25,
    time_filter: str = "day",
) -> list[dict[str, Any]]:
    """서브레딧의 게시글을 수집합니다."""
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
    params = {"limit": limit, "t": time_filter}

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [Reddit] r/{subreddit} 수집 실패: {e}")
        return []

    posts = []
    for child in data.get("data", {}).get("children", []):
        post = child.get("data", {})
        posts.append({
            "source": "reddit",
            "subreddit": subreddit,
            "title": post.get("title", ""),
            "selftext": (post.get("selftext", "") or "")[:500],
            "score": post.get("score", 0),
            "num_comments": post.get("num_comments", 0),
            "upvote_ratio": post.get("upvote_ratio", 0),
            "url": f"https://reddit.com{post.get('permalink', '')}",
            "created_utc": datetime.fromtimestamp(
                post.get("created_utc", 0), tz=timezone.utc
            ).isoformat(),
            "flair": post.get("link_flair_text", ""),
        })

    return posts


def crawl_all_reddit(limit_per_sub: int = 15) -> dict[str, list[dict]]:
    """모든 크립토/거시 서브레딧을 크롤링합니다."""
    results = {"crypto": [], "macro": []}

    print("[Reddit] 크립토 서브레딧 크롤링 시작...")
    for sub in CRYPTO_SUBREDDITS:
        posts = fetch_subreddit_posts(sub, sort="hot", limit=limit_per_sub)
        results["crypto"].extend(posts)
        print(f"  r/{sub}: {len(posts)}건")
        time.sleep(1.5)  # rate limit 방지

    print("[Reddit] 거시경제 서브레딧 크롤링 시작...")
    for sub in MACRO_SUBREDDITS:
        posts = fetch_subreddit_posts(sub, sort="hot", limit=limit_per_sub)
        results["macro"].extend(posts)
        print(f"  r/{sub}: {len(posts)}건")
        time.sleep(1.5)

    print(
        f"[Reddit] 완료: 크립토 {len(results['crypto'])}건, "
        f"거시경제 {len(results['macro'])}건"
    )
    return results
