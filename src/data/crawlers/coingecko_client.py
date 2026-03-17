"""크립토 시세 클라이언트 (멀티 소스 폴백).

CoinCap → CoinGecko → Google News 가격 추출 순으로 시도.
"""

import re
import time
import requests
from datetime import datetime, timezone, timedelta
from typing import Any

COINCAP_URL = "https://api.coincap.io/v2"
HEADERS = {"accept": "application/json", "User-Agent": "CryptoAgent/1.0"}
TIMEOUT = 10

# CoinCap ID 매핑
COINCAP_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "xrp",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche",
    "DOT": "polkadot",
    "LINK": "chainlink",
    "BNB": "binance-coin",
    "NEAR": "near-protocol",
    "SUI": "sui",
    "ARB": "arbitrum",
    "OP": "optimism",
    "MATIC": "polygon",
}


def get_price_data(ticker: str) -> dict[str, Any]:
    """CoinCap에서 현재 시세를 조회합니다."""
    coin_id = COINCAP_IDS.get(ticker.upper(), ticker.lower())
    url = f"{COINCAP_URL}/assets/{coin_id}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        d = resp.json().get("data", {})
    except Exception as e:
        print(f"  [CoinCap] {ticker} API 실패, 뉴스 기반 가격 추출 시도...")
        return _extract_price_from_news(ticker)

    price = float(d.get("priceUsd", 0))
    return {
        "ticker": ticker.upper(),
        "name": d.get("name", ""),
        "price_usd": round(price, 2),
        "price_krw": round(price * 1350, 0),  # 근사 환율
        "market_cap_usd": float(d.get("marketCapUsd", 0)),
        "market_cap_rank": int(d.get("rank", 0)),
        "total_volume_usd": float(d.get("volumeUsd24Hr", 0)),
        "change_24h_pct": round(float(d.get("changePercent24Hr", 0)), 2),
        "supply": float(d.get("supply", 0)),
        "max_supply": float(d.get("maxSupply", 0) or 0),
        "vwap_24h": float(d.get("vwap24Hr", 0) or 0),
        # CoinCap은 7d/30d 변동률을 직접 제공하지 않으므로 히스토리에서 계산
        "change_7d_pct": 0,
        "change_30d_pct": 0,
        "ath_usd": 0,
        "ath_change_pct": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def get_price_history(ticker: str, days: int = 30) -> list[dict]:
    """CoinCap에서 히스토리를 조회합니다."""
    coin_id = COINCAP_IDS.get(ticker.upper(), ticker.lower())

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    # interval: d1 = daily
    url = f"{COINCAP_URL}/assets/{coin_id}/history"
    params = {"interval": "d1", "start": start_ms, "end": end_ms}

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception as e:
        print(f"  [CoinCap] {ticker} 히스토리 실패, 뉴스 가격으로 단일 포인트 생성...")
        fallback = _extract_price_from_news(ticker)
        if fallback and fallback.get("price_usd"):
            return [{"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "price": fallback["price_usd"], "volume": 0}]
        return []

    history = []
    for point in data:
        ts = point.get("time", 0)
        price = float(point.get("priceUsd", 0))
        history.append({
            "date": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
            "price": round(price, 2),
            "volume": 0,  # CoinCap 히스토리는 거래량 미포함
        })

    return history


def enrich_with_history(price_data: dict, history: list[dict]) -> dict:
    """히스토리 데이터로 7d/30d 변동률을 보강합니다."""
    if not history or not price_data:
        return price_data

    current = price_data["price_usd"]
    if current == 0:
        return price_data

    if len(history) >= 7:
        p7d = history[-7]["price"]
        if p7d > 0:
            price_data["change_7d_pct"] = round((current - p7d) / p7d * 100, 2)

    if len(history) >= 30:
        p30d = history[0]["price"]
        if p30d > 0:
            price_data["change_30d_pct"] = round((current - p30d) / p30d * 100, 2)

    # ATH 근사 (히스토리 내 최고가)
    max_price = max(p["price"] for p in history)
    price_data["ath_usd"] = max_price
    if max_price > 0:
        price_data["ath_change_pct"] = round((current - max_price) / max_price * 100, 1)

    return price_data


def get_trending() -> list[dict]:
    """시총 상위 + 24h 변동률 상위 코인을 반환합니다."""
    url = f"{COINCAP_URL}/assets"
    params = {"limit": 20}
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception as e:
        print(f"  [CoinCap] 트렌딩 실패: {e}")
        return []

    return [
        {
            "name": c.get("name", ""),
            "symbol": c.get("symbol", ""),
            "market_cap_rank": int(c.get("rank", 0)),
            "change_24h": round(float(c.get("changePercent24Hr", 0)), 2),
        }
        for c in data
    ]


def _extract_price_from_news(ticker: str) -> dict:
    """Google News 타이틀에서 가격 정보를 추출합니다 (API 폴백)."""
    from urllib.parse import quote
    import xml.etree.ElementTree as ET

    NAMES = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "xrp"}
    name = NAMES.get(ticker.upper(), ticker.lower())

    url = f"https://news.google.com/rss/search?q={quote(name + ' price USD')}&hl=en&gl=US&ceid=US:en"
    try:
        resp = requests.get(url, headers={"User-Agent": "CryptoAgent/1.0"}, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as e:
        print(f"  [News Price] {ticker} 추출 실패: {e}")
        return {}

    # 종목별 합리적 가격 범위
    PRICE_RANGES = {
        "BTC": (10000, 500000),
        "ETH": (500, 50000),
        "SOL": (5, 5000),
        "XRP": (0.1, 50),
    }
    low, high = PRICE_RANGES.get(ticker.upper(), (0.01, 1000000))

    prices_found = []
    change_found = []
    for item in root.findall(".//item")[:20]:
        title = item.findtext("title", "")

        # $66,000 / $69K / $2,750 패턴 추출
        price_matches = re.findall(r'\$([0-9,]+(?:\.\d+)?)\s*[kK]?', title)
        for pm in price_matches:
            val = float(pm.replace(",", ""))
            if val < 1:
                continue
            # 범위 필터: 예측가/비현실적 가격 제거
            if low <= val <= high:
                prices_found.append(val)

        # 변동률 추출: -4.5%, +12%
        chg_matches = re.findall(r'([+-]?\d+\.?\d*)\s*%', title)
        for cm in chg_matches:
            change_found.append(float(cm))

    if not prices_found:
        print(f"  [News Price] {ticker} 가격 추출 실패")
        return {}

    # 중앙값 사용 (이상치 방지)
    prices_found.sort()
    median_price = prices_found[len(prices_found) // 2]
    avg_change = sum(change_found) / len(change_found) if change_found else 0

    print(f"  [News Price] {ticker} 추정가: ${median_price:,.2f} (뉴스 {len(prices_found)}건에서 추출)")

    return {
        "ticker": ticker.upper(),
        "name": name.title(),
        "price_usd": round(median_price, 2),
        "price_krw": round(median_price * 1350, 0),
        "market_cap_usd": 0,
        "market_cap_rank": 0,
        "total_volume_usd": 0,
        "change_24h_pct": round(avg_change, 2),
        "change_7d_pct": 0,
        "change_30d_pct": 0,
        "ath_usd": 0,
        "ath_change_pct": 0,
        "supply": 0,
        "max_supply": 0,
        "vwap_24h": 0,
        "source": "news_extraction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def format_price_summary(data: dict) -> str:
    """가격 데이터를 문자열로 변환합니다."""
    if not data:
        return "[데이터 없음]"

    source = " (뉴스 추정)" if data.get("source") == "news_extraction" else ""
    lines = [
        f"## {data['ticker']} ({data.get('name','')}) 시장 데이터{source}",
        f"- 현재가: ${data['price_usd']:,.2f}",
    ]
    if data.get("market_cap_rank"):
        lines.append(f"- 시총 순위: #{data['market_cap_rank']}")
    if data.get("total_volume_usd"):
        lines.append(f"- 24h 거래량: ${data['total_volume_usd']:,.0f}")
    lines.append(f"- 24h 변동: {data['change_24h_pct']:+.2f}%")
    if data.get("change_7d_pct"):
        lines.append(f"- 7일 변동: {data['change_7d_pct']:+.2f}%")
    if data.get("change_30d_pct"):
        lines.append(f"- 30일 변동: {data['change_30d_pct']:+.2f}%")
    return "\n".join(lines)
