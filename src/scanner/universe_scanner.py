"""Universe Scanner - 50+ 코인 OHLCV + 매크로 데이터 수집.

yfinance로 1시간봉(7일) 수집 후 tradability_scorer에 넘긴다.
5분봉은 단기 변동성/거래량 패턴 분석용으로 짧게 수집.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Any

# ── 유니버스 정의: 시가총액 상위 50+ 코인 ──
UNIVERSE = {
    # Tier 1: 메이저 (시총 Top 10)
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "BNB": "BNB-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
    "ADA": "ADA-USD",
    "DOGE": "DOGE-USD",
    "AVAX": "AVAX-USD",
    "DOT": "DOT-USD",
    "LINK": "LINK-USD",
    # Tier 2: 준메이저 (시총 11~25)
    "LTC": "LTC-USD",
    "UNI": "UNI7083-USD",
    "ATOM": "ATOM-USD",
    "APT": "APT21794-USD",
    "ARB": "ARB11841-USD",
    "OP": "OP-USD",
    "NEAR": "NEAR-USD",
    "FIL": "FIL-USD",
    "AAVE": "AAVE-USD",
    "ICP": "ICP-USD",
    "HBAR": "HBAR-USD",
    "VET": "VET-USD",
    "ALGO": "ALGO-USD",
    "RENDER": "RENDER-USD",
    "SAND": "SAND-USD",
    # Tier 3: 중소형 (시총 26~50, 거래량 활발한 것)
    "MANA": "MANA-USD",
    "AXS": "AXS-USD",
    "THETA": "THETA-USD",
    "EOS": "EOS-USD",
    "IOTA": "IOTA-USD",
    "XLM": "XLM-USD",
    "XTZ": "XTZ-USD",
    "EGLD": "EGLD-USD",
    "CHZ": "CHZ-USD",
    "CRV": "CRV-USD",
    "LDO": "LDO-USD",
    "SNX": "SNX-USD",
    "GRT": "GRT6719-USD",
    "MKR": "MKR-USD",
    "ENJ": "ENJ-USD",
    "GALA": "GALA-USD",
    "1INCH": "1INCH-USD",
    "DYDX": "DYDX-USD",
    "ZIL": "ZIL-USD",
    "KAVA": "KAVA-USD",
    "ENS": "ENS-USD",
    "RUNE": "RUNE-USD",
    "PENDLE": "PENDLE-USD",
    "WLD": "WLD-USD",
    "SEI": "SEI-USD",
    "SUI": "SUI20947-USD",
    "TIA": "TIA-USD",
    "JUP": "JUP29210-USD",
    "STX": "STX4847-USD",
    "INJ": "INJ-USD",
}


def scan_ohlcv(
    universe: dict[str, str] = None,
    period: str = "7d",
    interval: str = "1h",
) -> dict[str, pd.DataFrame]:
    """전체 유니버스 OHLCV 수집."""
    if universe is None:
        universe = UNIVERSE

    print(f"[Scanner] OHLCV 수집 ({interval}, {period}, {len(universe)} coins)...")
    results = {}
    fail_count = 0

    for coin, ticker in universe.items():
        try:
            df = yf.download(
                ticker, period=period, interval=interval,
                progress=False, auto_adjust=True,
            )
            if len(df) == 0:
                fail_count += 1
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            results[coin] = df
        except Exception:
            fail_count += 1

    print(f"[Scanner] {len(results)}/{len(universe)} OK, {fail_count} failed")
    return results


def scan_btc_dominance() -> float | None:
    """BTC dominance (CoinGecko)."""
    try:
        import requests
        resp = requests.get(
            "https://api.coingecko.com/api/v3/global",
            headers={"User-Agent": "CryptoScanner/1.0"},
            timeout=10,
        )
        data = resp.json().get("data", {})
        return data.get("market_cap_percentage", {}).get("btc", None)
    except Exception:
        return None


def scan_fear_greed() -> dict | None:
    """Fear & Greed Index."""
    try:
        import requests
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=1&format=json",
            timeout=10,
        )
        item = resp.json().get("data", [{}])[0]
        return {
            "value": int(item.get("value", 50)),
            "classification": item.get("value_classification", "Neutral"),
        }
    except Exception:
        return None


def scan_all() -> dict[str, Any]:
    """전체 스캔: 1h(7d) + 5m(2d) + 매크로."""
    print("\n" + "=" * 60)
    print("  UNIVERSE SCANNER - Short-term Profit Scan")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Universe: {len(UNIVERSE)} coins")
    print("=" * 60)

    # 1시간봉 7일 (핵심: 거래량 + 변동성 패턴)
    ohlcv_1h = scan_ohlcv(UNIVERSE, "7d", "1h")

    # 1시간봉 7일 추가 (단기 변동성 세부) — 4h 체제에서는 1h로 충분
    ohlcv_5m = scan_ohlcv(UNIVERSE, "7d", "1h")

    btc_dom = scan_btc_dominance()
    fear_greed = scan_fear_greed()

    if btc_dom:
        print(f"  BTC Dominance: {btc_dom:.1f}%")
    if fear_greed:
        print(f"  Fear & Greed: {fear_greed['value']} ({fear_greed['classification']})")

    return {
        "ohlcv_1h": ohlcv_1h,
        "ohlcv_5m": ohlcv_5m,
        "btc_dominance": btc_dom,
        "fear_greed": fear_greed,
        "scanned_at": datetime.now().isoformat(),
    }
