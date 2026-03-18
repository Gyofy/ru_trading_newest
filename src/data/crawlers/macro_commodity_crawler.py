"""거시경제 + 원자재 + 유동성 지표 수집기.

yfinance로 수집 가능한 모든 거시/원자재 데이터를 5분봉에 시간 정렬하여 제공합니다.

수집 대상:
- 원자재: 금, 은, WTI 원유, 브렌트 원유, 알루미늄
- ETF: 리튬(LIT), 희토류(REMX), 구리(COPX)
- 국채: 미국 10Y/30Y/5Y 수익률
- 지수: S&P500, VIX, USD Index, 나스닥
- 유동성: DeFiLlama TVL, Fear & Greed Index
- 환율: EUR/USD, USD/JPY, USD/KRW

시간 정렬 전략:
- 선물/외환: 거의 24시간 거래 → 1시간봉 → 5분봉 forward-fill
- 주식/ETF: 장 시간만 → 1시간봉 → 5분봉 forward-fill
- API 데이터: 일봉 → 5분봉 forward-fill
"""

import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Any

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CryptoResearch/3.0"
}

# ==================== yfinance 티커 정의 ====================

MACRO_TICKERS = {
    # v4: 5 tickers only — high crypto-correlation, low redundancy
    "gold": {"symbol": "GC=F", "name": "Gold Futures", "category": "commodity"},       # safe haven
    "vix": {"symbol": "^VIX", "name": "VIX Volatility Index", "category": "index"},    # fear gauge
    "usd_index": {"symbol": "DX-Y.NYB", "name": "US Dollar Index", "category": "index"},  # dollar strength
    "us_10y": {"symbol": "^TNX", "name": "US 10Y Treasury Yield", "category": "rates"},   # real rates
    "sp500": {"symbol": "^GSPC", "name": "S&P 500", "category": "index"},                 # risk appetite
}


# ==================== yfinance 데이터 수집 ====================

def fetch_macro_ohlcv(period: str = "60d", interval: str = "1h") -> dict[str, pd.DataFrame]:
    """모든 거시/원자재 티커의 OHLCV를 수집합니다.

    Args:
        period: 수집 기간 (yfinance 형식)
        interval: 봉 간격 (1h → 이후 5분봉에 forward-fill)

    Returns:
        {ticker_id: DataFrame(close, volume, returns, ...)}
    """
    print("[Macro] 거시경제/원자재 데이터 수집 시작...")
    results = {}

    for ticker_id, info in MACRO_TICKERS.items():
        try:
            df = yf.download(
                info["symbol"], period=period, interval=interval,
                progress=False, auto_adjust=True,
            )
            if len(df) == 0:
                print(f"  [EMPTY] {info['name']} ({info['symbol']})")
                continue

            # MultiIndex columns 처리 (yfinance 최신 버전)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # timezone-aware → naive (crypto 데이터와 통일)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # v4: 3 features per ticker (close, return_1h, vol_24h)
            out = pd.DataFrame(index=df.index)
            out[f"macro_{ticker_id}_close"] = df["Close"].astype(float)

            close = out[f"macro_{ticker_id}_close"]
            out[f"macro_{ticker_id}_return_1h"] = close.pct_change(1)
            out[f"macro_{ticker_id}_vol_24h"] = close.pct_change().rolling(24, min_periods=1).std()

            results[ticker_id] = out
            latest = float(close.iloc[-1])
            print(f"  [OK] {info['name']}: {len(out)} bars, latest={latest:.2f}")

        except Exception as e:
            print(f"  [FAIL] {info['name']} ({info['symbol']}): {e}")

    print(f"[Macro] 완료: {len(results)}/{len(MACRO_TICKERS)} 수집 성공")
    return results


# ==================== Fear & Greed Index ====================

def fetch_fear_greed(days: int = 60) -> pd.DataFrame:
    """Alternative.me Fear & Greed Index를 일봉으로 수집합니다."""
    print("[FearGreed] Fear & Greed Index 수집...")
    url = f"https://api.alternative.me/fng/?limit={days}&format=json"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception as e:
        print(f"  [FAIL] Fear & Greed: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    records = []
    for item in data:
        ts = datetime.fromtimestamp(int(item["timestamp"]))
        records.append({
            "datetime": ts,
            "macro_fear_greed_value": int(item["value"]),
            "macro_fear_greed_class": item.get("value_classification", ""),
        })

    df = pd.DataFrame(records).set_index("datetime").sort_index()

    # Fear & Greed 파생 features
    fgv = df["macro_fear_greed_value"].astype(float)
    df["macro_fear_greed_momentum"] = fgv.diff(1)
    df["macro_fear_greed_sma7"] = fgv.rolling(7, min_periods=1).mean()
    df["macro_fear_greed_extreme"] = ((fgv < 20) | (fgv > 80)).astype(float)

    print(f"  [OK] Fear & Greed: {len(df)} days, latest={int(fgv.iloc[-1])} ({df['macro_fear_greed_class'].iloc[-1]})")
    return df


# ==================== DeFiLlama TVL ====================

def fetch_defi_tvl() -> pd.DataFrame:
    """DeFiLlama에서 주요 프로토콜 TVL을 수집합니다."""
    print("[DeFiLlama] Total Value Locked 수집...")

    try:
        resp = requests.get("https://api.llama.fi/v2/historicalChainTvl", headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [FAIL] DeFiLlama TVL: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    records = []
    for item in data:
        ts = datetime.fromtimestamp(int(item["date"]))
        records.append({
            "datetime": ts,
            "macro_defi_tvl": float(item.get("tvl", 0)),
        })

    df = pd.DataFrame(records).set_index("datetime").sort_index()

    # TVL 파생 features
    tvl = df["macro_defi_tvl"]
    df["macro_defi_tvl_return_1d"] = tvl.pct_change(1)
    df["macro_defi_tvl_return_7d"] = tvl.pct_change(7)
    df["macro_defi_tvl_sma7_ratio"] = tvl / (tvl.rolling(7, min_periods=1).mean() + 1e-10)

    # 최근 60일만
    cutoff = datetime.now() - timedelta(days=65)
    df = df[df.index >= cutoff]

    latest_tvl = tvl.iloc[-1] / 1e9
    print(f"  [OK] DeFi TVL: {len(df)} days, latest=${latest_tvl:.1f}B")
    return df


# ==================== 시간 정렬 (5분봉) ====================

def align_macro_to_5min(
    macro_data: dict[str, pd.DataFrame],
    crypto_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """거시/원자재 데이터를 크립토 타임스탬프에 정렬합니다.

    전략:
    1. 원본 데이터를 크립토 인덱스에 reindex
    2. forward-fill: 새 데이터가 나올 때까지 직전 값 유지
    3. 장 마감/휴일 구간도 forward-fill로 커버
    4. 첫 구간 NaN은 backward-fill로 채움
    """
    # timezone 통일: 모두 naive로 변환
    if crypto_index.tz is not None:
        crypto_index = crypto_index.tz_localize(None)

    aligned = pd.DataFrame(index=crypto_index)

    for ticker_id, df in macro_data.items():
        if len(df) == 0:
            continue

        # timezone → naive 통일
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 5분봉 인덱스에 reindex + forward-fill
        for col in df.columns:
            if df[col].dtype == object:  # skip string columns
                continue
            # Merge on nearest timestamp (forward-fill)
            temp = df[[col]].reindex(crypto_index, method="ffill")
            temp = temp.fillna(0)  # no bfill: prevents future data leakage
            aligned[col] = temp[col]

    # NaN이 남은 컬럼은 0으로
    aligned = aligned.fillna(0)

    return aligned


# ==================== 통합 수집 함수 ====================

def crawl_all_macro_data(crypto_index: pd.DatetimeIndex = None) -> dict[str, Any]:
    """모든 거시경제/원자재/유동성 데이터를 수집합니다.

    Args:
        crypto_index: 크립토 5분봉 DatetimeIndex (시간 정렬용)

    Returns:
        {
            "raw": {ticker_id: DataFrame},  # 원본 1시간봉
            "fear_greed": DataFrame,         # 일봉
            "defi_tvl": DataFrame,           # 일봉
            "aligned": DataFrame,            # 5분봉 정렬된 전체 매크로 features
            "feature_count": int,            # 총 feature 수
        }
    """
    print("\n" + "=" * 70)
    print("  MACRO / COMMODITY / LIQUIDITY DATA COLLECTION")
    print("=" * 70)

    # 1. yfinance 거시/원자재 (1시간봉, 365일)
    macro_raw = fetch_macro_ohlcv("365d", "1h")

    # 2. Fear & Greed (일봉)
    fear_greed = fetch_fear_greed(365)

    # 3. DeFi TVL (일봉)
    defi_tvl = fetch_defi_tvl()

    # 4. 시간 정렬 (5분봉)
    if crypto_index is not None:
        all_data = dict(macro_raw)  # copy
        if len(fear_greed) > 0:
            all_data["fear_greed"] = fear_greed
        if len(defi_tvl) > 0:
            all_data["defi_tvl"] = defi_tvl

        aligned = align_macro_to_5min(all_data, crypto_index)
        feature_count = len(aligned.columns)
        print(f"\n[Macro] 시간 정렬 완료: {feature_count} features → {len(aligned)} rows")
    else:
        aligned = pd.DataFrame()
        feature_count = 0

    return {
        "raw": macro_raw,
        "fear_greed": fear_greed,
        "defi_tvl": defi_tvl,
        "aligned": aligned,
        "feature_count": feature_count,
    }


if __name__ == "__main__":
    result = crawl_all_macro_data()
    print(f"\nRaw tickers: {len(result['raw'])}")
    print(f"Fear & Greed: {len(result['fear_greed'])} rows")
    print(f"DeFi TVL: {len(result['defi_tvl'])} rows")
