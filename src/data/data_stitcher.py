"""Data Stitcher — 다시점 수집 데이터 병합.

yfinance 5분봉은 최대 60일만 제공하므로,
여러 날짜에 수집한 parquet 파일들을 시간순 병합하여
60일 이상의 연속 데이터를 구성합니다.

Usage:
    from src.data.data_stitcher import stitch_all_coins, collect_and_stitch

    # 기존 수집 데이터 병합
    merged = stitch_all_coins()  # data/raw/*/을 전부 탐색

    # 새로 수집 + 기존 데이터와 병합
    merged = collect_and_stitch()
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


RAW_DIR = Path("data/raw")
STITCHED_DIR = Path("data/stitched")

TOP10_COINS = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK", "BNB"]


def find_parquets(coin: str) -> list[Path]:
    """특정 코인의 모든 수집 시점 parquet 파일을 찾습니다."""
    files = []
    if not RAW_DIR.exists():
        return files
    for date_dir in sorted(RAW_DIR.iterdir()):
        if not date_dir.is_dir():
            continue
        pq = date_dir / f"{coin}_ohlcv_1h.parquet"
        if pq.exists():
            files.append(pq)
    return files


def stitch_coin(coin: str, files: list[Path] | None = None) -> pd.DataFrame | None:
    """한 코인의 다시점 parquet들을 시간순 병합 (중복 제거).

    Returns:
        병합된 DataFrame (DatetimeIndex, 중복 timestamp 제거)
        또는 데이터 없으면 None
    """
    if files is None:
        files = find_parquets(coin)
    if not files:
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            # 인덱스가 DatetimeIndex가 아니면 변환 시도
            if not isinstance(df.index, pd.DatetimeIndex):
                if "Datetime" in df.columns:
                    df = df.set_index("Datetime")
                elif "Date" in df.columns:
                    df = df.set_index("Date")
            # tz-aware → naive 통일
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            dfs.append(df)
        except Exception as e:
            print(f"  [Stitch] Skip {f}: {e}")
            continue

    if not dfs:
        return None

    # 병합 + 중복 제거 (같은 timestamp는 최신 수집분 우선)
    merged = pd.concat(dfs, axis=0)
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index()

    return merged


def stitch_all_coins(coins: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """전 코인의 다시점 데이터를 병합합니다.

    Returns:
        {coin: merged_DataFrame}
    """
    if coins is None:
        coins = TOP10_COINS

    result = {}
    for coin in coins:
        merged = stitch_coin(coin)
        if merged is not None and len(merged) > 0:
            result[coin] = merged

    if result:
        # 요약 출력
        print(f"\n  ── Data Stitcher: {len(result)} coins merged ──")
        for coin, df in result.items():
            days = (df.index[-1] - df.index[0]).days if len(df) > 1 else 0
            print(f"    {coin}: {len(df):,} bars, {days}d span "
                  f"({df.index[0].strftime('%m/%d')} ~ {df.index[-1].strftime('%m/%d')})")

    return result


def save_stitched(merged_data: dict[str, pd.DataFrame]) -> Path:
    """병합된 데이터를 stitched/ 디렉토리에 저장합니다."""
    STITCHED_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = STITCHED_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    for coin, df in merged_data.items():
        path = out_dir / f"{coin}_stitched.parquet"
        df.to_parquet(path)

    print(f"  Saved stitched data: {out_dir}")
    return out_dir


def load_latest_stitched() -> dict[str, pd.DataFrame] | None:
    """가장 최근 stitched 데이터를 로드합니다."""
    if not STITCHED_DIR.exists():
        return None
    dirs = sorted(STITCHED_DIR.iterdir())
    if not dirs:
        return None

    latest = dirs[-1]
    result = {}
    for pq in latest.glob("*_stitched.parquet"):
        coin = pq.stem.replace("_stitched", "")
        try:
            result[coin] = pd.read_parquet(pq)
        except Exception:
            continue

    return result if result else None


def collect_and_stitch() -> dict[str, pd.DataFrame]:
    """새로 OHLCV 수집 → 기존 데이터와 병합 → 저장.

    Returns:
        병합된 {coin: DataFrame}
    """
    from src.data.crawlers.crypto_ohlcv import fetch_all_top10, save_ohlcv_data

    print("\n  ── Collect Fresh OHLCV + Stitch with History ──")

    # 1. 새로 수집
    fresh = fetch_all_top10("365d", "1h")
    if fresh:
        save_ohlcv_data(fresh)

    # 2. 기존 수집분과 병합
    merged = stitch_all_coins()

    # 3. 저장
    if merged:
        save_stitched(merged)

    return merged
