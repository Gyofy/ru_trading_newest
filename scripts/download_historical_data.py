#!/usr/bin/env python3
"""
Binance Futures 15분봉/1시간봉 OHLCV 다운로드 → data/historical/ 저장.

사용법:
    python3 scripts/download_historical_data.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

COINS     = ["SOL", "XRP", "DOGE", "ADA", "BNB", "ETH", "DOT"]
TIMEFRAMES = ["15m", "1h"]
START_DATE = "2024-01-01"
END_DATE   = "2026-03-31"
OUT_DIR    = Path("data/historical")


async def fetch(exchange, symbol: str, timeframe: str, since_ms: int, end_ms: int) -> list:
    bars: list = []
    ccxt_tf = {"15m": "15m", "1h": "1h"}[timeframe]
    while since_ms < end_ms:
        try:
            chunk = await exchange._exchange.fetch_ohlcv(
                f"{symbol}/USDT:USDT", timeframe=ccxt_tf, since=since_ms, limit=1000
            )
        except Exception as e:
            print(f"  [{symbol}/{timeframe}] 오류: {e}")
            break
        if not chunk:
            break
        bars.extend(chunk)
        since_ms = chunk[-1][0] + 1
        await asyncio.sleep(0.1)
    return bars


async def main() -> None:
    from src.execution.exchange_adapter import ExchangeAdapter

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    exchange = ExchangeAdapter(
        mode="live",
        api_key=os.environ.get("BINANCE_API_KEY", ""),
        secret=os.environ.get("BINANCE_API_SECRET", ""),
    )
    await exchange.initialize()

    try:
        for tf in TIMEFRAMES:
            for coin in COINS:
                out_path = OUT_DIR / f"{coin}_{tf}_{START_DATE}_{END_DATE}.parquet"
                if out_path.exists():
                    print(f"[SKIP] {out_path.name} 이미 존재")
                    continue

                print(f"[FETCH] {coin}/{tf} {START_DATE} → {END_DATE} ...")
                since_ms = int(pd.Timestamp(START_DATE, tz="UTC").timestamp() * 1000)
                end_ms   = int(pd.Timestamp(END_DATE,   tz="UTC").timestamp() * 1000)

                bars = await fetch(exchange, coin, tf, since_ms, end_ms)
                if not bars:
                    print(f"  [{coin}/{tf}] 데이터 없음 — 건너뜀")
                    continue

                df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df.set_index("timestamp", inplace=True)
                df = df[df.index <= pd.Timestamp(END_DATE, tz="UTC")]
                df.to_parquet(out_path)
                print(f"  [{coin}/{tf}] {len(df)} bars → {out_path}")

    finally:
        await exchange.close()
        print("\n✅ 다운로드 완료")


if __name__ == "__main__":
    asyncio.run(main())
