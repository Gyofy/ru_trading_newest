"""Binance USDT-M Futures 연결 테스트.

API 키 없이도 퍼블릭 엔드포인트(시세, 펀딩률)는 테스트 가능.
프라이빗 기능(잔고, 주문)은 API 키 필요.

Usage:
    # 퍼블릭 테스트만:
    python test_binance_connection.py

    # API 키 포함:
    BINANCE_API_KEY=k4xwe6ILk8ogA12WSCX5CbpEY06q8MbAU31vXOfqLe9plcuiTZGvD2mCicgxfmyJ BINANCE_API_SECRET=RM0HTz27zlSfgK9LXAqHem1KoJfZ2hcdlPG7FM0kAcGWMCWCuEjglSAVnIFh13dG python test_binance_connection.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.execution.exchange_adapter import ExchangeAdapter


async def main():
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    mode = "sandbox" if api_key else "sandbox"  # testnet 사용

    print(f"=== Binance USDT-M Futures 연결 테스트 ===")
    print(f"Mode: {mode}")
    print(f"API Key: {'SET' if api_key else 'NOT SET (public only)'}")
    print()

    async with ExchangeAdapter(
        exchange="binance",
        mode=mode,
        api_key=api_key,
        secret=api_secret,
    ) as adapter:

        # 1. 마켓 로드 확인
        print(f"[OK] Markets loaded: {len(adapter._exchange.markets)}")

        # 2. 시세 조회 (public)
        for symbol in ["DOT", "ADA", "BTC"]:
            try:
                ticker = await adapter.fetch_ticker(symbol)
                print(
                    f"[OK] {symbol}: bid={ticker['bid']}, ask={ticker['ask']}, "
                    f"spread={ticker['spread_bps']:.1f}bps"
                )
            except Exception as e:
                print(f"[FAIL] {symbol} ticker: {e}")

        # 3. 펀딩률 (public)
        for symbol in ["DOT", "ADA"]:
            try:
                rate = await adapter.fetch_funding_rate(symbol)
                print(f"[OK] {symbol} funding rate: {rate*100:.4f}%")
            except Exception as e:
                print(f"[FAIL] {symbol} funding: {e}")

        # 4. 잔고 조회 (private — API 키 필요)
        if api_key:
            try:
                balance = await adapter.fetch_balance()
                print(f"\n[OK] Balance: {balance}")
            except Exception as e:
                print(f"\n[FAIL] Balance: {e}")

        # 5. Order ID 생성 테스트
        oid = ExchangeAdapter.make_order_id("DOT", "BUY", prefix="bnb")
        print(f"\n[OK] Sample order ID: {oid}")

    print("\n=== 테스트 완료 ===")


if __name__ == "__main__":
    asyncio.run(main())
