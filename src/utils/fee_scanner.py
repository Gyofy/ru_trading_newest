"""Fee Scanner — 수수료 0원 페어 탐지 및 우선순위 할당.

매일 1회 전체 심볼 수수료율을 스캔한다.
Taker 수수료가 0%인 USDT 선물 페어를 발견하면 zero_fee_pairs 목록을 업데이트한다.
이 목록은 전략 루프에서 참조하여 우선 진입 대상으로 사용할 수 있다.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.execution.exchange_adapter import ExchangeAdapter

log = logging.getLogger("fee_scanner")


class FeeScanner:
    """매일 1회 전체 심볼 수수료율 스캔. 0% 수수료 페어를 발견하면 우선순위 업데이트."""

    def __init__(
        self,
        exchange: "ExchangeAdapter",
        scan_interval: int = 86400,
    ) -> None:
        """
        Args:
            exchange: ExchangeAdapter 인스턴스.
            scan_interval: 스캔 주기(초). 기본 86400 = 24시간.
        """
        self.exchange = exchange
        self._scan_interval = scan_interval
        self.zero_fee_pairs: list[str] = []   # taker 수수료 0%인 코인 심볼 목록
        self.last_scan: float = 0.0           # 마지막 스캔 타임스탬프 (time.time())
        self._last_scan_result: dict = {}     # coin -> {"maker": float, "taker": float}

    async def scan(self) -> list[str]:
        """Binance tradeFee API 조회 → taker_fee == 0 인 USDT 선물 페어 반환.

        Returns:
            list[str]: taker 수수료 0%인 코인 심볼 목록 (예: ["BTC", "ETH"]).
                       API 미지원 시 빈 목록 반환 (graceful degrade).
        """
        log.info("[FeeScanner] 수수료율 스캔 시작...")
        zero_fee: list[str] = []
        fee_data: dict = {}

        # 방법 1: fapiPrivateGetCommissionRate (개별 심볼 조회 — 현재 계정 기준)
        # 방법 2: fetch_trading_fees() (전체 심볼 한 번에)
        # 방법 2를 우선 시도, 실패 시 방법 1로 폴백

        try:
            raw_fees = await self.exchange._exchange.fetch_trading_fees()
            # ccxt 반환 형식: {"BTC/USDT:USDT": {"maker": 0.0002, "taker": 0.0005}, ...}
            for symbol, rates in raw_fees.items():
                if not isinstance(rates, dict):
                    continue
                # USDT 선물 페어만 필터 (symbol 형식: "COIN/USDT:USDT")
                if not (symbol.endswith(":USDT") or symbol.endswith("USDT")):
                    continue
                taker = float(rates.get("taker", 9999) or 9999)
                maker = float(rates.get("maker", 9999) or 9999)
                # 코인 심볼 추출: "SOL/USDT:USDT" → "SOL"
                coin = symbol.split("/")[0] if "/" in symbol else symbol.replace("USDT", "")
                if not coin:
                    continue
                fee_data[coin] = {"maker": maker, "taker": taker}
                if taker == 0.0:
                    zero_fee.append(coin)
                    log.info(f"[FeeScanner] ZERO-FEE 발견: {coin} (taker={taker:.4%})")

        except Exception as e1:
            log.warning(f"[FeeScanner] fetch_trading_fees() 실패: {e1} — fapiPrivateGetCommissionRate 폴백 시도")
            # 폴백: Binance Futures Private API 직접 호출 (계정 현재 수수료율)
            try:
                sample_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
                for sym in sample_symbols:
                    try:
                        resp = await self.exchange._exchange.fapiPrivateGetCommissionRate(
                            {"symbol": sym}
                        )
                        taker = float(resp.get("takerCommissionRate", 0.0005) or 0.0005)
                        maker = float(resp.get("makerCommissionRate", 0.0002) or 0.0002)
                        coin = sym.replace("USDT", "")
                        fee_data[coin] = {"maker": maker, "taker": taker}
                        if taker == 0.0:
                            zero_fee.append(coin)
                    except Exception as _sym_e:
                        log.debug(f"[FeeScanner] {sym} 수수료 조회 실패: {_sym_e}")
            except Exception as e2:
                log.warning(f"[FeeScanner] 폴백 조회도 실패: {e2} — 스캔 생략 (기존 목록 유지)")
                return self.zero_fee_pairs  # graceful degrade: 기존 목록 유지

        # 결과 저장
        self._last_scan_result = fee_data
        self.zero_fee_pairs = zero_fee
        self.last_scan = time.time()

        if zero_fee:
            log.info(f"[FeeScanner] 스캔 완료 — ZERO-FEE 페어 {len(zero_fee)}개: {zero_fee}")
        else:
            log.info(f"[FeeScanner] 스캔 완료 — ZERO-FEE 페어 없음 (총 {len(fee_data)}개 심볼 조회)")

        return list(zero_fee)

    async def start_background(self) -> None:
        """백그라운드 루프 — 즉시 1회 스캔 후 scan_interval마다 반복."""
        log.info(
            f"[FeeScanner] 백그라운드 시작 (interval={self._scan_interval}s)"
        )
        # 시작 즉시 1회 실행
        try:
            await self.scan()
        except Exception as e:
            log.warning(f"[FeeScanner] 초기 스캔 오류: {e}")

        while True:
            try:
                await asyncio.sleep(self._scan_interval)
                await self.scan()
            except asyncio.CancelledError:
                log.info("[FeeScanner] 백그라운드 루프 종료")
                break
            except Exception as e:
                log.warning(f"[FeeScanner] 루프 오류 (계속 실행): {e}")

    def get_priority_coins(self) -> list[str]:
        """현재 zero_fee_pairs 목록을 반환 (우선 거래 대상)."""
        return self.zero_fee_pairs.copy()

    def is_zero_fee(self, coin: str) -> bool:
        """특정 코인이 zero-fee 페어인지 확인."""
        return coin in self.zero_fee_pairs

    def get_fee_rates(self, coin: str) -> dict:
        """특정 코인의 수수료율 반환. 데이터 없으면 기본값 반환."""
        default = {"maker": 0.0002, "taker": 0.0005}
        return self._last_scan_result.get(coin, default)
