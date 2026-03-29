"""BNB Keeper — BNB 잔고 자동 유지 (수수료 할인 10% 확보).

Binance Futures에서 수수료를 BNB로 납부하면 10% 추가 할인을 받는다.
BNB 잔고가 min_bnb_usdt 미만이면 시장가로 매수하여 잔고를 유지한다.

주의: Binance Futures 계좌의 BNB는 실제로 Spot 지갑에 보유해야 한다.
Futures 전용 계좌는 직접 BNB 구매가 불가하므로 경고만 출력한다.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.execution.exchange_adapter import ExchangeAdapter

log = logging.getLogger("bnb_keeper")


class BnbKeeper:
    """시작 시 + 주기적으로 BNB 잔고를 체크하고 부족하면 시장가로 매수."""

    def __init__(
        self,
        exchange: "ExchangeAdapter",
        min_bnb_usdt: float = 10.0,
        check_interval: int = 3600,
    ) -> None:
        """
        Args:
            exchange: ExchangeAdapter 인스턴스.
            min_bnb_usdt: BNB 잔고가 이 금액(USDT 기준) 미만이면 매수.
            check_interval: 체크 주기(초). 기본 3600 = 1시간.
        """
        self.exchange = exchange
        self.min_bnb_usdt = min_bnb_usdt
        self.check_interval = check_interval
        self._last_buy_ts: float = 0.0
        self._buy_cooldown: float = 300.0  # 5분 이내 중복 매수 방지

    async def check_and_buy(self) -> dict:
        """BNB 잔고 확인 후 필요시 매수.

        Returns:
            dict: {
                "bnb_qty": float,           — 현재 BNB 수량
                "bnb_usdt_value": float,    — USDT 환산 가치
                "bnb_price": float,         — 현재 BNB 가격
                "bought": bool,             — 이번 호출에서 매수했는지
                "buy_qty": float,           — 매수한 BNB 수량 (0이면 미매수)
                "error": str | None,        — 오류 메시지
            }
        """
        result: dict = {
            "bnb_qty": 0.0,
            "bnb_usdt_value": 0.0,
            "bnb_price": 0.0,
            "bought": False,
            "buy_qty": 0.0,
            "error": None,
        }

        # 1. BNB 잔고 조회
        try:
            raw_balance = await self.exchange._exchange.fetch_balance()
            bnb_free = float(
                raw_balance.get("BNB", {}).get("free", 0) or 0
            )
            bnb_total = float(
                raw_balance.get("BNB", {}).get("total", 0) or 0
            )
            result["bnb_qty"] = bnb_total
        except Exception as e:
            log.warning(f"[BnbKeeper] BNB 잔고 조회 실패: {e}")
            result["error"] = f"balance_fetch_failed: {e}"
            return result

        # 2. BNB/USDT 현재가 조회
        try:
            ticker = await self.exchange._exchange.fetch_ticker("BNB/USDT:USDT")
            bnb_price = float(ticker.get("last", 0) or 0)
            if bnb_price <= 0:
                # Spot 티커로 폴백
                ticker_spot = await self.exchange._exchange.fetch_ticker("BNB/USDT")
                bnb_price = float(ticker_spot.get("last", 0) or 0)
        except Exception as e:
            log.warning(f"[BnbKeeper] BNB 가격 조회 실패: {e}")
            result["error"] = f"price_fetch_failed: {e}"
            return result

        if bnb_price <= 0:
            log.warning("[BnbKeeper] BNB 가격이 0 — 스킵")
            result["error"] = "bnb_price_zero"
            return result

        result["bnb_price"] = bnb_price
        bnb_usdt_value = bnb_total * bnb_price
        result["bnb_usdt_value"] = bnb_usdt_value

        log.info(
            f"[BnbKeeper] BNB 잔고: {bnb_total:.4f} BNB "
            f"(≈${bnb_usdt_value:.2f} USDT) @ ${bnb_price:.2f}"
        )

        # 3. 잔고 충분 — 매수 불필요
        if bnb_usdt_value >= self.min_bnb_usdt:
            log.info(
                f"[BnbKeeper] 잔고 충분 (${bnb_usdt_value:.2f} >= ${self.min_bnb_usdt:.2f}) — 스킵"
            )
            return result

        # 쿨다운 체크 — 최근 5분 이내 매수했으면 스킵
        import time
        now = time.monotonic()
        if now - self._last_buy_ts < self._buy_cooldown:
            log.info(
                f"[BnbKeeper] 매수 쿨다운 중 ({self._buy_cooldown:.0f}s) — 스킵"
            )
            return result

        # 4. 매수 필요: min_bnb_usdt × 1.5 상당의 BNB 매수
        buy_usdt = self.min_bnb_usdt * 1.5
        buy_qty_raw = buy_usdt / bnb_price
        # Binance BNB 최소 수량: 0.01 BNB (정밀도 2자리)
        buy_qty = round(buy_qty_raw, 2)
        if buy_qty < 0.01:
            buy_qty = 0.01

        log.info(
            f"[BnbKeeper] BNB 부족 — ${buy_usdt:.2f} 상당 ({buy_qty:.2f} BNB) 시장가 매수 시도"
        )

        try:
            # Futures 계좌로 직접 BNB 구매 시도 (USDM Futures에서는 불가할 수 있음)
            order = await self.exchange._exchange.create_order(
                symbol="BNB/USDT",
                type="market",
                side="buy",
                amount=buy_qty,
            )
            filled_qty = float(order.get("filled", buy_qty) or buy_qty)
            cost = float(order.get("cost", filled_qty * bnb_price) or 0)
            result["bought"] = True
            result["buy_qty"] = filled_qty
            self._last_buy_ts = now
            log.info(
                f"[BnbKeeper] BNB 매수 완료: {filled_qty:.4f} BNB (${cost:.2f} USDT)"
            )
        except Exception as e:
            _err_msg = str(e)
            # Futures 계좌 직접 BNB 구매 불가 — Spot 지갑 이용 필요
            if any(kw in _err_msg.lower() for kw in ["spot", "not supported", "invalid", "margin"]):
                log.warning(
                    f"[BnbKeeper] Futures 계좌에서 직접 BNB 매수 불가 — "
                    f"Spot 지갑에서 BNB를 보유해야 수수료 할인이 적용됩니다. "
                    f"오류: {e}"
                )
            else:
                log.error(f"[BnbKeeper] BNB 매수 실패: {e}")
            result["error"] = f"buy_failed: {e}"

        return result

    async def start_background(self) -> None:
        """백그라운드 루프 — 즉시 1회 체크 후 check_interval마다 반복."""
        log.info(
            f"[BnbKeeper] 백그라운드 시작 "
            f"(min=${self.min_bnb_usdt:.2f} USDT, interval={self.check_interval}s)"
        )
        # 시작 즉시 1회 실행
        try:
            await self.check_and_buy()
        except Exception as e:
            log.warning(f"[BnbKeeper] 초기 체크 오류: {e}")

        while True:
            try:
                await asyncio.sleep(self.check_interval)
                await self.check_and_buy()
            except asyncio.CancelledError:
                log.info("[BnbKeeper] 백그라운드 루프 종료")
                break
            except Exception as e:
                log.warning(f"[BnbKeeper] 루프 오류 (계속 실행): {e}")
