"""한국투자증권 Open API 클라이언트.

KIS REST API를 통해 시세 조회, 주문 실행, 잔고 조회 등을 수행합니다.
토큰은 24시간 유효하며, 만료 시 자동 갱신합니다.
"""

import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any

import requests

from ..utils.logging import get_logger
from ..utils.config import load_settings

logger = get_logger("kis_client")

KST = timezone(timedelta(hours=9))


class KISClient:
    """KIS Open API REST 클라이언트."""

    def __init__(self, mode: str | None = None):
        settings = load_settings()
        self.mode = mode or settings.get("mode", "paper")

        self.app_key = os.environ.get("KIS_APP_KEY", "")
        self.app_secret = os.environ.get("KIS_APP_SECRET", "")
        self.account_no = os.environ.get("KIS_ACCOUNT_NO", "")

        # 모의투자 vs 실전 base URL
        if self.mode == "paper":
            self.base_url = "https://openapivts.koreainvestment.com:29443"
        else:
            self.base_url = "https://openapi.koreainvestment.com:9443"

        self._token: str | None = None
        self._token_expires: float = 0

    def _get_token(self) -> str:
        """접근 토큰 발급/갱신."""
        if self._token and time.time() < self._token_expires:
            return self._token

        url = f"{self.base_url}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        resp = requests.post(url, json=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        # 토큰 유효시간: 24시간이지만, 여유를 두고 23시간으로 설정
        self._token_expires = time.time() + 23 * 3600
        logger.info("KIS 토큰 발급 완료 (mode=%s)", self.mode)
        return self._token

    def _headers(self, tr_id: str) -> dict[str, str]:
        """API 요청 공통 헤더."""
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self._get_token()}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
        }

    def get_minute_candles(
        self, ticker: str, date: str, time_start: str = "090000", time_end: str = "153000"
    ) -> list[dict[str, Any]]:
        """1분봉 데이터 조회.

        Args:
            ticker: 종목 코드 (e.g. "005930")
            date: 조회일 (YYYYMMDD)
            time_start: 시작시각 (HHMMSS)
            time_end: 종료시각 (HHMMSS)

        Returns:
            1분봉 리스트 [{stck_bsop_date, stck_cntg_hour, stck_oprc, stck_hgpr, stck_lwpr, stck_clpr, cntg_vol, ...}]
        """
        # 국내주식 분봉 조회 API
        tr_id = "FHKST03010200"
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker,
            "FID_INPUT_HOUR_1": time_end,
            "FID_PW_DATA_INCU_YN": "Y",
        }
        resp = requests.get(url, headers=self._headers(tr_id), params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            logger.error("분봉 조회 실패: %s %s - %s", ticker, date, data.get("msg1"))
            return []

        return data.get("output2", [])

    def get_current_price(self, ticker: str) -> dict[str, Any]:
        """현재가 조회."""
        tr_id = "FHKST01010100"
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker,
        }
        resp = requests.get(url, headers=self._headers(tr_id), params=params, timeout=10)
        resp.raise_for_status()
        return resp.json().get("output", {})

    def get_balance(self) -> dict[str, Any]:
        """계좌 잔고 조회."""
        tr_id = "VTTC8434R" if self.mode == "paper" else "TTTC8434R"
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        acct_prefix = self.account_no[:8]
        acct_suffix = self.account_no[8:]
        params = {
            "CANO": acct_prefix,
            "ACNT_PRDT_CD": acct_suffix,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        resp = requests.get(url, headers=self._headers(tr_id), params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def place_order(
        self, ticker: str, side: str, qty: int, price: int = 0, order_type: str = "market"
    ) -> dict[str, Any]:
        """주문 실행.

        Args:
            ticker: 종목코드
            side: "buy" | "sell"
            qty: 수량
            price: 지정가 (시장가면 0)
            order_type: "market" | "limit"

        Returns:
            주문 결과
        """
        if self.mode == "paper":
            buy_tr = "VTTC0802U"
            sell_tr = "VTTC0801U"
        else:
            buy_tr = "TTTC0802U"
            sell_tr = "TTTC0801U"

        tr_id = buy_tr if side == "buy" else sell_tr
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"

        ord_dvsn = "01" if order_type == "market" else "00"  # 시장가: 01, 지정가: 00

        body = {
            "CANO": self.account_no[:8],
            "ACNT_PRDT_CD": self.account_no[8:],
            "PDNO": ticker,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(qty),
            "ORD_UNPR": str(price),
        }

        logger.info(
            "주문 실행: %s %s %s주 @ %s (%s mode)",
            side, ticker, qty, price if price else "시장가", self.mode,
        )

        resp = requests.post(url, headers=self._headers(tr_id), json=body, timeout=10)
        resp.raise_for_status()
        result = resp.json()

        if result.get("rt_cd") != "0":
            logger.error("주문 실패: %s", result.get("msg1"))

        return result
