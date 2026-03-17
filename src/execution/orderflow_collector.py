"""Real-time Order Flow Collector — Binance Futures Testnet.

바이낸스 서버에서 직접 수집하는 실시간 오더플로우 데이터:
  - Agg Trades    → 실제 CVD (buy/sell volume 정확히 분리)
  - Order Book    → OBI (호가 불균형), Spread, Depth
  - Open Interest → OI 변화
  - Funding Rate  → 펀딩 방향

4h 바 종가 시점에 집계된 통계를 반환해 trading signal에 활용.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger("orderflow")


class OrderFlowCollector:
    """실시간 오더플로우 수집 및 4h 집계.

    Usage:
        collector = OrderFlowCollector(exchange, symbols=["DOT", "ADA"])
        await collector.start()                # 백그라운드 수집 시작
        snap = await collector.get_snapshot("DOT")  # 현재 집계값 반환
        await collector.stop()
    """

    POLL_TRADES_SEC = 10    # agg trades 폴링 주기
    POLL_BOOK_SEC   = 30    # order book 폴링 주기
    POLL_OI_SEC     = 120   # open interest 폴링 주기
    MAX_TRADE_BUFFER = 2000 # 코인당 최대 보관 거래 수

    def __init__(self, exchange, symbols: list[str]):
        self._ex       = exchange
        self.symbols   = symbols
        self._running  = False
        self._tasks: list[asyncio.Task] = []

        # 수집 버퍼 (코인별)
        self._trades: dict[str, deque] = {s: deque(maxlen=self.MAX_TRADE_BUFFER)
                                          for s in symbols}
        self._book_snap: dict[str, dict] = {}
        self._oi_snap: dict[str, dict]   = {}
        self._funding: dict[str, float]  = {}

        # 마지막 처리 agg trade ID (중복 방지)
        self._last_trade_id: dict[str, int] = {s: 0 for s in symbols}

    # ── 공개 인터페이스 ─────────────────────────────────────

    async def start(self):
        """백그라운드 수집 태스크 시작."""
        self._running = True
        for sym in self.symbols:
            self._tasks.append(asyncio.create_task(self._poll_trades(sym)))
            self._tasks.append(asyncio.create_task(self._poll_book(sym)))
        self._tasks.append(asyncio.create_task(self._poll_oi_funding()))
        logger.info(f"[OrderFlow] 수집 시작: {self.symbols}")

    async def stop(self):
        self._running = False
        for t in self._tasks:
            t.cancel()
        self._tasks.clear()

    async def get_snapshot(self, symbol: str,
                           window_sec: int = 14400) -> dict:
        """4h(또는 지정 window) 집계 스냅샷 반환.

        Returns:
            {
                # CVD (진짜 체결 기반)
                "cvd_real"         : float,  # buy_vol - sell_vol 누적
                "buy_vol"          : float,  # 매수 체결량
                "sell_vol"         : float,  # 매도 체결량
                "buy_ratio"        : float,  # buy / total [0~1]
                "trade_count"      : int,
                "buy_count"        : int,
                "avg_trade_size"   : float,
                "large_buy_vol"    : float,  # 상위 10% 거래 매수량
                "large_sell_vol"   : float,

                # Order Book
                "bid_vol_top10"    : float,  # 상위 10레벨 호가 잔량
                "ask_vol_top10"    : float,
                "obi"              : float,  # (bid-ask)/(bid+ask) [-1,+1]
                "spread_bps"       : float,  # 스프레드 (bps)
                "mid_price"        : float,
                "book_depth_ratio" : float,  # bid_depth / ask_depth

                # Open Interest & Funding
                "oi"               : float,
                "funding_rate"     : float,

                # Derived
                "flow_pressure"    : float,  # CVD×OBI 복합 [-1,+1]
                "n_samples"        : int,    # 수집된 거래 수
            }
        """
        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - window_sec * 1000

        trades = [t for t in self._trades.get(symbol, [])
                  if t["T"] >= cutoff_ms]

        snap = self._compute_trade_stats(symbol, trades)
        snap.update(self._compute_book_stats(symbol))
        snap["oi"]           = self._oi_snap.get(symbol, {}).get("oi", 0.0)
        snap["funding_rate"] = self._funding.get(symbol, 0.0)
        snap["flow_pressure"] = self._compute_flow_pressure(snap)
        snap["n_samples"]    = len(trades)
        snap["window_sec"]   = window_sec
        snap["ts"]           = datetime.now(timezone.utc).isoformat()

        return snap

    # ── 폴링 루프 ───────────────────────────────────────────

    async def _poll_trades(self, symbol: str):
        """agg trades 주기적 수집."""
        ccxt_sym = f"{symbol}/USDT:USDT"
        binance_sym = f"{symbol}USDT"

        while self._running:
            try:
                last_id = self._last_trade_id[symbol]
                params = {"symbol": binance_sym, "limit": 500}
                if last_id > 0:
                    params["fromId"] = last_id + 1

                raw = await self._ex.fapiPublicGetAggTrades(params)

                new_count = 0
                for t in raw:
                    trade_id = int(t["a"])
                    if trade_id <= last_id:
                        continue
                    self._trades[symbol].append({
                        "T"  : int(t["T"]),       # timestamp ms
                        "p"  : float(t["p"]),     # price
                        "q"  : float(t["q"]),     # quantity
                        "m"  : t["m"],            # m=True: seller is maker (= sell taker)
                        "id" : trade_id,
                    })
                    self._last_trade_id[symbol] = max(
                        self._last_trade_id[symbol], trade_id)
                    new_count += 1

                if new_count > 0:
                    logger.debug(f"[OrderFlow] {symbol}: +{new_count} trades")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[OrderFlow] {symbol} trades poll error: {e}")

            await asyncio.sleep(self.POLL_TRADES_SEC)

    async def _poll_book(self, symbol: str):
        """order book 스냅샷 수집."""
        ccxt_sym = f"{symbol}/USDT:USDT"

        while self._running:
            try:
                ob = await self._ex.fetch_order_book(ccxt_sym, limit=20)
                self._book_snap[symbol] = {
                    "bids": ob["bids"][:20],
                    "asks": ob["asks"][:20],
                    "ts"  : ob.get("timestamp", int(time.time() * 1000)),
                }
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[OrderFlow] {symbol} book poll error: {e}")

            await asyncio.sleep(self.POLL_BOOK_SEC)

    async def _poll_oi_funding(self):
        """Open Interest + Funding Rate 수집 (전체 코인)."""
        while self._running:
            for symbol in self.symbols:
                ccxt_sym = f"{symbol}/USDT:USDT"
                try:
                    oi = await self._ex.fetch_open_interest(ccxt_sym)
                    self._oi_snap[symbol] = {
                        "oi": float(oi.get("openInterestAmount", 0)),
                        "ts": oi.get("timestamp", 0),
                    }
                except Exception:
                    pass
                try:
                    fr = await self._ex.fetch_funding_rate(ccxt_sym)
                    self._funding[symbol] = float(fr.get("fundingRate", 0))
                except Exception:
                    pass
                await asyncio.sleep(2)  # 코인 간 간격

            if not self._running:
                break
            await asyncio.sleep(self.POLL_OI_SEC)

    # ── 통계 계산 ───────────────────────────────────────────

    def _compute_trade_stats(self, symbol: str, trades: list) -> dict:
        if not trades:
            return {
                "cvd_real": 0.0, "buy_vol": 0.0, "sell_vol": 0.0,
                "buy_ratio": 0.5, "trade_count": 0, "buy_count": 0,
                "avg_trade_size": 0.0, "large_buy_vol": 0.0, "large_sell_vol": 0.0,
            }

        buy_vol  = sum(t["q"] for t in trades if not t["m"])  # m=False = buy taker
        sell_vol = sum(t["q"] for t in trades if t["m"])      # m=True  = sell taker
        total    = buy_vol + sell_vol + 1e-10
        buy_cnt  = sum(1 for t in trades if not t["m"])

        # 대형 거래 (상위 10% 이상)
        sizes = sorted([t["q"] for t in trades], reverse=True)
        top10_thresh = sizes[max(0, len(sizes)//10)] if sizes else 0
        large_buy  = sum(t["q"] for t in trades if not t["m"] and t["q"] >= top10_thresh)
        large_sell = sum(t["q"] for t in trades if t["m"]     and t["q"] >= top10_thresh)

        return {
            "cvd_real"      : round(buy_vol - sell_vol, 4),
            "buy_vol"       : round(buy_vol, 4),
            "sell_vol"      : round(sell_vol, 4),
            "buy_ratio"     : round(buy_vol / total, 4),
            "trade_count"   : len(trades),
            "buy_count"     : buy_cnt,
            "avg_trade_size": round(total / len(trades), 4),
            "large_buy_vol" : round(large_buy, 4),
            "large_sell_vol": round(large_sell, 4),
        }

    def _compute_book_stats(self, symbol: str) -> dict:
        book = self._book_snap.get(symbol)
        if not book or not book["bids"] or not book["asks"]:
            return {
                "bid_vol_top10": 0.0, "ask_vol_top10": 0.0,
                "obi": 0.0, "spread_bps": 0.0,
                "mid_price": 0.0, "book_depth_ratio": 1.0,
            }

        bids = book["bids"][:10]
        asks = book["asks"][:10]

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid      = (best_bid + best_ask) / 2
        spread   = (best_ask - best_bid) / mid * 10000  # bps

        bid_vol = sum(b[1] for b in bids)
        ask_vol = sum(a[1] for a in asks)
        total   = bid_vol + ask_vol + 1e-10
        obi     = (bid_vol - ask_vol) / total  # [-1, +1]

        return {
            "bid_vol_top10" : round(bid_vol, 4),
            "ask_vol_top10" : round(ask_vol, 4),
            "obi"           : round(obi, 4),
            "spread_bps"    : round(spread, 4),
            "mid_price"     : round(mid, 6),
            "book_depth_ratio": round(bid_vol / (ask_vol + 1e-10), 4),
        }

    def _compute_flow_pressure(self, snap: dict) -> float:
        """CVD와 OBI 결합 복합 오더플로우 압력 [-1, +1]."""
        buy_ratio = snap.get("buy_ratio", 0.5)
        obi       = snap.get("obi", 0.0)
        # buy_ratio: 0.5 기준 정규화 → [-1, +1]
        cvd_norm  = (buy_ratio - 0.5) * 2
        combined  = cvd_norm * 0.6 + obi * 0.4
        return round(float(np.clip(combined, -1.0, 1.0)), 4)
