"""Exchange Adapter — Binance USDT-M Futures ccxt 인터페이스.

설계 원칙:
  - Entry: Post-Only (maker fee only, GTX TIF)
  - Exit SL: closePosition=True STOP_MARKET → Position 탭 TP/SL 컬럼에 표시
  - Exit TP: closePosition=True TAKE_PROFIT_MARKET → Position 탭 TP/SL 컬럼에 표시
  - Single exchange instance 재사용
  - newClientOrderId로 idempotency 보장
  - Precision: load_markets() 후 amount/price 자동 반올림

거래소:
  - binance: USDT-M Futures (binanceusdm)
      sandbox/demo → testnet.binancefuture.com
      live         → fapi.binance.com
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

try:
    import ccxt.async_support as ccxt_async
    import ccxt as ccxt_sync
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

logger = logging.getLogger(__name__)


# Internal symbol → ccxt symbol mapping (linear perpetual / USDT-M futures)
SYMBOL_MAP = {
    "BTC": "BTC/USDT:USDT",
    "ETH": "ETH/USDT:USDT",
    "SOL": "SOL/USDT:USDT",
    "XRP": "XRP/USDT:USDT",
    "ADA": "ADA/USDT:USDT",
    "DOGE": "DOGE/USDT:USDT",
    "AVAX": "AVAX/USDT:USDT",
    "DOT": "DOT/USDT:USDT",
    "LINK": "LINK/USDT:USDT",
    "BNB": "BNB/USDT:USDT",
    "TAO": "TAO/USDT:USDT",
}

class ExchangeAdapter:
    """Binance USDT-M Futures 인터페이스 via ccxt.

    Modes:
        sandbox / demo — testnet.binancefuture.com (가상 자금)
        live           — fapi.binance.com (실거래)

    Usage:
        adapter = ExchangeAdapter(mode="sandbox", api_key="...", secret="...")
        await adapter.initialize()
        result = await adapter.place_post_only_entry("DOT", "BUY", 10, 8.5, "oid-1")
    """

    def __init__(
        self,
        mode: str = "sandbox",
        api_key: str = "",
        secret: str = "",
        exchange: str = "binance",  # 하위 호환용, 무시됨
    ):
        if not HAS_CCXT:
            raise ImportError("ccxt is required: pip install ccxt")

        self.exchange_name = "binance"
        self.mode = mode
        self._exchange: object | None = None
        self._sync_exchange: object | None = None
        self._markets_loaded = False

        self._exchange, self._sync_exchange = self._build_binance(mode, api_key, secret)

    # ── Factory ─────────────────────────────────────────────

    def _build_binance(self, mode: str, api_key: str, secret: str):
        """Binance USDT-M Futures (binanceusdm).

        demo: demo.binance.com — 실제 fapi.binance.com 엔드포인트 사용 (가상 자금)
        sandbox: testnet.binancefuture.com (개발자 테스트넷)
        """
        # Binance USDT-M Futures testnet base URL
        TESTNET_BASE = "https://testnet.binancefuture.com"

        config = {
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
                "fetchCurrencies": False,
            },
        }

        # demo.binance.com은 실제 fapi 엔드포인트를 사용 — URL 오버라이드 없음
        # sandbox만 testnet 엔드포인트로 강제 전환
        if mode == "sandbox":
            config["urls"] = {
                "api": {
                    "fapiPublic":         f"{TESTNET_BASE}/fapi/v1",
                    "fapiPublicV2":       f"{TESTNET_BASE}/fapi/v2",
                    "fapiPrivate":        f"{TESTNET_BASE}/fapi/v1",
                    "fapiPrivateV2":      f"{TESTNET_BASE}/fapi/v2",
                    "fapiPrivateV3":      f"{TESTNET_BASE}/fapi/v3",
                    "public":             f"{TESTNET_BASE}/fapi/v1",
                    "private":            f"{TESTNET_BASE}/fapi/v1",
                }
            }

        async_ex = ccxt_async.binanceusdm(config)
        sync_ex = ccxt_sync.binanceusdm(config)
        try:
            sync_ex.load_time_difference()
        except Exception:
            pass
        return async_ex, sync_ex

    # ── Order param helpers (Binance USDT-M) ────────────────

    def _client_id_key(self) -> str:
        return "newClientOrderId"

    def _fetch_by_client_id_key(self) -> str:
        return "origClientOrderId"

    def _post_only_params(self, order_id: str) -> dict:
        """Post-Only 진입 파라미터. ccxt가 postOnly=True → GTX TIF 자동 변환."""
        return {
            "postOnly": True,
            "newClientOrderId": order_id,
        }

    def _stop_loss_params(self, stop_price: float, side: str, order_id: str) -> tuple[str, dict]:
        """STOP_MARKET + closePosition=True → Binance Position 탭 TP/SL 컬럼에 표시."""
        return "stop_market", {
            "stopPrice": stop_price,
            "closePosition": True,   # 포지션 전체 청산 (qty 불필요, Position 탭 표시)
            "newClientOrderId": order_id,
        }

    def _take_profit_params(self, tp_price: float, side: str, order_id: str) -> tuple[str, dict]:
        """TAKE_PROFIT_MARKET + closePosition=True → Binance Position 탭 TP/SL 컬럼에 표시."""
        return "take_profit_market", {
            "stopPrice": tp_price,
            "closePosition": True,   # 포지션 전체 청산 (qty 불필요, Position 탭 표시)
            "newClientOrderId": order_id,
        }

    # ── Public access to underlying ccxt instance ────────────

    @property
    def async_exchange(self):
        """Expose async ccxt instance for direct API calls (e.g., data_hub)."""
        return self._exchange

    # ── Lifecycle ───────────────────────────────────────────

    async def initialize(self) -> None:
        """Load markets (must call once before any order)."""
        await self._exchange.load_time_difference()
        await self._exchange.load_markets()
        self._markets_loaded = True
        logger.info(
            f"Exchange initialized ({self.exchange_name}/{self.mode}): "
            f"{len(self._exchange.markets)} markets loaded"
        )

    def _ccxt_symbol(self, symbol: str) -> str:
        return SYMBOL_MAP.get(symbol, f"{symbol}/USDT:USDT")

    # ── Precision ───────────────────────────────────────────

    def round_qty(self, symbol: str, qty: float) -> float:
        return float(self._exchange.amount_to_precision(
            self._ccxt_symbol(symbol), qty
        ))

    def round_price(self, symbol: str, price: float) -> float:
        return float(self._exchange.price_to_precision(
            self._ccxt_symbol(symbol), price
        ))

    def round_all(
        self, symbol: str, qty: float, entry: float, sl: float, tp: float,
    ) -> tuple[float, float, float, float]:
        ccxt_sym = self._ccxt_symbol(symbol)
        return (
            float(self._exchange.amount_to_precision(ccxt_sym, qty)),
            float(self._exchange.price_to_precision(ccxt_sym, entry)),
            float(self._exchange.price_to_precision(ccxt_sym, sl)),
            float(self._exchange.price_to_precision(ccxt_sym, tp)),
        )

    def get_min_qty(self, symbol: str) -> float:
        ccxt_sym = self._ccxt_symbol(symbol)
        market = self._exchange.market(ccxt_sym)
        limits = market.get("limits") or {}
        amount_limits = limits.get("amount") or {}
        return amount_limits.get("min", 0) or 0

    def get_tick_size(self, symbol: str) -> float:
        ccxt_sym = self._ccxt_symbol(symbol)
        market = self._exchange.market(ccxt_sym)
        return market.get("precision", {}).get("price", 0.01)

    # ── Leverage ────────────────────────────────────────────

    def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set isolated margin mode and leverage for a symbol (sync). Best-effort."""
        ccxt_sym = self._ccxt_symbol(symbol)
        try:
            self._sync_exchange.set_margin_mode("isolated", ccxt_sym)
        except Exception:
            pass  # already isolated or unsupported — non-fatal
        try:
            self._sync_exchange.set_leverage(leverage, ccxt_sym)
            logger.info(f"[Leverage] {symbol} set to {leverage}x (isolated)")
        except Exception as e:
            logger.debug(f"[Leverage] {symbol} set_leverage failed: {e}")

    async def set_leverage_async(self, symbol: str, leverage: int) -> None:
        """Async wrapper for set_leverage — avoids blocking the event loop."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.set_leverage, symbol, leverage)

    # ── Entry Orders (Maker-First) ──────────────────────────

    async def place_post_only_entry(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        order_link_id: str,
    ) -> dict:
        """Post-Only limit entry. 즉시 체결될 상황이면 자동 취소."""
        ccxt_sym = self._ccxt_symbol(symbol)
        qty = self.round_qty(symbol, qty)
        price = self.round_price(symbol, price)

        try:
            order = await self._exchange.create_order(
                symbol=ccxt_sym,
                type="limit",
                side=side.lower(),
                amount=qty,
                price=price,
                params=self._post_only_params(order_link_id),
            )
            return {
                "success": True,
                "order_id": order_link_id,
                "exchange_order_id": order.get("id", ""),
                "status": order.get("status", "open"),
                "raw": order,
            }
        except Exception as e:
            # -5022: Post-Only 거부 → 일반 limit 주문으로 폴백
            if "-5022" in str(e):
                logger.warning(f"[Entry] {symbol} Post-Only FAILED, fallback to limit: {e}")
                try:
                    fallback_params = {"newClientOrderId": order_link_id}
                    order = await self._exchange.create_order(
                        symbol=ccxt_sym,
                        type="limit",
                        side=side.lower(),
                        amount=qty,
                        price=price,
                        params=fallback_params,
                    )
                    return {
                        "success": True,
                        "order_id": order_link_id,
                        "exchange_order_id": order.get("id", ""),
                        "status": order.get("status", "open"),
                        "raw": order,
                    }
                except Exception as e2:
                    logger.error(f"[Entry] {symbol} {side} limit fallback failed: {e2}")
                    return {"success": False, "order_id": order_link_id, "error": str(e2)}
            logger.error(f"[Entry] {symbol} {side} failed: {e}")
            return {"success": False, "order_id": order_link_id, "error": str(e)}

    # ── Protective Stop Loss ─────────────────────────────────

    async def place_protective_stop(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        order_link_id: str,
        parent_id: str = "",
    ) -> dict:
        """reduceOnly stop loss. Trigger 시 market 체결 보장."""
        ccxt_sym = self._ccxt_symbol(symbol)
        qty = self.round_qty(symbol, qty)
        stop_price = self.round_price(symbol, stop_price)

        order_type, params = self._stop_loss_params(stop_price, side, order_link_id)

        try:
            order = await self._exchange.create_order(
                symbol=ccxt_sym,
                type=order_type,
                side=side.lower(),
                amount=0,   # closePosition=True: 수량 불필요 (포지션 전체 청산)
                params=params,
            )
            return {
                "success": True,
                "order_id": order_link_id,
                "exchange_order_id": order.get("id", ""),
                "status": "conditional",
                "raw": order,
            }
        except Exception as e:
            logger.error(f"[SL] {symbol} {side} @ {stop_price} failed: {e}")
            return {"success": False, "order_id": order_link_id, "error": str(e)}

    # ── Take Profit ──────────────────────────────────────────

    async def place_take_profit(
        self,
        symbol: str,
        side: str,
        qty: float,
        tp_price: float,
        order_link_id: str,
        parent_id: str = "",
    ) -> dict:
        """reduceOnly take-profit."""
        ccxt_sym = self._ccxt_symbol(symbol)
        qty = self.round_qty(symbol, qty)
        tp_price = self.round_price(symbol, tp_price)

        order_type, params = self._take_profit_params(tp_price, side, order_link_id)

        try:
            order = await self._exchange.create_order(
                symbol=ccxt_sym,
                type=order_type,
                side=side.lower(),
                amount=0,   # closePosition=True: 수량 불필요 (포지션 전체 청산)
                params=params,
            )
            return {
                "success": True,
                "order_id": order_link_id,
                "exchange_order_id": order.get("id", ""),
                "status": "conditional",
                "raw": order,
            }
        except Exception as e:
            logger.error(f"[TP] {symbol} {side} @ {tp_price} failed: {e}")
            return {"success": False, "order_id": order_link_id, "error": str(e)}

    # ── Market Close ─────────────────────────────────────────

    async def market_close(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_link_id: str,
    ) -> dict:
        """reduceOnly market close. 즉시 체결."""
        ccxt_sym = self._ccxt_symbol(symbol)
        qty = self.round_qty(symbol, qty)

        try:
            order = await self._exchange.create_order(
                symbol=ccxt_sym,
                type="market",
                side=side.lower(),
                amount=qty,
                params={
                    "reduceOnly": True,
                    self._client_id_key(): order_link_id,
                },
            )
            return {
                "success": True,
                "order_id": order_link_id,
                "exchange_order_id": order.get("id", ""),
                "status": order.get("status", "closed"),
                "fill_price": order.get("average", 0),
                "raw": order,
            }
        except Exception as e:
            logger.error(f"[MKT_CLOSE] {symbol} {side} failed: {e}")
            return {"success": False, "order_id": order_link_id, "error": str(e)}

    # ── Order Management ─────────────────────────────────────

    async def cancel_order(
        self,
        symbol: str,
        order_link_id: str | None = None,
        exchange_order_id: str | None = None,
    ) -> dict:
        """주문 취소. 일반 주문 → Algo 조건부 주문 순으로 시도.

        Binance Futures Testnet은 SL/TP를 Algo 조건부 주문으로 처리하므로
        일반 cancel_order 실패 시 fapiPrivateDeleteAlgoFuturesOrder 로 재시도.
        """
        ccxt_sym = self._ccxt_symbol(symbol)
        mkt_id = self._exchange.market(ccxt_sym)["id"]  # e.g. "DOTUSDT"

        # ── 시도 1: 일반 주문 취소 (origClientOrderId) ────────
        if order_link_id:
            try:
                result = await self._exchange.cancel_order(
                    id=order_link_id,
                    symbol=ccxt_sym,
                    params={"origClientOrderId": order_link_id},
                )
                return {"success": True, "raw": result}
            except Exception as e1:
                logger.debug(f"[Cancel] regular cancel failed ({e1}), trying algo...")

        # ── 시도 2: exchange_order_id로 일반 취소 ─────────────
        if exchange_order_id:
            try:
                result = await self._exchange.cancel_order(
                    id=exchange_order_id,
                    symbol=ccxt_sym,
                )
                return {"success": True, "raw": result}
            except Exception as e2:
                logger.debug(f"[Cancel] orderId cancel failed ({e2}), trying algo...")

        # ── 시도 3: Algo 조건부 주문 취소 (fapiPrivateDeleteAlgoOrder) ──
        if exchange_order_id:
            try:
                result = await self._exchange.fapiPrivateDeleteAlgoOrder({
                    "symbol": mkt_id,
                    "algoId": int(exchange_order_id),
                })
                logger.info(f"[Cancel] {symbol} algo order cancelled: {exchange_order_id}")
                return {"success": True, "raw": result}
            except Exception as e3:
                err_str = str(e3).lower()
                if "not found" in err_str or "unknown order" in err_str or "-2011" in err_str:
                    logger.info(f"[Cancel] {symbol} order already gone")
                    return {"success": True, "already_gone": True}
                logger.error(f"[Cancel] {symbol} algo cancel failed: {e3}")
                return {"success": False, "error": str(e3)}

        return {"success": False, "error": "No order ID provided"}

    async def wait_fill_or_cancel(
        self,
        symbol: str,
        order_link_id: str,
        ttl_sec: float = 20.0,
        max_reprices: int = 1,
        poll_interval: float = 2.0,
    ) -> dict | None:
        """Wait for fill, or cancel and return None."""
        ccxt_sym = self._ccxt_symbol(symbol)
        start = time.time()

        while time.time() - start < ttl_sec:
            try:
                # 단건 직접 조회 (fetch_open_orders는 clientOrderId 필터 미지원)
                order = await self._fetch_order_by_link_id(symbol, order_link_id)
                if order:
                    if order.get("status") == "closed":
                        return {
                            "filled": True,
                            "fill_price": order.get("average", order.get("price", 0)),
                            "fill_qty": order.get("filled", 0),
                            "fee": (order.get("fee") or {}).get("cost", 0),
                        }
                    if order.get("status") in ("canceled", "cancelled", "rejected"):
                        return None
                await asyncio.sleep(poll_interval)

            except Exception as e:
                err_str = str(e)
                if "-1021" in err_str or "ahead of the server" in err_str or "Timestamp" in err_str:
                    logger.warning(f"[WaitFill] {symbol} time drift detected, resyncing...")
                    try:
                        await self._exchange.load_time_difference()
                    except Exception:
                        pass
                else:
                    logger.warning(f"[WaitFill] {symbol} poll error: {e}")
                await asyncio.sleep(poll_interval)

        logger.info(f"[WaitFill] {symbol} timeout ({ttl_sec}s), cancelling")
        await self.cancel_order(symbol, order_link_id=order_link_id)
        return None

    # ── Market Data ──────────────────────────────────────────

    async def fetch_ticker(self, symbol: str) -> dict:
        ccxt_sym = self._ccxt_symbol(symbol)
        ticker = await self._exchange.fetch_ticker(ccxt_sym)
        bid = ticker.get("bid") or 0
        ask = ticker.get("ask") or 0
        mid = (ask + bid) / 2
        spread_bps = ((ask - bid) / mid * 10000) if mid > 1e-10 else 0
        return {
            "bid": bid,
            "ask": ask,
            "last": ticker.get("last", 0),
            "spread_bps": round(spread_bps, 2),
        }

    async def fetch_balance(self) -> dict:
        balance = await self._exchange.fetch_balance()
        usdt = balance.get("USDT", {})
        return {
            "total": usdt.get("total", 0),
            "free": usdt.get("free", 0),
            "used": usdt.get("used", 0),
        }

    async def fetch_position(self, symbol: str) -> dict | None:
        ccxt_sym = self._ccxt_symbol(symbol)
        try:
            positions = await self._exchange.fetch_positions([ccxt_sym])
            for pos in positions:
                if pos.get("contracts", 0) > 0:
                    return {
                        "symbol": symbol,
                        "side": pos.get("side", ""),
                        "qty": pos.get("contracts", 0),
                        "entry_price": pos.get("entryPrice", 0),
                        "unrealized_pnl": pos.get("unrealizedPnl", 0),
                        "liquidation_price": pos.get("liquidationPrice", 0),
                    }
        except Exception as e:
            logger.error(f"[Position] {symbol} fetch failed: {e}")
        return None

    async def fetch_funding_rate(self, symbol: str) -> float:
        ccxt_sym = self._ccxt_symbol(symbol)
        try:
            funding = await self._exchange.fetch_funding_rate(ccxt_sym)
            return funding.get("fundingRate", 0.0)
        except Exception as e:
            logger.warning(f"[Funding] {symbol} fetch failed: {e}")
            return 0.0

    # ── Internal ─────────────────────────────────────────────

    async def _fetch_order_by_link_id(self, symbol: str, order_link_id: str) -> dict | None:
        ccxt_sym = self._ccxt_symbol(symbol)
        try:
            orders = await self._exchange.fetch_closed_orders(
                symbol=ccxt_sym,
                params={self._fetch_by_client_id_key(): order_link_id},
            )
            for order in (orders or []):
                # Binance가 필터를 무시하는 경우를 대비해 clientOrderId 직접 검증
                info = order.get("info", {})
                cid = info.get("clientOrderId") or info.get("origClientOrderId", "")
                if cid == order_link_id:
                    return order
        except Exception:
            pass
        return None

    # ── Maker Price ──────────────────────────────────────────

    async def get_maker_entry_price(self, symbol: str, side: str) -> float:
        """Return maker entry price (bid for BUY, ask for SELL).
        Falls back to last price if bid/ask is 0 (common on testnet).
        """
        ticker = await self.fetch_ticker(symbol)
        last = ticker.get("last", 0) or 0
        if side.upper() == "BUY":
            price = ticker["bid"] or last
        else:
            price = ticker["ask"] or last
        if price <= 0:
            raise ValueError(f"[{symbol}] maker price is 0 (bid={ticker['bid']}, ask={ticker['ask']}, last={last})")
        return float(price)

    # ── ID Generation ────────────────────────────────────────

    @staticmethod
    def make_order_id(
        symbol: str,
        side: str,
        seq: int = 1,
        prefix: str = "wf3",
    ) -> str:
        """Generate client order ID (max 36 chars, unique).

        Format: wf3-dot-0315-1830-b-0001
        """
        now = datetime.now(timezone.utc)
        s = side[0].lower()
        return f"{prefix}-{symbol.lower()}-{now.strftime('%m%d-%H%M')}-{s}-{seq:04d}"

    # ── Lifecycle ────────────────────────────────────────────

    async def close(self) -> None:
        if self._exchange:
            await self._exchange.close()

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, *args):
        await self.close()
