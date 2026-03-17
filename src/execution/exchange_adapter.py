"""Exchange Adapter — Bybit + ccxt 통합 인터페이스.

설계 원칙:
  - Entry: Post-Only (maker fee only)
  - Exit SL: reduceOnly + closeOnTrigger (체결 보장)
  - Exit TP: reduceOnly conditional (수수료 우선)
  - Single exchange instance 재사용
  - orderLinkId로 idempotency 보장
  - Precision: load_markets() 후 amount/price 자동 반올림
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


# Internal symbol → ccxt symbol mapping (linear perpetual)
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
}


class ExchangeAdapter:
    """Bybit unified interface via ccxt.

    Modes:
        sandbox — ccxt sandbox (set_sandbox_mode)
        demo    — Bybit demo trading (별도 API key/도메인)
        live    — 실거래

    Usage:
        adapter = ExchangeAdapter(mode="demo", api_key="...", secret="...")
        await adapter.initialize()
        result = await adapter.place_post_only_entry("BTC", "BUY", 0.001, 60000, "oid-1")
    """

    def __init__(
        self,
        mode: str = "demo",
        api_key: str = "",
        secret: str = "",
    ):
        if not HAS_CCXT:
            raise ImportError("ccxt is required: pip install ccxt")

        self.mode = mode
        self._exchange: ccxt_async.bybit | None = None
        self._sync_exchange: ccxt_sync.bybit | None = None
        self._markets_loaded = False

        # Exchange config
        config = {
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",        # linear perpetual
                "adjustForTimeDifference": True,
            },
        }

        if mode == "demo":
            # Bybit demo trading uses separate endpoints
            config["options"]["testnet"] = True
            config["urls"] = {
                "api": {
                    "public": "https://api-demo.bybit.com",
                    "private": "https://api-demo.bybit.com",
                },
            }

        self._exchange = ccxt_async.bybit(config)
        self._sync_exchange = ccxt_sync.bybit(config)

        if mode == "sandbox":
            self._exchange.set_sandbox_mode(True)
            self._sync_exchange.set_sandbox_mode(True)

    async def initialize(self) -> None:
        """Load markets (must call once before any order)."""
        await self._exchange.load_markets()
        self._markets_loaded = True
        logger.info(
            f"Exchange initialized ({self.mode}): "
            f"{len(self._exchange.markets)} markets loaded"
        )

    def _ccxt_symbol(self, symbol: str) -> str:
        """Internal symbol → ccxt symbol."""
        return SYMBOL_MAP.get(symbol, f"{symbol}/USDT:USDT")

    # ── Precision ───────────────────────────────────────────

    def round_qty(self, symbol: str, qty: float) -> float:
        """Round quantity to exchange precision."""
        return float(self._exchange.amount_to_precision(
            self._ccxt_symbol(symbol), qty
        ))

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to exchange precision."""
        return float(self._exchange.price_to_precision(
            self._ccxt_symbol(symbol), price
        ))

    def round_all(
        self, symbol: str, qty: float, entry: float, sl: float, tp: float,
    ) -> tuple[float, float, float, float]:
        """Round all values to exchange precision."""
        ccxt_sym = self._ccxt_symbol(symbol)
        return (
            float(self._exchange.amount_to_precision(ccxt_sym, qty)),
            float(self._exchange.price_to_precision(ccxt_sym, entry)),
            float(self._exchange.price_to_precision(ccxt_sym, sl)),
            float(self._exchange.price_to_precision(ccxt_sym, tp)),
        )

    def get_min_qty(self, symbol: str) -> float:
        """Get minimum order quantity."""
        ccxt_sym = self._ccxt_symbol(symbol)
        market = self._exchange.market(ccxt_sym)
        return market.get("limits", {}).get("amount", {}).get("min", 0)

    def get_tick_size(self, symbol: str) -> float:
        """Get price tick size."""
        ccxt_sym = self._ccxt_symbol(symbol)
        market = self._exchange.market(ccxt_sym)
        return market.get("precision", {}).get("price", 0.01)

    # ── Entry Orders (Maker-First) ──────────────────────────

    async def place_post_only_entry(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        order_link_id: str,
    ) -> dict:
        """Post-Only limit entry. 즉시 체결될 상황이면 자동 취소.

        Returns: {success, order_id, exchange_order_id, status, error}
        """
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
                params={
                    "postOnly": True,
                    "orderLinkId": order_link_id,
                    "timeInForce": "PostOnly",
                },
            )
            return {
                "success": True,
                "order_id": order_link_id,
                "exchange_order_id": order.get("id", ""),
                "status": order.get("status", "open"),
                "raw": order,
            }
        except Exception as e:
            logger.error(f"[Entry] {symbol} {side} failed: {e}")
            return {
                "success": False,
                "order_id": order_link_id,
                "error": str(e),
            }

    # ── Protective Stop Loss (Certainty-First) ──────────────

    async def place_protective_stop(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        order_link_id: str,
        parent_id: str = "",
    ) -> dict:
        """reduceOnly + closeOnTrigger stop loss.

        Trigger 시 market order로 실행 → 체결 보장.
        """
        ccxt_sym = self._ccxt_symbol(symbol)
        qty = self.round_qty(symbol, qty)
        stop_price = self.round_price(symbol, stop_price)

        try:
            order = await self._exchange.create_order(
                symbol=ccxt_sym,
                type="market",
                side=side.lower(),
                amount=qty,
                params={
                    "stopPrice": stop_price,
                    "triggerPrice": stop_price,
                    "triggerDirection": 2 if side.upper() == "SELL" else 1,
                    "reduceOnly": True,
                    "closeOnTrigger": True,
                    "orderLinkId": order_link_id,
                },
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
            return {
                "success": False,
                "order_id": order_link_id,
                "error": str(e),
            }

    # ── Take Profit (Separate Conditional) ──────────────────

    async def place_take_profit(
        self,
        symbol: str,
        side: str,
        qty: float,
        tp_price: float,
        order_link_id: str,
        parent_id: str = "",
    ) -> dict:
        """reduceOnly conditional TP. Trigger 시 limit order."""
        ccxt_sym = self._ccxt_symbol(symbol)
        qty = self.round_qty(symbol, qty)
        tp_price = self.round_price(symbol, tp_price)

        try:
            order = await self._exchange.create_order(
                symbol=ccxt_sym,
                type="limit",
                side=side.lower(),
                amount=qty,
                price=tp_price,
                params={
                    "stopPrice": tp_price,
                    "triggerPrice": tp_price,
                    "triggerDirection": 1 if side.upper() == "SELL" else 2,
                    "reduceOnly": True,
                    "orderLinkId": order_link_id,
                },
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
            return {
                "success": False,
                "order_id": order_link_id,
                "error": str(e),
            }

    # ── Market Close (Emergency / Time Stop) ────────────────

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
                    "orderLinkId": order_link_id,
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
            return {
                "success": False,
                "order_id": order_link_id,
                "error": str(e),
            }

    # ── Order Management ────────────────────────────────────

    async def cancel_order(
        self,
        symbol: str,
        order_link_id: str | None = None,
        exchange_order_id: str | None = None,
    ) -> dict:
        """Cancel by orderLinkId or exchange ID."""
        ccxt_sym = self._ccxt_symbol(symbol)
        try:
            if order_link_id:
                result = await self._exchange.cancel_order(
                    id=None,
                    symbol=ccxt_sym,
                    params={"orderLinkId": order_link_id},
                )
            elif exchange_order_id:
                result = await self._exchange.cancel_order(
                    id=exchange_order_id,
                    symbol=ccxt_sym,
                )
            else:
                return {"success": False, "error": "No order ID provided"}

            return {"success": True, "raw": result}
        except Exception as e:
            # OrderNotFound is idempotent (already cancelled or filled)
            if "OrderNotFound" in str(type(e).__name__) or "order not found" in str(e).lower():
                logger.info(f"[Cancel] {symbol} order already gone: {e}")
                return {"success": True, "already_gone": True}
            logger.error(f"[Cancel] {symbol} failed: {e}")
            return {"success": False, "error": str(e)}

    async def wait_fill_or_cancel(
        self,
        symbol: str,
        order_link_id: str,
        ttl_sec: float = 20.0,
        max_reprices: int = 1,
        poll_interval: float = 2.0,
    ) -> dict | None:
        """Wait for fill, reprice once, or cancel and return None."""
        ccxt_sym = self._ccxt_symbol(symbol)
        start = time.time()
        reprices = 0

        while time.time() - start < ttl_sec:
            try:
                orders = await self._exchange.fetch_open_orders(
                    symbol=ccxt_sym,
                    params={"orderLinkId": order_link_id},
                )
                # If no open orders, check if filled
                if not orders:
                    # Check order history
                    order = await self._fetch_order_by_link_id(symbol, order_link_id)
                    if order and order.get("status") == "closed":
                        return {
                            "filled": True,
                            "fill_price": order.get("average", order.get("price", 0)),
                            "fill_qty": order.get("filled", 0),
                            "fee": order.get("fee", {}).get("cost", 0),
                        }
                    # Order was rejected or cancelled externally
                    if order and order.get("status") in ("canceled", "cancelled", "rejected"):
                        return None
                    # May still be processing
                    await asyncio.sleep(poll_interval)
                    continue

                # Order still open — reprice?
                if reprices < max_reprices and time.time() - start > ttl_sec * 0.5:
                    # Could implement reprice logic here
                    # For now, just wait
                    pass

                await asyncio.sleep(poll_interval)

            except Exception as e:
                logger.warning(f"[WaitFill] {symbol} poll error: {e}")
                await asyncio.sleep(poll_interval)

        # Timeout — cancel
        logger.info(f"[WaitFill] {symbol} timeout ({ttl_sec}s), cancelling")
        await self.cancel_order(symbol, order_link_id=order_link_id)
        return None

    # ── Market Data ─────────────────────────────────────────

    async def fetch_ticker(self, symbol: str) -> dict:
        """Get current bid/ask/last."""
        ccxt_sym = self._ccxt_symbol(symbol)
        ticker = await self._exchange.fetch_ticker(ccxt_sym)
        bid = ticker.get("bid", 0)
        ask = ticker.get("ask", 0)
        spread_bps = ((ask - bid) / ((ask + bid) / 2) * 10000) if (bid and ask) else 0
        return {
            "bid": bid,
            "ask": ask,
            "last": ticker.get("last", 0),
            "spread_bps": round(spread_bps, 2),
        }

    async def fetch_balance(self) -> dict:
        """Get USDT equity."""
        balance = await self._exchange.fetch_balance()
        usdt = balance.get("USDT", {})
        return {
            "total": usdt.get("total", 0),
            "free": usdt.get("free", 0),
            "used": usdt.get("used", 0),
        }

    async def fetch_position(self, symbol: str) -> dict | None:
        """Get current position for symbol."""
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
        """Current funding rate."""
        ccxt_sym = self._ccxt_symbol(symbol)
        try:
            funding = await self._exchange.fetch_funding_rate(ccxt_sym)
            return funding.get("fundingRate", 0.0)
        except Exception as e:
            logger.warning(f"[Funding] {symbol} fetch failed: {e}")
            return 0.0

    # ── Internal ────────────────────────────────────────────

    async def _fetch_order_by_link_id(self, symbol: str, order_link_id: str) -> dict | None:
        """Fetch order by orderLinkId from closed orders."""
        ccxt_sym = self._ccxt_symbol(symbol)
        try:
            orders = await self._exchange.fetch_closed_orders(
                symbol=ccxt_sym,
                params={"orderLinkId": order_link_id},
            )
            if orders:
                return orders[0]
        except Exception:
            pass
        return None

    # ── Maker Price ─────────────────────────────────────────

    async def get_maker_entry_price(self, symbol: str, side: str) -> float:
        """Get best maker price (bid for BUY, ask for SELL).

        Post-Only 주문에 적합한 가격. 상대호가 기준.
        """
        ticker = await self.fetch_ticker(symbol)
        if side.upper() == "BUY":
            return ticker["bid"]  # bid에 걸어야 maker
        else:
            return ticker["ask"]  # ask에 걸어야 maker

    # ── ID Generation ───────────────────────────────────────

    @staticmethod
    def make_order_id(
        symbol: str,
        side: str,
        seq: int = 1,
        prefix: str = "wf3",
    ) -> str:
        """Generate orderLinkId (max 36 chars, unique).

        Format: wf3-xrp-0315-1830-b-0001
        """
        now = datetime.now(timezone.utc)
        s = side[0].lower()  # b or s
        return f"{prefix}-{symbol.lower()}-{now.strftime('%m%d-%H%M')}-{s}-{seq:04d}"

    # ── Lifecycle ───────────────────────────────────────────

    async def close(self) -> None:
        if self._exchange:
            await self._exchange.close()

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, *args):
        await self.close()
