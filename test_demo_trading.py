"""Binance Futures Testnet 데모 트레이딩 테스트.

전체 흐름:
  1. 잔고 확인
  2. DOT 현재가 조회
  3. 시장가 매수 (소량 — 1 DOT)
  4. 포지션 확인
  5. SL 주문 등록
  6. TP 주문 등록
  7. 오픈 주문 확인
  8. 포지션 시장가 청산 (SL/TP 취소)
  9. 최종 잔고 확인
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.execution.exchange_adapter import ExchangeAdapter

API_KEY    = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
SYMBOL     = "DOT"
QTY        = 1.0   # 1 DOT (최소 단위)

def sep(title=""):
    print(f"\n{'─'*50}")
    if title:
        print(f"  {title}")
        print(f"{'─'*50}")

async def main():
    print("=" * 50)
    print("  Binance Futures Testnet 데모 트레이딩 테스트")
    print("=" * 50)

    async with ExchangeAdapter(mode="sandbox", api_key=API_KEY, secret=API_SECRET) as adapter:

        # ── 1. 잔고 ──────────────────────────────────────────
        sep("1. 잔고 확인")
        bal = await adapter.fetch_balance()
        print(f"  USDT total : {bal['total']:.2f}")
        print(f"  USDT free  : {bal['free']:.2f}")
        print(f"  USDT used  : {bal['used']:.2f}")

        # ── 2. 현재가 ─────────────────────────────────────────
        sep(f"2. {SYMBOL} 현재가 조회")
        ticker = await adapter.fetch_ticker(SYMBOL)
        last_price = ticker["last"]
        print(f"  last  : {last_price}")
        print(f"  bid   : {ticker['bid']}  (testnet = 항상 None)")
        print(f"  ask   : {ticker['ask']}  (testnet = 항상 None)")

        if not last_price:
            print("  [WARN] last price도 None — testnet 시세 없음, 중단")
            return

        # 최소 노셔널 $5 충족하는 수량 자동 계산
        min_notional = 5.0
        qty_needed = max(QTY, min_notional / last_price * 1.1)  # 10% 여유

        # ── 3. 시장가 매수 ────────────────────────────────────
        sep(f"3. {SYMBOL} 시장가 매수 (최소 노셔널 ${min_notional} 충족)")
        entry_id = ExchangeAdapter.make_order_id(SYMBOL, "BUY", prefix="demo")
        qty = adapter.round_qty(SYMBOL, qty_needed)
        print(f"  order_id : {entry_id}")
        print(f"  qty      : {qty}")

        entry_result = await adapter._exchange.create_order(
            symbol=adapter._ccxt_symbol(SYMBOL),
            type="market",
            side="buy",
            amount=qty,
            params={"newClientOrderId": entry_id},
        )
        fill_price = entry_result.get("average") or entry_result.get("price") or last_price
        status     = entry_result.get("status", "?")
        exch_id    = entry_result.get("id", "?")
        print(f"  [OK] status={status}, exchange_id={exch_id}, fill≈{fill_price}")

        # ── 4. 포지션 확인 ────────────────────────────────────
        sep(f"4. 포지션 확인")
        await asyncio.sleep(1)
        pos = await adapter.fetch_position(SYMBOL)
        if pos:
            print(f"  side           : {pos['side']}")
            print(f"  qty            : {pos['qty']}")
            print(f"  entry_price    : {pos['entry_price']}")
            print(f"  unrealized_pnl : {pos['unrealized_pnl']:.4f} USDT")
        else:
            print("  [WARN] 포지션 미확인 — testnet 지연 가능")

        # ── 5. SL 주문 ────────────────────────────────────────
        sep("5. Stop-Loss 주문 등록")
        sl_price = adapter.round_price(SYMBOL, fill_price * 0.95)   # -5%
        sl_id    = ExchangeAdapter.make_order_id(SYMBOL, "SELL", seq=2, prefix="demo")
        print(f"  sl_price : {sl_price}  (entry × 0.95)")
        sl_result = await adapter.place_protective_stop(
            symbol=SYMBOL, side="SELL", qty=qty,
            stop_price=sl_price, order_link_id=sl_id,
        )
        sl_exch_id = sl_result.get("exchange_order_id", "")
        print(f"  [{'OK' if sl_result['success'] else 'FAIL'}] exchange_id={sl_exch_id}")

        # ── 6. TP 주문 ────────────────────────────────────────
        sep("6. Take-Profit 주문 등록")
        tp_price = adapter.round_price(SYMBOL, fill_price * 1.05)   # +5%
        tp_id    = ExchangeAdapter.make_order_id(SYMBOL, "SELL", seq=3, prefix="demo")
        print(f"  tp_price : {tp_price}  (entry × 1.05)")
        tp_result = await adapter.place_take_profit(
            symbol=SYMBOL, side="SELL", qty=qty,
            tp_price=tp_price, order_link_id=tp_id,
        )
        tp_exch_id = tp_result.get("exchange_order_id", "")
        print(f"  [{'OK' if tp_result['success'] else 'FAIL'}] exchange_id={tp_exch_id}")

        # ── 7. 오픈 주문 목록 ─────────────────────────────────
        sep("7. 오픈 주문 목록")
        open_orders = await adapter._exchange.fetch_open_orders(
            symbol=adapter._ccxt_symbol(SYMBOL)
        )
        if open_orders:
            for o in open_orders:
                print(f"  id={o['id']}  type={o['type']}  side={o['side']}  "
                      f"stopPrice={o.get('stopPrice') or o.get('triggerPrice')}  status={o['status']}")
        else:
            print("  (오픈 주문 없음)")

        # ── 8. SL/TP 취소 + 시장가 청산 ──────────────────────
        sep("8. SL/TP 취소 후 시장가 청산")
        for oid, exch_id, label in [
            (sl_id, sl_exch_id, "SL"),
            (tp_id, tp_exch_id, "TP"),
        ]:
            r = await adapter.cancel_order(
                SYMBOL,
                order_link_id=oid,
                exchange_order_id=exch_id,
            )
            print(f"  Cancel {label}: {'OK' if r['success'] else 'FAIL'}"
                  + (f" (already_gone)" if r.get("already_gone") else "")
                  + (f" ERROR: {r.get('error')}" if not r['success'] else ""))

        close_id = ExchangeAdapter.make_order_id(SYMBOL, "SELL", seq=9, prefix="demo")
        close_result = await adapter.market_close(
            symbol=SYMBOL, side="SELL", qty=qty, order_link_id=close_id,
        )
        close_price = close_result.get("fill_price", "?")
        print(f"  [{'OK' if close_result['success'] else 'FAIL'}] 청산 fill≈{close_price}")

        # ── 9. 최종 잔고 ──────────────────────────────────────
        sep("9. 최종 잔고")
        await asyncio.sleep(1)
        bal2 = await adapter.fetch_balance()
        pnl  = bal2["total"] - bal["total"]
        print(f"  USDT total : {bal2['total']:.2f}  (변동: {pnl:+.4f})")
        print(f"  USDT free  : {bal2['free']:.2f}")

    print("\n" + "=" * 50)
    print("  데모 트레이딩 테스트 완료")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
