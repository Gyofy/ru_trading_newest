"""Paper Trader 기본 동작 검증."""
import sys
sys.path.insert(0, "C:/Users/RJ/Desktop/CLAUDE_CRYPTO_AGENT")

from datetime import datetime, timezone, timedelta
from src.signals.contract import Signal, Action, Regime
from src.execution.paper_trader import PaperTrader, INITIAL_CAPITAL

def test_basic_flow():
    """진입 → TP 청산 → stats 확인."""
    trader = PaperTrader(initial_capital=1_000_000, fixed_size_pct=0.02)
    t0 = datetime(2026, 3, 14, 0, 0, tzinfo=timezone.utc)

    # Signal 생성
    sig = Signal(
        symbol="BTC", action=Action.LONG, size=0.02,
        take_profit=3.0, stop_loss=1.5, ttl_bars=6,
        confidence=0.6, p_up=0.7, pred_return=2.0,
    )

    prices = {"BTC": 100000.0, "DOT": 5.0, "DOGE": 0.1, "XRP": 0.5, "ADA": 0.3}

    # 진입
    ok = trader.on_signal(sig, prices, now=t0)
    assert ok, "Should enter position"
    assert "BTC" in trader.positions
    assert len(trader.positions) == 1

    # 동일 코인 재진입 불가
    ok2 = trader.on_signal(sig, prices, now=t0)
    assert not ok2, "Should reject duplicate position"

    # bar 1~3: 가격 상승 중 (TP 미도달)
    for i in range(1, 4):
        t = t0 + timedelta(hours=4*i)
        prices_mid = {**prices, "BTC": 100000 + i * 500}
        closed = trader.on_bar(prices_mid, now=t)
        assert len(closed) == 0

    # bar 4: TP 도달 (103000 >= 103000)
    t4 = t0 + timedelta(hours=16)
    prices_tp = {**prices, "BTC": 103100.0}
    closed = trader.on_bar(prices_tp, now=t4)
    assert len(closed) == 1
    assert closed[0].exit_reason == "TP"
    assert closed[0].pnl_pct > 0
    assert "BTC" not in trader.positions

    # Stats
    snap = trader.snapshot(prices_tp, now=t4)
    stats = trader.stats()
    assert stats["total_trades"] == 1
    assert stats["wins"] == 1
    assert stats["win_rate"] == 1.0
    print(f"[PASS] TP flow: PnL={closed[0].pnl_pct:.4%}, equity={snap.equity:,.0f}")

def test_stop_loss():
    """SL 청산."""
    trader = PaperTrader(initial_capital=1_000_000, fixed_size_pct=0.02)
    t0 = datetime(2026, 3, 14, 0, 0, tzinfo=timezone.utc)

    sig = Signal(
        symbol="DOT", action=Action.LONG, size=0.02,
        take_profit=3.0, stop_loss=1.5, ttl_bars=6,
        confidence=0.6, p_up=0.7, pred_return=2.0,
    )
    prices = {"BTC": 100000, "DOT": 5.0, "DOGE": 0.1, "XRP": 0.5, "ADA": 0.3}
    trader.on_signal(sig, prices, now=t0)

    # 가격 하락 → SL 도달
    prices_sl = {**prices, "DOT": 4.90}  # -2% < -1.5% SL
    closed = trader.on_bar(prices_sl, now=t0 + timedelta(hours=4))
    assert len(closed) == 1
    assert closed[0].exit_reason == "SL"
    assert closed[0].pnl_pct < 0
    print(f"[PASS] SL flow: PnL={closed[0].pnl_pct:.4%}")

def test_ttl_expiry():
    """TTL 만료 청산."""
    trader = PaperTrader(initial_capital=1_000_000, fixed_size_pct=0.02)
    t0 = datetime(2026, 3, 14, 0, 0, tzinfo=timezone.utc)

    sig = Signal(
        symbol="DOGE", action=Action.SHORT, size=0.02,
        take_profit=3.0, stop_loss=1.5, ttl_bars=3,
        confidence=0.6, p_up=0.3, pred_return=-1.5,
    )
    prices = {"BTC": 100000, "DOT": 5.0, "DOGE": 0.10, "XRP": 0.5, "ADA": 0.3}
    trader.on_signal(sig, prices, now=t0)

    # 3 bars, 가격 횡보
    for i in range(1, 3):
        closed = trader.on_bar(prices, now=t0 + timedelta(hours=4*i))
        assert len(closed) == 0

    # bar 3: TTL 만료
    closed = trader.on_bar(prices, now=t0 + timedelta(hours=12))
    assert len(closed) == 1
    assert closed[0].exit_reason == "TTL"
    print(f"[PASS] TTL flow: PnL={closed[0].pnl_pct:.4%}")

def test_exposure_limit():
    """총 노출 80% 제한."""
    trader = PaperTrader(initial_capital=1_000_000, fixed_size_pct=0.20)
    t0 = datetime(2026, 3, 14, 0, 0, tzinfo=timezone.utc)
    prices = {"BTC": 100000, "DOT": 5.0, "DOGE": 0.10, "XRP": 0.5, "ADA": 0.3}

    # fixed_size=20%이면 max_per_coin=5%로 제한됨
    # 5개 코인 × 5% = 25% — 모두 진입 가능
    for sym in ["BTC", "DOT", "DOGE", "XRP", "ADA"]:
        sig = Signal(
            symbol=sym, action=Action.LONG, size=0.05,
            take_profit=3.0, stop_loss=1.5, ttl_bars=6,
            confidence=0.6, p_up=0.7, pred_return=2.0,
        )
        trader.on_signal(sig, prices, now=t0)

    assert len(trader.positions) == 5
    exp = trader._total_exposure_pct()
    assert exp <= 0.80 + 0.001
    print(f"[PASS] Exposure limit: {exp:.1%} with {len(trader.positions)} positions")

def test_non_universe_rejected():
    """유니버스 외 코인 거부."""
    trader = PaperTrader(initial_capital=1_000_000)
    sig = Signal(
        symbol="ETH", action=Action.LONG, size=0.02,
        take_profit=3.0, stop_loss=1.5, ttl_bars=6,
        confidence=0.6, p_up=0.7, pred_return=2.0,
    )
    prices = {"ETH": 3000.0}
    ok = trader.on_signal(sig, prices)
    assert not ok, "ETH not in active universe, should reject"
    print("[PASS] Non-universe coin rejected")

def test_report_save():
    """리포트 저장."""
    import tempfile, os, json
    trader = PaperTrader(initial_capital=1_000_000, fixed_size_pct=0.02)
    t0 = datetime(2026, 3, 14, 0, 0, tzinfo=timezone.utc)
    prices = {"BTC": 100000, "DOT": 5.0, "DOGE": 0.10, "XRP": 0.5, "ADA": 0.3}

    sig = Signal(
        symbol="BTC", action=Action.LONG, size=0.02,
        take_profit=3.0, stop_loss=1.5, ttl_bars=2,
        confidence=0.6, p_up=0.7, pred_return=2.0,
    )
    trader.on_signal(sig, prices, now=t0)
    trader.snapshot(prices, now=t0)
    trader.on_bar({**prices, "BTC": 101000}, now=t0 + timedelta(hours=4))
    trader.on_bar({**prices, "BTC": 101500}, now=t0 + timedelta(hours=8))
    trader.snapshot({**prices, "BTC": 101500}, now=t0 + timedelta(hours=8))

    path = os.path.join(tempfile.gettempdir(), "paper_test_report.json")
    trader.save_report(path)
    with open(path) as f:
        report = json.load(f)
    assert report["stats"]["total_trades"] == 1
    assert len(report["equity_curve"]) == 2
    print(f"[PASS] Report saved: {path}")

if __name__ == "__main__":
    test_basic_flow()
    test_stop_loss()
    test_ttl_expiry()
    test_exposure_limit()
    test_non_universe_rejected()
    test_report_save()
    print("\n=== All Paper Trader tests passed ===")
