# CLAUDE_CRYPTO_AGENT

Crypto trading system — BTC global direction + relative strength alt selection.

**Binance USDT-M Futures | 10 coins | v6.0 Honest Portfolio | 1h resolution**

---

## Strategy: v6.0 → v6.1

### v6.0 (Current)
```
1. BTC 7-day return → Global Direction (LONG/SHORT/FLAT)
2. Alt 9 coins: relative strength vs BTC → top 2 only
3. RSI + CVD Q75 + OI filters
4. BTC direction flip → instant exit (DIR_FLIP)
5. TP=5×ATR, SL=2×ATR, TTL=24bars, max 2 positions, 2x leverage
```

### v6.1 (In Development)
```
Same strategy, 1h resolution instead of 4h:
- 진입: 1h 봉 기준 시그널 (4x more signals)
- 청산: 실시간 WebSocket 모니터링
- 목표: 조건 충족 즉시 진입 (봉 마감 대기 없음)
```

### Validated Performance

| | IS (9mo) | OOS (3.5mo) |
|---|---|---|
| per-trade Sharpe | +0.058 | +0.208 |
| WR | 42.4% | 47.2% |
| PF | 1.15 | 1.71 |
| MDD | -67.4% | -13.5% |

**IS + OOS 둘 다 양수** (이전 v5.3은 IS 마이너스)

### Why v6 (Fake Diversification Removed)
```
v5.3: 10코인 독립 시그널 → 상관관계 0.80, 방향 일치 99%
      = "BTC 1종목에 10x 베팅"과 동일
      Sharpe 4.88 → 실질 0.55 (8.6x 과대평가)

v6.0: BTC 방향 → 상대 강도 상위 2코인만
      = 정직한 포트폴리오
      per-trade Sharpe 0.208 (현실)
```

---

## Architecture

```
run_tsmom_paper.py              v6.0 Paper bot (yfinance/Binance dual)
run_live_bot_v2.py              v4.3 ML bot (Binance live, separate)

src/strategy/tsmom_core.py      Shared signal/backtest/metrics
src/execution/
  exchange_adapter.py           Binance USDT-M Futures (ccxt)
  sl_tp_monitor.py              SL/TP polling
  risk_engine.py                9-gate check
  position_store.py             Crash-safe persistence
  cost_model.py                 Fee + slippage model

src/rl/                         RL meta-layer (shadow mode)
experiments/                    Backtest scripts + archive

docs/
  v6_honest_portfolio_report.md v6.0 results
  adaptive_exit_rl_brainstorm.md RL exit research (30+ papers)
  binance_implementation_spec.md Binance API implementation plan
  strategy_v5_1r_final.md       v5.1r spec (superseded)
```

---

## Data Source

```bash
# Binance (production)
DATA_SOURCE=binance BINANCE_API_KEY=xxx BINANCE_API_SECRET=yyy python run_tsmom_paper.py

# yfinance (local dev fallback)
python run_tsmom_paper.py
```

---

## Binance Implementation Plan

See `docs/binance_implementation_spec.md` for full details:

1. **Data**: ccxt `fetch_ohlcv()` + WebSocket `watch_ticker()`
2. **Orders**: Post-Only entry + STOP_MARKET SL + TAKE_PROFIT_MARKET TP
3. **Real-time**: WebSocket price stream → SL/TP/DIR_FLIP instant
4. **Safety**: max 2 pos, 2% risk/trade, -10% daily halt

---

## Version History

| Version | Strategy | Real Metric | Status |
|---------|----------|-------------|--------|
| v4.0~4.3 | ML 2-Stage | INVALID (leakage) | Dead |
| v5.0~5.3 | TSMOM 10 coins | ptS 0.504 (fake, corr 0.80) | Dead |
| **v6.0** | **BTC dir + RS top 2** | **ptS 0.208 (honest)** | **Active** |
| v6.1 | 1h resolution | Testing | In dev |

---

## Environment

```
Python 3.10+ | ccxt | pandas | numpy | scikit-learn | lifelines | arch
pip install ccxt pandas numpy scikit-learn lifelines arch joblib pyyaml
```
