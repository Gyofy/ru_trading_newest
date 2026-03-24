# CLAUDE_CRYPTO_AGENT

Autonomous crypto trading system — Dual Regime TSMOM + CVD timing + OI crowding filter.

**Binance USDT-M Futures | 10 coins | A(sniper 4x) + B(steady 2x) | Paper Trading Active**

---

## Current Strategy: v5.3 Dual Regime

```
A (Sniper):  7d+28d TSMOM agree + RSI + CVD Q90 + OI → 4x leverage
B (Steady):  7d TSMOM only + RSI + CVD Q75 + OI      → 2x leverage

Barrier:     TP = 5×ATR, SL = 2×ATR, TTL = 24 bars
Coins:       BTC ETH SOL XRP ADA DOT LINK DOGE AVAX BNB
```

### A Sniper (확실한 자리, 강한 베팅)
- 7일 + 28일 모멘텀 방향 일치 (레짐 확인)
- CVD가 상위/하위 10% 극단 (Q90)
- OOS: 25 trades, **WR 80%**, PF 9.68, MDD -19%
- 월 2~3건, 4x 레버리지

### B Steady (잔잔한 추세 추종)
- 7일 모멘텀 방향만
- CVD가 상위/하위 25% (Q75)
- OOS: ~70 trades, WR 58%, avg +3.0%
- 월 ~10건, 2x 레버리지

### Validation

| Metric | IS (9개월) | OOS (3.5개월) |
|--------|-----------|-------------|
| per-trade Sharpe | +0.072 | +0.552 |
| Profit Factor | 1.19 | 3.73 |
| Permutation p-value | — | 0.0000 |

---

## Signal Logic

```
  7-day return → direction (LONG/SHORT)
       │
  RSI > 50 (LONG) / RSI < 50 (SHORT) → trend confirm
       │
  CVD extreme? → entry timing
    Q90 + 28d agree → A (sniper 4x)
    Q75 only        → B (steady 2x)
       │
  |OI z-score| < 2.0 → crowding check
       │
  ENTER: TP=5×ATR, SL=2×ATR, TTL=24bars
```

---

## Architecture

```
run_tsmom_paper.py              Paper bot v5.3 (ACTIVE, 10 coins, dual regime)
run_live_bot_v2.py              v4.3 ML live bot (separate session)

src/strategy/
  tsmom_core.py                 Shared signal/backtest/metrics (multiprocessing)

src/rl/
  bandit.py                     LinUCB 7-action sizing (shadow mode)
  state_builder.py              33-dim / 7-dim compact state
  signal_logger.py              JSONL signal logging
  rl_gate.py                    Shadow/active mode

src/data/crawlers/
  crypto_ohlcv.py               OHLCV + technical indicators
  signal_features.py            Wavelet/FFT/entropy
  microstructure_rollup.py      CVD/OFI/VPIN (BVC)
  binance_public_data_downloader.py   OI/LSR/Taker metrics
  binance_public_features.py    Binance feature integration

src/execution/
  exchange_adapter.py           Binance USDT-M Futures (ccxt)
  risk_engine.py                9-gate pre-trade check
  sl_tp_monitor.py              SL/TP polling
  cost_model.py                 Fee + slippage + funding
  position_store.py             Crash-safe persistence

experiments/
  test_paper_bot.py             Quick test
  download_and_integrate.py     Binance data pipeline
  v5_2_exit_optimization.py     Exit strategy comparison
  archive/                      Past experiments (9 scripts)

docs/
  strategy_v5_1r_final.md       v5.1r strategy spec
  v5_3_regime_dual_strategy.md  v5.3 dual regime design
  v5_2_exit_test_results.md     Exit optimization results
  v5_2_filter_relaxation_results.md   RSI/CVD sensitivity
  performance_summary_v5_1.md   Performance metrics
  brainstorm_v5_1_next_steps.md Data-driven analysis
```

---

## What We Tried and Rejected

| Approach | Result | Why Rejected |
|----------|--------|-------------|
| ML S2 direction (v4.0~4.3) | bal_acc 0.518 | Feature leakage |
| BTC spike strategy (v4.4) | avg +0.11% | Below cost |
| Dual TSMOM 7+28d filter (v5.1) | OOS 0 trades | 28d return ≈ 0 deadlock |
| GARCH dynamic SL | No improvement | ATR(14) sufficient at 4h |
| Trailing stop | Sharpe 3.16 (worse) | 4h noise triggers too often |
| Scale-out exit | MDD -10.7% (better) but avg -0.23%p | Net negative |
| Smart TTL | Sharpe 4.00 (worse) | Extensions hurt |
| ML S1 quality filter | CV 0.543 | Near random |
| 5x leverage | MDD -82% | Unviable |

---

## Paper Bot

```bash
# Start (single instance, lock enforced)
nohup python -X utf8 run_tsmom_paper.py > data/reports/tsmom_paper/nohup.log 2>&1 &

# Test single cycle
python -X utf8 experiments/test_paper_bot.py

# Check state
cat data/reports/tsmom_paper/state.json
cat data/reports/tsmom_paper/trades.jsonl
```

---

## Performance

```
Multiprocessing: 70% of CPU cores (12 → 8 workers)
Data fetch: 10 coins parallel (ThreadPoolExecutor)
Backtest: parallel per-coin (backtest_multi)
```

---

## Version History

| Version | Strategy | Result | Status |
|---------|----------|--------|--------|
| v4.0~4.3 | ML 2-Stage | INVALID (leakage) | Dead |
| v4.4 | BTC Spike | avg +0.11% | Dead |
| v5.0 | TSMOM 28d | OOS Sharpe 2.80 | Superseded |
| v5.1 | Dual 7+28d, 10 coins | OOS Sharpe 4.03 | Superseded (deadlock) |
| v5.1r | Single 7d, 10 coins | OOS Sharpe 4.88 | Superseded |
| **v5.3** | **Dual regime A(4x)+B(2x)** | **WR 80% (A), IS+OOS 양수** | **Active** |

---

## Environment

```
Python 3.10+ | Binance USDT-M Futures (ccxt) | scikit-learn | yfinance
pip install ccxt yfinance scikit-learn catboost xgboost ta pandas numpy joblib pyyaml requests tqdm arch
```
