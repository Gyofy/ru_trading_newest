# CLAUDE_CRYPTO_AGENT

Autonomous crypto trading system — TSMOM rule-based direction + ML/RL enhancement layer.

**Binance USDT-M Futures | 7 coins | TSMOM+RSI+CVD+OI | OOS Sharpe 2.80 | Paper Trading Active**

---

## Current Status (2026-03-23)

### v5.0 TSMOM Enhanced Strategy

Feature leakage discovery (2026-03-20) invalidated all ML direction predictions. v5.0 pivots to **rule-based direction** with ML/RL as enhancement layer.

```
Direction:  TSMOM 28-day momentum (volume-weighted)
Filter:     RSI > 50 (LONG valid) / RSI < 50 (SHORT valid)
Timing:     CVD Q75 extreme (counter-direction overextension)
Crowding:   OI z-score < 2.0 (Binance metrics)
Barrier:    TP = 5 x ATR, SL = 1.0 x ATR, TTL = 24 bars (96h)
Leverage:   2x (paper), max 3x (live)
```

### Validation Results

| Metric | IS (70%, 9mo) | OOS (30%, 3.5mo) |
|--------|---------------|-------------------|
| Configs tested | 6,480 grid | Top-20 IS |
| Best Sharpe | 2.67 | **2.80** |
| Best avg PnL | +1.12%/trade | **+2.31%/trade** |
| Positive configs | many | **20/20 (100%)** |

| Test | Result |
|------|--------|
| Permutation test | **p = 0.006** (statistically significant) |
| Bootstrap P(Sharpe > 1) | **99.8%** |
| Cost sensitivity | Positive up to **50 bps** roundtrip |
| Drop-one-out | All coins removable, min Sharpe 2.34 |
| Optimal leverage | **3x** (Monte Carlo, P(MDD>50%) < 8%) |

### OOS Performance by Coin (best config)

| Coin | Trades | WR | Avg PnL | Sharpe |
|------|--------|-----|---------|--------|
| BTC | 9 | 66.7% | +1.75% | 1.99 |
| ETH | 12 | 50.0% | +2.21% | 2.18 |
| SOL | 9 | 44.4% | +1.38% | 1.94 |
| XRP | 7 | 57.1% | +2.37% | 0.96 |
| ADA | 3 | 100% | +6.13% | - |
| DOT | 13 | 30.8% | -0.23% | -0.03 |
| LINK | 8 | 62.5% | +2.63% | 1.15 |

---

## Architecture (v5.0)

```
run_tsmom_paper.py              Paper bot (ACTIVE, RL logging enabled)
run_live_bot_v2.py              v4.3 ML bot (SUSPENDED)
run_btc_spike_paper.py          v4.4 BTC spike bot (SUPERSEDED by v5.0)

src/
  data/crawlers/
    crypto_ohlcv.py             OHLCV + technical indicators (causal)
    signal_features.py          Wavelet/FFT/entropy (leakage fixed)
    microstructure_rollup.py    CVD/OFI/VPIN/Roll Spread/Amihud (BVC)
    binance_public_data_downloader.py   Binance metrics downloader
  execution/
    exchange_adapter.py         Binance USDT-M Futures (ccxt)
    live_predictor.py           2-Stage Binary (v4.3, suspended)
    sl_tp_monitor.py            SL/TP polling
    risk_engine.py              9-gate pre-trade check
    position_store.py           Crash-safe persistence
    cost_model.py               Fee + slippage + funding
  models/
    masking_loop.py             Triple Barrier labeling
    regime_filter.py            4-state regime (TREND_UP/DOWN, RANGE_LOW/HIGH)
  rl/
    bandit.py                   LinUCB 7-action (v5.0)
    state_builder.py            33-dim state vector (v5.0 TSMOM)
    signal_logger.py            JSONL signal logging + counterfactual
    rl_gate.py                  Shadow/active mode with safety
    counterfactual.py           Rejected signal PnL estimation
    offline_train.py            CLI training pipeline
  signals/
    contract.py                 Signal dataclass
    policy.py                   SignalPolicy regime-aware filtering

experiments/
  tsmom_ml_enhanced.py          Phase 1: TSMOM base + filters
  tsmom_rsi_cvd_deep.py         Phase 2: 6,480 config grid search
  tsmom_rigorous_v2.py          Phase 3: IS/OOS split + permutation test
  download_and_integrate.py     Binance data download + integration

config/
  frozen_params_v4_3.yaml       Config (v4.3, RL section reused)

data/
  raw/binance_public/metrics/   OI/LSR/Taker (7 coins x 365d)
  reports/tsmom_paper/          Paper bot logs + signal_log
  reports/tsmom_*.csv           Backtest results
```

---

## Strategy Design (v5.0)

### Layer 1: Direction (Rule-Based)
```
TSMOM: sign(28-day return) -> LONG or SHORT
- Volume-weighted variant available
- No ML prediction (S2 = random after leakage fix)
- Academic basis: Sharpe 1.5-2.2 (Huang et al. 2024)
```

### Layer 2: Quality Filter (Rule + Optional ML)
```
RSI > 50 confirms LONG, RSI < 50 confirms SHORT
- RSI as trend filter, NOT overbought/oversold
- PMC study: RSI trend filter 773% vs B&H 275%
```

### Layer 3: Entry Timing (CVD Extreme)
```
SHORT signal + CVD > Q75 (buy-side overextended) = enter SHORT
LONG signal + CVD < Q25 (sell-side overextended) = enter LONG
- Spearman rho = -0.21 (CVD mean reversion)
- Counter-direction overextension = better fill price
```

### Layer 4: Crowding Filter (Binance OI)
```
|OI z-score| > 2.0 = skip (crowded positioning)
- Avoids entering at extreme OI where liquidation cascades likely
- Data: Binance public metrics (5-min, resampled to 4h)
```

### Layer 5: RL Enhancement (Shadow Mode)
```
LinUCB contextual bandit, 33-dim state, 7 actions
- Actions: REJECT, 0.5x, 0.75x, 1.0x, 1.25x, 1.5x, 2.0x
- State: TSMOM strength + RSI + CVD + OI + regime + microstructure
- Training: offline from signal_log.jsonl (200+ signals needed)
- Status: shadow mode (logging only, not applied)
```

---

## Development History

### v5.0 (2026-03-23) — TSMOM Enhanced
- Pivot from ML direction to rule-based TSMOM
- 6,480 config grid search + rigorous IS/OOS validation
- Permutation test p=0.006, Bootstrap P(Sharpe>1)=99.8%
- Binance OI/LSR/Taker data integrated
- RL state_builder adapted for TSMOM inputs

### v4.4 (2026-03-20) — BTC Spike
- Feature leakage discovered (STL, Ichimoku, SVD)
- 15,120 exhaustive search: only BTC spike survived
- Paper bot: avg +0.11%/trade (weak, superseded by v5.0)

### v4.3 (2026-03-18) — ML 2-Stage (INVALIDATED)
- 8 coins, Mega Search v3, RL meta-layer
- All results based on leaked features

### v4.0-4.2 (2026-03-17) — ML Evolution (INVALIDATED)
- 2-Stage Binary, Triple Barrier, walk-forward
- Invalidated by leakage discovery

---

## Paper Bot

```bash
# Start paper trading (v5.0 TSMOM strategy)
python run_tsmom_paper.py

# Signal log for RL training
data/reports/tsmom_paper/signal_log.jsonl

# Trade records
data/reports/tsmom_paper/trades.jsonl

# Bot state
data/reports/tsmom_paper/state.json
```

---

## Data Pipeline

### OHLCV (yfinance)
- 10 coins: BTC, ETH, SOL, XRP, ADA, DOGE, AVAX, DOT, LINK, BNB
- 1h fetch -> 4h resample
- 365 days history

### Binance Public Metrics
```bash
python src/data/crawlers/binance_public_data_downloader.py \
  --types funding_rate open_interest long_short_ratio \
  --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT ADAUSDT DOTUSDT LINKUSDT \
  --days 365
```

Data includes: OI, Long/Short Ratio, Top Trader Ratio, Taker Buy/Sell Volume (5-min resolution)

---

## Environment

- **Python** 3.10+ | **Exchange** Binance USDT-M Futures (ccxt)
- **ML** scikit-learn, CatBoost, XGBoost | **RL** LinUCB (custom)
- **Data** yfinance (paper) + Binance public data | **OS** Windows 11

```bash
pip install ccxt yfinance scikit-learn catboost xgboost ta pandas numpy joblib pyyaml requests tqdm
```
