# CLAUDE_CRYPTO_AGENT

Autonomous crypto trading system — event-driven BTC spike strategy + ML research platform.

**Binance USDT-M Futures · SOL/ADA/ETH/XRP · BTC Spike Trigger · Paper Trading Active**

---

## Current Status (2026-03-20)

### Critical Discovery: Feature Leakage

On 2026-03-20, **critical data leakage** was discovered in the ML pipeline:

| Source | Issue | Impact |
|--------|-------|--------|
| STL Decomposition | Centered moving average uses future data | S2 accuracy: 0.687 → 0.518 |
| Ichimoku `.shift(26)` | 26-bar lookahead | Direct future reference |
| SVD Interpolation | Stride + linear interpolation instability | Feature values change with new data |

**All previous backtest results (v4.0–v4.3) were invalidated.** The +1.5%/trade ML performance was entirely driven by leakage.

### Strategy Pivot

After removing leakage, **15,120 strategy combinations** were exhaustively tested:

```
Tested:  9 coins × 12 strategies × 7 TP/SL × 6 hold times = 15,120 combos
Result:  Only BTC spike → alt follow has consistent positive edge
         Momentum, RSI, EMA, volume, ML direction = ALL negative after costs
```

### Active Strategy: BTC Spike → Alt Follow

```
Trigger:  BTC 1h close-to-close |return| > 1.2%
Entry:    SOL/ETH/XRP/ADA in same direction as BTC, NEXT bar open
Filter:   Alt confirmation required (volume spike / big bar / direction align)
TP:       1.5 × ATR(14)
SL:       1.0 × ATR(14)
Hold:     Max 6 hours
Leverage: 3x (paper)
```

**Evidence (180 days, BTC -38.5% bear market):**

| Metric | Value |
|--------|-------|
| Total events | 276 (4 coins combined) |
| Win Rate | 48.6% |
| Avg PnL/trade | +0.11% |
| Not long-only bias | SHORT also positive (+0.087%) |
| FOLLOW vs COUNTER | +0.28% vs -0.52% (clear edge) |

---

## Architecture

```
run_btc_spike_paper.py          BTC spike paper trading bot (ACTIVE)
run_live_bot_v2.py              v4.3 ML bot (SUSPENDED - leakage fix applied)

src/
  execution/
    live_predictor.py           2-Stage Binary + predict_hybrid()
    exchange_adapter.py         Binance USDT-M Futures (ccxt)
    sl_tp_monitor.py            5s SL/TP/TTL polling
    risk_engine.py              9-gate pre-trade check
    position_store.py           Crash-safe JSON persistence
    cost_model.py               Fee + slippage + funding
  data/crawlers/
    crypto_ohlcv.py             OHLCV + causal decomposition (STL removed)
    signal_features.py          Wavelet/FFT/entropy (ichi leak fixed)
    microstructure_rollup.py    CVD/OFI/VPIN/Roll Spread/Amihud
  models/
    masking_loop.py             Triple Barrier labeling
  rl/
    rl_gate.py                  LinUCB gate (shadow mode)

config/
  frozen_params_v4_3.yaml       Config (leakage-fixed, rf=2%, deadzone=0.10)

scripts/                        Analysis & optimization scripts
rjh_20260320/                   2026-03-20 research archive
```

---

## Research History (2026-03-20)

### Phase 1: Leakage Discovery
- STL decomposition, Ichimoku shift(26), SVD interpolation → all leaked future data
- Permutation test: p=0.000 (model learns patterns, but from leaked features)
- Clean S2 accuracy: 0.518 (near random)

### Phase 2: Exhaustive Strategy Search
- 15,120 combinations across 9 coins
- Every traditional strategy negative after 0.18% round-trip cost
- Only BTC spike → alt follow survived

### Phase 3: Edge Validation
- Not bull market bias (tested in -38.5% BTC decline)
- Not "just buy BTC" (ALWAYS_LONG = negative)
- FOLLOW BTC: +0.28% vs COUNTER: -0.52%
- Edge source: momentum continuation + volatility clustering

### Phase 4: SOL/ADA Microstructure Deep Dive
- 300+ candle/volume/bar anomaly combinations tested
- Candle patterns alone: ALL negative
- Volume anomalies alone: ALL negative
- Only BTC spike + alt confirmation = positive

### Phase 5: ML Enhancement Attempt
- 28 features, 119 spike events, ExtraTrees + GBM
- Marginal improvement: WR +3.8% (not statistically significant)
- n too small for ML (need 500+ events)

---

## Leakage Fixes Applied

```python
# crypto_ohlcv.py: STL → EMA causal only
_add_decomposition() now uses EMA(adjust=False, center=False)
# No STL, no centered moving average

# signal_features.py: Ichimoku fix
ichi_above_cloud = (close > senkou_a)  # was: senkou_a.shift(26)

# crypto_ohlcv.py: SVD fix
_add_svd_features() computes every bar directly (no stride/interpolation)
```

---

## Paper Bot: `run_btc_spike_paper.py`

```bash
# Start paper trading
python run_btc_spike_paper.py

# Custom settings
python run_btc_spike_paper.py --threshold 0.015 --leverage 5 --equity 500
```

Features:
- yfinance fallback (no API key needed)
- Alt confirmation filter (volume spike / big bar / direction / body ratio)
- Score-based sizing (higher confirmation = larger position)
- Trade logging to `data/reports/btc_spike_paper/trades.jsonl`
- State persistence (`state.json`)

---

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| **v4.4** | **2026-03-20** | **Leakage fix, strategy pivot to BTC spike, paper bot** |
| v4.3 | 2026-03-18 | 8 coins, Mega Search v3, RL meta-layer (invalidated by leakage) |
| v4.2 | 2026-03-17 | 5 coins, Mega Search v2, 2-Stage Binary |
| v3.4 | 2026-02 | Triple Barrier labeling, trade-level EV |

---

## Environment

- **Python** 3.10+ · **Exchange** Binance USDT-M Futures (ccxt)
- **ML** scikit-learn · CatBoost · XGBoost
- **Data** yfinance (paper) / ccxt (live)
- **OS** Windows 11 / Linux

```bash
pip install ccxt yfinance scikit-learn catboost xgboost ta pandas numpy joblib pyyaml
```
