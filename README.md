# CLAUDE_CRYPTO_AGENT v4.3

Autonomous crypto trading system — ML ensemble prediction + RL meta-strategy.

**Binance USDT-M Futures · 8 Coins · 4h Bars · 2-Stage Binary · LinUCB RL Gate**
**Status: LIVE** (as of 2026-03-18)

---

## Pipeline

```
Every 4 hours:

  1. FETCH      Binance 4h OHLCV (500 bars, 8 coins + BTC/ETH reference)
  2. FEATURES   221 cols: technical + signal + microstructure (CVD/OFI/VPIN)
  3. MI SELECT  Top 120 features per coin (mutual information)
  4. TRAIN      2-Stage Binary — fresh retrain every cycle
  5. PREDICT    Stage 1: Trade/NoTrade → Stage 2: Long/Short
  6. RL GATE    LinUCB bandit: accept/reject + sizing multiplier
  7. SIZING     Confidence-tiered: 0.5% / 1.0% / 1.5% equity + DD brake
  8. RISK       9-gate pre-trade check
  9. EXECUTE    Post-Only entry + STOP_MARKET SL + TAKE_PROFIT_MARKET TP
 10. MONITOR    30s polling: SL/TP/TTL + 3-stage partial TP + MFE/MAE
```

---

## Model Results (Mega Search v3)

10 configs × 8 coins × 4 walk-forward windows.
Symmetric Triple Barrier labeling · No data leakage · TimeSeriesSplit CV.

| Rank | Coin | Best Combo | Avg PnL/trade | Win Rate | Max DD |
|:----:|------|:----------:|:-------------:|:--------:|:------:|
| 1 | **TAO** | CatBoost + XGBoost 50/50 | **+3.42%** | 53% | 6.7% |
| 2 | **DOT** | ET + XGBoost 70/30 | **+1.14%** | 41% | 9.2% |
| 3 | **XRP** | ET + TabM 70/30 | **+1.01%** | 47% | 4.6% |
| 4 | **ADA** | ET + TabM 70/30 | **+0.91%** | 36% | 9.0% |
| 5 | **ETH** | ET + TabM 70/30 | **+0.78%** | 40% | 6.4% |
| 6 | **BTC** | ET + CatBoost 50/50 | **+0.74%** | 44% | 4.3% |
| 7 | **SOL** | ET + TabM 70/30 | **+0.30%** | 29% | 7.8% |
| 8 | **LINK** | XGBoost solo | **+0.14%** | 26% | 10.1% |

All 8 coins positive EV after 0.2% round-trip cost deduction.

---

## Trading Strategy

### 2-Stage Binary Classification

```
Stage 1 — "Should we trade?"
  Ensemble → p(Trade) ≥ threshold (per-coin, 0.40–0.45) → proceed

Stage 2 — "Long or Short?"
  Ensemble → p(Long) > 0.5 → BUY, else SELL
  Trained only on Trade samples (HOLD excluded)
```

Both LONG and SHORT use symmetric R:R 5:1:
```
BUY:  TP = entry + 3.0 × ATR    SL = entry − 0.6 × ATR
SELL: TP = entry − 3.0 × ATR    SL = entry + 0.6 × ATR
TTL  = 48 hours (12 × 4h bars)
```

### 3-Stage Partial Take-Profit

```
TP1 = entry ± 1.0 × ATR  →  close 33%,  move SL → breakeven
TP2 = entry ± 2.0 × ATR  →  close 33%,  move SL → TP1
TP3 = entry ± 3.0 × ATR  →  close remaining
```

### Confidence-Tiered Sizing

```
p_trade > 0.65  → 1.5% of equity
p_trade 0.50–0.65 → 1.0%
p_trade < 0.50    → 0.5%
Daily DD > 1.5%   → halve all sizes
Daily DD > 2.0%   → kill switch (no new entries)
```

### RL Meta-Gate (LinUCB)

Sits between ML prediction and order execution. Learns *when to trust the signal* from live trade outcomes.

| Item | Detail |
|------|--------|
| State | 31 dimensions (signal quality, regime, microstructure, portfolio, coin history) |
| Actions | REJECT / ACCEPT_0.75 / ACCEPT_1.00 / ACCEPT_1.25 |
| Reward | Accept: realized PnL · Reject: −counterfactual PnL |
| Safety | Warmup 200 signals · fallback to no-gate if accept rate < 25% |
| Status | Shadow mode (logging only until warmup complete) |

---

## Risk Management

```
Gate 1   Regime filter    RANGE_LOW / UNKNOWN → block entry
Gate 2   Probability      p_trade < threshold → block
Gate 3   Spread           > 50 bps → block
Gate 4   Funding rate     > 0.1%/8h → block · > 0.05% → warn
Gate 5   Daily drawdown   > 2% → kill switch
Gate 6   Consecutive loss 3 losses on same coin → block
Gate 7   Alt bucket       Alt coin daily loss > 1% equity → block
Gate 8   Liquidation      SL beyond liq price → block (leverage > 1×)
Gate 9   Fee efficiency   Notional < 10× fee → block
```

---

## Retraining

Two timeframe variants are supported:

| Variant | Bars | Horizons | Script |
|---------|------|----------|--------|
| **v4.3 (active)** | 4h | 4h / 12h / 24h / 72h | bot retrains each cycle |
| v4.3-1m | 1m | 30m / 1h / 2h / 4h | `scripts/retrain_1m.py` |

```bash
# Retrain 1m models (all 8 coins in parallel, 4 workers)
python scripts/retrain_1m.py --mode live --workers 4
```

---

## Project Structure

```
run_live_bot_v2.py              Active trading bot (4h cycle, live + paper)
run_paper_sim.py                Offline paper simulator
run_frozen_oos_v2.py            Out-of-sample walk-forward validation

config/
  frozen_params_v4_3.yaml       Active 8-coin config (4h)
  frozen_params_v4_3_1m.yaml    1m variant config
  settings.yaml                 Runtime settings (bar_minutes, horizons, risk)
  settings_1m.yaml              1m runtime settings

src/
  execution/
    live_predictor.py           2-Stage + Multi-Model train/predict
    sl_tp_monitor.py            30s SL/TP/TTL + 3-stage partial TP + MFE/MAE
    position_store.py           Crash-safe JSON position persistence (.bak)
    risk_engine.py              9-gate pre-trade + drawdown accumulation
    exchange_adapter.py         Binance USDT-M Futures (ccxt async)
    order_ledger.py             SQLite order/fill/PnL ledger
    cost_model.py               Fee + slippage + funding cost model
  rl/
    rl_gate.py                  LinUCB gate with warmup + fallback
    state_builder.py            31-dim state vector
    signal_logger.py            Signal + outcome logging
    bandit.py                   LinUCB (Sherman-Morrison update)
    counterfactual.py           Rejected signal PnL estimation
  models/
    masking_loop.py             Triple Barrier labeling + ensemble training
    enhanced_ensemble.py        5-model weighted ensemble
  data/crawlers/
    crypto_ohlcv.py             Binance OHLCV + technical indicators
    microstructure_rollup.py    CVD / OFI / VPIN / Roll Spread / Amihud
    signal_features.py          Wavelet / FFT / entropy features
  signals/
    contract.py                 Signal dataclass
    policy.py                   Regime-aware signal filtering
  utils/
    config.py                   YAML config loader (SETTINGS_YAML_PATH override)
    feature_policy.py           Feature include/exclude rules

scripts/
  retrain_1m.py                 Parallel 1m model retraining (8 coins)
  discord_notifier.py           Discord webhook alerts
  git_push_agent.py             Automated git commit/push

data/models/
  live_v4_3/                    Active 4h models (8 coins)
  live_v4_3_1m/                 1m models (8 coins)

archive/
  v3.x/                         v3.2 – v3.4 configs, reports, walk-forward
  v4.0/ v4.1/ v4.2/             Previous version configs, models, reports
  experiments/                  TabPFN / TabM evaluation scripts + results

experiments/
  mega_search_v3.py             Latest hyperparameter search (active)
  mega_search_v2/run_mega_search_v2.py
```

---

## Quick Start

```bash
# Requirements
pip install ccxt yfinance scikit-learn catboost xgboost ta pandas numpy joblib pyyaml python-dotenv

# Environment
cp .env.example .env
# fill in BINANCE_API_KEY, BINANCE_API_SECRET, DISCORD_WEBHOOK_URL (optional)

# Paper trading (Binance testnet)
python run_live_bot_v2.py --mode paper --equity 10000

# Live trading (real money — starts after balance check + confirmation)
python run_live_bot_v2.py --mode live

# Retrain 1m models
python scripts/retrain_1m.py --mode live --workers 4

# Paper simulation (offline, no exchange)
python run_paper_sim.py --equity 10000 --days 90
```

---

## Code Quality

36 issues found and fixed across 3 audit passes:

| Category | Fixed |
|----------|:-----:|
| Data leakage (bfill, center=True, look-ahead) | 11 |
| Race conditions (double-close, async state) | 4 |
| Logic errors (barrier asymmetry, equity calc) | 8 |
| Strategy flaws (SHORT R:R, label bias) | 5 |
| Numerical / compatibility | 8 |

Zero `bfill()` calls in the codebase. All models retrained on clean pipeline.

---

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| **v4.3-1m** | 2026-03-18 | 1m bar retraining system, parallel multiprocessing, SETTINGS_YAML_PATH override |
| **v4.3** | 2026-03-18 | 8 coins (TAO/BTC/ETH), Mega Search v3, symmetric barriers (5:1 R:R), 36 audit fixes, RL meta-layer, confidence-tiered sizing, 3-stage partial TP, DrawdownTracker → RiskEngine |
| v4.2 | 2026-03-17 | 5 coins, Mega Search v2 (223 microstructure features), 2-Stage Binary |
| v4.0–4.1 | 2026-03 | Binance Futures live execution, walk-forward optimization |
| v3.4 | 2026-02 | Per-coin regime policy, trade-level EV, Triple Barrier labeling |

---

## Environment

- **Python** 3.10+ · **Exchange** Binance USDT-M Futures (ccxt)
- **ML** scikit-learn · CatBoost · XGBoost · TabM (optional)
- **GPU** RTX 3090 — XGB CUDA, TabM CUDA · CPU fallback supported
- **OS** Linux (WSL2) / Windows 11
