# CLAUDE_CRYPTO_AGENT v4.3

Autonomous crypto trading system — ML ensemble prediction + RL meta-strategy.

Binance USDT-M Futures | 8 Coins | 4h Bars | 2-Stage Binary | LinUCB RL Gate

---

## Pipeline

```
Every 2 hours:

  1. FETCH      Binance 4h OHLCV (500 bars, 8 coins)
  2. FEATURES   221 cols: technical + signal + microstructure
  3. MI SELECT  Top 120 features per coin
  4. PREDICT    2-Stage Binary (Trade/NoTrade → Long/Short)
  5. RL GATE    LinUCB bandit: accept/reject + sizing
  6. SIZING     Confidence-tiered: 0.5% / 1.0% / 1.5% + DD brake
  7. RISK       9-gate pre-trade check
  8. EXECUTE    Post-Only entry + STOP_MARKET SL + TP_MARKET TP
  9. MONITOR    30s polling: SL/TP/TTL + MFE/MAE tracking
```

---

## Model Search Results (Mega Search v3)

10 configs x 8 coins x 4 walk-forward windows. Clean pipeline: symmetric labeling, no data leakage, train/test split enforced.

| Rank | Coin | Best Combo | Avg PnL | Win Rate | DD |
|:----:|------|:----------:|:-------:|:--------:|:--:|
| 1 | **TAO** | CatBoost + XGBoost 50/50 | **+3.42%** | 53% | 6.7% |
| 2 | **DOT** | ET + XGBoost 70/30 | **+1.14%** | 41% | 9.2% |
| 3 | **XRP** | ET + TabM 70/30 | **+1.01%** | 47% | 4.6% |
| 4 | **ADA** | ET + TabM 70/30 | **+0.91%** | 36% | 9.0% |
| 5 | **ETH** | ET + TabM 70/30 | **+0.78%** | 40% | 6.4% |
| 6 | **BTC** | ET + CatBoost 50/50 | **+0.74%** | 44% | 4.3% |
| 7 | **SOL** | ET + TabM 70/30 | **+0.30%** | 29% | 7.8% |
| 8 | **LINK** | XGBoost solo | **+0.14%** | 26% | 10.1% |

All 8 coins show positive expected value after 0.2% round-trip cost deduction.

---

## Trading Strategy

### 2-Stage Binary Classification

```
Stage 1: "Should we trade?"
  ML models → p(Trade) — if above threshold (0.45) → proceed

Stage 2: "Long or Short?"
  ML models → p(Long) — if > 0.5 → BUY, else SELL
  Trained only on Trade samples (HOLD excluded)
```

Both LONG and SHORT use symmetric R:R = 5:1:
```
BUY:  TP = entry + 3.0 × ATR    SL = entry - 0.6 × ATR
SELL: TP = entry - 3.0 × ATR    SL = entry + 0.6 × ATR
TTL = 48 hours (12 bars)
```

### Confidence-Tiered Sizing

```
confidence > 0.65    → 1.5% of equity
confidence 0.50-0.65 → 1.0%
confidence < 0.50    → 0.5%
Daily DD > 1.5%      → halve all sizes
Daily DD > 2.0%      → kill switch (stop trading)
```

### RL Meta-Gate (LinUCB)

Sits between ML prediction and order execution. Learns "when to trust the signal" from trade outcomes.

- **State:** 31 dimensions (signal quality, market regime, microstructure, portfolio, coin history)
- **Actions:** REJECT / ACCEPT_0.75 / ACCEPT_1.00 / ACCEPT_1.25
- **Reward:** Symmetric — accept: realized PnL, reject: -counterfactual PnL
- **Safety:** Warmup 200 signals, v4.2 fallback if accept rate < 25%
- **Status:** Shadow mode (logging only, not affecting trades)

---

## Risk Management

```
Layer 1: Regime     RANGE_LOW → block entry
Layer 2: Threshold  p_trade < 0.45 → block
Layer 3: RL Gate    31-dim analysis → accept/reject (shadow)
Layer 4: Sizing     Confidence-tiered 0.5~1.5%
Layer 5: 9-Gate     Kill switch, spread, funding, DD, consecutive loss, sizing
Layer 6: DD Brake   DD > 1.5% → halve sizes
Layer 7: Kill       DD > 2% daily / 5% weekly → full stop
```

---

## Code Integrity

36 issues found and fixed across 3 audit passes (Opus 4.6):

| Category | Fixed |
|----------|:-----:|
| Data leakage (bfill, center=True, look-ahead) | 11 |
| Race conditions (double-close, shared state) | 4 |
| Logic errors (barrier asymmetry, equity calc) | 8 |
| Strategy flaws (SHORT R:R, label bias) | 5 |
| Numerical / compatibility | 8 |

Zero `bfill()` calls remain in the entire codebase. All models retrained on clean pipeline.

---

## Project Structure

```
run_live_bot_v2.py              -- Autonomous trading bot (2h cycle)
run_paper_sim.py                -- Offline simulator (yfinance + local CSV)

src/
  execution/
    live_predictor.py           -- 2-Stage + Multi-Model train/predict
    sl_tp_monitor.py            -- 30s SL/TP/TTL + MFE/MAE
    position_store.py           -- Crash-safe position persistence
    risk_engine.py              -- 9-gate + symmetric barriers
    exchange_adapter.py         -- Binance USDT-M (ccxt)
    order_ledger.py             -- SQLite ledger
    cost_model.py               -- Fee + slippage + funding

  rl/
    state_builder.py            -- 31-dim state vector
    signal_logger.py            -- Signal + result logging
    bandit.py                   -- LinUCB (Sherman-Morrison)
    rl_gate.py                  -- Gate with fallback
    counterfactual.py           -- Rejected signal PnL estimation
    offline_train.py            -- CLI training

  models/                       -- Labeling + ensemble pipeline
  data/crawlers/                -- Features + microstructure

config/frozen_params_v4_3.yaml  -- 8-coin frozen config
experiments/mega_search_v3.py   -- Model search script
```

## Usage

```bash
# Paper simulation (no exchange needed)
python run_paper_sim.py --equity 10000 --days 90

# Live bot (Binance testnet)
export BINANCE_API_KEY=... BINANCE_API_SECRET=...
python run_live_bot_v2.py --mode paper

# Live bot (real money)
python run_live_bot_v2.py --mode live

# RL offline training (after 200+ signals)
python -m src.rl.offline_train
```

## Version History

| Version | Key Changes |
|---------|------------|
| **v4.3** | 8 coins (TAO/BTC/ETH added), Mega Search v3, symmetric barriers (LONG+SHORT R:R 5:1), 36 audit fixes, RL meta-layer, confidence-tiered sizing |
| v4.2 | 5 coins, Mega Search v2, 2-Stage Binary, clean architecture |
| v4.0 | Binance Futures, walk-forward optimization |
| v3.4 | Per-coin regime policy, trade-level EV |

## Environment

- Python 3.10+ | PyTorch | scikit-learn | ccxt | yfinance
- GPU: RTX 3090 (XGB CUDA, TabM CUDA — CPU fallback supported)
- Windows 11
