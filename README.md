# CLAUDE_CRYPTO_AGENT v4.3

Autonomous crypto trading system with ML prediction + RL meta-strategy.

Binance USDT-M Futures | 5 Coins | 4h Bars | 2-Stage Binary ML | LinUCB RL Gate

## Architecture

```
Data (Binance 4h OHLCV, 500 bars)
  -> Feature Engineering (221 cols: technical + signal + microstructure)
  -> MI Feature Selection (top 120)
  -> 2-Stage ML Prediction
       Stage 1: Trade / NoTrade (ExtraTrees + CatBoost/XGBoost/TabM)
       Stage 2: Long / Short   (same combo, Trade samples only)
  -> RL Meta-Gate (LinUCB Contextual Bandit, 31-dim state)
       accept/reject + sizing [0.75x, 1.0x, 1.25x]
       cycle-level candidate ranking
  -> Risk Engine (9-gate pre-trade check)
  -> Execution (Post-Only entry, STOP_MARKET SL, TAKE_PROFIT_MARKET TP)
  -> SL/TP/TTL Monitor (30s background polling, MFE/MAE tracking)
```

## Performance (Paper Simulation, 90 days)

| Coin | Trades | Win Rate | Avg PnL | Total PnL | Max DD |
|------|:------:|:--------:|:-------:|:---------:|:------:|
| DOT  | 19     | 95%      | +2.36%  | +44.8%    | 1.39%  |
| ADA  | 32     | 97%      | +1.96%  | +62.6%    | 1.32%  |
| XRP  | 26     | 92%      | +1.75%  | +45.4%    | 1.08%  |
| SOL  | 25     | 100%     | +2.72%  | +68.0%    | 0.00%  |
| LINK | 25     | 100%     | +2.25%  | +56.3%    | 0.00%  |
| **Total** | **127** | **97%** | **+2.18%** | **+277%** | - |

Initial equity $10,000 -> $22,094. Cost-adjusted (0.2% round-trip deducted per trade).

## Coin-Specific Model Combos (Mega Search v2)

| Coin | Combo | Weights | S1 Threshold |
|------|-------|---------|:------------:|
| DOT  | ExtraTrees + TabM | 70/30 | 0.45 |
| ADA  | ExtraTrees + CatBoost | 50/50 | 0.40 |
| XRP  | ExtraTrees + CatBoost | 50/50 | 0.45 |
| SOL  | ExtraTrees + CatBoost | 50/50 | 0.45 |
| LINK | ExtraTrees + XGBoost | 50/50 | 0.45 |

Selected from 20 config x 5 coin x 4 window walk-forward search with 223 microstructure features.

## RL Meta-Strategy (LinUCB)

The RL layer sits between ML prediction and order execution. It learns "when to trust the ML signal" from actual trading outcomes.

- **Algorithm:** LinUCB Contextual Bandit (Sherman-Morrison update, gamma=0.995 forgetting)
- **State:** 31-dim (signal quality + market regime + microstructure + cost proxy + portfolio + coin history + coin identity + intercept)
- **Actions:** [REJECT, ACCEPT_0.75, ACCEPT_1.00, ACCEPT_1.25]
- **Reward:** Symmetric PnL (accept = realized, reject = -counterfactual)
- **Safety:** Warmup 200 signals, v4.2 fallback if accept rate < 25%, change cap +/-25%
- **Currently:** Shadow mode (logging decisions, not applying them)

## Project Structure

```
run_live_bot_v2.py              -- Main autonomous trading bot
run_paper_sim.py                -- Offline paper trading simulator

src/
  execution/
    exchange_adapter.py         -- Binance USDT-M Futures (ccxt)
    live_predictor.py           -- 2-Stage + Multi-Model train/predict
    sl_tp_monitor.py            -- 30s SL/TP/TTL polling + MFE/MAE
    position_store.py           -- Crash-safe JSON position persistence
    risk_engine.py              -- 9-gate pre-trade check + sizing
    order_ledger.py             -- SQLite order/fill/PnL ledger
    cost_model.py               -- Fee + slippage + funding cost model
    state_machine.py            -- Position FSM

  rl/                           -- RL Meta-Strategy Layer (v4.3)
    state_builder.py            -- 31-dim state vector
    signal_logger.py            -- Signal + result JSONL logging
    bandit.py                   -- LinUCB contextual bandit
    rl_gate.py                  -- Decision gate with safety fallback
    counterfactual.py           -- Rejected signal PnL estimation
    offline_train.py            -- CLI: signal_log -> trained LinUCB

  models/
    masking_loop.py             -- 2-Stage Binary labeling + ensemble
    model_store.py              -- Artifact save/load

  data/crawlers/                -- OHLCV + features + microstructure
  signals/                      -- Signal contract + policy
  evaluation/                   -- Trade-level simulation

config/
  frozen_params_v4_3.yaml       -- Current frozen config
```

## Usage

### Paper Simulation (no exchange needed)
```bash
python run_paper_sim.py --equity 10000 --days 90
```

### Live Bot (Binance testnet)
```bash
export BINANCE_API_KEY=your_key
export BINANCE_API_SECRET=your_secret
python run_live_bot_v2.py --mode paper
```

### Live Bot (real money)
```bash
python run_live_bot_v2.py --mode live --equity 10000
```

### Offline RL Training (after 200+ signals)
```bash
python -m src.rl.offline_train --alpha 1.0 --gamma 0.995
```

## Key Safety Features

- **Kill Switch:** Daily DD > 2% or Weekly DD > 5% halts all trading
- **Risk Engine:** 9-gate pre-trade check (probability, spread, funding, drawdown, consecutive losses, sizing)
- **SL/TP Monitor:** 30s polling (not 2h cycle dependent)
- **Position Persistence:** JSON + SQLite survive process crashes
- **RL Fallback:** Reverts to baseline when RL becomes too conservative
- **Post-Only Entry:** Maker orders only, 20s fill timeout

## Version History

| Version | Date | Key Change |
|---------|------|------------|
| v4.3 | 2026-03-18 | RL meta-layer (LinUCB), MFE/MAE tracking, 13 audit fixes |
| v4.2 | 2026-03-18 | Mega Search v2 best, 2-Stage Binary, 8 critical fixes |
| v4.1 | 2026-03-18 | 5-coin expansion, microstructure features |
| v4.0 | 2026-03-17 | Binance Futures, walk-forward optimization |
| v3.4 | 2026-03-17 | Per-coin regime policy, trade-level EV |

## Environment

- Python 3.10+ | PyTorch | scikit-learn | ccxt | yfinance
- GPU: RTX 3090 (XGBoost CUDA, TabM CUDA, optional)
- OS: Windows 11
