# Ru_trading

Crypto auto-trading system using ML ensemble prediction + asymmetric R:R exit strategy.

## Overview

| Item | Detail |
|------|--------|
| Target | DOT (Polkadot), ADA (Cardano) |
| Timeframe | 4-hour bars |
| Strategy | 2-Stage Binary Classification + Triple Barrier Exit |
| Models | 7-Model Ensemble + Stacking Meta-Learner |
| Risk | 0.5% equity per trade, stop-distance sizing |
| Backtest | 42 trades, +26.57% (8 weeks, 5M KRW seed) |

## How It Works

### Data Pipeline

```
yfinance (1h OHLCV, 365 days)
  |
  +-- Resample to 4h bars (~2,180 bars)
  +-- Signal Features: 84 (wavelet, FFT, Hurst, entropy, microstructure, ...)
  +-- Technical Indicators: 28 (RSI, MACD, BB, EMA, SMA, ATR, OBV, ...)
  +-- MI Feature Selection: top 80~120 from ~120 candidates
  |
  v
  135 features per coin per bar
```

**Top 5 predictive features** (by Mutual Information):
1. VWAP (volume-weighted average price)
2. Hurst exponent (trend persistence)
3. Parkinson volatility (high-low based)
4. SMA-50 (50-bar moving average)
5. OBV (on-balance volume)

**Removed features** (MI = 0, confirmed noise):
- All macro data (Gold, VIX, DXY, US10Y, S&P500, Fear&Greed, DeFi TVL)
- Hilbert transform features

### Prediction Model

```
Stage 1: Trade or No-Trade? (binary)
  +-- 7-model ensemble votes
  |   LightGBM (GPU) + XGBoost (GPU) + CatBoost (GPU)
  |   BalancedRandomForest + ExtraTrees + HistGradientBoosting
  |   + Stacking (LogisticRegression meta-learner)
  +-- P(trade) >= threshold --> proceed
  +-- DOT threshold: 0.50, ADA threshold: 0.52
  |
  v
Regime Filter: is market state acceptable?
  +-- 4 states: TREND_UP, TREND_DOWN, RANGE_LOW, RANGE_HIGH
  +-- RANGE_LOW --> BLOCKED (no entry)
  +-- ADA also blocks UNKNOWN regime
  |
  v
Stage 2: Long or Short? (binary)
  +-- Same ensemble architecture
  +-- P(UP) > 0.5 --> BUY, else SELL
```

### Position Management

```
Entry: close price at signal bar
Exit: Triple Barrier (ATR-14 based, asymmetric R:R)

DOT:
  Take Profit = entry + 3.0 * ATR  (R:R = 5.0)
  Stop Loss   = entry - 0.6 * ATR
  Time Limit  = 72 hours (18 bars)

ADA:
  Take Profit = entry + 3.0 * ATR  (R:R = 3.75)
  Stop Loss   = entry - 0.8 * ATR
  Time Limit  = 72 hours (18 bars)

Rules:
  - 1 position per coin (non-overlapping)
  - Same-bar TP+SL hit --> SL wins (conservative)
  - Sizing: 0.5% equity risk per trade
```

### Why It Makes Money

The strategy profits from **payoff geometry**, not prediction accuracy.

```
Win:  +2.4% equity (TP hit)
Loss: -0.5% equity (SL hit)
Ratio: 1 win = 4.8 losses recovered

Break-even win rate: 17~21% (depending on costs)
Actual win rate: 33~50%
Safety margin: 12~33 percentage points above BEP
```

Win rate 43% with R:R 5:1 = positive expected value.

## Cost Model

Bybit VIP0 Perpetual, 5-component breakdown:

| Component | Rate | Notes |
|-----------|------|-------|
| Entry fee | 0.02% | Post-Only (maker) |
| Exit fee (TP) | 0.02% | Limit order (maker) |
| Exit fee (SL) | 0.055% | Market order (taker) |
| Slippage | 0.03~0.05% | Entry + exit |
| Funding | 0.01%/8h | Per holding period |
| Miss-fill | 15% reject | Post-Only rejection penalty |

Realized cost share: DOT 13%, ADA 9% of gross PnL.

## Risk Management

### Per-Trade
- Max loss: 0.5% equity (fixed by stop-distance sizing)
- Stop loss on every position (no exceptions)
- Max holding: 72 hours (forced exit)

### Portfolio
- Daily loss limit: -2% equity
- Weekly loss limit: -5% equity
- Consecutive loss kill switch: 3 losses
- Max exposure: 80% of equity

### Pre-Entry Filters
- S1 threshold filters 50~70% of signals
- Regime filter blocks low-volatility sideways markets
- Non-overlapping positions prevent overexposure

## Validation Results

### Frozen OOS (v3.4, 8-week out-of-sample)

| Coin | Trades | Win% | Avg PnL | Total | MDD | Sharpe |
|------|--------|------|---------|-------|-----|--------|
| DOT | 21 | 43.5% | +0.89% | +18.72% | 1.83% | 8.6 |
| ADA | 12 | 50.0% | +0.77% | +9.26% | 0.58% | 8.8 |
| **Portfolio** | **33** | **45.5%** | **+0.85%** | **+27.99%** | **1.83%** | **8.6** |

### Backtest (5M KRW seed, 8 weeks)

```
Initial:   5,000,000 KRW
Final:     6,328,402 KRW
Profit:   +1,328,402 KRW (+26.57%)
MDD:       2.94%
Trades:    42 (18W / 24L)
```

### Regime Performance

| Regime | Trades | Avg PnL | Total |
|--------|--------|---------|-------|
| TREND_DOWN | 16 | +0.73% | +11.69% |
| TREND_UP | 12 | +0.26% | +3.15% |
| RANGE_HIGH | 2 | +2.43% | +4.85% |
| RANGE_LOW | 0 | blocked | blocked |

## Project Structure

```
Ru_trading/
|-- CLAUDE.md                    # Project rules & agent memory pointer
|-- config/
|   |-- frozen_params_v3_4.yaml  # Production params (FROZEN)
|   |-- live_promotion_criteria.yaml
|   +-- settings.yaml            # Central config
|
|-- src/
|   |-- data/crawlers/           # OHLCV, signal features, macro
|   |-- models/                  # Ensemble, labeling, regime filter
|   |-- execution/               # Cost model, exchange, risk, state machine
|   |-- evaluation/              # Trade-level EV, walk-forward, audit
|   |-- signals/                 # Signal contract, policy
|   +-- utils/                   # Config, logging, feature policy
|
|-- run_optimize_v3_netev.py     # Walk-forward optimizer (v3.2)
|-- run_frozen_oos_v2.py         # Frozen OOS validator
|-- run_paper_v3_4.py            # Paper trading (2-week forward)
|-- run_paper_sim_5m.py          # Backtest simulation
|-- run_strategy_diagnosis.py    # 4-step strategy analysis
|-- run_live_engine.py           # Live execution launcher
|
|-- docs/
|   |-- INDEX.md                 # Development memory index
|   |-- devlog/                  # Daily development logs
|   +-- checkpoints/             # Decision snapshots
|
+-- data/reports/                # OOS results, trade CSVs, diagnostics
```

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| DL (MOMENT) excluded | Training time too long, ML-only is practical |
| Summary EV abandoned | Overestimated 2-3x vs trade-level simulation |
| XRP suspended | S1 model cannot generate confident signals |
| Macro features removed | MI = 0 across all coins (confirmed noise) |
| Per-coin regime policy | DOT profits in UNKNOWN, ADA loses in UNKNOWN |
| Trade-level evaluation | Bar-by-bar barrier simulation replaces formula |

## Version History

| Version | Date | Change |
|---------|------|--------|
| v3.1 | 03-16 | Net EV optimizer, 9,390 evals |
| v3.2 | 03-17 | Trade-level evaluation, feature cleanup |
| v3.3 | 03-17 | ADA: threshold 0.52, k_lower 0.8, UNKNOWN blocked |
| **v3.4** | **03-17** | **Per-coin regime policy, final production config** |

## Tech Stack

- Python 3.10+
- LightGBM, XGBoost, CatBoost (GPU)
- scikit-learn, imbalanced-learn
- yfinance (data source)
- PyWavelets, SciPy (signal processing)
- ccxt (exchange adapter, blocked by firewall)
- NVIDIA RTX 3090 24GB

## Live Promotion Criteria

Paper trading must pass ALL before live deployment:

- [ ] 15+ trades in 2 weeks
- [ ] Net EV > 0
- [ ] MDD < 3%
- [ ] Cost share < 20%
- [ ] No single trade > 40% of total PnL
- [ ] Consecutive losses < 5
- [ ] Trades in 2+ regime types

## License

Private repository. All rights reserved.
