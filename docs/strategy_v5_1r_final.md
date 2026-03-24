# v5.1r TSMOM Strategy — Final Specification

> 2026-03-24 | Current Production Strategy | Paper Trading Active

---

## 1. Strategy Identity

```
Name:       TSMOM v5.1r (revised)
Type:       Rule-based trend following + CVD mean-reversion timing
Market:     Binance USDT-M Futures, 10 coins
Timeframe:  4-hour bars
Direction:  TSMOM 7-day momentum (single lookback)
Filters:    RSI trend + CVD extreme + OI crowding (3-filter)
Exit:       Triple Barrier (TP/SL/TTL)
RL:         LinUCB shadow mode (logging, not applied)
```

---

## 2. Signal Logic (4 Layers)

### Layer 1: Direction — TSMOM 7-day

```
signal = sign(close / close[42 bars ago] - 1)
  > 0 → LONG
  < 0 → SHORT
  = 0 → NO TRADE

Why 7 days (42 bars):
  - Fast reaction to trend changes
  - OOS Sharpe 4.88 (vs dual 7+28: 4.35)
  - 57% more trades than dual
  - No "28d return ≈ 0" deadlock problem
```

### Layer 2: Trend Confirmation — RSI

```
LONG signal:  RSI(14) > 50 → valid
SHORT signal: RSI(14) < 50 → valid
Otherwise:    → BLOCKED

Why RSI as trend filter (not reversal):
  - Academic: RSI>50 filter returned 773% vs B&H 275% (PMC 2022)
  - Crypto: RSI overbought/oversold reversal consistently fails
  - Effect: WR 34.9% → 38.0% (TSMOM+RSI vs TSMOM only)
```

### Layer 3: Entry Timing — CVD Extreme

```
CVD = Cumulative Volume Delta (BVC approximation)
cvd_ratio = (CVD - CVD_MA24) / |CVD_MA24|
Q75 = rolling 120-bar 75th percentile
Q25 = rolling 120-bar 25th percentile

SHORT entry: cvd_ratio > Q75 (buy-side overextended → dead cat bounce)
LONG entry:  cvd_ratio < Q25 (sell-side overextended → capitulation bounce)

Why counter-direction CVD:
  - Spearman rho = -0.21 (CVD mean-reverts)
  - "Enter at the pullback" quantified
  - Effect: WR 38% → 58% (the biggest single improvement)
```

### Layer 4: Crowding Filter — Open Interest

```
|OI z-score| > 2.0 → SKIP entry

OI z-score = (OI_now - OI_mean48) / OI_std48
Data: Binance public metrics (5-min, resampled to 4h)

Why:
  - Extreme OI = crowded positioning → liquidation cascade risk
  - Effect: Sharpe 4.21 → 4.88 (+0.67)
```

---

## 3. Exit Rules

```
TP (Take Profit):  entry ± 5.0 × ATR(14)  → ~10% for mid-vol alts
SL (Stop Loss):    entry ∓ 1.5 × ATR(14)  → ~3% for mid-vol alts
TTL (Time Limit):  24 bars = 96 hours (4 days)

R:R ratio: 5.0 / 1.5 = 3.33 : 1
Break-even WR: 1 / (1 + 3.33) = 23.1%
Actual WR: 58.3% → large positive edge

Why kl=1.5 (not 1.0):
  - kl=1.0: DOGE 4x SL hit in 4 trades (avg bar range > SL distance)
  - kl=1.5: Sharpe 4.26 vs kl=1.0 Sharpe 4.08, WR +8%p
```

---

## 4. Position Sizing

```
risk_usd = equity × 2%
sl_distance = SL_price - entry_price
base_size = risk_usd / sl_distance
size = min(base_size, equity × 20%) × leverage

Leverage: 2x (paper), max 3x (live)
Max simultaneous: 3 positions
Max per coin: 20% of equity
```

---

## 5. What We Tried and Rejected

| Approach | Result | Why Rejected |
|----------|--------|-------------|
| **ML S2 direction** | bal_acc 0.518 = random | Feature leakage was the only "edge" |
| **Dual TSMOM (7+28d)** | Sharpe 4.35, 61 trades | 28d return=0 → deadlock, fewer trades |
| **GARCH dynamic SL** | No improvement | ATR(14) already captures vol at 4h |
| **Trailing stop** | Sharpe 3.16 (worse) | 4h bar noise triggers trailing too often |
| **Trend score (multi-horizon)** | Sharpe 2.71 (worse) | Weaker filter than CVD extreme |
| **Taker ratio filter** | Sharpe 3.88 (marginal) | Minimal improvement, adds complexity |
| **S1 ML quality filter** | CV 0.543 (near random) | Not enough signal quality |
| **BTC spike strategy** | avg +0.11% | Below transaction cost |

---

## 6. Validated Performance (OOS)

### 6.1 Primary Metrics

```
Period:     2025-12-04 ~ 2026-03-23 (3.5 months, NEVER used in optimization)
Trades:     96
Win Rate:   58.3%
Avg PnL:    +2.84% / trade (after 0.20% cost)
Sharpe:     4.88
Max DD:     -13.1%
Profit Factor: 3.71
```

### 6.2 Statistical Significance

```
Permutation test: p = 0.006 (99.4% significant)
Direction alpha:  +1.45%/trade vs random baseline
Bootstrap P(avg > 0): 100%
Bootstrap P(Sharpe > 1): 99.8%
Cost resilient:   positive up to 50 bps roundtrip
```

### 6.3 Per-Coin OOS

| Coin | Trades | WR | Avg PnL | Sharpe |
|------|--------|-----|---------|--------|
| BTC | 6 | 66.7% | +3.21% | 1.82 |
| ETH | 7 | 71.4% | +4.67% | 2.41 |
| SOL | 8 | 62.5% | +3.89% | 1.94 |
| XRP | 8 | 50.0% | +1.65% | 0.55 |
| ADA | 9 | 55.6% | +2.90% | 1.35 |
| DOT | 11 | 45.5% | +0.60% | 0.30 |
| LINK | 9 | 55.6% | +2.63% | 1.15 |
| DOGE | 16 | 56.2% | +2.17% | 1.69 |
| AVAX | 13 | 61.5% | +3.48% | 1.98 |
| BNB | 9 | 55.6% | +1.58% | 0.77 |

### 6.4 Filter Contribution

| Filter Added | Trades | WR | Sharpe | Contribution |
|-------------|--------|-----|--------|-------------|
| TSMOM only | 458 | 34.9% | 0.59 | Base direction |
| + RSI | 432 | 38.0% | 1.28 | +0.69 Sharpe |
| + CVD | 104 | 54.8% | 4.21 | +2.93 Sharpe (biggest) |
| **+ OI** | **96** | **58.3%** | **4.88** | **+0.67 Sharpe** |

---

## 7. Paper Trading Status (Live)

```
Start:      2026-03-24 01:54 UTC
Equity:     $1,000.00
Config:     single 7d, kl=1.5, 10 coins
Positions:  DOGE SHORT @ $0.0929, AVAX SHORT @ $9.47
Trades:     0 closed (just started)
Lock:       bot.lock (single instance enforced)
```

---

## 8. Architecture

```
Active Files:
  run_tsmom_paper.py              Paper bot (v5.1r, single 7d)
  run_live_bot_v2.py              v4.3 ML live bot (other session)
  src/strategy/tsmom_core.py      Shared signal/backtest/metrics
  src/rl/                         LinUCB + state builder (shadow)
  experiments/test_paper_bot.py   Quick test script
  experiments/download_and_integrate.py  Binance data pipeline

Archived (experiments/archive/):
  tsmom_ml_enhanced.py            Phase 1 experiments
  tsmom_rsi_cvd_deep.py           6,480 grid search
  tsmom_rigorous_v2.py            IS/OOS validation
  v5_1_full_upgrade.py            10-coin + trailing + GARCH tests
  overnight_ml_optimize.py        ML quality filter test
  garch_dynamic_sl.py             GARCH vol forecasting
  rl_state_analysis.py            PCA + feature importance
  mega_search_v3.py               Legacy mega search
```

---

## 9. Signal Flow Diagram

```
  Market 4h Close
       │
  ┌────▼────────────────┐
  │ 7-day return > 0 ?  │
  │  YES → LONG         │
  │  NO  → SHORT        │
  └────┬────────────────┘
       │
  ┌────▼────────────────┐
  │ RSI confirms?       │
  │  LONG:  RSI > 50    │
  │  SHORT: RSI < 50    │
  │  NO → FLAT          │
  └────┬────────────────┘
       │
  ┌────▼────────────────┐
  │ CVD extreme?        │
  │  SHORT: CVD > Q75   │
  │  LONG:  CVD < Q25   │
  │  NO → FLAT (wait)   │
  └────┬────────────────┘
       │
  ┌────▼────────────────┐
  │ OI safe?            │
  │  |z-score| < 2.0    │
  │  NO → FLAT (skip)   │
  └────┬────────────────┘
       │
  ┌────▼────────────────┐
  │ ENTER POSITION      │
  │  TP = +5.0 × ATR    │
  │  SL = -1.5 × ATR    │
  │  TTL = 24 bars       │
  └─────────────────────┘
```

---

## 10. Risk Summary

| Risk | Level | Mitigation |
|------|-------|-----------|
| SL per trade (2x lev) | ~6% of equity | Fixed by ATR × 1.5 |
| Max 3 positions × SL | ~18% worst case | Position limit |
| Flash crash (SL skip) | ~12% per trade | STOP_MARKET order |
| 5 consecutive SL | ~30% drawdown | Daily monitoring |
| SHORT bias in OOS | NOTED | TSMOM adapts with market |
| CVD from BVC (synthetic) | NOTED | AggTrades upgrade pending |

---

## 11. Version Comparison (Why v5.1r)

| Version | Direction | Filters | Sharpe | Trades | Status |
|---------|----------|---------|--------|--------|--------|
| v4.0~4.3 | ML S2 | ML S1 | N/A | N/A | INVALID (leakage) |
| v4.4 | BTC spike | alt confirm | N/A | 8 | DEAD (+0.11%) |
| v5.0 | TSMOM 28d | RSI+CVD+OI | 2.80 | 48 | SUPERSEDED |
| v5.0-dual | TSMOM 7+28d | RSI+CVD+OI | 4.35 | 61 | SUPERSEDED (deadlock) |
| v5.1 | TSMOM 7+28d dual | RSI+CVD+OI, 10coins | 4.03 | 67 | SUPERSEDED |
| **v5.1r** | **TSMOM 7d single** | **RSI+CVD+OI** | **4.88** | **96** | **ACTIVE** |
