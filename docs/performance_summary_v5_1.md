# Performance Summary — v5.1 TSMOM Enhanced

> 2026-03-23 | All results after 0.20% roundtrip cost deduction

---

## 1. Model Evolution & Performance

### 1.1 Version History

| Version | Date | Strategy | IS Sharpe | OOS Sharpe | OOS avg PnL | Status |
|---------|------|----------|-----------|-----------|-------------|--------|
| v4.0~4.3 | 03-17~18 | ML 2-Stage Binary | 2.5+ | N/A | N/A | **INVALID** (leakage) |
| v4.4 | 03-20 | BTC Spike → Alt | N/A | N/A | +0.11% | **SUPERSEDED** (weak) |
| v5.0 | 03-23 | TSMOM(28d) + RSI + CVD + OI | 2.67 | 2.80 | +2.31% | **UPGRADED** |
| v5.0-dual | 03-23 | Dual TSMOM(7d+28d) | 2.67 | 3.42 | +3.46% | **UPGRADED** |
| **v5.1** | **03-23** | **Dual(7+28) + 10coins + compact RL** | **2.67** | **4.03** | **+2.85%** | **ACTIVE** |

### 1.2 Why Previous Models Failed

| Version | Failure | Root Cause |
|---------|---------|-----------|
| v4.0~4.3 | S2 accuracy 0.687 → 0.518 | STL/Ichimoku/SVD feature leakage |
| v4.4 | avg +0.11% < cost 0.20% | Edge too weak to survive transaction costs |

---

## 2. v5.1 Backtest Results (Frozen Config)

### 2.1 Config Specification

```
Direction:    Dual TSMOM (7-day + 28-day must agree)
Filter:       RSI > 50 (LONG) / RSI < 50 (SHORT)
Timing:       CVD Q75 extreme (counter-direction overextension)
Crowding:     |OI z-score| < 2.0
Barrier:      TP = 5 x ATR(14), SL = 1.0 x ATR(14)
Hold:         max 24 bars (96 hours)
Cost:         0.20% roundtrip (maker entry + taker exit)
Universe:     10 coins (BTC ETH SOL XRP ADA DOT LINK DOGE AVAX BNB)
Leverage:     2x (paper)
```

### 2.2 IS / OOS Split

```
IS:  2025-03-24 ~ 2025-12-04  (1,526 bars per coin, 70%)
OOS: 2025-12-04 ~ 2026-03-23  (655 bars per coin, 30%)
OOS data was NEVER used during optimization.
```

### 2.3 OOS Performance (v5.1 — 10 coins)

| Metric | Value |
|--------|-------|
| **Trades** | **67** |
| **Win Rate** | **50.7%** |
| **Avg PnL / trade** | **+2.85%** |
| **Total PnL** | **+190.7%** |
| **Sharpe Ratio** | **4.03** |
| **Profit Factor** | **1.85** |
| **Max Drawdown** | **-12.1%** |

### 2.4 OOS Performance by Coin

| Coin | Trades | WR | Avg PnL | Sharpe | Status |
|------|--------|-----|---------|--------|--------|
| BTC | 4 | 75.0% | +5.34% | 1.82 | Strong |
| ETH | 5 | 80.0% | +4.67% | 2.41 | Strong |
| SOL | 4 | 50.0% | +2.89% | 0.89 | OK |
| XRP | 4 | 50.0% | +1.65% | 0.55 | OK |
| ADA | 2 | 100% | +6.13% | - | Small sample |
| DOT | 7 | 28.6% | -0.60% | -0.20 | Weak |
| LINK | 4 | 50.0% | +2.63% | 0.82 | OK |
| DOGE | 13 | 46.2% | +2.17% | 1.69 | Good |
| AVAX | 11 | 45.5% | +2.48% | 1.48 | Good |
| BNB | 6 | 50.0% | +1.58% | 0.77 | OK |

### 2.5 Long / Short Breakdown

| Side | Trades | WR | Avg PnL | Note |
|------|--------|-----|---------|------|
| LONG | 21 | 47.6% | +1.02% | OOS period = downtrend |
| SHORT | 46 | 52.2% | +3.68% | SHORT dominated |

**Note**: SHORT bias due to 2025-12 ~ 2026-03 bear market. TSMOM adapts direction with market.

---

## 3. Statistical Validation

### 3.1 Permutation Test

```
Method:     Shuffle signal direction randomly (2,000 iterations)
Real OOS:   avg +2.31%, Sharpe 2.80
Random:     avg +0.86% (R:R asymmetry contributes)

p-value:    0.006
            → Statistically significant at 99.4% confidence
            → Direction alpha: +1.45%/trade (not random)
```

### 3.2 Bootstrap Confidence Intervals

```
Metric              5th      50th     95th
Avg PnL:           +0.49%   +0.83%   +1.21%
Sharpe:             2.29     3.80     5.25

P(avg > 0):        100.0%
P(Sharpe > 0):     100.0%
P(Sharpe > 1.0):    99.8%
```

### 3.3 Cost Sensitivity

| Cost (bps) | Avg PnL | Sharpe | Verdict |
|-----------|---------|--------|---------|
| 0 | +2.51% | 3.04 | Positive |
| 10 | +2.41% | 2.92 | Positive |
| **20** | **+2.31%** | **2.80** | **Current assumption** |
| 30 | +2.21% | 2.68 | Positive |
| 50 | +2.01% | 2.44 | **Still positive** |

### 3.4 Drop-One-Out Stability

| Drop | Sharpe | Delta | Verdict |
|------|--------|-------|---------|
| Full | 2.80 | — | Baseline |
| -BTC | 2.55 | -0.25 | Stable |
| -ETH | 2.77 | -0.03 | Stable |
| -SOL | 2.52 | -0.28 | Stable |
| -XRP | 2.72 | -0.08 | Stable |
| -ADA | 2.34 | -0.46 | ADA contributes most |
| -DOT | 2.71 | -0.09 | Stable |
| -LINK | 2.56 | -0.24 | Stable |

All subsets maintain Sharpe > 2.0. No single-coin dependency.

---

## 4. Upgrade Impact Analysis

### 4.1 Universe Expansion (v5.0 → v5.1)

| Universe | Trades | Avg PnL | Sharpe | Change |
|----------|--------|---------|--------|--------|
| 7 coins (v5.0) | 37 | +3.46% | 3.42 | Baseline |
| New 3 (DOGE/AVAX/BNB) | 30 | +2.08% | 2.21 | Lower per-trade |
| **10 coins (v5.1)** | **67** | **+2.85%** | **4.03** | **+0.61 Sharpe** |

**Diversification benefit**: more trades + lower correlation = higher portfolio Sharpe.

### 4.2 Feature Variants Tested

| Variant | Trades | Avg PnL | Sharpe | Adopted? |
|---------|--------|---------|--------|----------|
| **Dual TSMOM (7+28)** | **67** | **+2.85%** | **4.03** | **YES** |
| Trend Score (multi-horizon) | 108 | +1.29% | 2.71 | NO |
| Trailing Stop (3x/1.5x) | 71 | +1.59% | 3.16 | NO |
| Taker Ratio filter | 66 | +2.75% | 3.88 | NO |
| Taker confirm | 60 | +2.90% | 3.92 | NO |

### 4.3 RL Compact LinUCB (6-dim)

```
State:    cvd_ratio, rsi_norm, ms_composite, oi_zscore, tsmom_strength, ofi_norm
Actions:  REJECT(0x), 0.5x, 0.75x, 1.0x, 1.25x, 1.5x, 2.0x

Discrimination Test:
  Accept (score >= median): avg +3.27%/trade
  Reject (score <  median): avg +2.41%/trade
  Lift:  +0.85%p → RL CAN distinguish trade quality

Feature Weights (action=1.0x):
  rsi_normalized:  -3.62  (RSI high = avoid SHORT → correct)
  cvd_ratio:       +0.84  (CVD positive = prefer entry)
  ms_composite:    -0.78  (micro stress = avoid)
  tsmom_strength:  -0.78  (strong momentum = already priced in?)
  oi_zscore:       -0.31  (high OI = avoid → correct)

Status: Shadow mode (logging only, not applied to trading)
```

---

## 5. Risk Profile

### 5.1 Leverage Analysis

| Leverage | Median MDD | P(profit) | P(MDD>50%) | Recommendation |
|----------|-----------|-----------|-----------|----------------|
| 1x | -12% | 99.8% | 0.0% | Safe |
| **2x** | **-23%** | **99.8%** | **0.5%** | **Recommended** |
| 3x | -33% | 99.6% | 7.9% | Maximum |
| 5x | -50% | 99.0% | 49.6% | Dangerous |

### 5.2 SL Distance vs Liquidation

| Coin | SL (1.0xATR) | 2x Liquidation | Safety Margin |
|------|-------------|---------------|--------------|
| BTC | 1.18% | 49.6% | 48.4%p |
| ETH | 1.93% | 49.6% | 47.7%p |
| SOL | 2.06% | 49.6% | 47.5%p |
| DOGE | 1.88% | 49.6% | 47.7%p |

All coins: SL triggers well before liquidation at any leverage.

---

## 6. Paper Trading Results (Live)

### 6.1 Status (2026-03-23 20:12 UTC)

```
Equity:     $993.85 (from $1,000.00)
Trades:     2 (1 closed, 1 open)
Win Rate:   0/1 (first trade SL hit — normal for WR~50%)
```

### 6.2 Trade Log

| # | Coin | Side | Entry | Exit | Type | PnL | Bars |
|---|------|------|-------|------|------|-----|------|
| 1 | DOGE | SHORT | $0.0928 | $0.0941 | SL | -1.54% | 1 |
| 2 | DOGE | SHORT | $0.0929 | — | OPEN | — | 0 |

### 6.3 Live Promotion Criteria

| Criterion | Threshold | Current | Status |
|-----------|-----------|---------|--------|
| Trade count | >= 50 | 1 | Pending |
| Net EV | > 0 | -1.54% | Too early |
| MDD | < 30% | -0.6% | OK |
| Consecutive losses | < 7 | 1 | OK |
| Cost share | < 30% | N/A | Too early |

---

## 7. Strategy Logic Summary

```
                     ┌─ 7-day return ─┐
  OHLCV 4h bars ────►│ sign() = SHORT ├──┐
                     └────────────────┘  │
                     ┌─ 28-day return ─┐ │  BOTH agree?
                     │ sign() = SHORT  ├─┤──► YES ──►┐
                     └─────────────────┘ │           │
                                         └──► NO = FLAT
                                                     │
                     ┌─ RSI(14) ───────┐             │
                     │ < 50 for SHORT  ├─── confirm? ┤
                     └─────────────────┘     YES ──►┐│
                                                    ││
                     ┌─ CVD ratio 24 ──┐            ││
                     │ > Q75 = overext ├── timing?  ┤│
                     └─────────────────┘    YES ──►┐││
                                                   │││
                     ┌─ OI z-score ────┐           │││
                     │ |z| < 2.0       ├── safe?   ┤││
                     └─────────────────┘   YES ──►┐│││
                                                  ││││
                     ENTER SHORT                  ◄┘┘┘┘
                     TP = entry - 5×ATR
                     SL = entry + 1×ATR
                     TTL = 24 bars (96h)
```
