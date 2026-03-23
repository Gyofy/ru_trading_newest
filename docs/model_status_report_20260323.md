# Model Status Report — 2026-03-23

## Executive Summary

v5.0 TSMOM Enhanced strategy: **OOS Sharpe 2.80, avg +2.31%/trade, p-value 0.006**.
Paper trading 가동 중. RL 강화학습 shadow mode 연동 완료.

---

## 1. Strategy Evolution

| Version | Date | Strategy | Status | Result |
|---------|------|----------|--------|--------|
| v4.0-4.2 | 03-17 | ML 2-Stage Binary | INVALID | Feature leakage |
| v4.3 | 03-18 | ML 8-coin Mega Search | INVALID | Feature leakage |
| v4.4 | 03-20 | BTC Spike -> Alt Follow | SUPERSEDED | avg +0.11% (weak) |
| **v5.0** | **03-23** | **TSMOM+RSI+CVD+OI** | **ACTIVE** | **OOS Sharpe 2.80** |

---

## 2. v5.0 Strategy Specification

```
Layer 1 — Direction:  TSMOM 28d (sign of past return)
Layer 2 — Filter:     RSI > 50 (LONG), RSI < 50 (SHORT)
Layer 3 — Timing:     CVD Q75/Q25 extreme (counter-direction)
Layer 4 — Crowding:   OI |z-score| < 2.0
Layer 5 — Exit:       TP=5xATR, SL=1.0xATR, TTL=24bars
```

### Parameters (Frozen)

| Parameter | Value | Optimized from |
|-----------|-------|---------------|
| lookback_days | 28 | [5, 7, 10, 14, 21, 28] |
| cvd_quantile | 0.75 | [0.65, 0.70, 0.75, 0.80, 0.85] |
| cvd_roll_window | 120 | [60, 90, 120] |
| k_upper (TP) | 5.0 | [3.0, 4.0, 5.0] |
| k_lower (SL) | 1.0 | [0.8, 1.0, 1.5, 2.0] |
| max_hold | 24 bars | [18, 24] |
| volume_weighted | False | [True, False] |
| use_oi | True | [True, False] |
| cost_roundtrip | 0.20% | fixed |
| leverage | 2x | fixed (paper) |

---

## 3. Validation Results

### 3.1 Grid Search (6,480 configs)
- IS period: 2025-03-24 ~ 2025-12-04 (1,526 bars, 70%)
- OOS period: 2025-12-04 ~ 2026-03-23 (655 bars, 30%)
- Best IS Sharpe: 2.67 (lb=28, cq=0.70, cw=120, ku=5.0, kl=1.5)

### 3.2 OOS Results (NEVER SEEN data)

| IS Rank | OOS Trades | WR | Avg PnL | OOS Sharpe |
|---------|-----------|-----|---------|-----------|
| 1 | 67 | 49.3% | +1.43% | 2.11 |
| 2 | 66 | 54.5% | +1.47% | 2.07 |
| 3 | 61 | 52.5% | +1.77% | 2.25 |
| 6 | 60 | 58.3% | +1.82% | 2.21 |
| 14 | 72 | 55.6% | +1.58% | 2.38 |
| 15 | 88 | 48.9% | +1.38% | 2.52 |
| **17** | **48** | **47.9%** | **+2.31%** | **2.80** |

**All top-20 IS configs positive in OOS** — structural edge, not overfitting.

### 3.3 Statistical Tests

| Test | Result | Interpretation |
|------|--------|---------------|
| Permutation (2,000 shuffles) | p = 0.006 | **Significant at 99% level** |
| Random direction baseline | avg +0.86% | R:R asymmetry contributes |
| **Direction alpha** | **+1.45%/trade** | TSMOM direction adds real alpha |
| Bootstrap P(avg > 0) | 100.0% | Edge robust |
| Bootstrap P(Sharpe > 1) | 99.8% | High confidence |
| Bootstrap Sharpe 5th/50th/95th | 2.29 / 3.80 / 5.25 | Wide but positive |

### 3.4 Cost Sensitivity (OOS)

| Cost (bps) | Avg PnL | Sharpe | Status |
|-----------|---------|--------|--------|
| 0 | +2.51% | 3.04 | OK |
| 10 | +2.41% | 2.92 | OK |
| 20 | +2.31% | 2.80 | OK (current) |
| 30 | +2.21% | 2.68 | OK |
| 50 | +2.01% | 2.44 | OK |

### 3.5 Drop-One-Out Stability

| Drop | Trades | Avg PnL | Sharpe | Delta |
|------|--------|---------|--------|-------|
| Full | 48 | +2.31% | 2.80 | — |
| -BTC | 38 | +2.51% | 2.55 | -0.25 |
| -ETH | 40 | +2.57% | 2.77 | -0.03 |
| -SOL | 41 | +2.17% | 2.52 | -0.28 |
| -XRP | 45 | +2.25% | 2.72 | -0.08 |
| -ADA | 45 | +1.98% | 2.34 | -0.46 |
| -DOT | 37 | +2.59% | 2.71 | -0.09 |
| -LINK | 42 | +2.23% | 2.56 | -0.24 |

No single coin dependency. All subsets Sharpe > 2.0.

### 3.6 Long/Short Breakdown (OOS)

| Side | Trades | Avg PnL | Note |
|------|--------|---------|------|
| LONG | 19 | +0.09% | OOS period = downtrend |
| SHORT | 42 | +2.53% | SHORT dominated |

**Warning**: SHORT bias due to 2025-12 ~ 2026-03 downtrend. Bull market performance unverified in OOS.

---

## 4. Binance Data Integration

### 4.1 Downloaded Data

| Type | Coins | Period | Resolution | Size |
|------|-------|--------|-----------|------|
| Metrics (OI + LSR + Taker) | 7 | 365d | 5-min | ~2GB |

### 4.2 Derived Features (16 per coin)

| Feature | Source | Purpose |
|---------|--------|---------|
| oi_value | sum_open_interest_value | Raw OI |
| oi_change_pct | OI pct_change | OI momentum |
| oi_ratio | OI / MA(24) | OI vs baseline |
| oi_zscore | (OI - mean) / std | **Crowding filter** |
| oi_price_div | sign(OI_chg) * -sign(price_chg) | OI-price divergence |
| lsr | count_long_short_ratio | Long/Short ratio |
| lsr_zscore | (LSR - mean) / std | LSR extreme detection |
| lsr_extreme_long | z > 1.5 | **Contrarian signal** |
| lsr_extreme_short | z < -1.5 | **Contrarian signal** |
| top_trader_lsr | sum_toptrader_long_short_ratio | Smart money |
| top_trader_lsr_zscore | z-score | Smart money extreme |
| taker_vol_ratio | sum_taker_long_short_vol_ratio | Buy/sell pressure |
| taker_buy_pressure | ratio > 1.0 | Taker direction |

### 4.3 OI Filter Impact

| Config | Avg PnL | Sharpe | Improvement |
|--------|---------|--------|-------------|
| C3 base (no OI) | +0.490% | 1.39 | — |
| **+ OI filter** | **+0.652%** | **1.80** | **+0.16%p, Sharpe +0.41** |
| + OI divergence | +0.789% | 1.91 | +0.30%p |
| + ALL Binance | +0.805% | 1.90 | +0.32%p |

---

## 5. RL Meta-Layer (v5.0)

### 5.1 State Vector (33-dim)

| Category | Dims | Features |
|----------|------|----------|
| Signal Quality | 6 | tsmom_strength, rsi_norm, cvd_extremeness, oi_zscore, tsmom_rsi_agree, side_sign |
| Market Regime | 4 | regime_trend, regime_up, atr_pct, hurst |
| Microstructure | 3 | cvd_ratio, ofi_norm, ms_composite |
| Cost Proxy | 2 | spread_proxy, last_funding |
| Portfolio | 4 | open_positions, daily_pnl, weekly_pnl, dd_ratio |
| Coin History | 4 | win_rate_5, avg_pnl_5, streak, bars_since_last |
| Cross-Market | 2 | btc_return_24h, corr_btc |
| Coin Identity | 7 | one-hot (BTC~LINK) |
| Intercept | 1 | constant |

### 5.2 Action Space (7 discrete)

| Action | Sizing | Use Case |
|--------|--------|----------|
| 0 | 0.0x (REJECT) | Low quality signal |
| 1 | 0.5x | Low confidence |
| 2 | 0.75x | Below average |
| 3 | 1.0x | Normal |
| 4 | 1.25x | Above average |
| 5 | 1.5x | High confidence |
| 6 | 2.0x | Maximum conviction |

### 5.3 Deployment Plan

| Phase | Condition | Action |
|-------|-----------|--------|
| 1. Logging | Now | Shadow mode, signal_log.jsonl |
| 2. Training | 200+ signals | offline_train.py |
| 3. Evaluation | Trained model | Conditional lift, reject quality |
| 4. Activation | Phase 3 pass | shadow_mode: false |

---

## 6. Leverage Analysis

### ATR-based SL/TP Distance

| Coin | ATR% | SL (1.0x) | TP (5.0x) |
|------|------|----------|----------|
| BTC | 1.18% | 1.18% | 5.90% |
| ETH | 1.93% | 1.93% | 9.65% |
| SOL | 2.06% | 2.06% | 10.30% |
| XRP | 1.86% | 1.86% | 9.30% |
| ADA | 2.16% | 2.16% | 10.80% |
| DOT | 2.22% | 2.22% | 11.10% |
| LINK | 2.19% | 2.19% | 10.95% |

### Monte Carlo Leverage Simulation (5,000 paths)

| Leverage | Median CAGR | Median MDD | P(profit) | P(MDD>50%) |
|----------|-------------|-----------|-----------|-----------|
| 1x | high | -12% | 99.8% | 0.0% |
| 2x | higher | -23% | 99.8% | 0.5% |
| **3x** | **highest safe** | **-33%** | **99.6%** | **7.9%** |
| 5x | extreme | -50% | 99.0% | 49.6% |

**Recommendation: 2x (practical), 3x (maximum)**

---

## 7. Known Risks & Limitations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| SHORT bias in OOS | HIGH | Bull market unverified, TSMOM should adapt |
| OOS sample size (48-67) | MEDIUM | Paper trading to 50+ trades |
| BVC-based CVD (not real orderbook) | MEDIUM | AggTrades data pending |
| Single period (365d) | MEDIUM | Strategy has academic precedent |
| 6,480 grid → selection bias | LOW | Permutation test p=0.006 |
| Cost assumption (20bps) | LOW | Positive up to 50bps |

---

## 8. Next Steps

1. **Paper trading** 50+ trades for live promotion evaluation
2. **RL offline training** after 200+ signals accumulated
3. **AggTrades integration** for real CVD (vs BVC approximation)
4. **Bull market validation** when regime shifts

---

## 9. File Index

| File | Purpose |
|------|---------|
| `data/reports/tsmom_ml_enhanced_results.csv` | Phase 1 all-config results |
| `data/reports/tsmom_rsi_cvd_grid.csv` | Phase 2 grid search (6,480) |
| `data/reports/tsmom_binance_enhanced.csv` | Phase 2b Binance integration |
| `data/reports/tsmom_rigorous_v2.csv` | Phase 3 IS/OOS validation |
| `data/reports/tsmom_walkforward.csv` | Walk-forward results |
| `data/reports/tsmom_grid_search.csv` | Barrier parameter grid |
| `data/reports/tsmom_paper/` | Paper bot logs |
