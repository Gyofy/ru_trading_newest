# Comprehensive Status Report -- 2026-03-18

## 1. Project Overview

Crypto auto-trading system using ML ensemble prediction + asymmetric R:R exit strategy.
2-Stage Binary Classification (Trade/NoTrade -> Long/Short) with Triple Barrier exit.

---

## 2. Evolution Timeline

```
v3.1 (03-16) -> v3.4 (03-17) -> v4.0 (03-18) -> v4.1 (03-18)
```

| Version | Key Change | Status |
|---------|-----------|--------|
| v3.1 | 9,390 evals Walk-Forward, summary EV | Superseded |
| v3.4 | Trade-level EV, per-coin regime, DOT+ADA | Superseded |
| v4.0 | Microstructure module, Binance, LGB CPU, WF 444 evals | Superseded |
| **v4.1** | **5-coin, ET primary, TabPFN/TabM combos, MS features** | **Current** |

---

## 3. Microstructure Feature System

### 3.1 Architecture

```
OHLCV (open, high, low, close, volume)
  |
  +-- BVC (Bulk Volume Classification)
  |   buy_frac = (close - low) / (high - low)
  |   buy_vol = buy_frac * volume
  |   sell_vol = (1 - buy_frac) * volume
  |
  +-- 5 Feature Groups (~71 features total)
      |
      1. CVD (Cumulative Volume Delta)    ~19 features
      2. OFI (Order Flow Imbalance)       ~18 features
      3. VPIN (Informed Trading Prob)      ~15 features
      4. Roll Spread (Liquidity Proxy)     ~10 features
      5. Amihud Illiquidity               ~14 features
      + Composite Signals                  ~4 features
```

### 3.2 Feature Group Details

#### CVD (Cumulative Volume Delta)
- **Core**: buy_frac, vd (volume delta per bar), cvd (cumulative)
- **Rolling**: cvd_{3,6,12,24}, cvd_ma, cvd_ratio (normalized)
- **Divergence**: cvd_div (binary), cvd_divstrength (continuous)
- **Tick-rule**: cvd_tick (backup using price direction)
- **Alpha**: CVD rising + price falling = bullish divergence

#### OFI (Order Flow Imbalance)
- **Core**: ofi_raw = (close-open)/(high-low) * volume
- **Normalized**: ofi_norm (ATR-adjusted)
- **Rolling**: ofi_sum, ofi_mean, ofi_std, ofi_pct (volume-normalized)
- **Alpha**: Persistent positive OFI = institutional accumulation

#### VPIN (Volume-Synchronized Probability of Informed Trading)
- **BVC-based**: vpin_{12,24,48} (range [0,1])
- **CDF-based**: vpin_cdf (Easley et al. 2012 approximation)
- **Dynamics**: vpin_chg (rate of change), vpin_regime (75th pctl flag)
- **Direction**: vpin_buyfrac (average buy fraction)
- **Alpha**: VPIN spike = information asymmetry -> directional breakout imminent

#### Roll Spread
- **Core**: Roll (1984) = 2 * sqrt(max(-Cov(dr_t, dr_{t-1}), 0))
- **Variants**: roll_spread_{12,24,48}, roll_spread_bps, roll_spread_ratio
- **Covariance sign**: roll_cov_sign (+1=trending, -1=mean-reverting)
- **Alpha**: Spread spike + high VPIN = liquidity crisis warning

#### Amihud Illiquidity
- **Core**: ILLIQ = |return| / (volume * price) * 10^6
- **Rolling**: illiq_{3,6,12,24}, illiq_std, illiq_z (z-score)
- **Ratio**: illiq_ratio (short/long term)
- **Alpha**: ILLIQ surge = market makers withdrawing, slippage risk

#### Composite Signals
- **ms_flow_score**: CVD ratio + OFI pct z-score average
- **ms_informed_score**: VPIN average across windows
- **ms_liquidity_stress**: Roll spread + Amihud z-score
- **ms_composite**: Weighted blend (flow 50%, informed 30%, -stress 20%)

### 3.3 Integration Strategy for ML Model

**Current approach (v4.1):**
```
223 total features = 133 (v3.4 base) + ~90 (microstructure)
  -> MI-based selection: top 80-120
  -> ExtraTrees primary model
  -> Optional TabPFN/TabM blending
```

**Specific use cases in trading logic:**

| Signal | Condition | Action |
|--------|-----------|--------|
| CVD filter | cvd > Q75 + LONG signal | Reduce size 0.6x (overbought) |
| CVD filter | cvd < Q25 + LONG signal | Increase size 1.2x (accumulation) |
| OFI timing | ofi_sum_3 > Q67 | Confirm entry (momentum) |
| VPIN alert | vpin_24 > 0.7 | Flag high-info-asymmetry regime |
| Liquidity stress | ms_liquidity_stress > 2.0 | Block entry (high slippage risk) |

### 3.4 Learning Logic Design

**How microstructure features enter the model:**

```
Stage 1 (Trade/NoTrade):
  Input: base features (133) + microstructure (90) = 223
  MI selection: top max_features (80-120)
  Hypothesis: CVD divergence + VPIN spike -> "trade signal" detection

  Key MS features expected in MI top 20:
    - cvd_divstrength_{6,12} (divergence = directional opportunity)
    - vpin_chg_{12,24} (regime shift = volatility opportunity)
    - ms_flow_score (composite pressure)
    - ofi_pct_{3,6} (short-term order flow direction)

Stage 2 (Long/Short):
  Same feature set, different labels
  Hypothesis: CVD direction + OFI sign -> direction prediction

  Key MS features for direction:
    - cvd_ratio_{3,6} (positive = buy pressure = LONG)
    - ofi_norm (body/range * volume = directional force)
    - vpin_buyfrac_{12} (> 0.5 = more buy volume)

Regime filter augmentation:
  - ms_liquidity_stress > threshold -> block entry (any direction)
  - vpin_regime flag -> heightened caution (tighter threshold)
```

**What NOT to do with microstructure features:**
1. Do NOT use ms_composite directly as a trading signal (too many assumptions)
2. Do NOT use raw CVD (non-stationary, cumsum) -- use cvd_ratio or cvd_divstrength
3. Do NOT trust VPIN in low-volume periods (illiq_z high = VPIN unreliable)

---

## 4. Model Architecture (v4.1)

### 4.1 Mega Search v1 Results (47 configs, 141 experiments)

**Key finding: ExtraTrees is the strongest base model for all 3 coins.**

| Coin | Best Config | Avg PnL | Total | Trades | MDD |
|------|-------------|---------|-------|--------|-----|
| DOT | ET solo | +0.419% | +27.15% | 18 | 2.87% |
| ADA | ET+TabPFN 70/30 | +0.432% | +21.87% | 13 | 2.56% |
| XRP | ET+TabM 70/30 | +0.758% | +51.54% | 18 | 1.98% |

**Why ET dominates:**
- Tree-based, handles non-stationary data naturally
- Random splits = built-in regularization for small datasets (~1800 bars)
- class_weight="balanced" handles label imbalance
- n_jobs=6 parallel = fast iteration

**Why 7-model ensemble lost to ET solo:**
- Stacking meta-learner overfit on small OOF data (~300 trade samples)
- LGB/XGB/CB gave correlated predictions (all gradient boosting)
- HGB and BRF added noise without sufficient diversity
- ET's random splits provide MORE diversity than 7 correlated models

### 4.2 TabPFN Role (ADA)

TabPFN (Nature 2024, foundation model for tabular data):
- Zero-shot / few-shot: no gradient training needed
- Provides DIFFERENT probability estimates from tree models
- Signal agreement with ET: only 30.6% (high diversity)
- ADA 6-window Multi-OOS: Blend50 won 3/6 windows

Best use: **30% weight blended with ET 70%** for ADA only.

### 4.3 TabM Role (XRP)

TabM (ICLR 2025, parameter-efficient MLP ensemble):
- k=32 implicit MLPs sharing weights via BatchEnsemble adapters
- Weight sharing = effective regularization for small tabular data
- Signal agreement with ET: 42.7%
- XRP ADA Multi-OOS: Blend70 won 3/4 windows

Best use: **30% weight blended with ET 70%** for XRP only.

---

## 5. Current Configuration (v4.1)

### 5.1 Coin-Specific Settings

| Coin | Model Combo | Threshold | k_lower | R:R | Blocked Regimes | Status |
|------|-------------|-----------|---------|-----|-----------------|--------|
| DOT | ET solo | 0.45 | 0.6 | 5.0 | RANGE_LOW | Confirmed |
| ADA | ET+TabPFN 70/30 | 0.40 | 0.8 | 3.75 | RANGE_LOW, UNKNOWN | Confirmed |
| XRP | ET+TabM 70/30 | 0.45 | 0.6 | 5.0 | RANGE_LOW | Confirmed |
| SOL | ET solo (default) | 0.50 | 0.6 | 5.0 | RANGE_LOW | Pending v2 |
| LINK | ET solo (default) | 0.50 | 0.6 | 5.0 | RANGE_LOW | Pending v2 |

### 5.2 Feature Pipeline (223 features)

```
Layer 1: OHLCV base (49)
  open, high, low, close, volume + basic derived

Layer 2: Technical indicators (28)
  RSI, MACD, BB, EMA, SMA, ATR, ADX, OBV, MFI, CMF, etc.

Layer 3: Signal features (84)
  Wavelet(10), FFT(5), Hilbert(4), Entropy(3), Hurst/ACF(7),
  Microstructure_basic(13), CUSUM(4), Multi-TF(15), etc.

Layer 4: Microstructure rollup (~71)
  CVD(19), OFI(18), VPIN(15), Roll(10), Amihud(14), Composite(4)

Layer 5: Macro (removed -- MI=0)
  [EXCLUDED] Gold, VIX, DXY, US10Y, S&P500, Fear&Greed, DeFi TVL
```

### 5.3 Cost Model

Binance USDT-M Futures (also compatible with Bybit VIP0):

| Component | Rate |
|-----------|------|
| Maker fee (entry) | 0.02% |
| Taker fee (SL exit) | 0.055% |
| Slippage entry | 0.03% |
| Slippage exit market | 0.05% |
| Funding rate | 0.01%/8h |
| Miss-fill reject | 15% |

---

## 6. Validation History

| Test | Method | Result | Date |
|------|--------|--------|------|
| Walk-Forward v3.1 | 38 rounds, 9,390 evals | 3 coins converged | 03-16~17 |
| Frozen OOS v1 | 42-day OOS, trade-level | XRP 0 trades, DOT PASS | 03-17 |
| Strategy Diagnosis | 4-step analysis | Macro MI=0, ET best | 03-17 |
| ADA Optimization | th 0.52, k_l 0.8 | Cost 25%->9% | 03-17 |
| Frozen OOS v3.4 | 2-block, 8 weeks | Portfolio Sharpe 8.6 | 03-17 |
| Paper Sim (5M KRW) | 8-week replay | +26.57%, 42 trades | 03-17 |
| **Mega Search v1** | **47 configs, 4 OOS** | **ET dominance confirmed** | **03-17** |
| TabPFN Multi-OOS | 6 windows, ADA | Blend50 won 3/6 | 03-17 |
| TabM Multi-OOS | 4 windows, ADA | Blend70 won 3/4 | 03-17 |
| **Mega Search v2** | **16 configs, 5 coins, 223 feat** | **In progress** | **03-18** |

---

## 7. Risk Assessment

### 7.1 Known Strengths
- Payoff geometry (R:R 3.75~5.0) makes strategy profitable at 30%+ win rate
- ExtraTrees provides natural regularization for small datasets
- Per-coin model selection reduces model risk
- Regime filter blocks unprofitable market states
- Microstructure features add information layer without changing model

### 7.2 Known Risks
- 42 trades is statistically small (95% CI wide)
- DOT SHORT bias 2.2x (market direction dependent)
- XRP S1 threshold sensitivity (0.54+ cliff in v3.4, mitigated by lowering to 0.45)
- Microstructure BVC approximation may not match real L2 orderbook data
- Exchange connectivity limited (Binance accessible but network restrictions exist)

### 7.3 Open Questions
- Do microstructure features actually enter MI top 20? (Mega Search v2 will answer)
- Does SOL/LINK show positive EV with current parameters?
- Is ET solo actually better than ET+augmented for DOT? (or just data-specific)

---

## 8. Infrastructure

### 8.1 Data Sources
- **OHLCV**: yfinance (1h -> 4h resample, 365 days, all 10 coins)
- **Microstructure**: Pre-computed from OHLCV via BVC (data/microstructure/)
- **Macro**: Gold, VIX, DXY, US10Y, S&P500, F&G, TVL (MI=0, excluded from model)

### 8.2 Execution
- **Exchange**: Binance USDT-M Futures (confirmed accessible)
- **Backup**: Bybit (blocked by network), KIS API (domestic)
- **Mode**: Demo/Paper (live requires 2-week paper validation pass)

### 8.3 Hardware
- GPU: NVIDIA RTX 3090 24GB
- CPU: 16 cores, n_jobs=6
- TabPFN: local weights (tabpfn_v2_cls/)
- TabM: GPU training via PyTorch

---

## 9. Active Processes

| Process | Status | ETA |
|---------|--------|-----|
| Mega Search v2 (5 coins, 223 features) | Running | Hours |
| Paper Trading v3.4 | Stopped (KeyError bug) | Restart after v4.1 |
| v4.1 Config | Created | Ready |
| Frozen OOS v4.1 | Framework ready | After Mega Search v2 |

---

## 10. Next Steps (Priority Order)

1. **Mega Search v2 completion** -> SOL/LINK model combo + microstructure impact
2. **v4.1 config finalization** -> Update SOL/LINK based on v2 results
3. **Frozen OOS v4.1** -> 5-coin, 4-window validation with 223 features
4. **Paper Trading v4.1** -> 2-week forward validation (Binance demo)
5. **Live promotion decision** -> Based on paper results vs criteria
