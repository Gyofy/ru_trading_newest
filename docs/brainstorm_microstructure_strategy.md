# Microstructure Data Strategy Brainstorm
*Generated: 2026-03-18 | Based on 10-coin 4h data (4,378 bars, 222 features)*

---

## 1. Data Summary (GitHub: Gyofy/Ru_trading)

| Coin | Bars | Features | MS Features | NaN | Gap Fill |
|------|------|----------|-------------|-----|---------|
| BTC  | 4378 | 222 | 73 | 0 | 0.14% |
| ETH  | 4378 | 222 | 73 | 0 | 0.14% |
| SOL  | 4378 | 222 | 73 | 0 | 0.14% |
| XRP  | 4378 | 222 | 73 | 0 | 0.14% |
| ADA  | 4378 | 222 | 73 | 0 | 0.14% |
| DOGE | 4378 | 222 | 73 | 0 | 0.14% |
| AVAX | 4378 | 222 | 73 | 0 | 0.14% |
| DOT  | 4378 | 222 | 73 | 0 | 0.14% |
| LINK | 4378 | 222 | 73 | 0 | 0.14% |
| BNB  | 4378 | 222 | 73 | 0 | 0.14% |

**Period:** 2024-03-18 ~ 2026-03-17 (2 years)
**Gap fill method:** OHLCV ffill (no-trade bars → volume=0)

---

## 2. Key Empirical Findings

### A. CVD Mean-Reversion Effect (★★★ HIGH CONFIDENCE)
- `cvd_ratio_24` shows **negative** Spearman correlation (-0.21) with 24h forward returns
- When CVD is **high** (buy pressure overextended) → price mean-reverts DOWN
- When CVD is **low** (sell pressure overextended) → price mean-reverts UP

**Backtest Result (BTC, 48h hold, after 0.4% roundtrip fee):**
- SHORT high-CVD: **+0.334%/trade, 53.9% WR** ✅ (165% cumulative PnL)
- LONG low-CVD: -0.061%/trade, 51.8% WR ❌
- Effect strongest at 48h horizon, NOT 4h

**Cross-coin edge (24h horizon):**
| Coin | Short HIGH CVD | Long LOW CVD | Edge |
|------|---------------|--------------|------|
| ADA  | -1.251% | +0.551% | 1.80% |
| ETH  | -0.816% | +0.392% | 1.21% |
| SOL  | -0.889% | +0.144% | 1.03% |
| BTC  | -0.690% | +0.189% | 0.88% |

**Interpretation:** CVD at 4h bars is built from OHLCV proxy (BVC method, not real trades).
High CVD → HFT/market makers accumulate → release pressure creates reversals.

---

### B. OFI Short-Term Leading Indicator (★★ MODERATE)
- `ofi_sum_3` (3-bar rolling OFI) shows **positive** lead for 4-8h returns
- BTC: +0.078% (4h), +0.100% (8h), +0.175% (24h)
- ETH: +0.149% (4h), +0.233% (8h) — strongest OFI signal

**Strategy use:** OFI as entry timing filter (wait for OFI confirmation)

---

### C. Cross-Coin Correlation Structure
- **High correlation group (>0.85 with BTC):** ETH, SOL, LINK, BNB
- **Medium correlation (0.75-0.85):** XRP, AVAX, ADA, DOGE
- **Low correlation (0.59):** DOT — highest diversification value
- **MS Flow correlation** is lower than price correlation (more independent signals)

**DOT anomaly:** Beta=1.10 to BTC but correlation=0.65 → high idiosyncratic risk.
When BTC is neutral, DOT has positive alpha (+0.027%/4h).

---

### D. Time-of-Day Effects
- Best bar open (highest win rate): **20:00 UTC** (55.4% WR)
- Worst bar open: **00:00 UTC** (46.7% WR)
- **16:00 UTC** (our active bar): 48.2% WR — slightly below random

---

### E. Current Market Microstructure State
- BTC: MS composite = 0.175 (NEUTRAL), flow = -0.007 (slightly bearish flow)
- ETH: MS composite = 0.096 (BEARISH), flow = +0.129 (buy pressure building)
- DOT: MS composite = 0.135 (NEUTRAL), flow = -0.202 (strong sell flow)
- ADA: MS composite = 0.138 (NEUTRAL), flow = +0.194 (buy flow building)

---

## 3. Strategy Brainstorm

### Strategy 1: CVD Contrarian Overlay (Priority: HIGH)
**Concept:** Add CVD_ratio_24 as a binary filter on existing ML signal
- If ML signal = UP AND cvd_ratio_24 > Q75 → REDUCE size by 50% (overextended)
- If ML signal = DOWN AND cvd_ratio_24 < Q25 → REDUCE size by 50%
- If ML signal aligns with CVD mean-reversion → INCREASE size 1.3x

**Implementation:** Add to `calc_dynamic_risk()` as `cvd_filter_mult`

---

### Strategy 2: OFI Entry Timing (Priority: MEDIUM)
**Concept:** Use OFI to time entry within the 4h bar
- After 4h bar close signal, wait for OFI_sum_3 > 0 before entering LONG
- Estimated improvement: +0.1-0.2% per trade (OFI lead at 4h)
- Risk: miss-fill if OFI never turns positive

---

### Strategy 3: Cross-Coin Ensemble with Beta Adjustment (Priority: MEDIUM)
**Concept:** Use BTC signal as market-wide regime filter
- When BTC_cvd > Q75 (overbought): reduce all altcoin LONG exposure
- When BTC regime = TREND_UP + OFI > Q67: increase altcoin exposure

---

### Strategy 4: Real-Time OrderFlow Integration (Priority: HIGH - Phase 2)
**Concept:** Use `OrderFlowCollector` (implemented) to get real CVD from agg trades
- Real CVD from m-field is more accurate than OHLCV proxy
- Integrate flow_pressure into dynamic sizing multiplier
- flow_pressure > 0.3 AND ML=UP → size_mult × 1.2
- flow_pressure < -0.3 AND ML=UP → size_mult × 0.7

**Status:** `src/execution/orderflow_collector.py` complete, not yet integrated.

---

### Strategy 5: VPIN Volatility Regime Filter (Priority: LOW)
**Concept:** VPIN predicts information asymmetry → avoid trading in high-VPIN regimes
- High VPIN (>Q75) → 1.11x future volatility vs Low VPIN
- Reduce position size in high-VPIN periods
- Effect is weak (1.11x) in current OHLCV proxy — use real order flow for better VPIN

---

### Strategy 6: DOT-Specific Strategy (Priority: LOW)
**Concept:** DOT has lowest BTC correlation → ecosystem-specific signals matter
- Monitor Polkadot parachain slot auction events (external data needed)
- DOT ms_flow_score divergence from BTC → independent signal opportunity
- Current: DOT flow = -0.202 vs BTC flow = -0.007 → DOT underperforming relative to market

---

## 4. Implementation Priority

| Priority | Strategy | Expected Impact | Effort |
|----------|----------|----------------|--------|
| 1 | Real-Time OrderFlow → dynamic sizing | +0.1-0.3%/trade | Medium |
| 2 | CVD Contrarian Filter on ML signal | +0.05-0.15%/trade | Low |
| 3 | OFI Entry Timing | +0.05-0.1%/trade | Low |
| 4 | Cross-Coin BTC regime overlay | reduce drawdown | Medium |
| 5 | VPIN regime filter | reduce false positives | Low |
| 6 | DOT ecosystem monitoring | ecosystem alpha | High |

---

## 5. Data Gaps & Next Collection Steps

- **Missing:** Real-time order book data (bid/ask wall detection) → `OrderFlowCollector`
- **Missing:** Funding rate history (8h basis) → add to OI collector
- **Missing:** Liquidation data → Binance `fapi/v1/forceOrders`
- **Missing:** Large trade (>$100k) detection → whale alert proxy via agg trades
- **Upgrade:** VPIN should use real agg trades (not OHLCV proxy) for accurate signals
