# Frozen OOS Validation Report

Generated: 2026-03-17 11:40
OOS Period: 42 days (last 6 weeks)
Parameters: FROZEN (v3.1_netev R38)

---

## 1. Trade-level OOS Results

| Coin | Trades | Win% | TimExit% | Avg Net PnL | Total | MaxDD | Sharpe | 95% CI | Verdict |
|------|--------|------|----------|-------------|-------|-------|--------|--------|---------|
| XRP | 0 | - | - | - | - | - | - | - | NO TRADES |
| DOT | 18 | 33.3% | 11.1% | +0.6240% | +11.2314% | 2.2789% | 6.33 | [-0.0203%, +1.2682%] | PASS |
| ADA | 13 | 30.8% | 0.0% | +0.3343% | +4.3465% | 3.5649% | 4.61 | [-0.4242%, +1.0929%] | MARGINAL (CI includes zero) |

## 2. PnL Distribution

**DOT**: P5=-0.6066% P25=-0.5822% P50=-0.5654% P75=+2.4176% P95=+2.4324%
**ADA**: P5=-0.6019% P25=-0.5974% P50=-0.5932% P75=+2.4259% P95=+2.4283%

## 3. Cost Analysis

| Coin | Gross PnL | Total Cost | Cost Share | Net PnL |
|------|-----------|------------|------------|---------|
| DOT | +12.7081% | 1.4768% | 6.5% | +11.2314% |
| ADA | +5.5000% | 1.1535% | 8.0% | +4.3465% |

## 4. Regime Breakdown

**DOT**:
  - UNKNOWN: 18 trades, avg=+0.6240%, total=+11.2314%

**ADA**:
  - UNKNOWN: 13 trades, avg=+0.3343%, total=+4.3465%

## 5. Parameter Perturbation Stability

**XRP** (27 configs tested):
| k_u | k_l | R:R | th | Trades | Avg Net PnL | Verdict |
|-----|-----|-----|-----|--------|-------------|---------|
| 3.0 | 0.5 | 6.0 | 0.65 | 1 | +2.9554% | PASS (low N, cautious) |
| 2.5 | 0.7 | 3.57 | 0.65 | 2 | +1.7268% | PASS (low N, cautious) |
| 3.0 | 0.5 | 6.0 | 0.6 | 4 | +1.1667% | MARGINAL (CI includes zero) |
| 2.5 | 0.7 | 3.57 | 0.6 | 15 | +0.9105% | PASS |
| 2.5 | 0.5 | 5.0 | 0.6 | 11 | +0.8143% | PASS (low N, cautious) |
| 3.0 | 0.5 | 6.0 | 0.5499999999999999 | 6 | +0.5767% | MARGINAL (CI includes zero) |
| 2.5 | 0.6 | 4.17 | 0.5499999999999999 | 26 | +0.4959% | PASS |
| 2.5 | 0.5 | 5.0 | 0.65 | 6 | +0.4826% | MARGINAL (CI includes zero) |
| 2.5 | 0.6 | 4.17 | 0.65 | 5 | +0.4511% | MARGINAL (CI includes zero) |
| 2.5 | 0.7 | 3.57 | 0.5499999999999999 | 36 | +0.4493% | PASS |
| 2.5 | 0.5 | 5.0 | 0.5499999999999999 | 15 | +0.4339% | MARGINAL (CI includes zero) |
| 2.5 | 0.6 | 4.17 | 0.6 | 14 | +0.4247% | MARGINAL (CI includes zero) |
| 3.5 | 0.7 | 5.0 | 0.6 | 1 | -0.5506% | FAIL |
| 3.5 | 0.7 | 5.0 | 0.5499999999999999 | 2 | -0.5662% | FAIL |
| 3.0 | 0.7 | 4.29 | 0.5499999999999999 | 1 | -0.5878% | FAIL |
| 3.5 | 0.5 | 7.0 | 0.5499999999999999 | 1 | -0.6038% | FAIL |

**DOT** (27 configs tested):
| k_u | k_l | R:R | th | Trades | Avg Net PnL | Verdict |
|-----|-----|-----|-----|--------|-------------|---------|
| 3.5 | 0.6 | 5.83 | 0.55 | 9 | +1.2511% | PASS (low N, cautious) |
| 3.5 | 0.7 | 5.0 | 0.55 | 9 | +1.1985% | PASS (low N, cautious) |
| 3.5 | 0.6 | 5.83 | 0.45 | 10 | +1.1668% | PASS (low N, cautious) |
| 3.5 | 0.5 | 7.0 | 0.55 | 7 | +1.1197% | MARGINAL (CI includes zero) |
| 3.5 | 0.7 | 5.0 | 0.45 | 14 | +0.9822% | PASS (low N, cautious) |
| 3.5 | 0.6 | 5.83 | 0.5 | 11 | +0.9521% | PASS (low N, cautious) |
| 3.5 | 0.7 | 5.0 | 0.5 | 14 | +0.9432% | PASS (low N, cautious) |
| 2.5 | 0.5 | 5.0 | 0.55 | 22 | +0.8876% | PASS |
| 2.5 | 0.5 | 5.0 | 0.5 | 23 | +0.8224% | PASS |
| 3.5 | 0.5 | 7.0 | 0.5 | 8 | +0.7145% | MARGINAL (CI includes zero) |
| 2.5 | 0.7 | 3.57 | 0.55 | 15 | +0.6489% | PASS |
| 3.0 | 0.6 | 5.0 | 0.45 | 18 | +0.6240% | PASS |
| 3.0 | 0.6 | 5.0 | 0.5 | 18 | +0.6240% | PASS |
| 2.5 | 0.7 | 3.57 | 0.45 | 15 | +0.5861% | PASS |
| 2.5 | 0.7 | 3.57 | 0.5 | 15 | +0.5861% | PASS |
| 2.5 | 0.5 | 5.0 | 0.45 | 22 | +0.5840% | PASS |
| 3.5 | 0.5 | 7.0 | 0.45 | 11 | +0.5822% | MARGINAL (CI includes zero) |
| 2.5 | 0.6 | 4.17 | 0.45 | 19 | +0.5670% | PASS |
| 2.5 | 0.6 | 4.17 | 0.5 | 19 | +0.5670% | PASS |
| 3.0 | 0.7 | 4.29 | 0.45 | 20 | +0.5166% | PASS |
| 3.0 | 0.5 | 6.0 | 0.45 | 18 | +0.4592% | MARGINAL (CI includes zero) |
| 3.0 | 0.7 | 4.29 | 0.5 | 16 | +0.4592% | MARGINAL (CI includes zero) |
| 3.0 | 0.7 | 4.29 | 0.55 | 16 | +0.4592% | MARGINAL (CI includes zero) |
| 3.0 | 0.5 | 6.0 | 0.5 | 18 | +0.4591% | MARGINAL (CI includes zero) |
| 3.0 | 0.5 | 6.0 | 0.55 | 18 | +0.4381% | MARGINAL (CI includes zero) |
| 3.0 | 0.6 | 5.0 | 0.55 | 16 | +0.3993% | MARGINAL (CI includes zero) |
| 2.5 | 0.6 | 4.17 | 0.55 | 17 | +0.3978% | MARGINAL (CI includes zero) |

**ADA** (27 configs tested):
| k_u | k_l | R:R | th | Trades | Avg Net PnL | Verdict |
|-----|-----|-----|-----|--------|-------------|---------|
| 3.0 | 0.6 | 5.0 | 0.45 | 15 | +0.6138% | MARGINAL (CI includes zero) |
| 3.0 | 0.5 | 6.0 | 0.45 | 21 | +0.5692% | MARGINAL (CI includes zero) |
| 3.0 | 0.6 | 5.0 | 0.55 | 11 | +0.5034% | MARGINAL (CI includes zero) |
| 3.0 | 0.5 | 6.0 | 0.5 | 20 | +0.4504% | MARGINAL (CI includes zero) |
| 3.0 | 0.5 | 6.0 | 0.55 | 17 | +0.4265% | MARGINAL (CI includes zero) |
| 3.5 | 0.7 | 5.0 | 0.45 | 9 | +0.4193% | MARGINAL (CI includes zero) |
| 2.5 | 0.5 | 5.0 | 0.45 | 8 | +0.3955% | MARGINAL (CI includes zero) |
| 3.0 | 0.6 | 5.0 | 0.5 | 13 | +0.3343% | MARGINAL (CI includes zero) |
| 3.5 | 0.5 | 7.0 | 0.55 | 13 | +0.3218% | MARGINAL (CI includes zero) |
| 2.5 | 0.7 | 3.57 | 0.45 | 13 | +0.2992% | MARGINAL (CI includes zero) |
| 3.5 | 0.5 | 7.0 | 0.45 | 18 | +0.2876% | MARGINAL (CI includes zero) |
| 3.5 | 0.5 | 7.0 | 0.5 | 18 | +0.2876% | MARGINAL (CI includes zero) |
| 3.5 | 0.6 | 5.83 | 0.45 | 17 | +0.2154% | MARGINAL (CI includes zero) |
| 3.0 | 0.7 | 4.29 | 0.55 | 10 | +0.2121% | MARGINAL (CI includes zero) |
| 2.5 | 0.7 | 3.57 | 0.5 | 12 | +0.1814% | MARGINAL (CI includes zero) |
| 3.0 | 0.7 | 4.29 | 0.5 | 14 | +0.1747% | MARGINAL (CI includes zero) |
| 3.5 | 0.7 | 5.0 | 0.5 | 8 | +0.1636% | MARGINAL (CI includes zero) |
| 2.5 | 0.6 | 4.17 | 0.5 | 7 | +0.1499% | MARGINAL (CI includes zero) |
| 2.5 | 0.6 | 4.17 | 0.55 | 7 | +0.1499% | MARGINAL (CI includes zero) |
| 2.5 | 0.6 | 4.17 | 0.45 | 7 | +0.1496% | MARGINAL (CI includes zero) |
| 3.0 | 0.7 | 4.29 | 0.45 | 15 | +0.1240% | MARGINAL (CI includes zero) |
| 2.5 | 0.5 | 5.0 | 0.5 | 7 | +0.1061% | MARGINAL (CI includes zero) |
| 2.5 | 0.5 | 5.0 | 0.55 | 7 | +0.1061% | MARGINAL (CI includes zero) |
| 2.5 | 0.7 | 3.57 | 0.55 | 10 | +0.1034% | MARGINAL (CI includes zero) |
| 3.5 | 0.6 | 5.83 | 0.5 | 15 | +0.0940% | MARGINAL (CI includes zero) |
| 3.5 | 0.6 | 5.83 | 0.55 | 13 | -0.0661% | FAIL |
| 3.5 | 0.7 | 5.0 | 0.55 | 6 | -0.0895% | FAIL |

---

## Overall Verdict

- PASS: 1/3 coins
- Recommendation: Review model / EV calculation