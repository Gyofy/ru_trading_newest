# Frozen OOS v2 Report

Generated: 2026-03-17 13:28
Params: config\frozen_params_v3_3.yaml (vv3.3)
OOS Block A: validation (28d), Block B: holdout (28d)
Features excluded: 11 keywords
Regimes blocked: ['RANGE_LOW', 'UNKNOWN']

---

## 1. Per-Coin Results

### DOT

| Block | Trades | Win% | Avg Net | Total | MDD | Sharpe | 95% CI | Verdict |
|-------|--------|------|---------|-------|-----|--------|--------|---------|
| A | 8 | 50.0% | +0.9097% | +7.2777% | 1.2194% | 9.1 | [-0.1413%,+1.9607%] | PASS (low N) |
| B | 10 | 20.0% | +0.2534% | +2.5342% | 2.2687% | 3.0 | [-0.5380%,+1.0449%] | PASS (low N) |
| A+B | 18 | 33.3% | +0.5451% | +9.8119% | 3.4416% | 5.8 | [-0.1139%,+1.2041%] | PASS |

**Cost breakdown**: entry=0.1255% exit=0.2681% slip=0.4142% fund=0.1947% miss=0.4770% | total=1.4789% (13.1% of gross)
**Direction**: 1L / 17S

### ADA

| Block | Trades | Win% | Avg Net | Total | MDD | Sharpe | 95% CI | Verdict |
|-------|--------|------|---------|-------|-----|--------|--------|---------|
| A | 6 | 50.0% | +0.6137% | +3.6824% | 0.5797% | 7.3 | [-0.3470%,+1.5745%] | PASS (low N) |
| B | 6 | 50.0% | +0.9302% | +5.5813% | 0.5847% | 10.6 | [+0.0644%,+1.7960%] | PASS (low N) |
| A+B | 12 | 50.0% | +0.7720% | +9.2637% | 0.5847% | 8.8 | [+0.1191%,+1.4248%] | PASS |

**Cost breakdown**: entry=0.0708% exit=0.1379% slip=0.2177% fund=0.1501% miss=0.3180% | total=0.8937% (8.8% of gross)
**Direction**: 5L / 7S

## 2. Portfolio (All Coins Combined)

| Block | Trades | Avg Net | Total | MDD | Sharpe | Verdict |
|-------|--------|---------|-------|-----|--------|---------|
| A | 14 | +0.7829% | +10.9601% | 1.7899% | 8.3 | PASS |
| B | 16 | +0.5072% | +8.1155% | 2.2687% | 5.6 | PASS |
| A+B | 30 | +0.6359% | +19.0756% | 3.4416% | 6.8 | PASS |

## 3. Regime Analysis (All Coins, A+B)

| Regime | Trades | Avg Net | Total |
|--------|--------|---------|-------|
| TREND_DOWN | 16 | +0.7304% | +11.6857% |
| RANGE_HIGH | 2 | +2.1197% | +4.2394% |
| TREND_UP | 12 | +0.2625% | +3.1505% |