# Frozen OOS v2 Report

Generated: 2026-03-17 13:53
Params: config\frozen_params_v3_4.yaml (vv3.4)
OOS Block A: validation (28d), Block B: holdout (28d)
Features excluded: 11 keywords
Regimes blocked: ['RANGE_LOW']

---

## 1. Per-Coin Results

### DOT

| Block | Trades | Win% | Avg Net | Total | MDD | Sharpe | 95% CI | Verdict |
|-------|--------|------|---------|-------|-----|--------|--------|---------|
| A | 10 | 50.0% | +0.9119% | +9.1187% | 1.8326% | 8.9 | [-0.0263%,+1.8500%] | PASS |
| B | 11 | 45.5% | +0.8732% | +9.6051% | 1.7081% | 8.5 | [+0.0219%,+1.7244%] | PASS |
| A+B | 21 | 47.6% | +0.8916% | +18.7238% | 1.8326% | 8.7 | [+0.2604%,+1.5228%] | PASS |

**Cost breakdown**: entry=0.1557% exit=0.2984% slip=0.4743% fund=0.2784% miss=0.5565% | total=1.7626% (8.6% of gross)
**Direction**: 0L / 21S

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
| A | 16 | +0.8001% | +12.8011% | 1.8326% | 8.3 | PASS |
| B | 17 | +0.8933% | +15.1864% | 1.7081% | 9.0 | PASS |
| A+B | 33 | +0.8481% | +27.9875% | 1.8326% | 8.6 | PASS |

## 3. Regime Analysis (All Coins, A+B)

| Regime | Trades | Avg Net | Total |
|--------|--------|---------|-------|
| TREND_DOWN | 14 | +0.9215% | +12.9015% |
| UNKNOWN | 5 | +1.2045% | +6.0224% |
| TREND_UP | 12 | +0.4020% | +4.8242% |
| RANGE_HIGH | 2 | +2.1197% | +4.2394% |