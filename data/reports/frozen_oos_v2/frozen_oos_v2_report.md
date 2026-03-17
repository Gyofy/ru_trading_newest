# Frozen OOS v2 Report

Generated: 2026-03-17 13:10
Params: config\frozen_params_v3_2.yaml (vv3.2)
OOS Block A: validation (28d), Block B: holdout (28d)
Features excluded: 11 keywords
Regimes blocked: ['RANGE_LOW']

---

## 1. Per-Coin Results

### XRP

| Block | Trades | Win% | Avg Net | Total | MDD | Sharpe | 95% CI | Verdict |
|-------|--------|------|---------|-------|-----|--------|--------|---------|
| A | 0 | - | - | - | - | - | - | NO TRADES |
| B | 2 | 50.0% | +0.8838% | +1.7676% | 0.6528% | 5.8 | [-1.2458%,+3.0134%] | MARGINAL |
| A+B | 2 | 50.0% | +0.8838% | +1.7676% | 0.6528% | 5.8 | [-1.2458%,+3.0134%] | MARGINAL |

**Cost breakdown**: entry=0.0179% exit=0.0359% slip=0.0564% fund=0.0692% miss=0.0530% | total=0.2324% (11.6% of gross)
**Direction**: 0L / 2S

### DOT

| Block | Trades | Win% | Avg Net | Total | MDD | Sharpe | 95% CI | Verdict |
|-------|--------|------|---------|-------|-----|--------|--------|---------|
| A | 9 | 66.7% | +1.4127% | +12.7145% | 1.8326% | 13.3 | [+0.4779%,+2.3476%] | PASS (low N) |
| B | 14 | 28.6% | +0.4410% | +6.1734% | 2.8454% | 4.9 | [-0.2811%,+1.1630%] | PASS (low N) |
| A+B | 23 | 43.5% | +0.8212% | +18.8879% | 2.8454% | 8.1 | [+0.2174%,+1.4250%] | PASS |

**Cost breakdown**: entry=0.1670% exit=0.3225% slip=0.5119% fund=0.2928% miss=0.6095% | total=1.9029% (9.2% of gross)
**Direction**: 1L / 22S

### ADA

| Block | Trades | Win% | Avg Net | Total | MDD | Sharpe | 95% CI | Verdict |
|-------|--------|------|---------|-------|-----|--------|--------|---------|
| A | 4 | 50.0% | +0.9102% | +3.6407% | 0.5745% | 9.0 | [-0.5789%,+2.3993%] | MARGINAL |
| B | 10 | 20.0% | +0.0096% | +0.0962% | 2.9720% | 0.1 | [-0.7386%,+0.7579%] | PASS (low N) |
| A+B | 14 | 28.6% | +0.2669% | +3.7369% | 3.5654% | 3.5 | [-0.4487%,+0.9825%] | PASS (low N) |

**Cost breakdown**: entry=0.1122% exit=0.2577% slip=0.3903% fund=0.1324% miss=0.3710% | total=1.2631% (25.3% of gross)
**Direction**: 8L / 6S

## 2. Portfolio (All Coins Combined)

| Block | Trades | Avg Net | Total | MDD | Sharpe | Verdict |
|-------|--------|---------|-------|-----|--------|---------|
| A | 13 | +1.2581% | +16.3552% | 1.8326% | 11.8 | PASS |
| B | 26 | +0.3091% | +8.0372% | 3.5654% | 3.6 | PASS (low N) |
| A+B | 39 | +0.6254% | +24.3924% | 3.5654% | 6.4 | PASS |

## 3. Regime Analysis (All Coins, A+B)

| Regime | Trades | Avg Net | Total |
|--------|--------|---------|-------|
| TREND_UP | 15 | +0.5690% | +8.5352% |
| TREND_DOWN | 12 | +0.6633% | +7.9601% |
| RANGE_HIGH | 2 | +2.4260% | +4.8521% |
| UNKNOWN | 10 | +0.3045% | +3.0450% |