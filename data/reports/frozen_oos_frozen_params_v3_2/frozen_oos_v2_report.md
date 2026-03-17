# Frozen OOS v2 Report

Generated: 2026-03-17 13:29
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
| A | 9 | 66.7% | +1.4128% | +12.7151% | 1.8326% | 13.0 | [+0.4779%,+2.3477%] | PASS (low N) |
| B | 13 | 38.5% | +0.7036% | +9.1469% | 1.7081% | 7.3 | [-0.0712%,+1.4785%] | PASS |
| A+B | 22 | 50.0% | +0.9937% | +21.8620% | 1.8326% | 9.5 | [+0.3796%,+1.6078%] | PASS |

**Cost breakdown**: entry=0.1579% exit=0.2903% slip=0.4670% fund=0.2943% miss=0.5830% | total=1.7920% (7.6% of gross)
**Direction**: 2L / 20S

### ADA

| Block | Trades | Win% | Avg Net | Total | MDD | Sharpe | 95% CI | Verdict |
|-------|--------|------|---------|-------|-----|--------|--------|---------|
| A | 4 | 50.0% | +0.9091% | +3.6365% | 0.5745% | 8.5 | [-0.5789%,+2.3972%] | MARGINAL |
| B | 10 | 20.0% | +0.0096% | +0.0962% | 2.9720% | 0.1 | [-0.7386%,+0.7579%] | PASS (low N) |
| A+B | 14 | 28.6% | +0.2666% | +3.7327% | 3.5654% | 3.4 | [-0.4487%,+0.9820%] | PASS (low N) |

**Cost breakdown**: entry=0.1121% exit=0.2576% slip=0.3901% fund=0.1370% miss=0.3710% | total=1.2673% (25.4% of gross)
**Direction**: 8L / 6S

## 2. Portfolio (All Coins Combined)

| Block | Trades | Avg Net | Total | MDD | Sharpe | Verdict |
|-------|--------|---------|-------|-----|--------|---------|
| A | 13 | +1.2578% | +16.3516% | 1.8326% | 11.5 | PASS |
| B | 25 | +0.4404% | +11.0107% | 3.5654% | 4.9 | PASS |
| A+B | 38 | +0.7201% | +27.3623% | 3.5654% | 7.2 | PASS |

## 3. Regime Analysis (All Coins, A+B)

| Regime | Trades | Avg Net | Total |
|--------|--------|---------|-------|
| TREND_UP | 14 | +0.8230% | +11.5217% |
| TREND_DOWN | 12 | +0.6620% | +7.9435% |
| RANGE_HIGH | 2 | +2.4260% | +4.8521% |
| UNKNOWN | 10 | +0.3045% | +3.0450% |