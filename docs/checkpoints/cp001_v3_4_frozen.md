# Checkpoint CP-001: v3.4 Configuration Frozen

Date: 2026-03-17
Status: ACTIVE (production config)

## Frozen Artifacts
- `config/frozen_params_v3_4.yaml` -- DO NOT MODIFY
- `config/live_promotion_criteria.yaml` -- DO NOT MODIFY

## Configuration Summary

### DOT
| Param | Value |
|-------|-------|
| k_upper | 3.0 |
| k_lower | 0.6 |
| R:R | 5.0 |
| threshold | 0.50 |
| max_features | 80 |
| blocked_regimes | RANGE_LOW |

### ADA
| Param | Value |
|-------|-------|
| k_upper | 3.0 |
| k_lower | 0.8 |
| R:R | 3.75 |
| threshold | 0.52 |
| max_features | 120 |
| blocked_regimes | RANGE_LOW, UNKNOWN |

### Excluded Features
macro_gold, macro_vix, macro_usd_index, macro_us_10y, macro_sp500,
macro_fear_greed, macro_defi_tvl, hilbert, inst_phase, inst_freq, inst_amp

## Validation Results (Frozen OOS v3.4)
| Coin | Trades | Avg PnL | Total | MDD | Sharpe |
|------|--------|---------|-------|-----|--------|
| DOT | 21 | +0.892% | +18.72% | 1.83% | 8.6 |
| ADA | 12 | +0.772% | +9.26% | 0.58% | 8.8 |
| Portfolio | 33 | +0.848% | +27.99% | 1.83% | 8.6 |

## Next Gate
Paper trading 2 weeks (03-17 ~ 03-31)
-> live_promotion_criteria.yaml 자동 판정
-> PASS: live 전환 논의
-> FAIL: 재검토
