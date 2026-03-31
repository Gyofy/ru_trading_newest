# Strategy Analysis Report
Generated: 2026-03-31T01:07:36.417717+00:00

## Summary
- Total trades: 223
- Total PnL: -133.9919 USDT
- Overall win rate: 29.6%

## Per-Strategy
| Strategy | Trades | WR | PnL (USDT) | PF | Sharpe | MDD |
|----------|--------|----|-----------|----|----|-----|
| liquidation_fade | 41 | 41% | +54.25 | 1.29 | 0.69 | 104.1% |
| asymmetric_sniper | 33 | 45% | +8.08 | 1.08 | 1.18 | 180.9% |
| volume_impulse | 11 | 9% | -3.01 | 0.16 | -6.49 | 0.0% |
| cvd_extreme | 22 | 23% | -7.16 | 0.31 | -2.45 | 0.0% |
| vwap_reversion | 43 | 23% | -14.72 | 0.47 | -6.02 | 2056.5% |
| momentum_breakout | 7 | 0% | -61.69 | 0.00 | -20.66 | 0.0% |
| cvd_spike | 66 | 27% | -109.73 | 0.55 | -4.74 | 0.0% |

## Per-Coin Top 10
| Coin | Strategy | Trades | WR | PnL |
|------|----------|--------|----|-----|
| OP | liquidation_fade | 8 | 50% | +67.77 |
| DOT | liquidation_fade | 3 | 67% | +44.15 |
| OP | cvd_spike | 6 | 33% | +39.33 |
| AVAX | asymmetric_sniper | 2 | 50% | +32.74 |
| TAO | liquidation_fade | 1 | 100% | +18.22 |
| XRP | liquidation_fade | 1 | 100% | +14.57 |
| ADA | asymmetric_sniper | 3 | 67% | +14.55 |
| DOT | asymmetric_sniper | 5 | 60% | +14.08 |
| DOGE | asymmetric_sniper | 3 | 67% | +12.60 |
| SOL | liquidation_fade | 4 | 50% | +11.31 |

## Session Analysis
| Session | Trades | WR | Avg PnL |
|---------|--------|----|---------|
| Asia | 58 | 33% | +1.0897 |
| London | 41 | 17% | +0.1163 |
| NY | 119 | 32% | -1.8045 |
| Off | 5 | 40% | +2.5559 |

## Volatility Regime
| Regime | Trades | WR | Avg PnL |
|--------|--------|----|---------|
| HIGH | 60 | 28% | -1.2774 |
| LOW | 12 | 25% | -1.8342 |
| MED | 151 | 30% | -0.2340 |

## CVD Z-Score Buckets
| Z-Score | Trades | WR | Avg PnL |
|---------|--------|----|---------|
| z1-2 | 47 | 34% | +0.7913 |
| z2-3 | 37 | 30% | -0.6601 |
| z3-4 | 18 | 39% | +1.4232 |
| z<1 | 112 | 26% | -1.6643 |
| z>4 | 9 | 33% | +1.5589 |

## SL/TP Optimization Hints
| Strategy | Current SL ATR× | Suggested SL | Suggested TP | Note |
|----------|----------------|-------------|-------------|------|
| momentum_breakout | ATR×3.0 | ATR×2.0 | ATR×1.5 | SL 적절 |
| cvd_spike | ATR×3.0 | ATR×2.5 | ATR×3.5 | SL 적절 |
| asymmetric_sniper | ATR×3.0 | ATR×1.3 | ATR×2.2 | SL이 너무 넓음 — 리스크 과다 |
| liquidation_fade | ATR×3.0 | ATR×3.4 | ATR×3.4 | SL 적절 |
| vwap_reversion | ATR×4.0 | ATR×3.7 | ATR×5.0 | SL 적절 |
| cvd_extreme | ATR×4.0 | ATR×3.0 | ATR×6.2 | SL 적절 |
| volume_impulse | ATR×4.0 | ATR×2.5 | ATR×10.6 | SL 적절 |

## Top Recommendations
1. [volume_impulse] 승률 9% — 비활성화 또는 신호 필터 강화
2. [cvd_extreme] 승률 23% — 비활성화 또는 신호 필터 강화
3. [vwap_reversion] 승률 23% — 비활성화 또는 신호 필터 강화
4. [cvd_spike] 승률 27% — 비활성화 또는 신호 필터 강화
5. London 세션 승률 17% — 해당 세션 거래 중단 고려
6. 저변동성 레짐 승률 25% — 저변동성 시 진입 억제
7. [asymmetric_sniper] SL 조정: 현재 ATR×3.0 → 권장 ATR×1.3