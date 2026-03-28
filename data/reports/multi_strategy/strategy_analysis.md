# Strategy Analysis Report
Generated: 2026-03-28T15:56:03.825576+00:00

## Summary
- Total trades: 122
- Total PnL: -146.8598 USDT
- Overall win rate: 33.6%

## Per-Strategy
| Strategy | Trades | WR | PnL (USDT) | PF | Sharpe | MDD |
|----------|--------|----|-----------|----|----|-----|
| liquidation_fade | 25 | 48% | +37.60 | 1.21 | 1.58 | 104.1% |
| asymmetric_sniper | 24 | 46% | -13.04 | 0.88 | -1.38 | 180.9% |
| momentum_breakout | 7 | 0% | -61.69 | 0.00 | -20.66 | 0.0% |
| cvd_spike | 66 | 27% | -109.73 | 0.55 | -4.74 | 0.0% |

## Per-Coin Top 10
| Coin | Strategy | Trades | WR | PnL |
|------|----------|--------|----|-----|
| OP | liquidation_fade | 4 | 75% | +61.88 |
| DOT | liquidation_fade | 3 | 67% | +44.15 |
| OP | cvd_spike | 6 | 33% | +39.33 |
| AVAX | asymmetric_sniper | 2 | 50% | +32.74 |
| TAO | liquidation_fade | 1 | 100% | +18.22 |
| XRP | liquidation_fade | 1 | 100% | +14.57 |
| ADA | asymmetric_sniper | 3 | 67% | +14.55 |
| SOL | liquidation_fade | 1 | 100% | +14.50 |
| ETH | asymmetric_sniper | 6 | 50% | +11.11 |
| ADA | cvd_spike | 4 | 50% | +9.42 |

## Session Analysis
| Session | Trades | WR | Avg PnL |
|---------|--------|----|---------|
| Asia | 7 | 86% | +10.2068 |
| London | 11 | 27% | +1.8245 |
| NY | 104 | 31% | -2.2921 |

## Volatility Regime
| Regime | Trades | WR | Avg PnL |
|--------|--------|----|---------|
| HIGH | 30 | 33% | -2.7741 |
| LOW | 5 | 40% | -4.0993 |
| MED | 87 | 33% | -0.4958 |

## CVD Z-Score Buckets
| Z-Score | Trades | WR | Avg PnL |
|---------|--------|----|---------|
| z1-2 | 14 | 43% | +2.5988 |
| z2-3 | 13 | 46% | -2.5922 |
| z3-4 | 6 | 67% | +6.0765 |
| z<1 | 85 | 27% | -2.3534 |
| z>4 | 4 | 50% | +3.5086 |

## SL/TP Optimization Hints
| Strategy | Current SL ATR× | Suggested SL | Suggested TP | Note |
|----------|----------------|-------------|-------------|------|
| momentum_breakout | ATR×3.0 | ATR×2.0 | ATR×1.5 | SL 적절 |
| cvd_spike | ATR×3.0 | ATR×2.5 | ATR×3.5 | SL 적절 |
| asymmetric_sniper | ATR×3.0 | ATR×1.9 | ATR×2.2 | SL 적절 |
| liquidation_fade | ATR×3.0 | ATR×3.4 | ATR×3.4 | SL 적절 |

## Top Recommendations
1. [cvd_spike] 승률 27% — 비활성화 또는 신호 필터 강화
2. London 세션 승률 27% — 해당 세션 거래 중단 고려
3. CVD z3-4 구간에서 승률 67% — 이 구간 진입 우선화