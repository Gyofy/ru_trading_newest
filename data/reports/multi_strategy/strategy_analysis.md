# Strategy Analysis Report
Generated: 2026-03-29T05:41:48.079598+00:00

## Summary
- Total trades: 133
- Total PnL: -115.5279 USDT
- Overall win rate: 34.6%

## Per-Strategy
| Strategy | Trades | WR | PnL (USDT) | PF | Sharpe | MDD |
|----------|--------|----|-----------|----|----|-----|
| liquidation_fade | 28 | 50% | +51.36 | 1.28 | 2.14 | 104.1% |
| asymmetric_sniper | 32 | 44% | +4.53 | 1.04 | 0.83 | 180.9% |
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
| DOGE | asymmetric_sniper | 3 | 67% | +12.60 |
| SOL | liquidation_fade | 2 | 50% | +11.20 |
| ETH | asymmetric_sniper | 6 | 50% | +11.11 |

## Session Analysis
| Session | Trades | WR | Avg PnL |
|---------|--------|----|---------|
| Asia | 7 | 86% | +10.2068 |
| London | 11 | 27% | +1.8245 |
| NY | 110 | 32% | -1.9984 |
| Off | 5 | 40% | +2.5559 |

## Volatility Regime
| Regime | Trades | WR | Avg PnL |
|--------|--------|----|---------|
| HIGH | 32 | 38% | -2.1582 |
| LOW | 6 | 33% | -3.4161 |
| MED | 95 | 34% | -0.2733 |

## CVD Z-Score Buckets
| Z-Score | Trades | WR | Avg PnL |
|---------|--------|----|---------|
| z1-2 | 15 | 40% | +2.4255 |
| z2-3 | 20 | 45% | -0.8064 |
| z3-4 | 6 | 67% | +6.0765 |
| z<1 | 88 | 28% | -2.1168 |
| z>4 | 4 | 50% | +3.5086 |

## SL/TP Optimization Hints
| Strategy | Current SL ATR× | Suggested SL | Suggested TP | Note |
|----------|----------------|-------------|-------------|------|
| momentum_breakout | ATR×3.0 | ATR×2.0 | ATR×1.5 | SL 적절 |
| cvd_spike | ATR×3.0 | ATR×2.5 | ATR×3.5 | SL 적절 |
| asymmetric_sniper | ATR×3.0 | ATR×1.4 | ATR×2.2 | SL이 너무 넓음 — 리스크 과다 |
| liquidation_fade | ATR×3.0 | ATR×3.4 | ATR×3.4 | SL 적절 |

## Top Recommendations
1. [cvd_spike] 승률 27% — 비활성화 또는 신호 필터 강화
2. London 세션 승률 27% — 해당 세션 거래 중단 고려
3. 저변동성 레짐 승률 33% — 저변동성 시 진입 억제
4. CVD z3-4 구간에서 승률 67% — 이 구간 진입 우선화
5. [asymmetric_sniper] SL 조정: 현재 ATR×3.0 → 권장 ATR×1.4