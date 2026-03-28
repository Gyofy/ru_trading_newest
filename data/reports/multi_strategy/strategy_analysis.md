# Strategy Analysis Report
Generated: 2026-03-28T01:45:18.931696+00:00

## Summary
- Total trades: 94
- Total PnL: -159.5747 USDT
- Overall win rate: 29.8%

## Per-Strategy
| Strategy | Trades | WR | PnL (USDT) | PF | Sharpe | MDD |
|----------|--------|----|-----------|----|----|-----|
| liquidation_fade | 11 | 45% | +32.80 | 1.40 | 2.78 | 104.1% |
| asymmetric_sniper | 10 | 50% | -20.95 | 0.53 | -5.84 | 0.0% |
| momentum_breakout | 7 | 0% | -61.69 | 0.00 | -20.66 | 0.0% |
| cvd_spike | 66 | 27% | -109.73 | 0.55 | -4.74 | 0.0% |

## Per-Coin Top 10
| Coin | Strategy | Trades | WR | PnL |
|------|----------|--------|----|-----|
| OP | cvd_spike | 6 | 33% | +39.33 |
| OP | liquidation_fade | 3 | 67% | +36.47 |
| DOT | liquidation_fade | 1 | 100% | +26.49 |
| TAO | liquidation_fade | 1 | 100% | +18.22 |
| ADA | cvd_spike | 4 | 50% | +9.42 |
| SUI | cvd_spike | 2 | 100% | +5.32 |
| ADA | asymmetric_sniper | 2 | 50% | +4.84 |
| BTC | asymmetric_sniper | 2 | 100% | +4.02 |
| XRP | asymmetric_sniper | 1 | 100% | +1.96 |
| DOT | cvd_spike | 1 | 0% | +0.00 |

## Session Analysis
| Session | Trades | WR | Avg PnL |
|---------|--------|----|---------|
| Asia | 1 | 0% | -16.8350 |
| London | 5 | 0% | -1.3329 |
| NY | 88 | 32% | -1.5463 |

## Volatility Regime
| Regime | Trades | WR | Avg PnL |
|--------|--------|----|---------|
| HIGH | 19 | 32% | -2.3386 |
| LOW | 2 | 0% | -5.1128 |
| MED | 73 | 30% | -1.4372 |

## CVD Z-Score Buckets
| Z-Score | Trades | WR | Avg PnL |
|---------|--------|----|---------|
| z1-2 | 10 | 50% | +3.8219 |
| z2-3 | 9 | 44% | -0.5373 |
| z3-4 | 2 | 100% | +1.5549 |
| z<1 | 71 | 22% | -2.8850 |
| z>4 | 2 | 50% | +4.3842 |

## SL/TP Optimization Hints
| Strategy | Current SL ATR× | Suggested SL | Suggested TP | Note |
|----------|----------------|-------------|-------------|------|
| momentum_breakout | ATR×3.0 | ATR×2.0 | ATR×1.5 | SL 적절 |
| cvd_spike | ATR×3.0 | ATR×2.5 | ATR×3.5 | SL 적절 |
| asymmetric_sniper | ATR×3.0 | ATR×1.4 | ATR×2.2 | SL이 너무 넓음 — 리스크 과다 |
| liquidation_fade | ATR×3.0 | ATR×3.4 | ATR×3.4 | SL 적절 |

## Top Recommendations
1. [cvd_spike] 승률 27% — 비활성화 또는 신호 필터 강화
2. London 세션 승률 0% — 해당 세션 거래 중단 고려
3. [asymmetric_sniper] SL 조정: 현재 ATR×3.0 → 권장 ATR×1.4