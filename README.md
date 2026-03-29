# Binance Futures Multi-Strategy Bot — v9.0

**바이낸스 선물 자동매매 시스템 | 6개 전략 | Demo/Paper/Live 모드 | Edge 탐색 + 수수료 최적화**

> **현재 상태**: Demo 트레이딩 (Binance Testnet) — 6전략 edge 탐색 + 수수료 포함 수익 모델 검증

---

## 시스템 개요

| 항목 | 값 |
|------|-----|
| 버전 | **v9.0** (2026-03-29) |
| 모드 | demo (Binance testnet 실주문) |
| 거래소 | Binance USDM Futures (Testnet) |
| 활성 전략 | 6개 (아래 참조) |
| 초기 자본 | $5,000 USDT (demo) |
| 레버리지 | 3x 통일 |
| 왕복 수수료 | **0.15%** (Post-Only maker 0.02% + taker exit 0.05% + slippage 0.08%) |
| SL Floor | **max(fee x 2.5, 0.40%) = 0.45%** |
| Discord | 실시간 알림 (진입/청산/1시간 브리핑/EVGuardian) |

---

## 6개 전략

| # | 전략 | 유형 | 가설 | R:R | BE WR | 배분 |
|---|------|------|------|-----|-------|------|
| 1 | **CVD Extreme** | 역추세 | CVD Q92+Z1.5σ orderflow 소진 | 2.5:1 | 37% | $1,500 |
| 2 | **Cascade Fade** | 역추세 | OI급감 + 거래량 스파이크 → 청산 반전 | 2.0:1 | 46% | $1,500 |
| 3 | **VWAP Exhaustion** | 역추세 | VWAP 이탈 + 거래량 감소 = 소진 복귀 | 2.0:1 | 43% | $1,500 |
| 4 | **MTF Momentum** | 추세추종 | 1m+15m+1h EMA 방향 정렬 | 3.0:1 | 33% | $1,500 |
| 5 | **Volume Impulse** | 추세추종 | 거래량 3x 폭발 + 방향 추종 | 3.0:1 | 33% | $1,500 |
| 6 | **OI Divergence** | 혼합 | 가격-OI 괴리 4패턴 구조적 불균형 | 2.5:1 | 37% | $1,500 |

*BE WR = Break-Even Win Rate (수수료 0.15% 포함)*

### 수수료 구조 (Binance USDM Futures VIP 0)

```
수수료 = Notional x Rate (마진이 아닌 레버리지 포함 금액에 부과!)

Entry (Post-Only Maker):  0.02% + 0.03% slippage = 0.05%
Exit (Taker SL/TP):       0.05% + 0.05% slippage = 0.10%
왕복 합계:                0.15% of notional

3x 레버리지 시 자본 대비: 0.15% x 3 = 0.45%/건
```

---

## 아키텍처

```
[Binance Testnet] ← 시장 데이터 (OHLCV 1m/5m/15m/1h, OI, Funding)
      ↓
┌─────────────────────────────────────────────────────────┐
│ DataHub (캐시: OHLCV 58s, Ticker 10s)                   │
│ CVD / OFI / VWAP / VPIN / ATR 계산                      │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌──────────────────────┴──────────────────────────────────┐
│ EntryFilters: VPIN ≥ 0.3 + ATR Regime P15+ + Blacklist  │
└──────────────────────┬──────────────────────────────────┘
                       ↓
  ┌────────┬────────┬────────┬────────┬────────┬────────┐
  │CVD Ext │Cascade │VWAP Ex │MTF Mom │Vol Imp │OI Div  │
  │역추세  │역추세  │역추세  │추세추종│추세추종│혼합    │
  └───┬────┴───┬────┴───┬────┴───┬────┴───┬────┴───┬────┘
      └────────┴────────┴───┬────┴────────┴────────┘
                            ↓
┌───────────────────────────┴─────────────────────────────┐
│ PortfolioRisk (9 gates) + PositionSizer + EVGuardian    │
│ SL Floor 0.45% | Daily Cap 100 | Allocation Fallback    │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌──────────────────────┴──────────────────────────────────┐
│ ExchangeAdapter (ccxt + Binance Testnet)                │
│ Post-Only Entry → SL/TP FOR POSITION → SlTpMonitorV2    │
└─────────────────────────────────────────────────────────┘
```

---

## 설정

```yaml
# config/multi_strategy.yaml 핵심
version: "v9.0-cvd-extreme"
mode: "demo"
initial_equity: 5000.0

strategies:         # 균등 $1,500, 3x 통일
  cvd_extreme:      { enabled: true, R:R 2.5 }
  liquidation_fade: { enabled: true, R:R 2.0 }
  vwap_reversion:   { enabled: true, R:R 2.0 }
  funding_arb:      { enabled: true, R:R 3.0 }  # MTF Momentum
  volume_impulse:   { enabled: true, R:R 3.0 }
  oi_divergence:    { enabled: true, R:R 2.5 }

ev_guardian:
  ev_threshold: -0.001   # EV < -0.10% → 자동 차단 + allocation fallback
  ev_min_sample: 30
```

---

## 실행

```bash
nohup python3 run_multi_strategy.py --config config/multi_strategy.yaml >> logs/bot.log 2>&1 &
```

---

## 버전 이력

### v9.0 (2026-03-29) — 현재
- **6전략 edge 탐색 체제**: CVD Extreme, Cascade Fade, VWAP Exhaustion, MTF Momentum, Volume Impulse, OI Divergence
- **수수료 교정**: 0.18% → 0.15% (Post-Only maker 진입 반영)
- **SL Floor**: max(fee x 2.5, 0.40%) = 0.45% — 수수료보다 좁은 SL 원천 차단
- **EVGuardian 활성화**: 30건 이상 시 EV < -0.10% 전략 자동 차단 + allocation fallback
- 균등 배분 $1,500 x 6, 레버리지 3x 통일
- Bot daily cap 100건 + 50건 자동 checkpoint
- Cascade Fade: 실제 OI 연동 + 5m ATR 스케일링
- Volume Impulse: CVD 방향 일치 A/B 태그
- 214건 전체 거래 분석 리포트 (docs/v9_trade_analysis_report.md)

### v8.5 (2026-03-29)
- Discord 필수화, PID 락, EVGuardian yaml 주입, 데이터 수집 모드

### v8.0~8.4 (2026-03-20~27)
- Multi-strategy 아키텍처, Post-Only 주문, 거래소 SL/TP

---

*최종 업데이트: 2026-03-29 | Gyofy/ru_trading_newest*
