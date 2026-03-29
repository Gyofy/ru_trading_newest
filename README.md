# Binance Futures Multi-Strategy Bot — v9.1

**바이낸스 선물 자동매매 시스템 | 6개 전략 | StrategySolver v2 자동 최적화 | Demo/Paper/Live 모드**

> **현재 상태**: Demo 트레이딩 (Binance Testnet) — 6전략 edge 탐색 + StrategySolver v2 파라미터 자동 최적화

---

## 시스템 개요

| 항목 | 값 |
|------|-----|
| 버전 | **v9.1** (2026-03-29) |
| 모드 | demo (Binance testnet 실주문) |
| 거래소 | Binance USDM Futures (Testnet) |
| 활성 전략 | 6개 (아래 참조) |
| 초기 자본 | $5,000 USDT (demo) |
| 레버리지 | 3x 통일 |
| 왕복 수수료 | **0.15%** (Post-Only maker 0.02% + taker exit 0.05% + slippage 0.08%) |
| SL Floor | **max(fee x 2.5, 0.40%) = 0.45%** |
| 자동 최적화 | **StrategySolver v2** (3시간 주기, train/test 검증) |
| Discord | 실시간 알림 (진입/청산/1시간 브리핑/EVGuardian/Solver) |

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

## StrategySolver v2 — 자동 파라미터 최적화

### 동작 원리

3시간마다 `trade_context.jsonl`의 완결 거래를 분석하여 전략별 진입 필터 임계치와 R:R을 자동 조정.

```
trade_context.jsonl (완결 거래 50건+)
    ↓
Temporal Split (70% train / 30% test)
    ↓
Train: 신호 메타데이터 분포에서 최적 필터 임계치 탐색
    ↓
Test: 동일 임계치 적용 시 EV > 0 검증 (일반화 확인)
    ↓
통과 시: config.extra 자동 조정 (tighter-only, ±15%)
    ↓
Discord 알림 + solver_state.json 영속화
```

### v1 → v2 주요 개선

| 항목 | v1 | v2 | 근거 |
|------|-----|-----|------|
| 검증 방식 | In-sample only | **Train/Test split (70/30)** | Pardo (2008) Walk-Forward |
| 최소 거래 수 | 30건 | **50건** | Harvey (2014) 표본 크기 |
| WR 개선 기준 | 2%p | **5%p** | 다중 검정 보정 |
| 조정 방향 | 양방향 ±20% | **Tighter-only ±15%** | Liakopoulos (2019) Cautious OCO |
| Config 매핑 | phantom keys | **실제 config.extra 키** | 버그 수정 |
| TP 키 | 전략 무관 tp_rr | **per-strategy** (tp_rr/tp_atr_mult) | liq_fade 호환 |
| 영속화 | 없음 (재시작 시 소실) | **solver_state.json** | 조정값 보존 |
| R:R 하한 | 1.0 | **1.5** | 수수료 감안 현실적 최소 |
| EVGuardian 연동 | 없음 | **suspended 전략 skip** | 충돌 방지 |

### 안전장치

- **50건 이상** 완결 거래 필요 (train 30 + test 10 최소)
- **변경폭 ±15%** — 한 번에 급격한 변경 차단
- **1회 1파라미터** — 다변량 최적화의 curse of dimensionality 회피
- **Tighter-only** — 진입 임계치는 엄격해지는 방향으로만 (완화 차단)
- **Test-set EV > 0 필수** — in-sample에서만 좋은 파라미터 차단
- **MFE 경로 의존성** — R:R 최적화는 upper bound로 표기 (caveat 명시)
- **재시작 생존** — solver_state.json에 조정 이력 영속화, 복원

### 전략별 최적화 대상

| 전략 | 분석 피처 | Config 키 | TP 키 |
|------|----------|-----------|-------|
| CVD Extreme | cvd_z_score | cvd_sigma_mult | tp_rr |
| Cascade Fade | signal_strength, ofi_value | oi_sigma_threshold, taker_spike_mult | tp_atr_mult |
| VWAP Exhaustion | signal_strength | sigma_mult | tp_rr |
| Volume Impulse | signal_strength | volume_mult | tp_rr |
| OI Divergence | signal_strength | oi_change_threshold | tp_rr |

### 이론적 배경 (References)

| 주제 | 핵심 레퍼런스 | 솔버 적용 |
|------|-------------|----------|
| Walk-Forward | Pardo, *Eval & Optim of Trading Strategies* (2008) | Temporal train/test split |
| MFE/MAE | Sweeney, *Maximum Adverse Excursion* (1997) | MFE 분포 기반 R:R 최적화 |
| Data Snooping | White, *Reality Check* (2000); Harvey+Liu+Zhu, RFS (2016) | MIN_WR_IMPROVEMENT 5%p |
| Deflated Sharpe | Bailey & Lopez de Prado, JPM (2014) | 다중 검정 보정 |
| PBO | Bailey & Lopez de Prado, JCF (2017) | Train/test 일반화 검증 |
| Parameter Robustness | Masters, *Testing & Tuning* (2018) | ±15% clamp, plateau test |
| Monotonic OCO | Liakopoulos et al., ICML (2019) | Tighter-only 제약 |
| Sample Size | Harvey, JPM 40th Anniv (2014) | MIN_TRADES 50 |
| Purged CV | Lopez de Prado, AFML Ch.7 & 12 (2018) | Temporal split (purge 적용) |
| Path Dependency | Palomar, HKUST (2024) | MFE 시뮬레이션 caveat |
| Adaptive Tuning | Chan, *Machine Trading* (2017) | 3시간 주기 + regime gate |
| Kelly Criterion | Thorp (2006); Vince (1992) | R:R 최적화 후 사이징 연동 |

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
│ StrategySolver v2 (3h 주기 자동 최적화)                  │
│ Train/Test Split → Tighter-Only → ±15% Clamp → Persist  │
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

### v9.1 (2026-03-29) — 현재
- **StrategySolver v2**: 거래 데이터 기반 파라미터 자동 최적화
  - Temporal train/test split (70/30) — in-sample overfitting 방지
  - Tighter-only 제약 — 진입 필터 엄격화 방향만 허용
  - Per-strategy TP 키 분리 (tp_rr vs tp_atr_mult)
  - solver_state.json 영속화 — 재시작 시 조정값 복원
  - EVGuardian suspended 전략 skip — 시스템 간 충돌 방지
  - 12개 학술 레퍼런스 기반 설계 (Pardo, Sweeney, White, Harvey 등)

### v9.0 (2026-03-29)
- **6전략 edge 탐색 체제**: CVD Extreme, Cascade Fade, VWAP Exhaustion, MTF Momentum, Volume Impulse, OI Divergence
- **수수료 교정**: 0.18% → 0.15% (Post-Only maker 진입 반영)
- **SL Floor**: max(fee x 2.5, 0.40%) = 0.45% — 수수료보다 좁은 SL 원천 차단
- **EVGuardian 활성화**: 30건 이상 시 EV < -0.10% 전략 자동 차단 + allocation fallback
- 균등 배분 $1,500 x 6, 레버리지 3x 통일
- Bot daily cap 100건 + 50건 자동 checkpoint
- 214건 전체 거래 분석 리포트

### v8.5 (2026-03-29)
- Discord 필수화, PID 락, EVGuardian yaml 주입, 데이터 수집 모드

### v8.0~8.4 (2026-03-20~27)
- Multi-strategy 아키텍처, Post-Only 주문, 거래소 SL/TP

---

*최종 업데이트: 2026-03-29 | Gyofy/ru_trading_newest*
