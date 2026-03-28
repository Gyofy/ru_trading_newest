# Binance Futures Multi-Strategy Bot — v8.5

**바이낸스 선물 자동매매 시스템 | 4개 전략 | Demo/Paper/Live 모드 | 데이터 수집 최적화**

> **현재 상태**: Demo 트레이딩 실행 중 (Binance Testnet 실주문) — 알고리즘 개선을 위한 데이터 수집 단계

---

## 목차

1. [시스템 개요](#시스템-개요)
2. [아키텍처](#아키텍처)
3. [전략 상세](#전략-상세)
4. [실거래 성과 분석](#실거래-성과-분석)
5. [설정](#설정)
6. [실행 방법](#실행-방법)
7. [안전장치 및 리스크 관리](#안전장치-및-리스크-관리)
8. [모니터링 및 알림](#모니터링-및-알림)
9. [버전 이력](#버전-이력)

---

## 시스템 개요

| 항목 | 값 |
|------|-----|
| 버전 | **v8.5** (2026-03-29) |
| 모드 | demo (Binance testnet 실주문) |
| 거래소 | Binance USDⓈ-M Futures (Testnet) |
| 활성 전략 | CVD Spike, Liquidation Fade, Asymmetric Sniper |
| 비활성 전략 | Momentum Breakout (enabled: false) |
| 초기 자본 | $5,000 USDT |
| 현재 자본 | ~$4,600 USDT (2026-03-29 기준) |
| Discord | 실시간 알림 연동 (진입/청산/1시간 브리핑) |

---

## 아키텍처

```
[Binance Testnet]
      ↓ 시장 데이터 (OHLCV, 오더북, 펀딩비)
┌─────────────────────────────────────────────────────────────┐
│  DataHub                                                    │
│  - 멀티코인 OHLCV/CVD/OFI/ATR 계산                         │
│  - CoinProfileStore (변동성 프로파일 관리)                  │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  EntryFilters (진입 전 검증)                                │
│  - VPIN 필터 (>= 0.3)                                       │
│  - ATR 레짐 필터 (Percentile >= 10%)                        │
│  - 블랙리스트 (24h 내 SL 2회 이상 -> 거래 정지)             │
└───────────────────────┬─────────────────────────────────────┘
                        ↓ 신호 생성
    ┌───────────────────┼───────────────────┐
    ↓                   ↓                   ↓
[CVD Spike]   [Liquidation Fade]  [Asymmetric Sniper]
  CVD 극단값     청산 캐스케이드     CVD Z-score > 2sigma
  OFI 확인       OI 급감 포착        비대칭 진입
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  PositionSizer + PortfolioRisk                              │
│  - risk_pct_per_trade: 15%                                  │
│  - 최대 노셔널 노출: 500%                                   │
│  - EVGuardian: EV 기반 전략 차단 (데이터 수집 모드 해제)    │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│  ExchangeAdapter (ccxt + Binance Testnet)                   │
│  - SL/TP 거래소 등록 (STOP_MARKET / TAKE_PROFIT_MARKET)     │
│  - SlTpMonitorV2: 15초 폴링 + Trailing SL + 수수료 breakeven│
└─────────────────────────────────────────────────────────────┘
                        ↓
             Discord 실시간 알림
```

---

## 전략 상세

### 1. CVD Spike Reactor (`cvd_spike`)
**목적**: CVD(Cumulative Volume Delta) 극단값 발생 시 역추세 진입

| 파라미터 | 값 |
|---------|-----|
| Allocation | $9,000 |
| Leverage | 6x |
| Max Positions | 10 |
| SL | ATR x 5.0 |
| TP | SL x 3.0 (R:R 3:1) |
| CVD Quantile | 0.92 (상위 8% 돌파) |
| OFI Quantile | 0.92 |
| Volume Spike | 1.5x 평균 |
| Trailing SL | ATR x 0.7 |

**신호 조건**: CVD 극단값 + OFI 확인 + 볼륨 스파이크 + OI 하락 (-0.2% 이상)

---

### 2. Liquidation Fade (`liquidation_fade`)
**목적**: 청산 캐스케이드 후 반등 포착

| 파라미터 | 값 |
|---------|-----|
| Allocation | $4,500 |
| Leverage | 5x |
| Max Positions | 6 |
| SL | ATR x 2.5 |
| TP | ATR x 7.5 |
| OI Sigma Threshold | 1.2sigma |
| Taker Spike | 1.2x |

**신호 조건**: OI 급감 (1.2sigma 이상) + Taker 매도 스파이크 + 스윙 저점 근접

---

### 3. Asymmetric Sniper (`asymmetric_sniper`)
**목적**: CVD Z-score 2sigma 이상 극단 신호에서 비대칭 진입

| 파라미터 | 값 |
|---------|-----|
| Allocation | $9,000 |
| Leverage | 8x |
| Max Positions | 6 |
| SL | ATR x 1.0 |
| Risk/Trade | $200 |
| CVD Z-score | >= 1.5sigma |
| CVD Quantile | 0.92 |
| Trailing SL (초기) | ATR x 0.8 |
| Trailing SL (타이트) | ATR x 0.4 (이익 0.8 ATR 이상 시) |
| 쿨다운 | 5봉 |
| 일일 최대 거래 | 20건 |

---

### 4. Momentum Breakout (`momentum_breakout`) — 비활성
**목적**: 볼륨 확인된 레인지 돌파 (현재 데이터 수집 목적으로 비활성화)

---

## 실거래 성과 분석

> **기간**: 2026-03-27 ~ 2026-03-29 (Testnet Demo Trading)
> **데이터**: 153건 거래 중 104건 완전 수수료 기록 기준

### 전략별 성과

| 전략 | 거래수 | 승률 | Gross PnL | 수수료 | Net PnL |
|------|--------|------|-----------|--------|---------|
| CVD Spike | 49 | 20.2% | -$132 | $491 | **-$623** |
| Liquidation Fade | 27 | **48.3%** | **+$152** | $190 | -$37 |
| Asymmetric Sniper | 24 | 7.1% | -$16 | $533 | **-$549** |
| Momentum Breakout | 4 | 8.3% | -$77 | $34 | -$112 |
| **합계** | **104** | **22.2%** | **-$74** | **$1,247** | **-$1,321** |

### 핵심 발견: 수수료가 손실의 94.4%

```
총 손실:    -$1,321
수수료:     -$1,247  (손실의 94.4%)
Gross PnL:    -$74  (수수료 제외 시 거의 브레이크이븐)

결론: 전략의 방향성은 거의 맞다.
      수수료가 수익을 완전히 잠식하는 구조 문제.
```

### 수수료 문제 원인 분석

1. **SL 거리가 수수료 대비 너무 좁음**
   - 평균 SL 거리: 0.265% vs 왕복 수수료 0.18%
   - SL 히트 시 수수료가 손실의 68%를 차지
   - 27% 거래에서 SL < 0.20% (수수료와 거의 같은 수준)

2. **Asymmetric Sniper 레버리지 과다**
   - 8x 레버리지 -> 노셔널 증가 -> 수수료 절댓값 증가
   - 24건에 $533 수수료 = 건당 $22 (가장 높음)

3. **TP 히트율 극단적으로 낮음**
   - 전체 TP 히트: 11/87 = **12.6%**
   - SL 히트: 76/87 = **87.4%**

### 개선 방향 (다음 버전)

| 문제 | 현재 | 개선 방향 |
|------|------|----------|
| SL 거리 | ~0.27% 평균 | ATR 최소 0.5% 이상 보장 |
| Asymmetric Sniper SL | ATR x 1.0 | ATR x 2.0 이상으로 확대 |
| Asymmetric Sniper 레버리지 | 8x | 5-6x로 낮춰 수수료 절댓값 감소 |
| TP hit rate | 12.6% | 신호 품질 향상, 진입 시점 최적화 |

> **Liquidation Fade**: 유일하게 Gross +$152 기록 — 신호 품질과 R:R 방향이 올바름

### 청산 사유 분포 (153건 전체)

| 사유 | 건수 | 비율 |
|------|------|------|
| SL_HIT | 124 | 81.0% |
| TP_HIT | 12 | 7.8% |
| EARLY_EXIT_NO_MFE | 9 | 5.9% |
| TIME_STOP | 4 | 2.6% |
| GHOST_CLEANUP | 4 | 2.6% |

### 수수료 구조 (Binance USDM Futures VIP 0)

| 항목 | 비율 |
|------|------|
| Taker Fee | 0.05% (편도) |
| Maker Fee | 0.02% (편도) |
| 진입 슬리피지 | 0.03% |
| 청산 슬리피지 | 0.05% |
| **왕복 총비용 (ROUND_TRIP)** | **0.18%** |
| 강제청산 수수료 | 0.5% (별도) |

---

## 설정

### `config/multi_strategy.yaml` 주요 파라미터

```yaml
mode: "demo"
initial_equity: 5000.0
daily_loss_limit: 0.20

position_sizing:
  risk_pct_per_trade: 0.15
  max_factor: 3.0

exchange:
  taker_fee: 0.0005      # VIP 0 taker 0.05%
  maker_fee: 0.0002      # VIP 0 maker 0.02%
  slippage_entry: 0.0003
  slippage_exit: 0.0005
  liquidation_fee: 0.005 # 강제청산 0.5%

ev_guardian:
  fee_budget_pct: 100.0  # 데이터 수집 모드: 사실상 비활성화
  ev_threshold: -1.0     # F1_EV 차단 비활성화 (데이터 수집 모드)
  ev_min_sample: 9999    # EV 판단 유예
```

---

## 실행 방법

### 기본 실행

```bash
cd /home/wlsry/ru_trading_newest
nohup python3 run_multi_strategy.py --config config/multi_strategy.yaml >> logs/bot.log 2>&1 &
```

**시작 시 자동 검증:**
1. PID 파일 락 확인 — 중복 실행 시 즉시 종료
2. Discord webhook 연결 확인 — 실패 시 즉시 종료

### 환경변수 (`.env`)

```
BINANCE_TESTNET_API_KEY=...
BINANCE_TESTNET_API_SECRET=...
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

### 봇 종료

```bash
kill $(cat data/reports/multi_strategy/bot.pid)
```

### Config 핫리로드 (재시작 없이)

```bash
kill -HUP $(cat data/reports/multi_strategy/bot.pid)
```

### EVGuardian 리셋 (전략 정지 해제 후 재시작)

```bash
touch data/reports/multi_strategy/ev_reset.flag
kill $(cat data/reports/multi_strategy/bot.pid)
# 재시작 시 자동 적용
```

---

## 안전장치 및 리스크 관리

| 항목 | 설정값 | 설명 |
|------|--------|------|
| 중복 실행 방지 | PID 파일 락 | 동시에 두 봇 실행 불가 |
| Discord 연동 필수 | 시작 시 체크 | webhook 실패 시 봇 시작 안 함 |
| 일일 손실 한도 | 20% | 초과 시 신규 진입 차단 |
| 전략별 손실 한도 | 40% | 전략별 독립 차단 |
| Max Exposure | 500% | 전체 노셔널 합산 |
| SL 필수 | 항상 | 모든 포지션에 거래소 SL 등록 |
| 수수료 브레이크이븐 SL | 자동 | entry x (1 + 0.18%) 이상에서만 Trailing SL 이동 |
| 강제청산 수수료 | 0.5% 별도 계상 | worst-case 반영 |

---

## 모니터링 및 알림

### Discord 알림 종류
- **진입 알림**: 코인, 전략, 진입가, SL/TP, R:R
- **청산 알림**: PnL (Gross + 수수료 + Net), MFE/MAE, 세션 통계
- **1시간 브리핑**: 잔고, 오픈 포지션, EVGuardian 상태
- **15분 신호 스캔**: 변동성 상위 30개 전체 신호 분석
- **긴급 알림**: SL 5회 실패 시 즉시 전송

### 로그 파일
```
data/reports/multi_strategy/
├── bot.log              # 구조화 로그 (RotatingFile 10MB x 5)
├── bot.pid              # 실행 중 PID (중복 실행 방지)
├── trades.jsonl         # 거래 기록 (진입~청산 완전 이력)
├── trade_context.jsonl  # EVGuardian EV 평가 데이터
├── equity_state.json    # 세션 자본 상태
├── positions.json       # 오픈 포지션 (봇 재시작 시 복구)
├── ev_report.json       # EVGuardian 최신 EV 통계
└── strategy_analysis.md # 전략별 자동 분석 리포트
```

---

## 버전 이력

### v8.5 (2026-03-29) — 현재
- **Discord 연동 필수화**: 봇 시작 시 webhook 연결 확인 → 실패 시 즉시 종료
- **중복 실행 원천 차단**: PID 파일 락 (`bot.pid`) 추가
- **EVGuardian 설정 가능화**: `ev_threshold`, `ev_min_sample`을 yaml에서 주입 가능
- **데이터 수집 모드**: `ev_threshold: -1.0` -> F1_EV 전략 차단 비활성화
- 수수료 기반 Trailing SL breakeven: `entry x (1 + ROUND_TRIP_FEE_RATE)`
- PnL 정확도 수정: SL/TP 히트 시 `pos.sl_price` / `pos.tp_price` 사용
- CancelledError ghost position 버그 수정 (asyncio 명시적 처리)
- 로그 RotatingFileHandler (10MB x 5 = 최대 60MB)
- JSONL atomic write (fsync 보장)

### v8.4 (2026-03-27)
- Binance 수수료 오류 수정: taker 0.00055 -> 0.0005 (0.055% -> 0.05%)
- SL 최소 거리: `ROUND_TRIP x 1.5` -> `1.0`
- 진입 조건 완화: CVD Q97->Q92, Volume 2.0x->1.5x, OI -0.5%->-0.2%
- 일일 수수료 cap 비활성화 (`fee_budget_pct: 100.0`)
- TradeLogger: `pnl_net_usdt`, `fee_usdt` 완전 기록

### v8.0 ~ v8.3 (2026-03-20 ~ 26)
- Multi-strategy 아키텍처 (CVD Spike / Liquidation Fade / Momentum Breakout / Asymmetric Sniper)
- PortfolioRiskManager, EVGuardian, EntryFilters 구현
- 동적 코인 선택 (변동성 상위 30개)
- Post-Only 주문 + taker fallback
- 1시간 브리핑 + 15분 신호 스캔 Discord 리포트

---

*최종 업데이트: 2026-03-29 | Gyofy/ru_trading_newest*
