# Binance Futures Multi-Strategy Bot — v8.5

**바이낸스 선물 자동매매 시스템 | 3개 활성 전략 | Demo/Paper/Live 모드 지원**

> v8.5는 v8.4 기반에서 수수료 기반 순수익 정확도를 높이고, 진입 조건 완화를 통해 데이터 수집 효율을 높였다.
> CRITICAL 버그 2건(PnL 오기록, CancelledError ghost position) 수정, trailing SL 수수료 기반 breakeven 적용,
> 신호 임계값 완화(Q97→Q92), ATR 레짐 필터 완화(P20→P10), 일일 수수료 cap 비활성화.

---

## 목차

1. [시스템 개요](#시스템-개요)
2. [아키텍처](#아키텍처)
3. [전략 상세](#전략-상세)
4. [설정](#설정)
5. [실행 방법](#실행-방법)
6. [리스크 관리](#리스크-관리)
7. [모니터링 및 로그](#모니터링-및-로그)
8. [버전 이력](#버전-이력)
9. [환경 설정](#환경-설정)

---

## 시스템 개요

| 항목 | 값 |
|------|-----|
| 버전 | v8.5 (multi-strategy demo trading) |
| 기본 모드 | demo (Binance testnet 실주문) |
| 초기 가상 자본 | $5,000 |
| 전략 수 | 4개 (병렬 동시 실행) |
| 평가 대상 코인 | 13개 base + 동적 확장 (APT/TAO 제외) |
| 사이클 | 1분 (전략별) |
| 포지션 진입 방식 | Post-Only Maker (GTX), LIMIT 폴백 |
| SL/TP 처리 | 거래소 측 closePosition=True (demo/live) |
| SL/TP 재등록 | 모니터가 15초마다 누락 여부 자동 감지·재등록 |
| 수수료 반영 | TP = SL × RR + round-trip fee (0.190%) |

### 운영 모드

| 모드 | 설명 | 주문 |
|------|------|------|
| `paper` | 시뮬레이션 체결 (bid/ask 가격) | 실제 주문 없음 |
| `demo` | Binance testnet 실주문 | 가상 자금 |
| `live` | 실계좌 실매매 | **실제 자금 손실 가능** |

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│  run_multi_strategy.py  (진입점)                            │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  DataHub  — 1분/5분 OHLCV + CVD/OFI/청산 데이터 수집 │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │ asyncio.gather (15 coins × 4 전략) │
│                         ▼                                   │
│  ┌────────────┐  ┌──────────────┐  ┌───────────────┐  ┌───────────────┐
│  │ CVD Spike  │  │  Liq. Fade   │  │ Mom. Breakout │  │ Asym. Sniper  │
│  │  1분 사이클 │  │  5분 사이클  │  │  5분 사이클   │  │  1분 사이클   │
│  │ $900 / 3x  │  │  $600 / 2x   │  │  $600 / 3x    │  │ $2400 / 5x    │
│  └─────┬──────┘  └──────┬───────┘  └───────┬───────┘  └──────┬────────┘
│        └────────────────┴──────────────────┴──────────────────┘
│                         │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │  PortfolioRiskManager                                │  │
│  │  - exposure_cap: 250%  - same_dir_max: 8            │  │
│  │  - daily_loss_limit: -20%  - strategy_pause: -40%   │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │  MultiPositionManager + PositionSizer                │  │
│  │  CoinProfileStore (코인별 adaptive params)           │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │  ExchangeAdapter (ccxt Binance USDM Futures)         │  │
│  │  Post-Only 진입 → SL/TP 거래소 등록 → 포지션 모니터 │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │  SlTpMonitorV2 (15초 폴링)                           │  │
│  │  OrderLedger + TradeLogger + StrategyAnalyzer        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

Discord Webhook ← 진입/청산/에러 알림
data/reports/multi_strategy/ ← 로그, 상태 파일
```

---

## 전략 상세

### A. CVD Spike Reactor (`cvd_spike`)

**개념:** Order Flow Imbalance(OFI)와 Cumulative Volume Delta(CVD)가 동시에 극단적 수치에 도달할 때 역추세 포지션을 취한다.

| 파라미터 | 값 |
|---------|-----|
| 사이클 | 1분 |
| 할당 자본 | $900 |
| 레버리지 | 3x |
| CVD/OFI 임계 | Q92 (상위 8%) |
| CVD 롤링 윈도우 | 240봉 (4시간) |
| SL | ATR × 3.0 |
| 트레일링 SL | ATR × 1.5 |
| 최대 동시 포지션 | 8개 |

**진입 조건:**
- CVD 단기 스파이크 ≥ Q92 분위수
- OFI ≥ Q92 분위수
- 두 조건 동시 충족 시 반대 방향 진입 (매수 스파이크 → SHORT)

---

### B. Liquidation Fade (`liquidation_fade`)

**개념:** 강제 청산 캐스케이드로 인한 일시적 가격 왜곡을 이용한다. 청산 급증 + Taker 스파이크 발생 직후 역행 포지션을 취한다.

| 파라미터 | 값 |
|---------|-----|
| 사이클 | 5분 |
| 할당 자본 | $600 |
| 레버리지 | 2x |
| OI 변화 임계 | 1.5σ |
| Taker 스파이크 | 평균 × 1.5배 |
| 스윙 조회 기간 | 24봉 |
| SL | ATR × 2.5 |
| TP | ATR × 3.75 (RR ≈ 1.5) |
| 최대 동시 포지션 | 4개 |

**진입 조건:**
- Open Interest 급감 (청산 발생 감지)
- Taker 매수/매도량 이상 스파이크
- 스윙 고/저점 근처에서 진입

---

### C. Momentum Breakout (`momentum_breakout`)

**개념:** 거래량이 동반된 가격 돌파를 추세추종한다. 12시간 레인지를 기준으로 볼륨 확인된 상방/하방 돌파 시 진입한다.

| 파라미터 | 값 |
|---------|-----|
| 사이클 | 5분 |
| 할당 자본 | $600 |
| 레버리지 | 3x |
| 볼륨 배수 임계 | 평균 × 2.0배 |
| 레인지 기준 | 12시간 고/저점 |
| 최소 이동폭 | 0.5% |
| SL | ATR × 2.0 |
| 트레일링 SL | ATR × 2.0 |
| 최대 동시 포지션 | 4개 |

**진입 조건:**
- 12시간 고점/저점 돌파
- 현재 거래량 ≥ 20봉 평균 × 2.0
- 최소 0.5% 이동 확인

---

### D. Asymmetric Sniper (`asymmetric_sniper`)

**개념:** CVD Q97 + 3σ 수준의 극단적 orderflow 이상 발생 시 저위험 역추세 포지션. 단위 트레이드당 고정 달러 리스크로 운영된다.

| 파라미터 | 값 |
|---------|-----|
| 사이클 | 1분 |
| 할당 자본 | $2,400 |
| 레버리지 | 5x |
| CVD 임계 | Q97 (상위 3%) |
| CVD 시그마 | 2.0σ 이상 |
| 트레이드당 리스크 | $60 고정 |
| 최소 RR | 3.0 (testnet: 비활성) |
| SL | ATR × 1.0 |
| 쿨다운 | 15분 |
| 최대 일일 트레이드 | 8회 |
| 트레일링 SL (초기) | ATR × 2.0 |
| 트레일링 SL (긴축) | ATR × 1.0 (수익 ATR × 1.5 초과 시) |
| 최대 동시 포지션 | 4개 |

**진입 조건:**
- CVD ≥ Q97 분위수 AND ≥ 2σ (극단적 쏠림)
- Funding rate < 0.01% (testnet: 비활성)
- 쿨다운 경과 확인
- 일일 거래 한도 미초과

---

## 설정

설정 파일: `config/multi_strategy.yaml`

### 포트폴리오 설정

```yaml
version: "v8.2-demo"
mode: "demo"          # paper | demo | live
initial_equity: 5000.0
daily_loss_limit: 0.20  # -20% 시 전략 중지

portfolio:
  total_exposure_pct: 2.5    # 250% 노출 상한 (공격적)
  same_direction_max: 8      # 동방향 최대 포지션 수
  daily_loss_pct: 0.20       # 일일 손실 한도
  strategy_loss_pct: 0.40    # 전략별 일시 중단 손실 임계
  min_notional_usdt: 5.0     # 최소 주문 금액
  max_funding_rate: 0.003    # 펀딩비 게이트

position_sizing:
  risk_pct_per_trade: 0.15   # 트레이드당 리스크 15%
  vol_adjust_enabled: true
  min_factor: 0.7
  max_factor: 3.0             # 최대 포지션 크기 배수
```

### Base 코인 목록 (15개)

```
XRP / SOL / TAO / DOGE / ADA / BTC / ETH / BNB / DOT / AVAX / LINK / OP / ARB / SUI / APT
```

동적 코인 기능이 활성화된 경우 변동성 상위 30개 풀에서 추가 선택된다 (15분 갱신).

### 전략별 자본 배분 요약

| 전략 | 할당 | 레버리지 | 사이클 |
|------|------|---------|--------|
| CVD Spike | $900 | 3x | 1분 |
| Liquidation Fade | $600 | 2x | 5분 |
| Momentum Breakout | $600 | 3x | 5분 |
| Asymmetric Sniper | $2,400 | 5x | 1분 |
| **합계** | **$4,500** | | |

### 수수료 설정

```yaml
exchange:
  maker_fee: 0.0002     # 0.02% (Post-Only 진입)
  taker_fee: 0.00055    # 0.055% (SL/TP 청산)
  slippage_entry: 0.0003
  slippage_exit: 0.0005
```

---

## 실행 방법

### 필수 환경 변수

```bash
export BINANCE_API_KEY="your_testnet_api_key"
export BINANCE_API_SECRET="your_testnet_api_secret"
export DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."   # 선택
```

### 실행

```bash
# Demo 모드 (기본, testnet 실주문)
python3 run_multi_strategy.py

# Paper 모드 (완전 시뮬레이션)
python3 run_multi_strategy.py --mode paper

# 커스텀 설정 파일 사용
python3 run_multi_strategy.py --config config/my_config.yaml

# Live 모드 (실계좌 — 주의)
python3 run_multi_strategy.py --mode live
```

### Live 모드 전환 체크리스트

Live 모드는 `BINANCE_TESTNET_API_KEY` 환경 변수가 **설정되지 않은** 경우 활성화된다.

- [ ] Demo 모드에서 최소 2주 이상 가동 확인
- [ ] 총 거래 50건 이상, WR > 45%, net PnL > 0 확인
- [ ] `initial_equity` 실제 계좌 잔고로 변경
- [ ] `mode: "live"` 설정
- [ ] `position_sizing.risk_pct_per_trade` 축소 (0.05 권장 시작)
- [ ] Discord 알림 webhook 설정 완료
- [ ] `daily_loss_limit: 0.05` (초기 5% 제한 권장)

---

## 리스크 관리

### 포트폴리오 레벨 게이트 (PortfolioRiskManager)

| 조건 | 임계값 | 처리 |
|------|--------|------|
| 일일 손실 | -20% | 전 전략 진입 중지 |
| 전략별 손실 | -40% | 해당 전략 일시 중단 |
| 총 노출도 초과 | 250% 이상 | 신규 진입 차단 |
| 동방향 포지션 초과 | 8개 초과 | 신규 진입 차단 |
| 펀딩비 초과 | 0.3% 초과 | 해당 포지션 방향 진입 차단 |

### 포지션 레벨 안전장치

| 항목 | 설명 |
|------|------|
| SL | 모든 포지션 필수 (전략별 ATR 배수) |
| 트레일링 SL | CVD Spike, Momentum Breakout, Asymmetric Sniper 적용 |
| 최소 주문금액 | $5 (notional 기준) |
| Post-Only 진입 | Maker 수수료 보장, GTX 거부 시 일반 LIMIT 폴백 |
| Rate Limit | asyncio.Semaphore(2) — 동시 API 호출 2개 제한 |

### 코인별 적응형 파라미터 (CoinProfileStore)

각 코인의 변동성, 스프레드, 유동성 특성을 학습하여 포지션 크기를 자동 조정한다.

```
고변동성 코인 (예: TAO) → 포지션 크기 축소
저변동성 코인 (예: BTC) → 포지션 크기 확대 또는 유지
```

---

## 모니터링 및 로그

### 로그 파일 위치

```
data/reports/multi_strategy/
├── bot.log                  # 메인 로그 (INFO/ERROR)
├── positions.json           # 현재 오픈 포지션 상태
├── portfolio_state.json     # 포트폴리오 요약 (equity, PnL)
├── trades.jsonl             # 전체 거래 내역 (JSONL)
├── strategy_stats.json      # 전략별 성과 통계
└── coin_profiles.json       # 코인별 학습된 파라미터
```

### Discord 알림 이벤트

| 이벤트 | 조건 |
|--------|------|
| 진입 알림 | 신규 포지션 진입 시 |
| 청산 알림 | SL/TP 체결 또는 수동 청산 시 |
| 일일 손실 경고 | 손실 -10% 초과 시 |
| 전략 중단 알림 | 전략별 손실 임계 초과 시 |
| 에러 알림 | 연결 끊김, 주문 실패 등 |

### 헬스체크

봇은 60초마다 heartbeat 로그를 출력하며, SL/TP 상태는 15초 주기로 폴링한다.

```bash
# 로그 실시간 확인
tail -f data/reports/multi_strategy/bot.log

# 현재 포지션 확인
cat data/reports/multi_strategy/positions.json | python3 -m json.tool

# 거래 내역 최근 10건
tail -n 10 data/reports/multi_strategy/trades.jsonl | python3 -m json.tool
```

---

## 버전 이력

| 버전 | 전략 | 주요 변경 | 상태 |
|------|------|---------|------|
| v4.x ~ v5.x | ML 2-Stage Binary | 데이터 leakage 확인, Sharpe 과대평가 | 폐기 |
| v6.x | TSMOM BTC Dir + RS | BTC 방향 + 상대강도 선택 (1h) | 폐기 |
| v8.1 | Multi-Strategy 초기 | CVD/OFI/청산/모멘텀 프레임워크 구축 | 구버전 |
| v8.2 | Multi-Strategy | asyncio 병렬 평가, funding 필터 우회, R:R 오버라이드 | 구버전 |
| v8.3 | Multi-Strategy | 수수료 반영 TP/SL, SL 3회 실패 강제청산, trailing 축소, BTC macro 로그 | 구버전 |
| v8.4 | Multi-Strategy | 레버리지 +1/수량 2배, SL/TP 자동재등록, naked position 보존, shutdown SL/TP 취소, APT/TAO 제외 | 구버전 |
| **v8.5** | **Multi-Strategy 현재** | CRITICAL 버그 수정, 수수료 기반 trailing SL, 신호 임계값 완화, 데이터 수집 모드 | **현재** |

### v8.5 주요 변경점 (vs v8.4)

**CRITICAL 버그 수정**
- `run_multi_strategy.py`: SL_HIT/TP_HIT 시 `ticker["last"]` 대신 `pos.sl_price`/`pos.tp_price` 사용 → PnL 정확도 향상 및 데일리 손실 한도 정확도 보장
- `run_multi_strategy.py`: `market_close` await 중 `asyncio.CancelledError` 명시적 처리 → ghost position 방지

**수수료 기반 Trailing SL (`sl_tp_monitor_v2.py`)**
- Breakeven trigger: 고정 0.3% → 동적 `(round-trip fee 0.19% + trail_dist%)` 계산
- Breakeven SL: entry → `entry × (1 + 0.19%)` — 청산 시 수수료 보전 보장
- `bars_held == 5` 버그 수정 → `>= 5` + `_zero_mfe_checked` 플래그 (1회만 발동)

**신호 임계값 완화 (데이터 수집 효율화)**
- CVD Spike: cvd/ofi quantile 0.97→0.92, volume_spike_mult 2.0→1.5, oi_decline_pct -0.5→-0.2, sl_atr_mult 3.0→5.0
- Asymmetric Sniper: cvd_sigma_mult 1.8→1.5, cooldown_bars 10→5, max_daily_trades 12→20
- ATR 레짐 필터: atr_min_percentile 20→10
- SL 최소 거리 임계: 1.5× fee → 1.0× fee

**PnL 수수료 반영 통일**
- 세션 통계, Discord 알림, Ledger, 종료 요약 모두 `pnl_net_usdt` (수수료 차감 순수익) 기준으로 통일
- 일일 수수료 cap 비활성화 (`fee_budget_pct: 100.0`) — 데이터 수집 모드에서 불필요한 차단 방지

**기타**
- `set_leverage()` → `set_leverage_async()`: 이벤트 루프 blocking 해소
- `exclude_coins: [APT, TAO]`: testnet WaitFill timeout 코인 완전 제외 (base + dynamic 양쪽)
- Config 버전 v8.3-demo → v8.4-demo

---

## 환경 설정

### 의존성 설치

```bash
pip install ccxt pandas numpy pyyaml aiohttp websockets
```

### 프로젝트 구조

```
ru_trading_newest/
├── run_multi_strategy.py           # 진입점
├── config/
│   ├── multi_strategy.yaml         # 메인 설정 (전략/포트폴리오/코인)
│   └── settings.yaml               # 거래소 연결 설정
├── src/
│   ├── execution/
│   │   ├── exchange_adapter.py     # ccxt Binance USDM Futures 래퍼
│   │   ├── order_ledger.py         # 주문 장부 관리
│   │   └── sl_tp_monitor_v2.py     # SL/TP 실시간 폴링 (15초)
│   └── strategies/
│       ├── base.py                 # StrategyConfig 기반 클래스
│       ├── cvd_spike.py            # CVD Spike Reactor
│       ├── liquidation_fade.py     # Liquidation Fade
│       ├── momentum_breakout.py    # Momentum Breakout
│       ├── asymmetric_sniper.py    # Asymmetric Sniper
│       ├── multi_position_manager.py
│       ├── portfolio_risk.py       # PortfolioRiskManager
│       ├── data_hub.py             # 시장 데이터 수집기
│       ├── coin_profile.py         # CoinProfileStore
│       ├── position_sizer.py       # PositionSizer
│       ├── trade_logger.py         # JSONL 거래 기록
│       └── strategy_analyzer.py   # 성과 분석
├── data/
│   └── reports/multi_strategy/    # 로그 및 상태 파일
└── push_logs.sh                   # 거래 로그 GitHub push
```

### 로그 원격 동기화

```bash
# 거래 로그를 GitHub에 push (다른 머신에서 확인용)
bash push_logs.sh

# 다른 머신에서 내려받기
git pull
```

---

## 주의사항

- **Demo/Live 전환 시 반드시 `initial_equity`를 실제 잔고에 맞게 수정한다.**
- `daily_loss_limit: 0.20`은 가상매매용 공격적 설정이다. Live 전환 시 0.05~0.10으로 낮춘다.
- `total_exposure_pct: 2.5` (250%)는 가상매매 전용이다. Live 전환 시 0.5~1.0 권장.
- Asymmetric Sniper의 testnet 파라미터(`rr_check_enabled: false`, `funding_filter_enabled: false`)는 testnet 데이터 품질 이슈로 인한 우회 설정이다. Live 모드에서는 반드시 활성화해야 한다.
- Post-Only(GTX) 주문이 거부될 경우 일반 LIMIT 주문으로 자동 폴백된다. 이 경우 Taker 수수료가 적용된다.
