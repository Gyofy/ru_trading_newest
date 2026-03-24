# CLAUDE_CRYPTO_AGENT

Crypto trading system — BTC global direction + relative strength alt selection.

**Binance USDT-M Futures | 10 coins | v6.1 | 1h resolution | Continuous Trading**

---

## Strategy: v6.1

```
1. BTC 7-day return → Global Direction (LONG / SHORT / FLAT)
2. Alt 9 coins: BTC 대비 상대 강도 랭킹 → 상위 2개만 선택
3. RSI + CVD Q75 + OI z-score 필터
4. BTC 방향 전환 → 즉시 청산 (DIR_FLIP)
5. TP=5×ATR, SL=2×ATR, TTL=96 1h-bars (4일)
6. 1h 주기 시그널 체크, 조건 충족 시 즉시 진입
7. 최대 2 포지션, 2x 레버리지, 포지션당 리스크 2%
```

### Validated Performance

| | IS (9개월) | OOS (3.5개월) |
|---|---|---|
| per-trade Sharpe | +0.058 | +0.208 |
| WR | 42.4% | 47.2% |
| Profit Factor | 1.15 | 1.71 |
| MDD | -67.4% | -13.5% |

**IS + OOS 둘 다 양수.** 이전 v5.3은 IS 마이너스 + Sharpe 8.6배 과대평가.

---

## Binance API Adaptation (실거래 전환 프로세스)

### Step 1: 환경 설정

```bash
# 필수 환경 변수
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
export DATA_SOURCE="binance"

# 선택 (기본값 있음)
export BINANCE_TESTNET="true"       # true: testnet, false: 실거래
```

### Step 2: Testnet 검증 (1주일)

```bash
# Testnet으로 먼저 실행 (가상 자금)
BINANCE_TESTNET=true DATA_SOURCE=binance python run_tsmom_paper.py
```

**체크리스트:**
- [ ] Binance ccxt 연결 성공 (1h OHLCV fetch)
- [ ] Post-Only 진입 주문 체결
- [ ] STOP_MARKET SL 주문 정상 배치
- [ ] TAKE_PROFIT_MARKET TP 주문 정상 배치
- [ ] DIR_FLIP 시 MARKET 즉시 청산
- [ ] 포지션 크기 계산 정확 (risk 2%, leverage 2x)
- [ ] 1h 주기 사이클 안정 가동
- [ ] trade_analysis.jsonl 기록 정상

### Step 3: Live 소액 (2주)

```bash
# 실거래 $50~100로 시작
BINANCE_TESTNET=false DATA_SOURCE=binance python run_tsmom_paper.py
```

**제한 설정:**
- `max_positions: 1` (1개만)
- `equity_risk_pct: 0.01` (1%로 축소)
- 목표: 10+ 거래, WR > 35%, net PnL > 0

### Step 4: Scale Up

```
10건 통과 → max_positions: 2, risk: 2%
50건 통과 → 자본 증가 검토
WR < 30% or MDD > 20% → 즉시 중단, 전략 재검토
```

### 데이터 흐름 (Binance 모드)

```
Binance API
  │
  ├─ fetch_ohlcv("1h", limit=500) ─→ 1h 봉 (매 사이클)
  │    └─ ATR(14), RSI(14), CVD 계산
  │
  ├─ fetch_ticker() ─→ 실시간 가격 (SL/TP 모니터링)
  │
  ├─ create_limit_order(GTX) ─→ Post-Only 진입
  │    └─ 거부 시 → 일반 limit 폴백
  │
  ├─ create_order(STOP_MARKET) ─→ SL 배치
  │
  └─ create_order(TAKE_PROFIT_MARKET) ─→ TP 배치

로컬 저장
  ├─ trades.jsonl          거래 내역
  ├─ trade_analysis.jsonl  진입/청산 분석 + 손익 원인
  ├─ trajectories.jsonl    bar-by-bar 궤적 (RL 학습용)
  ├─ signal_log.jsonl      RL 시그널 기록
  └─ state.json            현재 상태

GitHub (push_logs.sh)
  └─ 위 파일 전부 → 다른 머신에서 내려받기 가능
```

### 주문 타입 상세

| 용도 | 주문 타입 | 수수료 | 비고 |
|------|----------|--------|------|
| 진입 | `LIMIT` + `timeInForce: GTX` (Post-Only) | Maker 0.02% | 거부 시 일반 LIMIT 폴백 |
| SL | `STOP_MARKET` | Taker 0.05% | 즉시 체결 보장 |
| TP | `TAKE_PROFIT_MARKET` | Taker 0.05% | |
| DIR_FLIP | `MARKET` | Taker 0.05% | BTC 방향 전환 시 |
| TTL | `MARKET` | Taker 0.05% | 96바 시간 초과 시 |

### 안전장치

| 항목 | 설정 | 설명 |
|------|------|------|
| 최대 포지션 | 2개 | 동시 보유 상한 |
| 포지션당 리스크 | equity × 2% | SL 히트 시 최대 손실 |
| 총 리스크 | equity × 4% | 2포지션 × 2% |
| 레버리지 | 2x 고정 | 변동 없음 |
| 일일 손실 한도 | -10% | → 전체 포지션 청산 + 당일 중단 |
| 쿨다운 | 4 1h-bars | 청산 후 같은 코인 재진입 차단 |
| 단일 인스턴스 | bot.lock | 중복 프로세스 방지 |

### 기존 모듈 재사용

```
src/execution/exchange_adapter.py  → 주문 실행 (이미 Binance ccxt)
src/execution/sl_tp_monitor.py     → SL/TP 실시간 모니터링
src/execution/risk_engine.py       → 9-gate 리스크 체크
src/execution/position_store.py    → 포지션 영속화 (crash recovery)
src/execution/cost_model.py        → 수수료 계산
```

---

## Architecture

```
run_tsmom_paper.py              v6.1 Bot (1h, Binance/yfinance)
push_logs.sh                    거래 로그 → GitHub push

src/strategy/tsmom_core.py      시그널/백테스트/메트릭
src/execution/                  주문/리스크/모니터링 (Binance ccxt)
src/rl/                         RL (shadow mode, 데이터 수집 중)

docs/
  binance_implementation_spec.md  Binance 구현 상세
  v6_honest_portfolio_report.md   v6.0 성과 리포트
  adaptive_exit_rl_brainstorm.md  RL exit 연구 (30+ 논문)
```

---

## Trade Logs (GitHub 자동 저장)

```bash
# 로그 push
bash push_logs.sh

# 다른 머신에서 내려받기
git pull
```

| 파일 | 내용 |
|------|------|
| `trades.jsonl` | 모든 거래 (진입/청산/PnL/레버리지) |
| `trade_analysis.jsonl` | 진입/청산 시장 상태 + 손익 원인 자동 분류 |
| `trajectories.jsonl` | bar-by-bar 포지션 궤적 (미래 RL 학습용) |
| `signal_log.jsonl` | RL state/action 기록 |
| `state.json` | 현재 equity/포지션 |

### 손익 원인 자동 분류

| 원인 | 조건 | 의미 |
|------|------|------|
| `volatility_spike` | 청산 ATR > 진입 ATR × 1.5 | 변동성 급증으로 SL |
| `trend_reversal` | RSI 극단 (< 30 or > 70) | 추세 반전으로 SL |
| `noise_stop` | 그 외 SL | 일반 노이즈 |
| `trend_continuation` | TP 히트 | 추세 지속으로 수익 실현 |
| `btc_direction_change` | DIR_FLIP | BTC 방향 전환 |
| `no_momentum` | TTL + PnL < 0 | 모멘텀 부재 |
| `slow_profit` | TTL + PnL > 0 | 느리지만 수익 |

---

## Version History

| Version | Strategy | per-trade Sharpe | Status |
|---------|----------|-----------------|--------|
| v4.0~4.3 | ML 2-Stage | INVALID (leakage) | Dead |
| v5.0~5.3 | TSMOM 10 coins | 0.504 (fake, corr 0.80) | Dead |
| v6.0 | BTC dir + RS top 2 (4h) | 0.208 (honest) | Superseded |
| **v6.1** | **BTC dir + RS top 2 (1h)** | **Testing** | **Active** |

---

## Environment

```bash
pip install ccxt pandas numpy scikit-learn lifelines arch joblib pyyaml
# yfinance는 로컬 개발용 (선택)
pip install yfinance  # optional
```

```bash
# 실행
DATA_SOURCE=binance python run_tsmom_paper.py     # Binance
python run_tsmom_paper.py                          # yfinance (기본)
```
