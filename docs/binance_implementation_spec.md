# Binance API 연결 시 구현 사항 — 상세 기술서

> 2026-03-24 | 지금까지 토의 내용 종합

---

## 1. 현재 상태 vs 목표

```
현재 (로컬 개발):
  데이터: yfinance (지연, 1h/4h, 10코인)
  실행: paper bot (가상 체결)
  주기: 4h 봉 마감마다 체크

목표 (Binance 실거래):
  데이터: Binance ccxt/WebSocket (실시간)
  실행: 실제 주문 (USDT-M Futures)
  주기: 실시간 (신호 즉시 → 즉시 주문)
```

---

## 2. 구현해야 할 것들 (우선순위순)

### 2.1 데이터 소스 전환 (yfinance → Binance)

```python
# 현재: yfinance (제거 대상)
df = yf.download("BTC-USD", period="60d", interval="1h")

# 변경: Binance ccxt
ohlcv = exchange.fetch_ohlcv("BTC/USDT:USDT", "1h", limit=500)
```

**구현 상세:**

| 항목 | 설명 |
|------|------|
| 라이브러리 | `ccxt` (동기) 또는 `ccxt.async_support` (비동기) |
| 심볼 | USDT-M Perpetual: `{COIN}/USDT:USDT` |
| 타임프레임 | `1h` (진입 시그널), 실시간 (SL/TP 모니터링) |
| 히스토리 | `fetch_ohlcv(symbol, "1h", limit=500)` = 20일치 |
| 인증 | `BINANCE_API_KEY`, `BINANCE_API_SECRET` 환경변수 |

### 2.2 실시간 가격 모니터링 (WebSocket)

```python
# ccxt pro (WebSocket)
import ccxt.pro as ccxtpro

exchange = ccxtpro.binanceusdm({...})
while True:
    ticker = await exchange.watch_ticker("SOL/USDT:USDT")
    price = ticker["last"]
    # SL/TP 체크
    if price <= position.sl_price:
        await close_position("SL")
```

**구현 상세:**

| 항목 | 설명 |
|------|------|
| 방식 | `watch_ticker()` 또는 `watch_ohlcv()` |
| 주기 | 실시간 (WebSocket push, 폴링 아님) |
| 용도 | SL/TP 히트 감지, 진입 조건 실시간 체크 |
| 폴백 | WebSocket 끊기면 REST `fetch_ticker()` 5초 폴링 |

### 2.3 주문 실행

```python
# 진입: Post-Only limit (maker fee)
order = await exchange.create_limit_order(
    symbol="SOL/USDT:USDT",
    side="sell",  # SHORT
    amount=qty,
    price=entry_price,
    params={"timeInForce": "GTX"}  # Post-Only
)

# SL: STOP_MARKET (즉시 체결)
sl_order = await exchange.create_order(
    symbol, "STOP_MARKET", "buy", qty,
    params={"stopPrice": sl_price, "closePosition": False}
)

# TP: TAKE_PROFIT_MARKET
tp_order = await exchange.create_order(
    symbol, "TAKE_PROFIT_MARKET", "buy", qty,
    params={"stopPrice": tp_price, "closePosition": False}
)
```

**주문 타입:**

| 용도 | 주문 타입 | 수수료 |
|------|----------|--------|
| 진입 | Post-Only LIMIT (GTX) | Maker 0.02% |
| 진입 폴백 | LIMIT (Post-Only 거부 시) | Maker 0.02% |
| SL | STOP_MARKET | Taker 0.05% |
| TP | TAKE_PROFIT_MARKET | Taker 0.05% |
| DIR_FLIP | MARKET | Taker 0.05% |

### 2.4 v6.0 전략 로직 (Binance 버전)

```
매 1h:
  1. BTC 1h OHLCV fetch → 7d return 계산 → Global Direction
  2. Global Direction = 0 → 진입 안 함
  3. 알트 9개 1h OHLCV 동시 fetch (asyncio)
  4. 각 코인: RSI + CVD + OI 계산
  5. BTC 대비 상대 강도 랭킹
  6. 상위 2개 선택 → 즉시 진입 (Post-Only)

실시간 (WebSocket):
  7. 보유 포지션 SL/TP 모니터링
  8. BTC 방향 전환 감지 → DIR_FLIP 즉시 청산
  9. TTL 체크 (96 1h-bars)
```

### 2.5 포지션 관리

```python
# 기존 exchange_adapter.py 재사용
from src.execution.exchange_adapter import ExchangeAdapter

adapter = ExchangeAdapter(mode="live")
await adapter.initialize()

# 포지션 조회
positions = await adapter.get_positions()

# 주문 실행
await adapter.place_entry(coin, side, qty, entry_price, sl, tp)
```

**안전장치:**

| 항목 | 설정 |
|------|------|
| 최대 포지션 | 2개 |
| 포지션당 리스크 | equity × 2% |
| 레버리지 | 2x (고정) |
| 일일 손실 한도 | equity × -10% → 전체 중단 |
| Post-Only 거부 | → limit 폴백 → 실패 시 skip |

---

## 3. 실시간 진입 (봉 마감 안 기다림)

**현재**: 1h 봉 마감 → 시그널 체크 → 진입
**목표**: 조건 충족 즉시 진입

```python
# 매 1분 or WebSocket ticker update마다:
async def on_price_update(coin, price):
    # 1h 기준 지표는 마지막 확정 봉 기준 (현재 봉은 미확정)
    if not signal_conditions_met(coin):
        return

    # 조건 충족 → 즉시 주문
    await place_entry(coin, direction, price)
```

**주의:** RSI, CVD는 확정 봉 기준으로 계산해야 함 (미확정 봉 포함 시 look-ahead).
→ 지표는 1h 확정 봉, 진입 타이밍은 실시간 가격.

---

## 4. OI/Funding Rate 실시간 통합

```python
# Binance REST (5분마다 갱신)
oi = exchange.fetch_open_interest(symbol)
funding = exchange.fetch_funding_rate(symbol)

# 또는 data/raw/binance_public/metrics/ 에서 로드 (이미 다운로드됨)
```

---

## 5. 코드 구조 (Binance 전용)

```
run_v6_live.py              ← 신규 작성 (Binance 전용 봇)
  ├── __init__
  │   ├── ExchangeAdapter 초기화
  │   ├── WebSocket 연결
  │   └── State 로드
  │
  ├── run() [메인 루프]
  │   ├── 매 1h: OHLCV fetch → 시그널 계산 → 진입
  │   ├── 실시간: WebSocket → SL/TP/DIR_FLIP 모니터링
  │   └── 매 5분: OI/Funding 갱신
  │
  └── 기존 모듈 재사용:
      ├── src/execution/exchange_adapter.py  (주문)
      ├── src/execution/sl_tp_monitor.py     (SL/TP 폴링)
      ├── src/execution/position_store.py    (영속화)
      ├── src/execution/risk_engine.py       (9-gate)
      └── src/strategy/tsmom_core.py         (시그널 로직)
```

---

## 6. yfinance 제거 계획

| 파일 | 현재 | 변경 |
|------|------|------|
| `run_tsmom_paper.py` | yfinance + Binance 듀얼 | **Binance 전용** (`run_v6_live.py`) |
| `src/strategy/tsmom_core.py` | yfinance `fetch_ohlcv()` | Binance `fetch_ohlcv()` |
| `experiments/*.py` | yfinance 로드 | Binance or 로컬 캐시 |

**GitHub 푸시 시**: yfinance 의존성 제거, `ccxt` 필수.

---

## 7. 환경 변수

```bash
# 필수
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret

# 선택
DATA_SOURCE=binance          # (기본값으로 변경)
BINANCE_TESTNET=false        # true면 testnet 사용
MAX_POSITIONS=2
LEVERAGE=2
EQUITY_RISK_PCT=0.02
```

---

## 8. 테스트 단계

```
Phase 1: Testnet (가상 자금)
  → Binance Futures Testnet에서 v6.0 로직 실행
  → 주문 체결, SL/TP 작동 확인
  → 1주일 가동

Phase 2: Live (소액)
  → 실 자금 $50~100로 시작
  → max_pos=1로 제한
  → 2주 가동, 10+ 거래 후 평가

Phase 3: Scale up
  → 실적 확인 후 자본 증가
  → max_pos=2 복원
```

---

## 9. 기존 인프라 재사용 가능 여부

| 모듈 | 재사용 | 수정 필요 |
|------|--------|----------|
| `exchange_adapter.py` | **100%** | 없음 (이미 Binance ccxt) |
| `sl_tp_monitor.py` | **90%** | WebSocket 모드 추가 |
| `position_store.py` | **100%** | 없음 |
| `risk_engine.py` | **100%** | 없음 |
| `order_ledger.py` | **100%** | 없음 |
| `cost_model.py` | **100%** | 없음 |
| `tsmom_core.py` | **70%** | fetch를 ccxt로 변경 |
| `state_builder.py` | **100%** | 없음 |
| `signal_logger.py` | **100%** | 없음 |
