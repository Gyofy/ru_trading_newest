# CLAUDE_CRYPTO_AGENT

Autonomous crypto trading system — 2-Stage Binary ML + Microstructure + Binance Public Data + RL Sizing.

**Binance USDT-M Futures | 4 coins | 1분봉 ML Ensemble | LIVE 실거래 중 (2026-03-24~)**

---

## Current Status (2026-03-24)

### v4.3-1m ML Bot — LIVE 실거래 중

```
Bot:        run_live_bot_v2.py
Config:     config/frozen_params_v4_3_1m.yaml
Mode:       LIVE (Binance USDT-M Futures)
Coins:      SOL, XRP, ADA, DOT  (4코인)
Timeframe:  1분봉, 1000 bars
Equity:     ~65 USDT
Position:   자본의 10% per trade
Daily Halt: -10% 손실 시 당일 거래 중단
SL/TP:      10s polling (Post-Only → Limit 폴백)
```

### ML Pipeline

```
Phase 1: Data Collection
  ├── OHLCV: Binance ccxt 1분봉 1000bars (4코인 + BTC/ETH 참조)
  ├── 마이크로스트럭처: CVD, OFI, VPIN, Roll Spread, Amihud
  └── Binance 공개 데이터: OI, Long/Short Ratio, Taker Vol, Funding Rate

Phase 2: Feature Engineering
  ├── 기술지표 ~60개 → technical_analysis.py
  ├── 시그널 피처 → signal_features.py (wavelet, FFT, entropy)
  ├── 마이크로스트럭처 ~71개 → microstructure_rollup.py
  └── Binance Public Features → binance_public_features.py

Phase 3: 2-Stage Binary ML (per-coin)
  ├── Stage 1: Trade/NoTrade (ET + CatBoost)
  ├── Stage 2: Long/Short   (ET + CatBoost)
  ├── S2 Deadzone: |p_long - 0.5| < 0.10 → HOLD
  └── CV: TimeSeriesSplit(n_splits=3, gap=12)

Phase 4: Execution
  ├── Post-Only entry → -5022 시 limit 폴백
  ├── STOP_MARKET SL + TAKE_PROFIT_MARKET TP
  ├── 10s SL/TP polling
  └── Risk engine 9-gate pre-trade check
```

### Risk Settings

| 항목 | 값 |
|------|-----|
| 포지션 크기 | 자본의 10% (notional) |
| 일일 손실 한도 | -10% halt |
| 주간 손실 한도 | 비활성화 |
| 레버리지 | 4x |
| 최소 SL 거리 | 0.18% |

### 거래 코인 선정 이유

Binance USDT-M Futures 최소 주문 금액 기준:

| 코인 | 최소 주문 | 선정 |
|------|-----------|------|
| SOL | 5 USDT | ✅ |
| XRP | 5 USDT | ✅ |
| ADA | 5 USDT | ✅ |
| DOT | 5 USDT | ✅ |
| ETH | 20 USDT | ❌ (자본 부족) |
| LINK | 20 USDT | ❌ (자본 부족) |
| BTC | 100 USDT | ❌ (자본 부족) |

---

## Architecture

```
run_live_bot_v2.py              LIVE ML bot (v4.3-1m, ACTIVE)

src/
  data/crawlers/
    crypto_ohlcv.py             OHLCV + technical indicators
    signal_features.py          Wavelet/FFT/entropy
    microstructure_rollup.py    CVD/OFI/VPIN/Roll Spread/Amihud
    binance_public_features.py  OI/LSR/Taker/Funding (merge_asof)
    binance_public_data_downloader.py  Binance 공개 데이터 다운로더
  execution/
    exchange_adapter.py         Binance USDT-M Futures (ccxt, Post-Only→Limit 폴백)
    live_predictor.py           2-Stage Binary ML combo
    sl_tp_monitor.py            10s SL/TP polling
    risk_engine.py              9-gate pre-trade check + sizing
    position_store.py           Crash-safe JSON persistence
    cost_model.py               Fee + slippage + funding
    order_ledger.py             SQLite order/fill/PnL
  models/
    masking_loop.py             Triple Barrier labeling
  rl/
    bandit.py                   LinUCB 7-action (shadow mode)
    signal_logger.py            JSONL signal logging
    rl_gate.py                  Shadow/active mode + safety
  signals/
    contract.py                 Signal dataclass
    policy.py                   SignalPolicy regime-aware filtering

config/
  frozen_params_v4_3_1m.yaml   현재 운영 config

trading_result/
  daily_pnl.csv                 일별 손익
  equity_state.json             현재 자산 상태
  fills.csv                     체결 기록
  orders.csv                    주문 기록
  events.jsonl                  이벤트 로그
```

---

## Quick Start

```bash
# 의존성 설치
pip install ccxt scikit-learn catboost xgboost ta pandas numpy joblib pyyaml requests tqdm python-dotenv

# .env 설정
BINANCE_API_KEY=...
BINANCE_API_SECRET=...

# Binance 공개 데이터 다운로드 (선택 — 없으면 graceful skip)
python src/data/crawlers/binance_public_data_downloader.py \
  --types metrics funding_rate \
  --symbols SOLUSDT XRPUSDT ADAUSDT DOTUSDT BTCUSDT ETHUSDT LINKUSDT \
  --days 365

# 봇 시작 (LIVE)
python run_live_bot_v2.py --mode live --equity <USDT잔고> --yes

# 로그 확인
tail -f logs/live_bot_*.log
```

---

## Development History

### 2026-03-24 — v4.3-1m LIVE 첫 실거래

| 항목 | 내용 |
|------|------|
| Post-Only 폴백 | -5022 거부 시 limit 주문 자동 재시도 |
| Binance 공개 데이터 통합 | OI/L&S/Taker/Funding → compute_features() |
| datetime dtype 버그 수정 | ms vs us 불일치 → datetime64[ns] 통일 |
| 일일 손실 한도 | 6% → 10% |
| 포지션 크기 | 자본 150% 상한 → 자본 10% |
| --yes 플래그 | 백그라운드 실행 시 확인 프롬프트 우회 |
| Discord 종료 알림 | 봇 종료 시 시작 알림과 동일 형식으로 발송 |
| 거래 코인 | SOL → SOL/XRP/ADA/DOT (최소주문금액 기준 4코인) |
| 첫 체결 | SOL SELL @ 89.67 / SL=89.83 / TP=89.11 |

### 2026-03-23 — v5.1 TSMOM (Paper Bot, 별도 운영)

- 피처 누수 발견(2026-03-20) 후 rule-based 방향으로 피봇
- Dual TSMOM(7d+28d) + RSI + CVD + OI 필터
- 10코인 paper trading (`run_tsmom_paper.py`)
- OOS Sharpe 4.03 (permutation p=0.006)

### 2026-03-20 — Feature Leakage Discovery

- STL decomposition, Ichimoku .shift(26), SVD → 미래 데이터 누수
- v4.0~v4.3 백테스트 결과 무효화
- 15,120 조합 전수 검색 → BTC spike만 유일한 edge

### 2026-03-18 — v4.3 ML 2-Stage

- Mega Search v2: ET+CB 콤보 최적화
- 1분봉 재학습 (v4.3-1m)
- RL meta-layer 7-action (shadow mode)

### 2026-03-17 — v3.4 → v4.0 진화

- 5코인 DOT/ADA/XRP/SOL/LINK
- 2-Stage Binary, Triple Barrier 라벨링

---

## Key Rules

1. `paper` 모드 검증 없이 `live` 전환 금지
2. 데이터 5분 이상 지연 시 주문 차단
3. 일일 손실 -10% 초과 시 당일 거래 중단
4. OHLCV → feature → prediction → signal → order 파이프라인 순서 준수

---

## Environment

- **Python** 3.10+ | **Exchange** Binance USDT-M Futures (ccxt)
- **ML** scikit-learn, CatBoost, XGBoost | **RL** LinUCB (custom, shadow mode)
- **OS** Linux (WSL2)
