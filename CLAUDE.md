# CLAUDE_CRYPTO_AGENT

## Agent Memory (ALWAYS READ FIRST)
- **Development Index**: `docs/INDEX.md` -- devlog, checkpoints, brainstorm
- **Current Config**: `config/frozen_params_v4_3_1m.yaml` (v4.3-1m — 1분봉, 4코인)
- **Live Bot**: `run_live_bot_v2.py` (v4.3-1m, LIVE 실거래 중 — 2026-03-24~)
- **Active Coins**: SOL, XRP, ADA, DOT (4코인 — Binance 최소주문금액 5 USDT 이하)

## Overview
크립토 4코인(SOL/XRP/ADA/DOT) 자동 예측·매매 시스템.
Binance USDT-M Futures 1분봉 + 마이크로스트럭처(CVD/OFI/VPIN) + Binance 공개 데이터(OI/L&S/Taker/Funding),
2-Stage Binary(Trade/NoTrade → Long/Short) ML Ensemble로 방향 분류.
Claude Code를 오케스트레이터로 사용하고, 데이터·모델·주문은 분리된 파이썬 서비스로 운영.

## Current Pipeline (v4.3-1m)
```
Phase 1: Data Collection
  ├── OHLCV: Binance ccxt 1분봉 1000bars (4코인 + BTC/ETH 참조)
  ├── 마이크로스트럭처: CVD, OFI, VPIN, Roll Spread, Amihud
  └── Binance 공개 데이터: OI, Long/Short Ratio, Taker Vol, Funding Rate

Phase 2: Feature Engineering
  ├── 기술지표 ~60개 → technical_analysis.py
  ├── 시그널 피처 → signal_features.py (wavelet, FFT, entropy)
  ├── 마이크로스트럭처 ~71개 → microstructure_rollup.py
  └── Binance Public Features → binance_public_features.py (merge_asof)

Phase 3: 2-Stage Binary ML (per-coin, et+cb combo)
  ├── Stage 1: Trade/NoTrade (ExtraTrees + CatBoost)
  ├── Stage 2: Long/Short   (ExtraTrees + CatBoost)
  ├── S2 Deadzone: |p_long - 0.5| < 0.10 → HOLD
  └── CV: TimeSeriesSplit(n_splits=3, gap=12)

Phase 4: Execution
  ├── Post-Only entry → -5022 시 limit 폴백
  ├── STOP_MARKET SL + TAKE_PROFIT_MARKET TP
  ├── 10s SL/TP polling
  └── Risk engine 9-gate pre-trade check
```

## Target Assets
- SOL, XRP, ADA, DOT (LIVE 거래)
- BTC, ETH (OHLCV 참조용 — 크로스 에셋 피처)

## Architecture Principles
- **역할 분리**: 리서치(뉴스/웹) ↔ 모델링(GPU) ↔ 실행(주문) ↔ 감사(읽기전용)는 절대 합치지 않음
- **Prompt Injection 방어**: 외부 콘텐츠를 읽는 에이전트에게 주문 권한을 주지 않음
- **데이터 누수 방지**: point-in-time join, 확정 바만 사용, TimeSeriesSplit + gap
- **비용 반영**: 수수료(0.1%) + 슬리피지(0.05%) + 버퍼(0.05%) = ±0.2% fee-aware 라벨링

## Directory Layout
```
src/
  data/
    crawlers/    # Binance OHLCV, 마이크로스트럭처, Binance 공개 데이터
  models/
    masking_loop.py           # 2-Stage Binary labeling + ensemble training
    model_store.py            # artifact save/load (joblib)
  execution/
    exchange_adapter.py       # Binance USDT-M Futures (ccxt, Post-Only→Limit 폴백)
    live_predictor.py         # 2-Stage + Multi-Model combo train/predict
    sl_tp_monitor.py          # 10s background SL/TP/TTL polling
    position_store.py         # crash-safe JSON position persistence
    risk_engine.py            # 9-gate pre-trade check + sizing
    order_ledger.py           # SQLite order/fill/PnL ledger
    cost_model.py             # fee + slippage + funding + miss-fill cost
  rl/
    bandit.py                 # LinUCB 7-action (shadow mode)
    signal_logger.py          # JSONL signal logging
    rl_gate.py                # Shadow/active mode + safety gates
  signals/
    contract.py               # Signal dataclass
    policy.py                 # SignalPolicy regime-aware filtering
  utils/       # config, logging, feature_policy
config/        # frozen_params_v4_3_1m.yaml (현재 운영)
trading_result/ # daily_pnl, equity_state, fills, orders, events
run_live_bot_v2.py            # v4.3-1m autonomous trading bot (ACTIVE)
```

## Key Rules
1. `paper` 모드에서 충분한 검증 없이 `live` 전환 금지
2. 데이터 5분 이상 지연 시 주문 차단
3. 일일 손실 -10% 초과 시 당일 거래 중단
4. OHLCV → feature → prediction → (signal → order) 파이프라인 순서 준수

## ⛔ 절대 원칙: 모든 포지션에 TP/SL FOR POSITION 필수

**어떠한 예외도 없다. 모든 포지션은 반드시 SL + TP 양쪽을 FOR POSITION 타입으로 거래소에 등록해야 한다.**

### 규칙 상세

| 항목 | 규칙 |
|------|------|
| SL 타입 | `STOP_MARKET` + `closePosition=True` (FOR POSITION) |
| TP 타입 | `TAKE_PROFIT_MARKET` + `closePosition=True` (FOR POSITION) |
| 등록 시점 | fill 확인 후 **1초 대기** 후 등록 (Binance position propagation delay) |
| 재시도 | 최대 5회 × 1초 간격 |
| 실패 시 | SL 5회 실패 → 즉시 포지션 강제청산 (naked position 절대 금지) |
| TP 5회 실패 → 즉시 포지션 강제청산 (SL만 있는 포지션도 금지) |
| 확인 방법 | `positions.json` 내 `sl_exchange_id`, `tp_exchange_id` 모두 non-empty여야 함 |

### 왜 FOR POSITION 타입인가
- Binance Position 탭 TP/SL 컬럼에 표시 → 거래소 UI에서 즉시 확인 가능
- `closePosition=True` → 포지션 전체 일괄 청산 보장 (수량 불일치 없음)
- `reduceOnly=True` + 수량 지정 방식은 수량 변경 시 TP/SL 무효화 위험

### 위반 시나리오 (과거 사례)
- **-4509 오류**: fill 직후 즉시 SL 등록 시도 → Binance position DB 미반영 → 거부
  - 해결: fill 후 1초 대기 추가, retry 5회로 증가
- **-2022 오류**: 포지션이 없는데 market_close 시도 (ghost position)
  - 해결: GHOST_CLEANUP early return으로 PnL 기록 없이 tracker만 제거

### 모든 코드 변경 시 체크리스트
- [ ] `place_protective_stop()` 호출 전 fill 완료 확인
- [ ] SL/TP 등록 후 `pos.sl_exchange_id`, `pos.tp_exchange_id` 비어있지 않음 확인
- [ ] SL/TP 없는 포지션이 `pos_manager`에 남지 않도록 강제청산 분기 유지
- [ ] `closePosition=True` 파라미터 절대 제거 금지

## Labeling & Evaluation (v4.3)
- **라벨링**: Triple Barrier (k_upper=3.0×ATR, k_lower=1.0×ATR, max_hold=60bars)
- **2-Stage Binary**: S1(Trade/NoTrade) → S2(Long/Short, Trade samples only)
- **S2 Deadzone**: |p_long - 0.5| < 0.10 → HOLD
- **CV**: TimeSeriesSplit(n_splits=3, gap=12) — StratifiedKFold 사용 금지
- **평가지표**: balanced_accuracy (주), trade-level PnL, MDD

## ML Ensemble (v4.3-1m)
- SOL: ExtraTrees(50%) + CatBoost(50%)
- XRP: ExtraTrees(50%) + CatBoost(50%)
- ADA: ExtraTrees(50%) + CatBoost(50%)
- DOT: ExtraTrees(50%) + CatBoost(50%)
- Feature Selection: MI 기반 top 120

## Risk Settings
- **포지션 크기**: 자본의 10% (notional cap)
- **일일 손실 한도**: -10%
- **레버리지**: 4x
- **최소 SL 거리**: min_barrier_pct × 0.9

## Discord 알림
- 봇 시작 시: 🚀 봇 시작 embed (잔고, 모드, 버전, 코인)
- 봇 종료 시: 🛑 봇 종료 embed (동일 형식)
- 진입 시: 진입 정보 embed
- 청산 시: PnL, 누적 WR embed

## Data Sources
```
OHLCV:       Binance ccxt (1분봉)
마이크로스트럭처: BVC-based CVD/OFI/VPIN
Binance 공개: data.binance.vision (OI, L/S Ratio, Taker Vol, Funding Rate)
```

## Environment Variables Required
```
BINANCE_API_KEY     # Binance API key (live)
BINANCE_API_SECRET  # Binance API secret
GEMINI_API_KEY      # 선택 (미사용 시 내부 분석만)
```

## Implementation Status
```
[완료] src/data/crawlers/            — OHLCV + 마이크로스트럭처 + Binance 공개 데이터
[완료] src/models/                   — 2-Stage Binary + ML Ensemble
[완료] src/execution/                — 주문실행 + 리스크엔진 + 비용모델 + Post-Only 폴백
[완료] src/rl/                       — LinUCB 7-action (shadow mode)
[완료] src/signals/                  — Signal contract + SignalPolicy
[완료] run_live_bot_v2.py            — v4.3-1m LIVE bot
```
