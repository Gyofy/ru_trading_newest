# CLAUDE_CRYPTO_AGENT

## Agent Memory (ALWAYS READ FIRST)
- **Development Index**: `docs/INDEX.md` -- devlog, checkpoints, brainstorm
- **Current Config**: `config/frozen_params_v4_2.yaml` (v4.2 — Mega Search v2 best 반영)
- **Previous Config**: `config/frozen_params_v4_1.yaml`, `config/frozen_params_v3_4.yaml` (보존)
- **Live Bot**: `run_live_bot_v2.py` (v4.2, 2-Stage + Multi-Model + 30s SL/TP)
- **Live Criteria**: `config/live_promotion_criteria.yaml`
- **Active Coins**: SOL (단일 운영 — 2026-03-19, 백테스트 결과 SOL만 손익분기 근접)

## Overview
크립토 5코인(DOT/ADA/XRP/SOL/LINK) 자동 예측·매매 시스템.
Binance USDT-M Futures 4h봉 + 마이크로스트럭처(CVD/OFI/VPIN) 223개 피처,
2-Stage Binary(Trade/NoTrade → Long/Short) ML Ensemble로 방향 분류.
Claude Code를 오케스트레이터로 사용하고, 데이터·모델·주문은 분리된 파이썬 서비스로 운영.

## Current Pipeline (v4.2)
```
Phase 1: Data Collection
  ├── OHLCV: Binance ccxt 4h봉 500bars (5 코인 + BTC/ETH)
  └── 마이크로스트럭처: CVD, OFI, VPIN, Roll Spread, Amihud

Phase 2: Feature Engineering
  ├── 기술지표 ~60개 → technical_analysis.py
  ├── 시그널 피처 → signal_features.py (wavelet, FFT, entropy)
  ├── 마이크로스트럭처 ~71개 → microstructure_rollup.py
  └── MI 기반 Feature Selection: top 120개

Phase 3: 2-Stage Binary ML (per-coin combo)
  ├── Stage 1: Trade/NoTrade (ET + CatBoost/XGBoost/TabM weighted)
  ├── Stage 2: Long/Short   (same combo, Trade samples only)
  ├── CV: TimeSeriesSplit(n_splits=3, gap=12)
  └── Coin-specific combos: DOT(et+tabm), ADA/XRP/SOL(et+cb), LINK(et+xgb)

Phase 4: Execution
  ├── 30s SL/TP/TTL monitoring (background asyncio)
  ├── Post-Only entry + STOP_MARKET SL + TAKE_PROFIT_MARKET TP
  └── Risk engine 9-gate pre-trade check
```

## Target Assets
- BTC, ETH, SOL, XRP, ADA, DOGE, AVAX, DOT, LINK, BNB
- 24시간 거래 (장 시간 제한 없음)

## Architecture Principles
- **역할 분리**: 리서치(뉴스/웹) ↔ 모델링(GPU) ↔ 실행(주문) ↔ 감사(읽기전용)는 절대 합치지 않음
- **Prompt Injection 방어**: 외부 콘텐츠를 읽는 에이전트에게 주문 권한을 주지 않음
- **데이터 누수 방지**: point-in-time join, 확정 바만 사용, TimeSeriesSplit + gap
- **비용 반영**: 수수료(0.1%) + 슬리피지(0.05%) + 버퍼(0.05%) = ±0.2% fee-aware 라벨링
- **내부 분석 + 외부 검증**: Claude가 전략+리스크 직접 수행, Gemini는 최종 검증 1회만

## Directory Layout
```
src/
  data/
    crawlers/    # Binance OHLCV, 미디어 크롤러, 매크로/원자재, 마이크로스트럭처
  models/
    masking_loop.py           # 2-Stage Binary labeling + ensemble training
    model_store.py            # artifact save/load (joblib)
    enhanced_ensemble.py      # 5-model weighted ensemble
  execution/
    exchange_adapter.py       # Binance USDT-M Futures (ccxt)
    live_predictor.py         # 2-Stage + Multi-Model combo train/predict
    sl_tp_monitor.py          # 30s background SL/TP/TTL polling
    position_store.py         # crash-safe JSON position persistence
    risk_engine.py            # 9-gate pre-trade check + sizing
    order_ledger.py           # SQLite order/fill/PnL ledger
    cost_model.py             # fee + slippage + funding + miss-fill cost
    state_machine.py          # position FSM (IDLE→FILLED→PROTECTED→...)
  signals/
    contract.py               # Signal dataclass
    policy.py                 # SignalPolicy regime-aware filtering
  evaluation/
    trade_level_ev.py         # bar-by-bar Triple Barrier simulation
  utils/       # config, logging, feature_policy
config/        # YAML 설정 (frozen_params_v4_2.yaml)
data/          # raw/reports/models (일자별)
run_live_bot_v2.py            # v4.2 autonomous trading bot
```

## Key Rules
1. `paper` 모드에서 충분한 검증 없이 `live` 전환 금지
2. 데이터 5분 이상 지연 시 주문 차단
3. 코인당 계좌의 5%, 총 노출 계좌의 80% 초과 금지
4. 당일 손실 계좌의 2% 초과 시 전체 거래 중단
5. OHLCV → feature → prediction → (signal → order) 파이프라인 순서 준수

## Labeling & Evaluation (v4.3)
- **라벨링**: Triple Barrier (k_upper=3.0×ATR, k_lower=1.0×ATR, max_hold=12bars)
- **2-Stage Binary**: S1(Trade/NoTrade) → S2(Long/Short, Trade samples only)
- **S2 Deadzone**: |p_long - 0.5| < 0.10 → HOLD (WR +4.2%p, MDD -5.6%p)
- **CV**: TimeSeriesSplit(n_splits=3, gap=12) — StratifiedKFold 사용 금지
- **평가지표**: balanced_accuracy (주), trade-level PnL, MDD
- **Horizons**: 4h bars × [1, 3, 6, 18] = 4h, 12h, 24h, 72h

## ML Ensemble (v4.2 -- Mega Search v2 best per coin)
- DOT: ExtraTrees(70%) + TabM(30%)
- ADA: ExtraTrees(50%) + CatBoost(50%)
- XRP: ExtraTrees(50%) + CatBoost(50%)
- SOL: ExtraTrees(50%) + CatBoost(50%)
- LINK: ExtraTrees(50%) + XGBoost(50%)
- Feature Selection: MI 기반 top 120 (223개 중, 마이크로스트럭처 포함)

## Data Sources
```
OHLCV:     yfinance 5분봉 (BTC, ETH, SOL, XRP, ADA, DOGE, AVAX, DOT, LINK, BNB)
뉴스:      Google News (크립토+거시), CoinDesk, CoinTelegraph, TheBlock, Decrypt, Coinness
SNS:       Reddit (7 subreddits), X/Twitter, YouTube, TikTok(proxy), Instagram(proxy)
온체인:    Glassnode, Messari, Tiger Research, On-chain metrics
매크로:    금, 은, WTI, 브렌트, 알루미늄, LIT(리튬), REMX(희토류), COPX(구리)
국채:      US 10Y/30Y/5Y Treasury Yield
지수:      S&P500, NASDAQ, VIX, USD Index
환율:      EUR/USD, USD/JPY, USD/KRW
유동성:    Fear & Greed Index, DeFi TVL (DeFiLlama)
```

## Team Discussion Rules (Claude 내부 + Gemini 최종 검증)
<rule name="team_discussion">
6. 시그널 확신도 70% 미만, 포지션 3% 초과, 당일 손실 1% 초과 시 팀 디스커션 자동 발동
7. Claude가 리스크매니저 역할을 직접 수행 — `risk_level: extreme` 판정 시 무조건 거래 중단 (거부권)
8. 합의 신뢰도(consensus_confidence) 0.5 미만이면 보류(hold) 처리
9. Gemini는 최종 검증 단계에서 1회만 호출 (비용 최소화, max_tokens=512)
10. Gemini disagree + high confidence 시 합의 신뢰도 60% 할인
11. Gemini 응답은 절대 주문 명령으로 직접 변환하지 않음 — signal-ensemble 경유 필수
12. GPT/Claude API는 사용하지 않음 (비용 절감)
</rule>

## Structured Prompting Standards
<rule name="xml_tag_structure">
- 모든 스킬/에이전트 프롬프트에 XML 태그로 영역 구분: `<role>`, `<responsibility>`, `<constraints>`, `<output_format>`, `<error_handling>`
- 복잡한 로직 실행 전 `<thinking>` 단계에서 설계 → 실행 분리
- 디스커션 프롬프트에 `<context>`, `<task>`, `<cross_review>` 태그 사용
</rule>

## Atomic Tool Design
<rule name="atomic_tools">
- 하나의 모듈/스킬이 두 가지 이상의 책임을 갖지 않음
- fetch_data ↔ calculate_indicator ↔ generate_signal ↔ execute_order 철저 분리
- 에러 발생 시 정확한 모듈 특정 → self-healing 용이성 확보
- 각 모듈은 독립적으로 테스트 가능해야 함
</rule>

## Error Handling & Safety
<rule name="error_handling">
- API 응답 5초 초과 지연 시 해당 요청 타임아웃 처리
- API 연속 3회 실패 시 해당 서비스 비활성화 + 알림
- Gemini API 장애 시 내부 분석만으로 디스커션 속행 (graceful degradation)
- 모든 예외는 `logs/` 에 기록, 심각도별 알림 수준 차등
</rule>

## Context & Memory Management
<rule name="context_management">
- CLAUDE.md는 '살아있는 문서' — 전략 변경/설정 변경 시 즉시 업데이트
- 중요 의사결정, 성공 패턴, 디버깅 노하우는 memory/ 에 별도 파일로 저장
- 디스커션 결과는 `data/reports/{date}/discussion_{session}.json` 에 전량 기록
- 대화 컨텍스트 과부하 방지: 핵심 결론만 추출하여 다음 단계에 전달
</rule>

## Implementation Status
```
[완료] src/data/crawlers/            — OHLCV + 마이크로스트럭처 + 시그널피처
[완료] src/models/                   — 2-Stage Binary + ML Ensemble + model_store
[완료] src/execution/                — 주문실행 + 리스크엔진 + 비용모델
[완료] src/execution/live_predictor  — 2-Stage + Multi-Model combo (v4.2)
[완료] src/execution/sl_tp_monitor   — 30s 실시간 SL/TP 모니터링
[완료] src/execution/position_store  — 포지션 영속화 (crash recovery)
[완료] src/signals/                  — Signal contract + SignalPolicy
[완료] src/evaluation/               — trade-level bar-by-bar simulation
[완료] src/utils/                    — config, logging, feature_policy
[완료] run_live_bot_v2.py            — v4.2 autonomous trading bot
[완료] run_overnight_loop.py         — 자동 반복 실행
[완료] experiments/mega_search_v2    — 20 configs × 5 coins 최적화
```

## Environment Variables Required
```
# Gemini (최종 검증용, 선택 — 없으면 내부 분석만으로 운영)
GEMINI_API_KEY      # Google Gemini 2.5 Pro

# Binance USDT-M Futures (현재 활성)
BINANCE_API_KEY     # Binance API key (testnet 또는 live)
BINANCE_API_SECRET  # Binance API secret

# KIS API (예약 — 실행 모듈 구현 시 필요)
KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT_NO

# 미사용 (비용 절감)
# OPENAI_API_KEY    — 사용 안 함
# ANTHROPIC_API_KEY — 사용 안 함
```
