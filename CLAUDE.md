# CLAUDE_CRYPTO_AGENT

## Agent Memory (ALWAYS READ FIRST)
- **Development Index**: `docs/INDEX.md` -- devlog, checkpoints, brainstorm
- **Current Config**: `config/frozen_params_v4_0.yaml` (v4.0 — WF 444eval 반영, th 하향)
- **Previous Config**: `config/frozen_params_v3_4.yaml` (보존)
- **Live Criteria**: `config/live_promotion_criteria.yaml`
- **Active Coins**: DOT, ADA (XRP suspended)

## Overview
크립토 Top 10 코인 자동 예측·매매 시스템.
yfinance 5분봉 + 15개 미디어 + 18개 매크로/원자재 데이터를 수집하고,
MOMENT(self-supervised pretrain) DL + 5-Model ML Ensemble로 방향(UP/HOLD/DOWN)을 분류.
Claude Code를 오케스트레이터로 사용하고, 데이터·모델·주문은 분리된 파이썬 서비스로 운영.

## Current Pipeline (v3)
```
Phase 1: Data Collection
  ├── OHLCV: yfinance 5분봉 60일 (10 코인)
  ├── 미디어: 15개 소스 (뉴스/SNS/온체인/리서치)
  └── 매크로: 18개 yfinance 티커 + Fear&Greed + DeFi TVL

Phase 2: Feature Engineering
  ├── 기술지표 51개 → technical_analysis.py
  ├── 미디어 감성 81개 → sentiment_analyzer.py
  ├── 크로스소스 7개
  └── 매크로 134개 → macro_commodity_crawler.py
  → MI 기반 Feature Selection: top 80개

Phase 3a: Deep Learning
  ├── MOMENT pretrain (SelectiveMasking 30ep)
  └── OLinear + RevIN + MediaAttention fine-tune (80ep)

Phase 3b: ML Ensemble
  ├── LightGBM (GPU) + XGBoost (GPU) + CatBoost (GPU)
  ├── RandomForest + ExtraTrees (CPU, n_jobs=6)
  ├── GridSearch: TimeSeriesSplit(n_splits=3, gap=12)
  └── balanced_accuracy 기반 가중 앙상블

Phase 4: Report (JSON + Markdown)
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
    crawlers/    # yfinance OHLCV, 15개 미디어 크롤러, 매크로/원자재 수집
    kis_client.py  # 한국투자증권 API (예약 — 실행 모듈 구현 시 연결)
  models/
    multimodal_classifier.py  # MOMENT DL (PatchEmbed + SelectiveMasking + OLinear + RevIN)
    masking_loop.py           # 5-Model ML Ensemble + GridSearch + Fee-aware labeling
    run_masking_pipeline.py   # 전체 파이프라인 오케스트레이터
  discussion/  # 멀티 AI 팀 디스커션 (Claude 내부 + Gemini 검증)
  signals/     # (미구현) 앙상블 시그널 생성
  execution/   # (미구현) 주문 실행, paper/live 전환
  evaluation/  # (미구현) holdout 검증, walk-forward 백테스트
  utils/       # 로깅, 설정 로드
config/        # YAML 설정
data/          # raw/reports (일자별)
.claude/
  skills/      # 13개 스킬
  agents/      # 5개 서브에이전트
```

## Key Rules
1. `paper` 모드에서 충분한 검증 없이 `live` 전환 금지
2. 데이터 5분 이상 지연 시 주문 차단
3. 코인당 계좌의 5%, 총 노출 계좌의 80% 초과 금지
4. 당일 손실 계좌의 2% 초과 시 전체 거래 중단
5. OHLCV → feature → prediction → (signal → order) 파이프라인 순서 준수

## Labeling & Evaluation (v3)
- **라벨링**: Fee-aware dynamic — UP > +0.2%, DOWN < -0.2%, HOLD = 사이
- **CV**: TimeSeriesSplit(n_splits=3, gap=12) — StratifiedKFold 사용 금지
- **평가지표**: balanced_accuracy (주), MCC, Brier, PR-AUC, F1-macro, confusion matrix
- **목적함수**: balanced_accuracy 단일 기준 (GridSearch, 앙상블 가중, 리포트 통일)
- **Horizons**: 4개 (5min, 15min, 30min, 60min)

## DL Architecture (MOMENT v3)
- PatchEmbedding: patch_len=8(40min), stride=4(50% overlap) → 11 patches
- SelectiveMasking: random → hard-mining → curriculum (3-phase)
- MaskedAutoEncoder: pretrain 30ep (self-supervised)
- OLinear(NormLin) + RevIN + MediaSourceAttention + CrossModalFusion
- Fine-tune 80ep, AMP FP16, batch=512, torch.compile(backend="eager")

## ML Ensemble (v3)
- LightGBM (GPU, device=gpu)
- XGBoost (GPU, device=cuda)
- CatBoost (GPU, task_type=GPU)
- RandomForest (CPU, n_jobs=6)
- ExtraTrees (CPU, n_jobs=6)
- Feature Selection: MI 기반 top 80 (273개 중)

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
[완료] src/data/crawlers/     — OHLCV + 15 미디어 + 매크로/원자재 수집
[완료] src/models/            — MOMENT DL + 5-Model ML Ensemble
[완료] src/discussion/        — Claude 내부 + Gemini 검증 (미연결)
[완료] src/utils/             — config, logging
[완료] run_overnight_loop.py  — 9시까지 자동 반복 실행
[미구현] src/signals/ensemble — ML+DL 시그널 결합
[미구현] src/execution/       — 주문 실행 (paper/live)
[미구현] src/evaluation/      — holdout 검증, walk-forward
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
