# CLAUDE_CRYPTO_AGENT Development History

Generated: 2026-03-17

---

## Phase 1: Foundation (2026-03-09 ~ 03-11)

### 목표
크립토 Top 10 코인 자동 예측 시스템 초기 구축.

### 구현
- **OHLCV 수집**: yfinance 기반 (거래소 API 방화벽 차단 대응)
- **미디어 크롤러 15개**: Google News, CoinDesk, CoinTelegraph, TheBlock, Decrypt,
  Coinness, Reddit(7 subreddits), X/Twitter, YouTube, TikTok(proxy), Instagram(proxy),
  Glassnode, Messari, Tiger Research, On-chain
- **매크로/원자재**: Gold, Silver, WTI, Brent, LIT, REMX, COPX, 국채(US 10Y/30Y/5Y),
  S&P500, NASDAQ, VIX, DXY, EUR/USD, USD/JPY, USD/KRW, Fear&Greed, DeFi TVL
- **기술지표 51개**: RSI, MACD, Bollinger, ATR 등 (technical_analysis.py)
- **감성분석 81개**: sentiment_analyzer.py
- **MOMENT DL**: PatchEmbedding + SelectiveMasking + OLinear + RevIN + MediaAttention
- **5-Model ML Ensemble**: LightGBM(GPU) + XGBoost(GPU) + CatBoost(GPU) + RF + ET
- **Fee-aware labeling**: UP > +0.2%, DOWN < -0.2%, HOLD = 사이
- **run_overnight_loop.py**: 9시까지 자동 반복 실행

### 결과
- 10코인 학습, balanced_accuracy 기반 평가
- DL(MOMENT) 학습 시간 과다 -> 실용성 의문 제기

### 기술 이슈 해결
- PyTorch CUDA cu128 force-reinstall
- torch.compile: backend="eager" (Windows Triton 미지원)
- n_jobs=-1 -> 6: CPU 100% 방지

---

## Phase 2: Walk-Forward v2 + Architecture v4 (2026-03-12 ~ 03-13)

### 목표
시계열 교차검증 도입, DL 배제 결정, 아키텍처 재설계.

### 주요 결정
- **DL 배제**: MOMENT 학습 시간 과다 -> ML 2-Stage Binary만으로 전환
- **2-Stage Binary**: Stage1(Trade/NoTrade) -> Stage2(Long/Short)
- **TimeSeriesSplit**: StratifiedKFold 사용 금지, gap=12 bars
- **Architecture v4 설계**: 3계층 분리 (Research / Modeling / Execution)

### 구현
- **Signal Features 84개 추가**: wavelet, FFT, Hilbert, entropy, Hurst, ACF,
  ADX, Stochastic, OBV, MFI, CMF, Ichimoku, microstructure, CUSUM, multi-TF
- **7-Model Ensemble + Stacking**: BalancedRF + HistGB 추가, LogisticRegression 메타러너
- **Regime Filter**: 4상태 (TREND_UP/DOWN, RANGE_LOW/HIGH)
- **Scanner 모듈**: universe_scanner, hot_scanner, tradability_scorer
- **Signal 모듈**: contract.py, policy.py

### Walk-Forward v2 결과 (9코인)
- 대부분 balanced_accuracy 50% 미만 -> 비용 고려하면 수익 불가
- 평가 지표 전환 필요성 인식: accuracy -> net EV

### 기술 이슈
- 타임프레임 통일: 5분봉 -> 1h fetch -> 4h resample (bar_minutes=240)
- 미디어 감성 데이터: 노이즈 과다 -> 실질적 예측력 기여 의문

---

## Phase 3: Walk-Forward v3 + 코인 압축 (2026-03-14 ~ 03-15)

### 목표
5코인으로 집중, stability 기반 최적화, 파라미터 공간 탐색.

### 구현
- **5코인 집중**: DOT, DOGE, BTC, XRP, ADA (ETH, SOL, AVAX, LINK, BNB 제외)
- **Triple Barrier Labeling**: ATR 기반, k_upper/k_lower 튜닝
- **Stage1 threshold 분리 최적화**: 0.40~0.65 범위
- **MAX_FEATURES 확대**: 80 -> 150 (MI 기반 선택)
- **Augmented windows**: train 90~300d, test 21~45d, stride 7d
- **Stability metric**: mean_combined - 0.5 * std + 0.1 * MCC

### v3 Extended 결과 (6라운드, 2,244 evals)
| Coin | EV/trade | 판정 |
|------|----------|------|
| XRP | +0.215% | 1순위 |
| DOT | +0.117% | 검증 대상 |
| ADA | +0.061% | 경계선 |
| DOGE | +0.033% | 노이즈 |
| BTC | -0.109% | 제외 |

### 핵심 인사이트
- accuracy/MCC만으로는 실전 수익성 판단 불가
- 비용 단위 불일치: R-multiple vs flat % vs equity% 혼재
- "감으로 올리면 안 된다" -> 숫자 기반 승급 기준 수립

---

## Phase 4: Execution Layer v1 + Cost Engine (2026-03-15 ~ 03-16)

### 목표
실행 계층 구축, 비용 모델 정밀화, net EV 기반 재평가.

### 구현 (8개 모듈)
- **cost_model.py**: FeeSchedule + FundingConfig + MissFillConfig -> CostBreakdown
  - Bybit VIP0: maker 0.02%, taker 0.055%, funding 0.01%/8h
  - stop-distance sizing -> notional/equity ratio -> equity% 환산
  - 5개 비용 항목: entry_fee, exit_fee, slippage, funding, miss_fill
- **exchange_adapter.py**: Bybit + ccxt 통합
- **live_engine.py**: 메인 실행 루프
- **order_ledger.py**: 주문 기록 (SQLite)
- **risk_engine.py**: 일간/주간 DD, 연속 손실 kill switch
- **state_machine.py**: IDLE -> ENTRY -> OPEN -> EXIT 상태 전이
- **watchdog.py**: heartbeat, 이상 감지
- **paper_trader.py**: 시뮬레이션 모드

### 비용 구조 확립
```
entry_fee  = maker_fee * (risk_frac / stop_dist_pct)     [equity%]
exit_fee   = taker_fee * (risk_frac / stop_dist_pct)     [equity%]
slippage   = (slip_entry + slip_exit) * notional_ratio    [equity%]
funding    = rate * (holding_hours / 8) * notional_ratio  [equity%]
miss_fill  = reject_prob * missed_ev / (1 - reject_prob)  [equity%]
```

### 주요 결정
- **3코인 압축**: XRP, DOT, ADA (DOGE 관찰, BTC 제외)
- **사이징**: stop-distance 기반 0.5% risk/trade
- **평가 지표 전환**: accuracy/MCC -> net EV(equity%/trade) 단일 기준

---

## Phase 5: v3.1_netev Optimization (2026-03-16 ~ 03-17)

### 목표
비용 후 순 EV 최적화, 비대칭 R:R 구조 탐색, 장기 자동 실행.

### 구현
- **run_optimize_v3_netev.py**: 전체 파이프라인 통합
  - CostModel 내장, RegimeFilter 연동, TradeAuditor 출력
  - composite score = netEV * 10000 * (1 - 0.5*std) + MCC * 10
  - 비대칭 R:R 중심 파라미터 공간 (k_upper > k_lower)
  - 자동 체크포인트 + resume 기능

### 최적화 이력

**Session 1 (R1-R10, 6h)**
- XRP: score 61 -> 126.2 (+106%)
- ADA: score 97 -> 119.5 (+23%)
- DOT: score 77 -> 93.1 (+21%)

**Session 2 (R11-R38, 5h, resume)**
- DOT: score 93 -> 101 (R15) -> 114 (R27) -> 117 (R37)
- XRP, ADA: R10 best 유지 (수렴)

**Session 3 (DEADLINE 연장, 30min)**
- 변동 없음 -> 수렴 확정

### 최종 결과 (R38, 9,390 evals)
| Coin | Net EV | Score | R:R | Margin | S2% | k_u/k_l |
|------|--------|-------|-----|--------|-----|---------|
| XRP | +1.30% | 126.2 | 5.0 | 43.3% | 64.2% | 3.0/0.6 |
| ADA | +1.23% | 119.5 | 5.0 | 41.1% | 61.3% | 3.0/0.6 |
| DOT | +1.20% | 116.6 | 5.0 | 40.0% | 60.0% | 3.0/0.6 |

### 수렴 패턴
- 3코인 모두 k_upper=3.0, k_lower=0.6 (R:R=5.0) 동일 구조
- 낮은 BEP (~20%) + 높은 margin (40%+) = 비용 변동에 강건

---

## Current Project Structure (Post-Cleanup)

```
CLAUDE_CRYPTO_AGENT/
|
|-- CLAUDE.md                      # 프로젝트 규칙/설정 (살아있는 문서)
|-- run_optimize_v3_netev.py       # [MAIN] v3.1 net EV 최적화
|-- run_live_engine.py             # 실행 엔진 런처
|-- run_overnight_loop.py          # 야간 자동 반복
|
|-- config/
|   +-- settings.yaml              # 중앙 설정 (2-tier timeframe, costs, risk)
|
|-- src/
|   |-- data/
|   |   |-- crawlers/
|   |   |   |-- crypto_ohlcv.py    # yfinance OHLCV + signal features
|   |   |   |-- signal_features.py # 84 signal features
|   |   |   |-- macro_commodity_crawler.py  # Gold, VIX, DXY, US10Y, S&P500
|   |   |   |-- technical_analysis.py      # 51 technical indicators
|   |   |   |-- sentiment_analyzer.py      # 감성 분석 (미사용)
|   |   |   |-- google_news_crawler.py     # 뉴스 크롤러 (미사용)
|   |   |   |-- reddit_crawler.py          # Reddit (미사용)
|   |   |   |-- x_crawler.py              # X/Twitter (미사용)
|   |   |   +-- youtube_crawler.py         # YouTube (미사용)
|   |   |-- kis_client.py          # 한국투자증권 API (예약)
|   |   +-- data_stitcher.py       # 데이터 병합
|   |
|   |-- models/
|   |   |-- enhanced_ensemble.py   # [CORE] 7-Model + Stacking
|   |   |-- masking_loop.py        # [CORE] Triple barrier, metrics
|   |   |-- regime_filter.py       # 4-state regime classifier
|   |   |-- multimodal_classifier.py  # MOMENT DL (비활성)
|   |   |-- model_store.py         # 모델 저장/로드
|   |   +-- run_masking_pipeline.py   # 내부 파이프라인
|   |
|   |-- execution/
|   |   |-- cost_model.py          # [CORE] 비용 엔진 (5항목)
|   |   |-- exchange_adapter.py    # Bybit + ccxt
|   |   |-- live_engine.py         # 메인 실행 루프
|   |   |-- order_ledger.py        # SQLite 주문 기록
|   |   |-- risk_engine.py         # DD/kill switch
|   |   |-- state_machine.py       # IDLE->ENTRY->OPEN->EXIT
|   |   |-- watchdog.py            # 헬스체크
|   |   +-- paper_trader.py        # 시뮬레이션
|   |
|   |-- evaluation/
|   |   |-- trade_audit.py         # 감사표 출력
|   |   |-- walk_forward.py        # WF 프레임워크
|   |   +-- report.py              # 리포트 생성
|   |
|   |-- discussion/                # Claude 내부 + Gemini 검증 (미연결)
|   |-- signals/                   # 시그널 정책 (contract, policy)
|   |-- scanner/                   # 유니버스/핫 스캐너
|   |-- regime/                    # 레짐 감지 (detector)
|   +-- utils/                     # config, logging
|
+-- data/
    |-- raw/20260313/              # 최신 raw 데이터
    +-- reports/
        |-- v3_1_netev_final_report.md  # [CURRENT] 최종 리포트
        |-- DEVELOPMENT_HISTORY.md      # [CURRENT] 이 문서
        +-- walkforward_v3_1/           # v3.1 체크포인트 (R1-R38)
```

---

## Key Technical Decisions Log

| Date | Decision | Reason |
|------|----------|--------|
| 03-09 | yfinance 사용 | 거래소 API 방화벽 차단 |
| 03-11 | DL(MOMENT) 배제 | 학습 시간 과다, RTX 3090에서도 비실용적 |
| 03-12 | 2-Stage Binary 전환 | 3-class 직접 분류보다 안정적 |
| 03-12 | TimeSeriesSplit 강제 | StratifiedKFold -> 미래 누수 |
| 03-13 | 7-Model Stacking 도입 | 5-Model 가중평균 대비 안정성 개선 |
| 03-14 | 5코인 집중 | 10코인 분산 -> 상위 5 집중 |
| 03-15 | net EV 전환 | accuracy/MCC로는 수익성 판단 불가 |
| 03-15 | stop-distance sizing | 고정 비율 -> SL 폭 기반 동적 사이징 |
| 03-16 | 비용 엔진 5항목 | maker/taker/slippage/funding/miss-fill 분리 |
| 03-16 | 3코인 압축 | DOGE 노이즈, BTC 음수 EV |
| 03-17 | k=3.0/0.6 수렴 | R:R=5.0 비대칭 구조가 비용 후 EV 최적 |

---

## Version History

| Version | Date | Focus | Evals |
|---------|------|-------|-------|
| v1 | 03-09~11 | 초기 구축, 10코인, DL+ML | - |
| v2 | 03-12~13 | Walk-Forward, 9코인, stability | ~500 |
| v3 | 03-14~15 | 5코인, Signal Features, 7-Model | 2,244 |
| v3 Extended | 03-15~16 | XRP+ADA 추가, 6라운드 | 2,244 |
| **v3.1_netev** | **03-16~17** | **3코인, CostModel, net EV** | **9,390** |
