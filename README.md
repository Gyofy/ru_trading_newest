# Binance Futures Trading Bot — v12.1

**1D TSMOM 28d/7d | 27개월 검증 완료 | Triple Eval 3중 검증 체계**

> **현재 상태**: v12.1-dual-tsmom-1d — 27개월 백테스트 4/4 WF Pass, Paper Trading 준비 완료

---

## 시스템 개요

| 항목 | 값 |
|------|-----|
| 버전 | **v12.1-dual-tsmom-1d** (2026-04-02) |
| 거래소 | Binance USDM Futures |
| 활성 전략 | **2개** (1D TSMOM 28d/7d + 28d/14d) |
| 타임프레임 | **일봉 (1D)** — 1h OHLCV에서 리샘플 |
| 초기 자본 | $5,000 USDT |
| 레버리지 | 3x |
| 왕복 수수료 | 0.20% (수수료 비중: ATR 대비 **5%**) |
| 코인 | SOL, XRP, ADA, DOT, DOGE (5개) |
| 거래 빈도 | ~16건/월 (~4건/주) |
| WR | 49.8% (full), **55.8% (OOS)** |
| 검증 | **3/3 WF Pass** (OOS + 양반기 + 반전) |

---

## 검증된 전략: 1D TSMOM

### 로직

```
28일 모멘텀 > 3% + 7일(또는 14일) 확인 모멘텀 동일 방향 → 진입
+ Volume filter: 당일 거래량 >= 10일 평균의 80%
+ Day filter: 월요일/토요일 진입 제외
+ SL = 2.0 × daily ATR, TP = 3.0 × daily ATR, TTL = 7일
```

### 27개월 백테스트 (2024-01-01 ~ 2026-03-31)

| 항목 | Config B (Vol + Skip Mon/Sat) |
|------|------|
| 전체 거래 | 418건 |
| Win Rate | **49.8%** |
| OOS WR (30%) | **55.8%** |
| OOS Net | **양수 (PASS)** |
| 양반기 | 모두 양수 (PASS) |
| 반전 시그널 | 손실 (PASS) |
| 파라미터 강건성 | 22/25 (88%) |

### 듀얼 포트폴리오

| 인스턴스 | 확인 기간 | 배분 |
|----------|----------|------|
| **tsmom_1d** | 28d + **7d** 확인 | $1,500 |
| **tsmom_1d_slow** | 28d + **14d** 확인 | $1,500 |

두 변형의 시그널은 비중첩 → 단독 대비 +13% 거래 기회 증가

---

## 27개월 전수 탐색 결과

700+ 전략/파라미터 조합을 검증한 결과:

| Timeframe | 전략 유형 | 조합 수 | 수익 조합 | OOS Pass |
|-----------|----------|---------|----------|----------|
| 1m | 6전략 (CVD/VWAP/OFI 등) | 150 | 0 | - |
| 1h | 23전략 + Adaptive + Ensemble | 500+ | 1 (PF 1.05) | 불가 |
| 4h | TSMOM/MeanRev/Regime Switch | 100+ | 27 | 전부 FAIL |
| **1d** | **TSMOM 28d/7d** | **35+** | **25** | **PASS** |

**핵심 발견**: Timeframe이 가장 중요한 변수. 1m/1h는 수수료 > ATR로 수학적 불가능. 4h는 regime decay. **1d만 27개월 OOS 생존.**

---

## Triple Evaluation (3중 검증)

```
python run_triple_eval.py --mode all     # Paper + Demo + Backtest 동시
python run_triple_eval.py --mode paper   # Paper만
python run_triple_eval.py --mode backtest # 백테스트만
python run_triple_eval.py --mode compare  # 결과 비교
```

30분마다 3개 모드 성과 교차 비교 → Discord 알림

---

## 아키텍처 (v12.1)

```
[Binance] ← OHLCV 1h (→ 1D 리샘플) + OI + Funding
      ↓
┌─────────────────────────────────────────────┐
│ DataHub (1h → 1D resample)                   │
│ 28d/7d momentum + volume + day filter        │
└──────────────────┬──────────────────────────┘
                   ↓
        ┌──────────┴──────────┐
        │ tsmom_1d (28d/7d)   │ tsmom_1d_slow (28d/14d)
        └──────────┬──────────┘
                   ↓
┌──────────────────┴──────────────────────────┐
│ PortfolioRisk (9 gates) + PositionSizer      │
│ SL 2.0×ATR | TP 3.0×ATR | TTL 7d            │
└──────────────────┬──────────────────────────┘
                   ↓
┌──────────────────┴──────────────────────────┐
│ ExchangeAdapter + SlTpMonitorV2              │
│ SL/TP FOR POSITION 필수 (closePosition=True) │
└─────────────────────────────────────────────┘
```

---

## 백테스트 인프라 (backtest/)

| 스크립트 | 용도 |
|----------|------|
| `run_27m_full_search.py` | 27개월 23전략 전수 탐색 |
| `run_brainstorm.py` | 5라운드 브레인스토밍 (35 변형) |
| `run_full_validation.py` | 9/9 WF 검증 스위트 |
| `run_htf_backtest.py` | 15m/1h HTF 파라미터 스윕 |
| `run_adaptive_solver.py` | 8시그널 적응형 솔버 |
| `run_ensemble_vote.py` | 앙상블 다수결 투표 |
| `engine.py` / `engine_htf.py` | 바 단위 백테스트 엔진 |
| `data_loader.py` | CSV/Parquet 데이터 로더 |
| `report.py` | 수수료 분석 + 데이터 누수 검증 |

---

## 설정

```yaml
# config/multi_strategy.yaml
version: "v12.1-dual-tsmom-1d"
initial_equity: 5000.0

strategies:
  tsmom_1d:          # 28d momentum + 7d confirmation
    enabled: true
    allocation_usdt: 1500.0
    leverage: 3
    cycle_seconds: 3600
    extra:
      sl_mult: 2.0
      tp_mult: 3.0
      min_move_pct: 0.03
      mom_slow: 28
      mom_fast: 7
      vol_filter: true
      skip_days: [0, 5]    # Monday, Saturday

  tsmom_1d_slow:     # 28d momentum + 14d confirmation
    enabled: true
    allocation_usdt: 1500.0
    leverage: 3
    cycle_seconds: 3600
    extra:
      sl_mult: 2.0
      tp_mult: 3.0
      min_move_pct: 0.03
      mom_slow: 28
      mom_fast: 14
      vol_filter: true
      skip_days: [0, 5]
```

---

## 실행

```bash
# 메인 봇
python run_multi_strategy.py --config config/multi_strategy.yaml

# Triple Eval (3중 검증)
python run_triple_eval.py --mode all
```

---

## 버전 이력

### v12.1 (2026-04-02) — 현재
- **1D TSMOM 28d/7d + 28d/14d 듀얼 포트폴리오**
- 27개월 백테스트 전수 탐색 (700+ 조합) → 유일한 검증 통과 전략
- Volume filter + Day filter (Skip Mon/Sat) → WR 49.8%, OOS 55.8%
- 백테스트 인프라 13개 스크립트
- Triple eval 통합 (backtest/paper/demo 3중 검증)

### v10.0 (2026-03-30)
- Meta modules: DrawdownThrottle, FeeEVGate, DSR, ClusterTracker
- DataHub: taker_buy_ratio, realized_vol

### v9.1-data (2026-03-30)
- Data-Gathering Mode (시장가, SL 1.5%, 15포지션/전략)
- StrategySolver v2 (train/test, tighter-only, 12 학술 레퍼런스)

### v9.0 (2026-03-29)
- 6전략 체제, SL Floor 0.45%, EVGuardian

### v8.0~v8.5 (2026-03-27~29)
- Multi-strategy, Post-Only, Fixed TP

### v3.4~v6.1 (2026-03-17~25)
- ML 2-Stage Binary → leakage 발견 → 마이크로스트럭처 피봇 → 첫 실거래

---

## 핵심 교훈 (18일간 검증)

1. **Timeframe이 전부**: 1m 불가(ATR < Fee) → 1h 불가 → 4h 불안정 → **1d만 유효**
2. **14일 검증 = 무효**: 14일 9/9 Pass가 27개월에서 전멸
3. **단순 > 복잡**: Adaptive/Ensemble/Kelly 전부 단순 TSMOM보다 나쁨
4. **수수료 = 물리 법칙**: ATR 대비 20% 넘으면 어떤 시그널도 무효
5. **Regime decay 실재**: 모든 전략은 특정 regime에서만 작동

---

*최종 업데이트: 2026-04-02 | Gyofy/ru_trading_newest*
