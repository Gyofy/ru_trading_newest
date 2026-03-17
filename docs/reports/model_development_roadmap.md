# 모델 개발 로드맵
*작성: 2026-03-18 | 기반: Walk-Forward v3.1 최적화 결과 + 마이크로스트럭처 분석*

---

## 현재 모델 상태 (v3.4 frozen)

### 아키텍처
```
Stage 1 (Trade/NoTrade):  LightGBM CPU × 2-Stage Ensemble
Stage 2 (Long/Short):     LightGBM CPU × 2-Stage Ensemble
Labeling:                 Triple Barrier (k_upper/k_lower × ATR)
CV:                       TimeSeriesSplit(n_splits=3, gap=12)
Objective:                balanced_accuracy → net EV (post-cost)
Features:                 222개 (기술지표 49 + 신호 89 + MS 73 + 매크로 24)
```

### 현재 성능 (frozen_params_v3_4)
- DOT: stage1_threshold=0.50, k_upper≈1.5, k_lower=1.0
- ADA: stage1_threshold=0.52, k_upper≈1.5, k_lower=0.8
- 가상거래: 16:00 UTC 바 HOLD (s1_prob < threshold)

---

## v4.0 개발 계획

### v4.1 — 파라미터 재설정 (즉시)
444 eval 분석 결과 기반:

```yaml
# 권장 업데이트
DOT:
  k_upper: 3.0        # R:R=5.0이 최고 netEV (+1.083% avg)
  k_lower: 0.6        # 좁은 손절 = 높은 EV
  stage1_threshold: 0.45

ADA:
  k_upper: 3.0
  k_lower: 0.6
  stage1_threshold: 0.40  # 더 공격적 진입
```

**근거:**
- R:R 4.5~6.0 구간 avg netEV = +1.083% vs R:R 0~1.5 구간 +0.068%
- k_lower=0.6 avg netEV = +0.554% vs k_lower=1.5 avg +0.156%
- th=0.45가 전체 최적 (noise 방지 + 충분한 신호)

---

### v4.2 — 마이크로스트럭처 통합 (1주일)

#### A. CVD Contrarian Filter
```python
# generate_signal()에 추가
cvd_ratio = df['cvd_ratio_24'].iloc[-1]
cvd_q25 = df['cvd_ratio_24'].quantile(0.25)
cvd_q75 = df['cvd_ratio_24'].quantile(0.75)

if cvd_ratio > cvd_q75:
    # 매수세 과열 → LONG 사이즈 축소, SHORT 우대
    cvd_mult = 0.6 if side == 'BUY' else 1.3
elif cvd_ratio < cvd_q25:
    # 매도세 과열 → SHORT 사이즈 축소, LONG 우대
    cvd_mult = 1.3 if side == 'BUY' else 0.6
else:
    cvd_mult = 1.0
```
**예상 효과:** +0.334%/trade at 48h (실증됨)

#### B. OFI Entry Timing
```python
ofi = df['ofi_sum_3'].iloc[-1]
ofi_q67 = df['ofi_sum_3'].quantile(0.67)
ofi_timing = 1.1 if ofi > ofi_q67 else 1.0
```
**예상 효과:** +0.08~0.15%/trade at 4-8h

#### C. MS Composite Score → Dynamic Sizing
```python
# ms_composite > 0.3 → bullish microstructure
# ms_composite < 0.1 → bearish microstructure
ms_mult = np.clip(0.5 + df['ms_composite'].iloc[-1], 0.5, 1.5)
```

---

### v4.3 — OrderFlowCollector 통합 (구현 완료, 연결 필요)

**파일:** `src/execution/orderflow_collector.py`

```python
# run_demo_trading.py main loop에 추가
collector = OrderFlowCollector(exchange, symbols=ACTIVE_COINS)
await collector.start()

# 바 종가 시점에
snap = await collector.get_snapshot(coin)
flow_pressure = snap['flow_pressure']  # [-1, +1]

# dynamic sizing에 반영
flow_mult = np.clip(1.0 + flow_pressure * 0.3, 0.7, 1.3)
```

**수집 데이터:**
- Real CVD (agg trades m-field) — OHLCV 프록시보다 정확
- L2 Order Book (bid/ask wall)
- Open Interest, Funding Rate

---

### v4.4 — 크로스코인 BTC 레짐 필터 (선택적)

```python
# BTC 레짐이 알트 거래에 영향
btc_cvd = btc_df['cvd_ratio_24'].iloc[-1]
btc_q75 = btc_df['cvd_ratio_24'].quantile(0.75)

if btc_cvd > btc_q75:  # BTC 과열
    for alt in ['DOT', 'ADA']:
        if alt_signal == 'LONG':
            size_mult *= 0.7  # 알트 롱 축소
```

---

### v4.5 — 모델 아키텍처 개선 (중기)

1. **EnhancedEnsemble 완전 활용 (non-lightweight)**
   - 최적화: lightweight=True (LGB only) → 빠른 파라미터 탐색
   - 최종 학습: lightweight=False → 7모델 풀 앙상블 (XGB GPU + CB GPU 활용)
   - XGBoost GPU: CPU 대비 75배 빠름 (검증됨)

2. **Stacking Meta-Learner 활성화**
   - 현재 OOF → LogisticRegression
   - 개선: OOF → LightGBM meta-learner (비선형 조합)

3. **Temporal Feature Engineering**
   - 4h 바 시간대별 수익률 편차 반영 (20:00 UTC 최고 55.4% WR)
   - CVD 48h 시그널을 별도 feature로 추가

4. **Kelly Criterion 포지션 사이징**
   - 현재: risk_frac × conf × regime × drawdown
   - 개선: half-Kelly(f* = edge/odds) — 30회 이상 거래 후 활성화

---

## 모델 개발 우선순위

| 우선순위 | 작업 | 예상 개선 | 소요 시간 |
|---------|------|---------|---------|
| ★★★ | v4.1 파라미터 업데이트 | +0.5%/trade netEV | 30분 |
| ★★★ | CVD Contrarian Filter | +0.3%/trade | 2시간 |
| ★★☆ | OrderFlowCollector 연결 | real-time 정확도↑ | 3시간 |
| ★★☆ | OFI Entry Timing | +0.1%/trade | 1시간 |
| ★☆☆ | 풀 앙상블 최종 학습 | 정확도↑ | 4시간 |
| ★☆☆ | Kelly Criterion | 리스크 최적화 | 30+ trades 필요 |

---

## 현재 병목 및 제약

1. **가상거래 신호 없음**: threshold 높고 현재 시장이 HOLD 편향
   → v4.1 파라미터 업데이트 즉시 적용 권장

2. **LightGBM OpenCL 없음**: PyPI 빌드 OpenCL 의존
   → CPU 16스레드로 해결 (현재 적용됨)

3. **CatBoost GPU 소용량 데이터 느림**: 초기화 오버헤드
   → lightweight=True에서 제외, full ensemble에서만 사용

4. **XRP 거래 중단 (CLAUDE.md)**: 최적화 대상이나 실거래 비활성
   → 결과 참고용으로만 사용
