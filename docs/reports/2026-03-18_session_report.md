# Session Report — 2026-03-18
*생성: 2026-03-18 01:30 UTC | 세션 기간: 2026-03-17 14:00 ~ 2026-03-18 09:30 KST*

---

## 1. 인프라 수정 (이번 세션)

### GPU/CPU 최적화
| 항목 | 이전 | 수정 후 |
|------|------|--------|
| LightGBM | `device='gpu'` (OpenCL, RTX4070에 없음 → 전 평가 None) | `device='cpu'`, `num_threads=16` |
| RF / ET | `n_jobs=6` | `n_jobs=-1` (16코어 전부) |
| CPU 모델 학습 | 순차 | `ThreadPoolExecutor(4)` 병렬 |
| 윈도우 평가 | 15회 순차 | `ProcessPoolExecutor(4워커)` 병렬 |

**결과:** CPU 4워커 × ~280% = 총 ~1100% CPU 활용. 이전 0건 → 444건/세션 평가.

### 데이터 수집 완료 (GitHub 업로드 완료)
- `data/microstructure/` : 10코인 × 4파일 = 40 CSV (4,378 bars, 222 features, NaN=0)
- `src/execution/orderflow_collector.py` : 실시간 Binance agg trades/orderbook 수집기
- `docs/brainstorm_microstructure_strategy.md` : 마이크로스트럭처 전략 분석

---

## 2. Walk-Forward 최적화 결과 (v3.1 net EV)

### 세션 통계
- **총 평가 수:** 444건 (R1~R5, 각 라운드 12 param × 15 window × 3 coins)
- **평균 평가 시간:** 7.8s/eval
- **양수 netEV 비율:** XRP 96%, DOT 90%, ADA 90%

### 코인별 최적 파라미터 (현재까지)

| 코인 | 최고 netEV | S2 정확도 | MCC | k_upper | k_lower | threshold |
|------|-----------|---------|-----|---------|---------|-----------|
| **ADA** | **+1.2325%** | 61.3% | 0.131 | 3.0 | 0.6 | 0.40 |
| **XRP** | **+1.0664%** | 56.6% | 0.084 | 3.0 | 0.6 | 0.45 |
| **DOT** | **+0.9251%** | 51.0% | 0.003 | 3.0 | 0.6 | 0.45 |

### 파라미터 패턴 분석 (444 evals)

**R:R 비율별 평균 netEV (핵심 발견):**
```
R:R 0.0 ~ 1.5 :  n=135, avg=+0.068%  ← 낮은 R:R은 비효율
R:R 1.5 ~ 2.5 :  n=138, avg=+0.243%
R:R 2.5 ~ 3.5 :  n=120, avg=+0.540%
R:R 3.5 ~ 4.5 :  n=36,  avg=+0.813%
R:R 4.5 ~ 6.0 :  n=15,  avg=+1.083%  ← 높은 R:R이 압도적으로 우수
```
→ **k_upper=3.0, k_lower=0.6 (R:R=5.0)이 일관되게 최고**

**k_lower별 평균 netEV:**
```
k_lower=0.6 :  avg=+0.554%  ← 손절 tight = 결과 최고
k_lower=0.8 :  avg=+0.427%
k_lower=1.0 :  avg=+0.304%
k_lower=1.2 :  avg=+0.249%
k_lower=1.5 :  avg=+0.156%  ← 손절 wide = 결과 최저
```
→ **손절을 좁게 (k_lower=0.6 ATR) 설정할수록 net EV 상승 — 소손절 다익절 전략 유효**

**stage1_threshold별 평균 netEV:**
```
th=0.40 :  avg=+0.212%  ← 너무 낮으면 noise 진입
th=0.45 :  avg=+0.434%  ← 최적
th=0.50 :  avg=+0.367%
th=0.55 :  avg=+0.330%
th=0.60 :  avg=+0.391%
```
→ **th=0.45가 최적 균형점 (기존 frozen 0.50~0.52보다 약간 낮춰야 함)**

### 비교: 이전 frozen_params_v3_4 vs 이번 최적화

| 코인 | frozen k_upper | 신규 k_upper | frozen k_lower | 신규 k_lower | frozen th | 신규 th |
|------|--------------|-------------|--------------|-------------|----------|--------|
| DOT  | 1.5 (추정) | **3.0** | 1.0 (추정) | **0.6** | 0.50 | **0.45** |
| ADA  | 1.5 (추정) | **3.0** | 0.8 | **0.6** | 0.52 | **0.40** |

---

## 3. 가상 거래 현황 (Binance Testnet)

### 진행 상황
- **기간:** 2026-03-17 14:39 ~ 진행 중
- **거래 수:** 0건 (모델 HOLD 판정 지속)
- **잔고:** 4,999.97 USDT (수수료 0.03 USDT 차감)
- **처리된 4h 바:** 16:00 UTC (DOT $1.5925, ADA $0.2864)

### 신호 없음 원인 분석
```
DOT: regime=TREND_UP, s1_prob < 0.50 → HOLD
ADA: regime=TREND_UP, s1_prob < 0.52 → HOLD
```
- 현재 frozen_params의 threshold(0.50/0.52)가 최적화 권장값(0.45)보다 높음
- → **threshold 낮추면 신호 발생 빈도 증가 가능**
- BTC Fear&Greed=28 (공포), 전반적 약보합장 → 모델이 HOLD 편향 정상

---

## 4. 마이크로스트럭처 분석 핵심 발견

### A. CVD 역추세 효과 (★★★ 검증됨)
- HIGH CVD(buy 과열) → 48h 후 하락, SHORT edge +0.334%/trade (53.9% WR, 수수료 후)
- 6개 코인 모두 일관성: ADA +1.80%, ETH +1.21%, SOL +1.03%, BTC +0.88%
- **현재 시스템에 적용 미완: CVD 필터 오버레이 추가 필요**

### B. OFI 단기 선행 (+4-8h)
- `ofi_sum_3` HIGH → 4h +0.078%, 8h +0.100% (BTC)
- **진입 타이밍 필터로 활용 권장**

### C. 현재 시장 마이크로스트럭처
```
BTC: ms_composite=0.175 (NEUTRAL), flow=-0.007
ETH: ms_composite=0.096 (BEARISH), flow=+0.129
DOT: ms_composite=0.135 (NEUTRAL), flow=-0.202 ← 약세 흐름
ADA: ms_composite=0.138 (NEUTRAL), flow=+0.194 ← 매수세 형성 중
```

---

## 5. 모델 개발 방향 (브레인스토밍 → 실행 계획)

### Phase 1: 파라미터 업데이트 (즉시 적용 가능)
```yaml
# frozen_params_v3_4.yaml 업데이트 권고
DOT:
  k_upper: 3.0        # 1.5 → 3.0 (+100%)
  k_lower: 0.6        # 1.0 → 0.6 (-40%)
  stage1_threshold: 0.45  # 0.50 → 0.45
ADA:
  k_upper: 3.0        # → 3.0
  k_lower: 0.6        # 0.8 → 0.6
  stage1_threshold: 0.40  # 0.52 → 0.40
```
**예상 효과:** 신호 발생 빈도 증가 + net EV +0.5~0.8%/trade 개선

### Phase 2: 마이크로스트럭처 신호 통합 (1주일)
```python
# dynamic sizing에 CVD 필터 추가
cvd_filter_mult = 0.6 if cvd_ratio_24 > Q75 else (1.2 if cvd_ratio_24 < Q25 else 1.0)
ofi_timing_mult = 1.1 if ofi_sum_3 > Q67 else 1.0
final_size = base_size * cvd_filter_mult * ofi_timing_mult
```

### Phase 3: OrderFlowCollector 통합 (구현됨, 연결 필요)
- `OrderFlowCollector.get_snapshot()` → real CVD/OBI
- `flow_pressure` → dynamic sizing multiplier
- 실제 체결 데이터 기반 VPIN → 변동성 예측

### Phase 4: 48h CVD Contrarian Layer (별도 전략)
- SHORT k_upper=3.0 when cvd_ratio_24 > Q75 AND 48h horizon
- 별도 포지션 관리 (기존 4h 전략과 분리)

### Phase 5: 크로스코인 BTC 레짐 필터
- BTC cvd > Q75 → 모든 알트 LONG 사이즈 × 0.7
- BTC ms_flow > 0.3 AND TREND_UP → 알트 LONG 사이즈 × 1.2

---

## 6. 시스템 현황

```
[최적화] PID 152955  Round 5+ 진행 중 (25h 남음, 444 evals 완료)
[데모]   PID 156125  모델 재학습 중 (16:00 UTC 바 처리 완료)
[GPU]    RTX 4070   VRAM 1197/12282MiB (10%), CatBoost GPU 대기
[CPU]    16코어     ~1100% 사용 중 (4 optimizer workers)
```

### 다음 이벤트
- **20:05 UTC** (오전 5:05 KST): 다음 4h 바 — s1_prob 디버그 값 최초 출력
- **2026-03-19 02:00 UTC** (오전 11:00 KST): 최적화 마감 → 결과 집계
