# rjh_20260320: Feature Leakage Discovery + Strategy Pivot

## Timeline

### 10:00 — Feature Leakage Discovery
- STL decomposition (centered MA): **미래 period/2 바 참조**
- `ichi_above_cloud` `.shift(26)`: **26바 미래 참조**
- SVD stride + interpolation: **보간 불안정**
- S2 방향 예측: 누수 포함 **0.687** → 제거 후 **0.518** (거의 랜덤)

### 11:00 — Clean Feature 재검증
- Permutation test: p=0.000 (패턴 학습은 사실이지만 누수에서 옴)
- 5코인 clean walk-forward: **전부 마이너스** (avg -0.17%/trade)
- **이전 모든 백테스트 수익은 누수의 산물**

### 12:00 — 전략 전수 검색
- 15,120 조합 (9코인 × 12전략 × 7 TP/SL × 6 hold time)
- 양수: 2,379개 (15.7%), 랜덤 기대 ~30%보다 낮음
- **BTC spike 관련만 일관적 양수, 나머지 전부 마이너스**

### 13:00 — BTC Spike Edge 검증
- "BTC 롱" 편향 아님 (180일 BTC -38.5% 하락장)
- FOLLOW BTC: +0.28% vs COUNTER: -0.52% (명확한 차이)
- Edge 원인: momentum continuation + volatility clustering (lag 아님)

### 14:00 — 레버리지/청산 리스크 분석
- 3x + 24h hold: 청산확률 ~0%
- 5x + 2h hold: 청산확률 ~0%
- Kelly: 17.8%, Half Kelly: 8.9%

### 15:00 — SOL/ADA 마이크로스트럭처 Deep Dive
- 캔들 패턴(doji, engulfing, marubozu): 전부 마이너스
- 볼륨 이상(spike, 압축/확장): 전부 마이너스
- **BTC spike + 알트 확인(볼륨+바크기)만 양수**
- SOL: n=67, WR 56.7%, avg +0.20%

### 16:00 — ML Enhancement + Paper Bot
- 28-feature ML 필터: marginal 개선 (+0.08%, 통계 부족)
- Paper bot v2: 확인 필터 + score 기반 사이징
- 백그라운드 실행 시작

## 파일 목록

| File | Description |
|------|-------------|
| `run_btc_spike_paper.py` | BTC spike paper trading bot (확인 필터 포함) |
| `exhaustive_search.py` | 15,120조합 전수 검색 |
| `brainstorm_2h.py` | 2h hold 전략 탐색 (6개 패러다임) |
| `brainstorm_full.py` | 레버리지/청산 리스크 + 5개 옵션 분석 |
| `btc_bias_check.py` | "BTC 롱 편향" 검증 (기각) |
| `coin_deep_dive.py` | SOL/XRP/ADA 코인별 성격 (변동성/상관/시간대) |
| `coin_deep_v2.py` | Round 2 비판적 검증 (SHORT 분리, 기간 분할) |
| `deep_think.py` | Edge 본질 분석 (lag vs momentum vs volatility) |
| `strategy_500.py` | $500 equity 전략 분석 |
| `strategy_6h.py` | 6h hold 종합 검색 (6개 전략 × 4코인) |
| `stress_test.py` | Tail risk + 연속 손실 + 자본 경로 |
| `ml_spike_optimizer.py` | ML spike quality scoring (ET/GBM) |
| `sol_ada_3h_microstructure.py` | 캔들/볼륨/바 이상 300+ 조합 |

## 핵심 교훈

```
1. Feature leakage는 "좋아 보이는 모든 것"을 만들 수 있다
2. Clean feature에서의 방향 예측은 비용을 이기지 못한다
3. 가격 패턴(캔들, 모멘텀, RSI)은 단독으로 edge가 없다
4. 유일한 edge: BTC 급변 시 알트 추종 (momentum continuation)
5. ML은 n>500 이벤트가 있어야 의미 있다
```
