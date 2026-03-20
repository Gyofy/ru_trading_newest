# rjh_20260320: Feature Leakage Discovery + Strategy Pivot

## 핵심 발견

### 1. Feature Leakage (데이터 누수)
- STL decomposition (centered MA): 미래 데이터 참조
- ichi_above_cloud (.shift(26)): 26바 미래 참조
- SVD stride+interpolation: 보간 불안정
- **S2 방향 예측: 누수 포함 0.687 → 제거 후 0.518 (거의 랜덤)**
- **이전 모든 백테스트 수익은 누수의 산물**

### 2. 수정 적용
- `crypto_ohlcv.py`: STL → EMA causal decomposition
- `signal_features.py`: ichi_above_cloud .shift(26) 제거
- `crypto_ohlcv.py`: SVD stride/interpolation 제거, 매 바 직접 계산
- `live_predictor.py`: predict_hybrid() 추가 (S1 + momentum direction)

### 3. 전수 검색 (15,120 조합)
- 9코인 × 12 전략 × 7 TP/SL × 6 hold time
- 양수 전략: BTC spike 관련만 일관적
- 모멘텀/RSI/EMA/볼륨 = 비용 차감 후 전부 마이너스
- 유일한 edge: BTC 1h >1.2% spike → 알트 추종

### 4. BTC Spike Paper Bot
- `run_btc_spike_paper.py`: 실행 중
- BTC 1h |ret| > 1.2% → SOL/ETH/XRP/ADA 진입
- TP=1.5*ATR, SL=1.0*ATR, 6h hold, 3x leverage

## 파일 목록
- `run_btc_spike_paper.py` - BTC spike paper trading bot
- `exhaustive_search.py` - 15,120조합 전수 검색
- `brainstorm_2h.py` - 2h hold 전략 탐색
- `brainstorm_full.py` - 레버리지/청산 분석 + 5옵션
- `btc_bias_check.py` - "BTC 롱" 편향 검증
- `coin_deep_dive.py` - SOL/XRP/ADA 코인별 성격 분석
- `coin_deep_v2.py` - Round 2 비판적 검증
- `deep_think.py` - edge 본질 분석 (lag vs momentum)
- `strategy_500.py` - $500 equity 전략 분석
- `strategy_6h.py` - 6h hold 종합 검색
- `stress_test.py` - tail risk + 연속 손실 분석
