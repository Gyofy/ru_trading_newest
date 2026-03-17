# feature-pipeline

정제본에서 기술지표, 거래량/변동성, 감성, 거시 feature를 생성하는 스킬.

user-invocable: true
context: fork
disable-model-invocation: true

## 수행 단계
1. `data/processed/{date}/` 에서 1시간봉 로드
2. 기술지표 생성: SMA, EMA, RSI, MACD, Bollinger, ATR
3. 거래량 feature: VWAP, 거래대금 z-score, volume ratio
4. 변동성 feature: realized vol, Parkinson, 호가 스프레드
5. 감성 feature: 뉴스 감성 점수 rolling average
6. 거시 feature: 이벤트 플래그, 금리, 환율 변화율
7. 결측 처리: missing_flag 컬럼 유지, 무조건 채우지 않음
8. `data/features/{date}/features.parquet` 저장

## allowed-tools
- Bash(python src/features/*)
- Read
- Glob
