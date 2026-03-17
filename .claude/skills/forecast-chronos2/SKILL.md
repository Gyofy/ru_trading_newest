# forecast-chronos2

주력 예측 스킬. Amazon Chronos-2 기반 멀티변량 + covariate-aware forecasting.

user-invocable: true
context: fork
disable-model-invocation: true

## 역할
- Feature store에서 입력 구성
- Chronos-2 모델로 1시간봉 방향 예측 + quantile forecast
- 종목별 방향 확률, 기대수익, 예측 불확실성(quantile width) 출력

## 입력
- `data/features/{date}/features.parquet`
- 과거 context window: 168시간 (7일 × 24시간)

## 출력
- `data/predictions/{date}/{hour}_chronos2.json`
```json
{
  "ticker": "005930",
  "direction_prob": 0.62,
  "expected_return": 0.0031,
  "q10": -0.012,
  "q50": 0.003,
  "q90": 0.018,
  "uncertainty": 0.030,
  "model": "chronos2-base"
}
```

## allowed-tools
- Bash(python src/models/chronos2_forecast.py*)
- Read
- Glob
