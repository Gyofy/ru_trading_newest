# forecast-moirai-risk

확률분포/quantile 기반 하방리스크 계산 스킬. Salesforce Moirai 기반.

user-invocable: true
context: fork
disable-model-invocation: true

## 역할
- Moirai의 mixture distribution forecast로 하방 꼬리위험 추정
- VaR, CVaR 수준의 리스크 지표 산출
- 방향 예측보다 리스크 경고에 특화

## 입력
- `data/features/{date}/features.parquet`

## 출력
- `data/predictions/{date}/{hour}_moirai_risk.json`
```json
{
  "ticker": "005930",
  "var_95": -0.025,
  "cvar_95": -0.038,
  "tail_prob": 0.08,
  "distribution_type": "mixture",
  "model": "moirai-base"
}
```

## allowed-tools
- Bash(python src/models/moirai_risk.py*)
- Read
- Glob
