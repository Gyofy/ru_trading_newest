# forecast-timesfm-benchmark

기준선/적응형 벤치마크 스킬. Google TimesFM + ICF 기반.

user-invocable: true
context: fork
disable-model-invocation: true

## 역할
- TimesFM을 기준선 예측기로 사용
- 새 종목군/섹터 적응성 테스트 겸용
- Chronos-2 결과와의 편차 모니터링

## 입력
- `data/features/{date}/features.parquet`

## 출력
- `data/predictions/{date}/{hour}_timesfm.json`
- 동일 스키마: direction_prob, expected_return, q10/q50/q90, uncertainty

## allowed-tools
- Bash(python src/models/timesfm_forecast.py*)
- Read
- Glob
