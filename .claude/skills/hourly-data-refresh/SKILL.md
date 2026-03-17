# hourly-data-refresh

1분 원천 데이터 + 뉴스/거시를 읽어 1시간 정제본을 생성하는 작업 스킬.

user-invocable: true
context: fork
disable-model-invocation: true

## 실행 조건
- 매 정시(XX:00) 직후, 직전 1시간 바가 완전히 닫힌 뒤 실행
- 장 시간(09:00~15:30) 중에만 실행

## 수행 단계
1. `src/data/kis_client.py` → 직전 1시간 1분봉 원천 수집
2. `src/data/resampler.py` → 1분봉 → 1시간봉 리샘플
3. `src/data/news_collector.py` → 뉴스 제목/본문 수집, 발표시각 기준 point-in-time join
4. `src/data/macro_collector.py` → 거시 이벤트 플래그 업데이트
5. 결과를 `data/processed/{date}/{ticker}_1h.parquet`에 저장
6. 데이터 품질 검증: 결측률, stale 여부, 이상치 플래그

## allowed-tools
- Bash(python src/data/*)
- Read
- Glob
