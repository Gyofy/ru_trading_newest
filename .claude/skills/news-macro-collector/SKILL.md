# news-macro-collector

뉴스/거시 데이터 수집 전용 스킬. 주문 권한 없음.

user-invocable: true
context: fork

## 역할
- 웹/뉴스 MCP를 통해 종목 관련 뉴스, 공시, 거시 이벤트 수집
- 구조화된 JSON 요약만 출력 (원문 URL, 제목, 발표시각, 감성 라벨)
- 주문/실행 도구에는 절대 접근 불가

## 수행 단계
1. 대상 종목 리스트 로드
2. 종목별 최근 1시간 뉴스 검색
3. 거시 캘린더 (금리, CPI, 고용 등) 이벤트 확인
4. 감성 분류: positive / neutral / negative
5. `data/raw/{date}/news.json` 저장

## allowed-tools
- WebSearch
- WebFetch
- Read
- Bash(python src/data/news_collector.py*)
- Bash(python src/data/macro_collector.py*)

## 금지
- kis-trading MCP 사용 금지
- 주문 관련 스킬 호출 금지
