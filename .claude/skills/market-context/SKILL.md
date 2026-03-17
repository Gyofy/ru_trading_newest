# market-context

배경지식 스킬: 시장 운영 규칙과 프로젝트 데이터 규약을 제공합니다.

user-invocable: false

## 거래시간
- KRX 정규장: 09:00~15:30 KST
- NXT (확장 시 추가): 08:00~20:00 KST (선정 종목만)
- 장 외 시간에는 주문 스킬 호출 금지

## 데이터 freeze 규칙
- 1시간봉은 해당 시간이 완전히 닫힌 뒤(XX:00 시점) 확정
- 1분봉 원천은 `data/raw/` 에 `{date}/{ticker}_1m.parquet`
- 1시간 정제본은 `data/processed/` 에 `{date}/{ticker}_1h.parquet`
- Feature store는 `data/features/` 에 `{date}/features.parquet`
- 예측 결과는 `data/predictions/` 에 `{date}/{hour}_pred.json`
- 주문 기록은 `data/orders/` 에 `{date}/orders.json`

## 종목 범위 (v1)
- KOSPI200 구성종목
- 대표 ETF: KODEX200, TIGER200, KODEX 레버리지, KODEX 인버스
- 유동성 기준: 최근 20일 평균 거래대금 10억원 이상

## 리포트 형식
- 모든 내부 데이터 교환은 JSON 또는 Parquet
- 사용자용 리포트는 Markdown 테이블
