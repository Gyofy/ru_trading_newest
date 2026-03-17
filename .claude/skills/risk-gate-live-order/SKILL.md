# risk-gate-live-order

실거래 주문 스킬. 최대한 격리하여 운영.

user-invocable: true
context: fork
disable-model-invocation: true

## 보안 원칙
- Claude가 자동으로 이 스킬을 호출하면 안 됨 (disable-model-invocation)
- 외부 웹/Search MCP 완전 제거
- signal-ensemble의 주문 제안 JSON만 입력으로 받음
- 모든 주문은 pre-order gate를 통과해야 함

## 수행 단계
1. `data/predictions/{date}/{hour}_signal.json` 로드
2. pre-order gate 체크 (risk-policy 스킬 규칙 적용):
   - 장 시간 내?
   - 데이터 최신성 5분 이내?
   - 포지션 한도 이내?
   - 당일 손실 한도 이내?
   - 불확실성 허용 범위?
   - 유동성 기준 통과?
3. 통과한 신호만 KIS API로 주문 실행 (또는 paper 모드 기록)
4. 체결 결과를 `data/orders/{date}/orders.json`에 기록
5. 차단된 주문은 사유와 함께 `data/orders/{date}/blocked.json`에 기록

## allowed-tools
- Bash(python src/execution/order_executor.py*)
- Bash(python src/execution/risk_gate.py*)
- Read

## 금지
- WebSearch, WebFetch 사용 금지
- 뉴스/리서치 관련 도구 사용 금지
- 데이터 수집 스크립트 실행 금지
