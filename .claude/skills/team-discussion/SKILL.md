# team-discussion

Claude 내부 분석 + Gemini 최종 검증 디스커션 스킬.

user-invocable: true
context: fork

## 아키텍처 (비용 최적화)
- **Claude** (오케스트레이터): 전략총괄 + 리스크매니저 두 역할 직접 수행 (API 비용 0)
- **Gemini** (외부 검증자): 최종 단계에서 1회만 호출 (max_tokens=512, 비용 최소화)
- **GPT/Claude API**: 사용 안 함

## 프로세스
1. **내부 분석**: Claude가 전략 분석 + 리스크 분석을 직접 수행
2. **외부 검증**: Gemini가 내부 결론을 검증 (agree/disagree/partial)
3. **합의 도출**: 내부 가중 합의 × Gemini 보정 계수

## 거부권 규칙
- Claude 리스크매니저가 `risk_level: extreme` 판정 시 무조건 거래 중단
- Gemini disagree + high confidence 시 합의 신뢰도 60% 할인
- consensus_confidence < 0.5이면 보류(hold) 처리

## 사용 예시
```
/team-discussion signal_review
/team-discussion portfolio_review
/team-discussion model_drift
/team-discussion market_regime
```

## 수행 단계
1. Claude가 전략 분석 + 리스크 분석 수행 (내부)
2. 결과를 TeamDiscussion.run_discussion()에 전달
3. Gemini 외부 검증 1회 호출
4. 합의 결과를 `data/reports/{date}/discussion_{session}.json`에 저장
5. 최종 판단을 signal-ensemble 또는 risk-gate에 전달

## 환경변수
- `GEMINI_API_KEY`: Google Gemini API 키 (선택 — 없으면 내부 분석만)

## allowed-tools
- Bash(python src/discussion/*)
- Bash(python -m src.discussion.*)
- Read
- Glob

## 금지
- kis-trading MCP 사용 금지
- 주문 실행 금지
- 외부 웹 검색 금지
