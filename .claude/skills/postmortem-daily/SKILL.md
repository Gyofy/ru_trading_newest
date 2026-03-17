# postmortem-daily

일일 사후 리뷰 스킬. 예측 vs 실현 비교, 체결 분석, 리스크 점검.

user-invocable: true
context: fork

## 수행 단계
1. 당일 예측 결과 vs 실제 가격 비교
2. 체결/미체결/차단 주문 분석
3. 리스크 차단 사유 집계
4. 모델별 성과 비교 (chronos2 vs timesfm vs moirai vs lgbm)
5. 모델 drift 여부 점검 (최근 5일 hit rate 급락 감지)
6. stale-data 차단 횟수 집계
7. 비용 분석: 수수료 + 슬리피지 실현치
8. 일일 P&L, 누적 P&L, MDD 업데이트

## 출력
- `data/reports/{date}/postmortem.json`
- `data/reports/{date}/postmortem.md` (사용자용 요약)

## 알림 조건
- 당일 손실 > 계좌 1.5% → 경고
- 모델 hit rate < 45% (5일 연속) → drift 경고
- 체결 실패율 > 20% → 유동성 필터 재조정 권고

## allowed-tools
- Bash(python src/evaluation/*)
- Read
- Glob
