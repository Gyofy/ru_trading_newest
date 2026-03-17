# backtest-walkforward

Walk-forward 백테스트 + 누수 점검 스킬.

user-invocable: true
context: fork

## 핵심 원칙
- 매 시점에 실제로 알 수 있었던 정보만 사용 (point-in-time)
- 미래 데이터 참조 절대 금지
- 학습/평가 데이터 누수 자동 점검

## 수행 단계
1. 기간/종목/모델 파라미터 입력
2. walk-forward window 설정 (train: 60일, val: 10일, step: 5일)
3. 각 window별:
   a. feature 재구성 (point-in-time)
   b. 모델 추론 (학습된 가중치만 사용)
   c. 신호 생성
   d. 시뮬레이션 체결 (슬리피지 + 수수료 + 세금 반영)
4. 누수 점검: 미래 정보 사용 여부 자동 검증
5. 리포트 생성

## 평가 지표
- 방향 hit rate
- 기대수익 vs 실현수익
- MDD (Maximum Drawdown)
- Turnover
- 비용 반영 Sharpe / Sortino
- Calibration error
- 주문 거절률

## 출력
- `data/reports/{date}/backtest_report.json`
- `data/reports/{date}/backtest_summary.md`

## allowed-tools
- Bash(python src/evaluation/*)
- Read
- Glob
