# signal-ensemble

여러 모델 출력 + 유동성 필터를 합쳐 주문 제안 JSON을 생성하는 스킬.
주문 권한 없음.

user-invocable: true
context: fork

## 수행 단계
1. 예측 결과 로드: chronos2, timesfm, moirai_risk, (lgbm baseline)
2. 가중 앙상블: 모델별 최근 성과 기반 동적 가중
3. 4중 필터 적용:
   - 방향 확률 ≥ 60%
   - 기대수익 > 수수료(0.015%) + 슬리피지(0.05%) + 세금(0.23%)
   - 불확실성(quantile width) ≤ 임계값
   - 유동성/스프레드 기준 통과
4. 포지션 사이징: Kelly fraction (half-Kelly) + 한도 캡
5. **팀 디스커션 게이트**: 아래 조건 충족 시 team-discussion 스킬 발동
   - 시그널 확신도 < 70%
   - 포지션 비중 > 3%
   - 당일 누적 손실 > 1%
6. 디스커션 결과 반영: 합의가 abort/hold면 시그널 제거, reduce_size면 사이징 축소
7. 주문 제안 JSON 출력

## 출력
- `data/predictions/{date}/{hour}_signal.json`
```json
{
  "signals": [
    {
      "ticker": "005930",
      "action": "buy",
      "size_pct": 0.03,
      "expected_return": 0.0028,
      "confidence": 0.65,
      "risk_score": 0.12,
      "reason": "chronos2+timesfm consensus, low uncertainty"
    }
  ]
}
```

## allowed-tools
- Bash(python src/signals/*)
- Read
- Glob

## 금지
- kis-trading MCP 사용 금지
- 주문 실행 금지
