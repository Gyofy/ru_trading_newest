# Adaptive Exit RL — 종합 브레인스토밍

> 2026-03-24 | 3개 도메인(트레이딩/의료/ML이론) 레퍼런스 종합

---

## 1. 문제 정의

```
현재:  고정 TP=5×ATR, SL=2×ATR, TTL=24bars
문제:  시장 상황에 따라 최적 청산 타이밍이 다름
       - 강한 추세: TP를 더 열어야 함 (수익 반납 방지)
       - 약한 추세: 빨리 나와야 함 (시간 낭비 방지)
       - 변동성 급변: SL 동적 조절 필요

목표:  RL이 매 바마다 {HOLD, TIGHTEN_SL, CLOSE_NOW}를 결정
       → 고정 배리어보다 높은 per-trade Sharpe
```

---

## 2. 도메인별 핵심 인사이트

### 의료 (혈당/ICU) → 트레이딩 매핑

| 의료 개념 | 트레이딩 매핑 | 출처 |
|-----------|-------------|------|
| 혈당 궤적 모니터링 | PnL 궤적 모니터링 | 인슐린 PAINT |
| "치료 윈도우"(농도 범위) | "수익 윈도우"(PnL 범위) | PK/PD 모델 |
| 안전 정책 롤백 | 고정 SL/TP로 자동 복귀 | PAINT framework |
| 중간 보상(SOFA 점수) | 미실현 PnL 곡선 형태 | AI Clinician |
| JITAI 수용성 윈도우 | 마이크로스트럭처 이벤트 시점 | HeartSteps |
| Changepoint 감지 | PnL 레짐 전환 감지 | BOCPD |

**핵심 통찰**: 인슐린 투여 ≈ 포지션 청산. 둘 다 "궤적을 모니터링하다가 적절한 시점에 개입"하는 문제.

### ML 이론 → 최적 방법론

| 방법론 | 적합도 | 핵심 역할 |
|--------|--------|----------|
| **IQN + CVaR** | 최고 | 꼬리 리스크 최적화 → SL 자동 조절 |
| **Option-Critic** | 최고 | "포지션 유지"를 하나의 option으로, termination = exit |
| **Cal-QL (offline→online)** | 최고 | 백테스트→paper→live 전환 안정성 |
| **Meta-RL** | 높음 | 레짐 변화 시 빠른 적응 |
| **Decision Transformer** | 중간 | 과거 거래 패턴에서 학습, 해석 가능 |
| **BOCPD** | 높음 | PnL 궤적의 구조적 변화 감지 |

---

## 3. 제안 아키텍처: Adaptive Exit RL v1

### 3.1 프레임워크: Option-Critic + CVaR

```
포지션 = "Option" (시작: 진입, 종료: 청산)

Option 내부:
  매 바마다 Termination 확률 β(s) 계산
  β(s) > threshold → CLOSE
  β(s) < threshold → HOLD (+ SL 조정)

학습:
  Termination Gradient: 현재 option이 비효율적이면 β↑
  CVaR 제약: 최악 10% 시나리오 손실 < X%
```

### 3.2 State Space (15-dim)

포지션 진입 후 매 바마다 관측:

```
[포지션 상태] 5dim
  unrealized_pnl_pct      현재 미실현 PnL (%)
  bars_held / max_hold     정규화된 보유 시간
  pnl_vs_sl               현재 가격과 SL 사이 거리 (ATR 단위)
  pnl_vs_tp               현재 가격과 TP 사이 거리 (ATR 단위)
  max_favorable_excursion  최대 유리 움직임 (진입 후 최고점)

[시장 상태] 6dim
  atr_ratio               현재 ATR / 진입 시 ATR (변동성 변화)
  cvd_delta               진입 후 CVD 변화량
  rsi_current             현재 RSI
  volume_ratio            현재 거래량 / 20바 평균
  tsmom_7d_current        현재 7일 모멘텀 (추세 유지?)
  oi_change               진입 후 OI 변화

[리스크] 3dim
  bocpd_run_length        PnL 궤적의 BOCPD 런렝스 (레짐 나이)
  portfolio_dd            포트폴리오 전체 drawdown
  consecutive_bars_losing  연속 손실 바 수

[포지션 메타] 1dim
  strategy_type           A_sniper(1.0) / B_steady(0.0)
```

### 3.3 Action Space (4 discrete)

```
0: HOLD           — 유지, 변경 없음
1: TIGHTEN_SL     — SL을 1×ATR로 축소 (수익 보호)
2: MOVE_SL_TO_BE  — SL을 본절가(entry)로 이동 (무위험화)
3: CLOSE_NOW      — 즉시 시장가 청산
```

### 3.4 Reward Function

**Intermediate Reward (매 바, ICU AI Clinician 방식):**

```python
def intermediate_reward(state):
    # 1. PnL 방향 보상 (추세 지속 시 작은 양수)
    pnl_reward = unrealized_pnl_change * 10

    # 2. 리스크 페널티 (SL 근접 시 음수)
    risk_penalty = -max(0, 1 - pnl_vs_sl) * 0.5

    # 3. 시간 비용 (보유 시간 증가 → 소폭 감소)
    time_cost = -0.01

    return pnl_reward + risk_penalty + time_cost
```

**Terminal Reward (청산 시):**

```python
def terminal_reward(final_pnl, action):
    # Differential: 고정 배리어 대비 개선분
    baseline_pnl = what_fixed_barrier_would_have_given()
    alpha = final_pnl - baseline_pnl  # RL의 marginal contribution
    return alpha * 100
```

### 3.5 Safety: CVaR Constraint + Rollback

```
제약: CVaR(10%) of per-trade drawdown < -3%
  → Lagrange multiplier λ가 자동으로 SL 강도 조절
  → λ 증가 = 더 보수적 (SL 타이트)
  → λ 감소 = 더 공격적 (SL 느슨)

Rollback:
  if rolling_10_trade_sharpe < 0:
    revert to fixed SL/TP (v5.3 baseline)
    log("adaptive exit suspended, safe policy active")
```

---

## 4. 학습 파이프라인

### Phase 1: Offline Pre-training (Cal-QL)

```
데이터: 365일 × 10코인 백테스트 거래 (~300건)
       각 거래의 bar-by-bar state trajectory 기록

방법:  Cal-QL (Conservative + Calibrated Q-Learning)
       → overestimation 방지
       → offline→online 전환 시 성능 유지

도구:  d3rlpy 라이브러리 (scikit-learn 스타일 API)
```

### Phase 2: Paper Trading Fine-tuning

```
환경:  run_tsmom_paper.py (현재 가동 중)
방법:  RLPD (50% offline replay + 50% live data)
       → 기존 학습 유지하면서 실시간 적응

평가:
  - Conditional lift: RL exit avg PnL > fixed exit avg PnL
  - CVaR compliance: 최악 10% < -3%
  - Rollback trigger: 10-trade rolling Sharpe < 0 시 중단
```

### Phase 3: Live Deployment

```
조건:  50+ paper trades with RL exit
       Sharpe(RL) > Sharpe(fixed) × 0.9
       CVaR compliant

적용:  run_live_bot_v2.py의 sl_tp_monitor에 통합
       check_exit() → rl_exit_decision() 대체
```

---

## 5. BOCPD 통합: PnL 궤적 Changepoint

```python
# 매 바마다 PnL 궤적에 BOCPD 적용
from bayesian_changepoint import BOCPD

class PnLChangeDetector:
    def __init__(self):
        self.bocpd = BOCPD(hazard=1/50)  # 평균 50바마다 changepoint

    def update(self, pnl_change):
        run_length_probs = self.bocpd.update(pnl_change)

        # 런렝스가 bimodal이면 (현재 + 0 모두 높으면) → 레짐 전환
        p_change = run_length_probs[0]  # P(run_length = 0)

        if p_change > 0.3:  # 30% 이상이면 changepoint 시그널
            return "REGIME_CHANGE"
        return "STABLE"
```

**활용**:
- STABLE → RL이 자유롭게 결정
- REGIME_CHANGE → 즉시 CLOSE_NOW (추세가 끝남)

---

## 6. Event-Driven Exit Evaluation (JITAI 방식)

```
현재:  매 4h bar마다 exit 체크 (fixed timer)
개선:  아래 이벤트 발생 시에만 RL exit 평가

트리거 이벤트:
  1. CVD 방향 반전 (누적 매수→매도 or 반대)
  2. 거래량 > 3× 평균 (비정상 활동)
  3. OI 급변 (|change| > 5%)
  4. ATR 급변 (현재 > 1.5× 진입 시)
  5. BOCPD changepoint 감지

비트리거 시: HOLD (아무것도 안 함)
→ 노이즈 결정 감소, 의미 있는 시점에만 집중
```

---

## 7. 구현 로드맵

| Phase | 작업 | 난이도 | 기간 | 전제 |
|-------|------|--------|------|------|
| **0** | bar-by-bar state trajectory 기록 시작 | 낮음 | 1일 | paper bot 가동 중 |
| **1** | BOCPD 구현 + RL state에 추가 | 중간 | 2일 | — |
| **2** | Cal-QL offline training (d3rlpy) | 중간 | 3일 | 300+ 거래 trajectory |
| **3** | Paper bot에 adaptive exit 통합 (shadow) | 중간 | 2일 | Phase 2 완료 |
| **4** | Event-driven evaluation 추가 | 낮음 | 1일 | Phase 3 완료 |
| **5** | Rollback safety + CVaR constraint | 중간 | 2일 | Phase 3 완료 |
| **6** | Online fine-tuning (RLPD) | 높음 | 지속 | Phase 5 완료 |

### 즉시 시작 가능한 것 (Phase 0)

paper bot의 포지션 보유 중 매 바의 state를 기록:

```python
# run_tsmom_paper.py의 check_exit() 내부에 추가
exit_state = {
    "unrealized_pnl": (current_close - entry) / entry * side,
    "bars_held": pos.bars_held,
    "atr_ratio": current_atr / entry_atr,
    "cvd_delta": current_cvd - entry_cvd,
    "rsi": current_rsi,
    "volume_ratio": current_vol / vol_ma,
    "action_taken": "HOLD" or "SL" or "TP" or "TTL",
}
# → position_trajectory.jsonl에 append
```

이 데이터가 **50포지션 × 평균12바 = 600 state-action pairs** 쌓이면 Cal-QL 학습 가능.

---

## 8. 기존 인프라 재활용

| 기존 모듈 | 새 역할 |
|-----------|---------|
| `src/rl/bandit.py` | 사이징 전용 유지, exit에는 별도 agent |
| `src/rl/state_builder.py` | exit state builder 추가 |
| `src/rl/signal_logger.py` | trajectory logger 추가 |
| `src/rl/counterfactual.py` | fixed barrier 대비 marginal PnL 계산 |
| `src/rl/rl_gate.py` | adaptive exit의 shadow/active 제어 |
| `sl_tp_monitor.py` | event-driven trigger 추가 |

---

## 9. 냉정한 평가

### 이게 작동할 가능성

| 요소 | 판단 | 근거 |
|------|------|------|
| 고정 배리어 대비 개선 가능? | **가능** | 의료 RL에서 고정 프로토콜 대비 20-30% 개선 보고 |
| 데이터 충분? | **부족** | 300 거래 × 12바 = 3,600 pairs (최소 수준) |
| 과적합 위험? | **높음** | 15-dim state + 4-action → 최소 1,000+ pairs 필요 |
| Rollback으로 안전? | **가능** | 의료 PAINT 프레임워크 검증됨 |
| 실전 효과? | **불확실** | 백테스트에서 +0.3%p 정도 기대, 혁신적 변화 아님 |

### 솔직한 결론

> **Adaptive exit RL은 이론적으로 유효하나, 현재 데이터(300 거래)로는 학습이 불안정할 가능성 높음.**
>
> **가장 현실적인 접근:**
> 1. Phase 0(trajectory 기록)부터 시작 → 데이터 축적
> 2. BOCPD를 규칙 기반으로 먼저 구현 (RL 없이)
> 3. 데이터 1,000+ pairs 축적 후 Cal-QL 시도

---

## 10. 트레이딩 도메인 실증 결과 (3번째 에이전트)

### 가장 중요한 발견: Positional Context (PPO, 2024)

```
논문: DRL with Positional Context for Intraday Trading
결과: Sharpe 2.759, MDD -6.4% (상품 포트폴리오)
      백금: Sharpe 3.812 vs buy-and-hold -0.156

핵심: 4개의 "포지션 컨텍스트" 피처가 exit 성능을 결정
  1. 남은 시간 (time remaining)
  2. 현재 포지션 방향 (-1/0/+1)
  3. 미실현 PnL (net of costs)
  4. 일일 누적 수익률

이 4개를 빼면 → 수수료 차감 후 마이너스 전략으로 전락
```

**우리 시스템 적용**: state에 이 4개가 이미 설계되어 있음 (unrealized_pnl, bars_held, strategy_type). 검증된 구조.

### Asymmetric Reward Dampening (Sadighian, 2020)

```
7가지 reward 함수 비교 결과:
  최적: asymmetric dampening (eta=0.35)
  → 미실현 손실에 2x 페널티, 미실현 이익은 감쇄
  → 너무 높으면 조기 청산, 너무 낮으면 drawdown 방치

Event-driven training: 1bps 가격 변동 시에만 학습 → 훈련 시간 70% 단축
```

### Optimal Stopping = Exit Timing (수학적 등가)

```
IQN (Implicit Quantile Network)이 최적 정지 문제에서
Least-Squares Monte Carlo 대비 974% 개선

포지션 exit = 최적 정지 문제의 특수 사례
→ IQN/distributional RL이 이론적으로 가장 적합
```

### AdaptiveTrend Trailing Stop (가장 강한 실증)

```
ATR 기반 trailing stop (alpha=2.5):
  Sharpe 2.41, MDD -12.7%, Calmar 3.18
  trailing 제거 시: Sharpe -0.73, MDD +9.7%p

최적 timeframe: 6h (크립토)
alpha 범위 [2.0, 3.5]에서 robust

→ "adaptive exit가 전략 성과의 지배적 요인"이라는 가장 명확한 증거
```

### 실무 Reward Shaping 교훈 (tr8dr, Denny Britz)

```
문제: buy/sell이 전체 timestep의 극소 비율 → reward 극히 희소
해법:
  1. shaped intermediate rewards (매 바 보상)
  2. asymmetric penalty (하락 2x, 상승 1x)
  3. time decay (-0.01/bar)
  4. eta=0.35 dampening

실패한 접근:
  - 순수 terminal reward → 너무 sparse, 학습 안 됨
  - 순수 unrealized PnL reward → 단기 편향, 과적합
```

---

## 11. 3개 도메인 최종 종합

### 방법론 우선순위 (실증 근거 순)

| 순위 | 방법 | 근거 | 기대 효과 |
|------|------|------|----------|
| **1** | **Positional Context + PPO** | Sharpe 3.81, 논문 검증 | exit 성능의 핵심 |
| **2** | **IQN + CVaR** | optimal stopping 974% 개선 | 꼬리 리스크 자동 조절 |
| **3** | **Asymmetric Reward (eta=0.35)** | 7가지 비교 최적 | 학습 안정성 |
| **4** | **Cal-QL offline→online** | 학습 망각 방지 | 배포 안전성 |
| **5** | **BOCPD changepoint** | PnL 레짐 전환 감지 | 즉시 적용 가능 |
| **6** | **Event-driven evaluation** | 훈련 70% 단축 | 노이즈 감소 |
| **7** | **Safe Rollback (PAINT)** | 의료 검증 | 실패 시 안전망 |

### 킬러 조합 (구현 제안)

```
Entry:   v5.3 그대로 (TSMOM + RSI + CVD + OI)
Exit:    PPO (positional context state)
         + IQN critic (CVaR 꼬리 리스크)
         + asymmetric reward (eta=0.35)
         + BOCPD changepoint trigger
         + Cal-QL pre-training
         + safe rollback to fixed SL/TP
```

---

## 12. 참고 문헌 (주요)

**의료/혈당:**
- PAINT: Offline RL for blood glucose (arXiv 2501.15972)
- AI Clinician: Optimal ICU treatment (Komorowski et al.)
- Distributional RL for glucose after cardiac surgery (Nature, 2025)

**ML 이론:**
- IQN + CVaR for futures trading (arXiv 2501.04421, 2025)
- Option-Critic Architecture (Bacon et al., 2017)
- Cal-QL: Calibrated Offline Pre-Training (NeurIPS 2023)
- DreamerV3: World Models (Nature, 2025)
- Decision Transformer (arXiv 2106.01345)

**Changepoint/시계열:**
- BOCPD (Adams & MacKay, 2007)
- BOCPD for order flow regime detection (Quantitative Finance, 2024)
- Hierarchical RL with changepoint detection (arXiv 2510.24988)
