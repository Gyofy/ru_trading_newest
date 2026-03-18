# v4.3 Glucose RL 문헌 → 트레이딩 적용 판단

## 논문 6편에서 추출한 패턴 vs v4.3 현재 설계

### 패턴 1: Offline RL > Online RL

**논문:** Emerson 2023 — TD3-BC가 온라인 RL 대비 1/10 샘플로 더 안전한 정책 달성.
100일 데이터(5만 샘플)면 PID 대비 우위.

**v4.3 현재:** LinUCB Contextual Bandit (사실상 offline-friendly linear model).

**판단: 유지하되 로드맵에 추가**
- LinUCB는 이미 offline 학습 구조 (signal_log → offline_train → deploy)
- 200건이면 LinUCB 충분. TD3-BC는 1,000건+ 이후 비교 실험으로 미룸
- **즉시 반영할 것:** offline_train.py에 train/eval chronological split (이미 구현됨)

---

### 패턴 2: FQE (Fitted Q-Evaluation) 사전 배포 평가

**논문:** Beolet 2024, GLUCOSE 2025 — 배포 전에 FQE로 정책 rank.
실제 환자에 적용하기 전에 "이 정책이 기존보다 나은가" 정량 판단.

**v4.3 현재:** Shadow mode에서 conditional lift, calibration by bucket으로 수동 평가.

**판단: Phase 3에 경량 OPE 추가 (즉시 채택)**

이유: shadow mode에서 "이 정책을 live로 올려도 되는가"를 사람이 판단하는 것보다,
정량적 OPE 점수가 있으면 go/no-go 기준이 명확해짐.

구현:
```python
# src/rl/ope.py — lightweight importance-weighted OPE

def importance_weighted_ope(
    signal_log: list[dict],   # 전체 시그널 로그 (state, action, reward, ...)
    new_policy: LinUCB,       # 평가할 새 정책
    behavior_policy_action: int = 2,  # 기존 정책 = 항상 ACCEPT_1.00
) -> dict:
    """
    Importance Sampling OPE.
    behavior policy: v4.2 baseline (항상 action=2)
    target policy: new LinUCB

    Advantage: 기존 데이터로 새 정책의 예상 reward를 추정.
    단점: variance 높음 → clipped IS 사용.
    """
    weighted_rewards = []
    for rec in signal_log:
        if not rec.get("resolved"):
            continue
        state = np.array(rec["state"])
        actual_action = rec["action"]
        reward = rec["result_reward"]

        # Target policy probability (deterministic → 1 if matches, 0 if not)
        target_action = new_policy.select_action(state)

        if target_action == actual_action:
            # IS ratio = π_new(a|s) / π_old(a|s)
            # π_old = 1.0 for action=2, 0.0 for others (deterministic baseline)
            if actual_action == behavior_policy_action:
                is_ratio = 1.0  # both agree
            else:
                is_ratio = 0.0  # behavior never took this action
        else:
            is_ratio = 0.0

        # Clipped IS (cap at 5.0 for stability)
        is_ratio = min(is_ratio, 5.0)
        weighted_rewards.append(is_ratio * reward)

    if not weighted_rewards:
        return {"ope_value": 0.0, "n_samples": 0}

    return {
        "ope_value": np.mean(weighted_rewards),
        "ope_std": np.std(weighted_rewards),
        "n_samples": len(weighted_rewards),
        "effective_n": sum(1 for w in weighted_rewards if w != 0),
    }
```

**문제:** 기존 정책이 deterministic(항상 action=2)이면, 다른 action의
reward를 추정할 수 없음 → IS가 사실상 작동 안 함.

**해결:** Direct Method (DM) 사용 — LinUCB theta 자체가 Q-function 근사이므로,
```
OPE_DM = mean(max_a x^T theta_a) over all states in log
```
이게 더 실용적. FQE는 데이터가 1,000건+ 쌓인 후에 도입.

**v4.3 채택 범위:**
- offline_train.py에 `evaluate()` 함수 이미 있음 → DM 기반 OPE 점수 추가
- Phase 3 합격 기준에 "DM OPE > baseline mean reward" 조건 추가

---

### 패턴 3: Task Decomposition + Change Cap (Jafar 2024)

**논문:** multi-agent Q로 high-fat meal / exercise를 분리.
daily recommendation을 바로 쓰지 않고 **weekly median** + **±20% cap**.

**v4.3 현재:** 단일 gate (accept/reject+sizing 통합).
변경 제약 없음.

**판단: 변경 cap 즉시 채택, task 분리는 v4.4로**

이유: RL이 갑자기 극단적으로 바뀌는 것을 막는 inertia 장치가 필요.
Jafar의 20% cap 아이디어를 sizing에 적용하면:

```python
# rl_gate.py에 추가
def _apply_change_cap(self, new_action: int, last_week_actions: list[int]) -> int:
    """직전 주 평균 sizing 대비 ±25% 이내로 제한."""
    if not last_week_actions:
        return new_action
    avg_sizing = np.mean([SIZING_MAP[a] for a in last_week_actions if a > 0])
    new_sizing = SIZING_MAP[new_action]
    if avg_sizing > 0:
        ratio = new_sizing / avg_sizing
        if ratio > 1.25:
            # 가장 가까운 낮은 action으로 clamp
            return 3  # ACCEPT_1.25
        elif ratio < 0.75:
            return 1  # ACCEPT_0.75
    return new_action
```

**v4.3 채택 범위:**
- RLGate에 weekly action history 유지
- 새 action이 직전 주 평균 대비 ±25% 초과하면 clamp

---

### 패턴 4: Meta-Learning + Embedding (ARLPE 2023)

**논문:** meta-training → probabilistic patient embedding → 25 샘플로 빠른 적응.
새 환자에게 patient embedding이 context로 들어감.

**v4.3 현재:** coin one-hot 5차원 (is_dot, is_ada, ...).

**판단: 현재 one-hot 유지. 학습 embedding은 v4.5+로**

이유:
- 코인 5개, 각각 고유 특성이 config에 이미 반영됨 (threshold, blocked_regimes)
- one-hot이면 LinUCB theta가 코인별 bias를 자연스럽게 학습함
- probabilistic embedding은 수백 "환자"가 있을 때 의미 있음
  → 코인 5개에는 과도
- **만약 코인을 10~20개로 늘린다면** 그때 embedding 전환 검토

---

### 패턴 5: Prediction + Planning (G2P2C 2024)

**논문:** PPO + auxiliary glucose dynamics model + short-horizon planning.
미래 glucose trajectory를 시뮬레이션해서 policy 보정.

**v4.3 현재:** ML 예측(p_trade, p_direction)만 state에 넣고 RL이 판단.
"향후 가격 경로 분포"에 대한 정보 없음.

**판단: MFE/MAE 추정 모듈을 Phase 2에 추가 (중기 채택)**

이유: G2P2C의 "planning" 핵심은 "미래를 짧게 시뮬레이션해서 리스크 사전 평가".
트레이딩 버전은:

```
시그널 발생 시:
  → 현재 시장 상태에서 이 포지션의 MFE(Maximum Favorable Excursion),
    MAE(Maximum Adverse Excursion), TP/SL 도달 확률을 추정
  → 이 추정치를 RL state에 추가
```

이건 **기존 학습 데이터에서 이미 계산 가능**:
```python
# 과거 유사 상태에서의 MFE/MAE 통계
# regime + confidence bucket별로 테이블 만들기
mfe_table = {
    ("TREND_UP", "high_conf"): {"mean_mfe": 0.025, "mean_mae": 0.008},
    ("RANGE_HIGH", "low_conf"):  {"mean_mfe": 0.012, "mean_mae": 0.015},
    ...
}
```

**v4.3 채택 범위:**
- signal_log에 MFE/MAE 기록 필드 추가 (Phase 1 확장)
- Phase 2 학습 시 regime×confidence bucket별 MFE/MAE 테이블 구축
- 테이블이 충분히 차면 state vector에 `expected_mfe`, `expected_mae` 2차원 추가
- **지금 당장은 불가** (데이터 없음). Phase 1 로깅부터 시작

---

### 패턴 6: Distributional RL / Safety-Focused Reward (GLUCOSE 2025)

**논문:** distributional offline RL로 return 분포 자체를 모델링.
평균만 보면 tail risk를 놓침. CVaR 최적화.

**v4.3 현재:** reward = pnl_pct × 100 (단순 스칼라).

**판단: reward 함수에 tail penalty 추가 (즉시 채택 가능)**

이유: full distributional RL은 과도하지만, reward에 tail risk를 반영하는 것은 간단함.

```python
# 현재
reward = pnl_pct * 100

# 수정 (v4.3.1)
reward = pnl_pct * 100
if pnl_pct < -0.01:  # 1% 이상 손실
    reward -= abs(pnl_pct) * 50  # 추가 페널티 (tail risk)
```

하지만 초기에는 **reward를 단순하게 유지**하라는 원칙과 충돌.

**결론:** Phase 2 학습 시 두 가지 reward variant로 비교 실험:
- R1: `pnl_pct × 100` (순수 PnL)
- R2: `pnl_pct × 100 - λ × max(0, -pnl_pct - 0.01) × 50` (tail penalty)
- 두 모델 학습 → OPE로 비교 → 더 나은 것 선택

**v4.3 채택 범위:**
- offline_train.py에 `--reward_mode` 플래그 추가 (pnl / pnl_tail)
- Phase 2에서 비교 실험으로 결정

---

## 종합: v4.3에 즉시 반영할 것 vs 미룰 것

### 즉시 반영 (Phase 1 확장)

| 패턴 | 구현 | 공수 |
|------|------|------|
| FQE → DM OPE | offline_train.py에 DM 점수 추가 | 소 |
| Change cap | RLGate에 weekly history + ±25% clamp | 소 |
| MFE/MAE 로깅 | signal_logger에 필드 추가 + close 시 계산 | 소 |
| Reward variant | offline_train에 --reward_mode 플래그 | 소 |

### Phase 2~3에서 검증

| 패턴 | 시점 | 조건 |
|------|------|------|
| DM OPE 기반 go/no-go | Phase 3 | OPE > baseline |
| Tail penalty reward | Phase 2 | R1 vs R2 비교 |
| MFE/MAE state 추가 | Phase 2 | 200건 이상 축적 후 |

### v4.4+ 연기

| 패턴 | 이유 |
|------|------|
| TD3-BC / CQL | 1,000건+ 필요 |
| Full FQE | 복잡, IS variance |
| Probabilistic embedding | 코인 5개에 과도 |
| G2P2C planning | auxiliary model 별도 학습 필요 |
| Distributional RL | full distribution → LinUCB 구조와 호환 안 됨 |

---

## Sources
- [Emerson 2023 — Offline RL for Glucose Control](https://arxiv.org/abs/2204.03376)
- [Jafar 2024 — Personalized Insulin Dosing](https://www.nature.com/articles/s41467-024-50764-5)
- [GLUCOSE 2025 — Distributional Offline RL](https://www.nature.com/articles/s41746-025-01709-9)
- [G2P2C 2024 — PPO + Planning](https://www.sciencedirect.com/science/article/pii/S1746809423012727)
- [ARLPE 2023 — Meta-RL + Embedding](https://www.sciencedirect.com/science/article/pii/S0957417423006589)
- [FQE Overview](https://www.emergentmind.com/topics/fitted-q-evaluation-fqe)
