# v4.3 RL Meta-Strategy Layer — Final Design (v2)

> v1 피드백 반영: credit assignment 통합, state vector 보강, reward 대칭화,
> portfolio ranking, fallback 정책, LinUCB 실전 디테일.

---

## 1. v4.2 현재 흐름 (변경 없음)

```
2h Cycle:
  Data(ccxt 4h) → Features(221cols) → MI Select(120)
  → Train(daily, 2-Stage S1→S2, per-coin combo)
  → per coin:
      regime_filter → predict_2stage → risk_gate(9) → micro_sizing → order
  Background: SL/TP/TTL monitor (30s)
```

### Decision Point 상세

```
_process_coin(coin):
  ① regime = detect_regime(df)
     → blocked? skip
  ② pred = predict_2stage(combo, df, threshold)
     → HOLD? skip
  ③ ticker = fetch_ticker()     → entry_price, spread_bps
  ④ barriers = compute_barriers(entry, atr, side)
  ⑤ funding = fetch_funding_rate()
  ⑥ check = pre_trade_gate(9 gates)
     → rejected? skip
  ⑦ qty = micro_sizing(qty, cvd, ofi)
  ⑧ place_order()
```

### 현재 한계 (v4.3가 해결할 것)

| 한계 | 현상 |
|------|------|
| 맹목적 진입 | ML 시그널 + risk gate 통과 = 무조건 진입 |
| 코인간 우선순위 없음 | 5코인 동시 시그널 → 전부 진입 |
| 최근 성과 미반영 | 3연패 후에도 동일 기준 |
| threshold 고정 | 시장 상황과 무관 |

---

## 2. RL 구조: 단일 Gate, 통합 Action

### 2.1 핵심 변경 (v1 대비)

| v1 설계 | 문제 | v2 수정 |
|---------|------|---------|
| Gate B(accept/reject) + Gate C(sizing) 분리 | Credit assignment 꼬임 | **단일 gate, 통합 action space** |
| State 20차원 | side/margin/coin_id/cost 누락 | **28차원** |
| Reward 비대칭 | accept/reject 스케일 불일치 | **대칭 reward** |
| per-signal 독립 판단 | 동시 시그널 우선순위 없음 | **cycle-level ranking** |
| min_accept_rate 강제 accept | 최악 시점에 억지 진입 | **v4.2 fallback** |

### 2.2 Gate 위치

```
predict_2stage() 이후
  → RL Gate (accept/reject + sizing 통합)
    → risk_gate (기존 9-gate 그대로)
      → order
```

RL Gate는 `②`와 `③` 사이에 위치. Ticker/funding 정보는 직접 fetch하지 않지만,
**최근 bar 기반 proxy**를 state에 포함하여 비용 감각을 갖게 함.

### 2.3 통합 Action Space

```python
ACTIONS = [
    "REJECT",         # 0: 진입하지 않음
    "ACCEPT_0.75",    # 1: base qty × 0.75
    "ACCEPT_1.00",    # 2: base qty × 1.00 (기본)
    "ACCEPT_1.25",    # 3: base qty × 1.25 (확대)
]

SIZING_MAP = {0: 0.0, 1: 0.75, 2: 1.0, 3: 1.25}
```

**왜 4개인가:**
- Gate B(2) × Gate C(6) = 12개 조합보다 학습 효율 3배
- 200 signals / 4 actions = action당 50 samples (충분)
- 1.5x는 제거: 초기 RL이 sizing을 과도하게 키우는 것 방지
- v4.4에서 검증 후 action 추가 가능

---

## 3. State Vector (28차원)

```python
@dataclass
class RLState:
    # ── Signal quality (6) ──────────────────────
    p_trade: float            # S1 확률 [0, 1]
    p_direction: float        # S2 확률 [0, 1]
    confidence: float         # p_trade × p_direction
    trade_margin: float       # p_trade - s1_threshold (threshold 대비 여유)
    dir_margin: float         # |p_direction - 0.5| × 2 (방향 확신도)
    side_sign: float          # +1.0 = BUY, -1.0 = SELL

    # ── Market regime (4) ───────────────────────
    regime_trend: float       # 1.0 if TREND_UP/DOWN, 0.0 if RANGE
    regime_up: float          # 1.0 if TREND_UP, 0.0 otherwise
    atr_pct: float            # atr_14 / close (변동성 수준)
    hurst: float              # Hurst exponent (추세 지속성)

    # ── Microstructure (3) ──────────────────────
    cvd_ratio: float          # cvd_ratio_6
    ofi_norm: float           # ofi_sum_3 / volume_sma_20
    ms_composite: float       # ms_composite [-1, 1]

    # ── Cost proxy (2) ─────────────────────────
    spread_proxy: float       # (high - low) / close of last bar (bid-ask proxy)
    last_funding: float       # 마지막 알려진 funding rate (직전 cycle 캐시)

    # ── Portfolio state (4) ─────────────────────
    open_positions: float     # 현재 오픈 수 / 5
    daily_pnl_pct: float      # 당일 누적 PnL / equity, clipped [-0.05, 0.05]
    weekly_pnl_pct: float     # 주간 PnL / equity, clipped [-0.10, 0.10]
    dd_ratio: float           # current DD / daily_limit [0, 1]

    # ── Coin history (4) ───────────────────────
    coin_win_rate_5: float    # 최근 5건 승률 [0, 1]
    coin_avg_pnl_5: float     # 최근 5건 평균 PnL%, clipped [-0.05, 0.05]
    coin_streak: float        # 연속 승(+)/패(-), clipped [-5, 5] / 5
    bars_since_last: float    # 마지막 거래 후 경과 bars / max_horizon

    # ── Cross-market (2) ───────────────────────
    btc_return_24h: float     # BTC 24h return, winsorized [-0.10, 0.10]
    corr_btc: float           # 코인-BTC rolling 상관계수

    # ── Coin identity (5) ──────────────────────
    # global model 1개 + coin one-hot (per-coin bandit 대신)
    is_dot: float             # 1.0 if DOT
    is_ada: float
    is_xrp: float
    is_sol: float
    is_link: float

    # ── Intercept (1) ──────────────────────────
    intercept: float = 1.0    # 상수항 (LinUCB bias term)

    def to_array(self) -> np.ndarray:
        """28-dim + 1 intercept = 29-dim vector."""
        return np.array([
            self.p_trade, self.p_direction, self.confidence,
            self.trade_margin, self.dir_margin, self.side_sign,
            self.regime_trend, self.regime_up, self.atr_pct, self.hurst,
            self.cvd_ratio, self.ofi_norm, self.ms_composite,
            self.spread_proxy, self.last_funding,
            self.open_positions, self.daily_pnl_pct, self.weekly_pnl_pct, self.dd_ratio,
            self.coin_win_rate_5, self.coin_avg_pnl_5, self.coin_streak, self.bars_since_last,
            self.btc_return_24h, self.corr_btc,
            self.is_dot, self.is_ada, self.is_xrp, self.is_sol, self.is_link,
            self.intercept,
        ], dtype=np.float64)  # shape: (30,)
```

### 3.1 Feature Normalization 정책

| Feature | Range | Clipping | 비고 |
|---------|-------|----------|------|
| p_trade, p_direction, confidence | [0, 1] | 자연 범위 | |
| trade_margin | [-0.5, 0.5] | clip | threshold=0.45면 p=0.95 → margin=0.50 |
| dir_margin | [0, 1] | 자연 범위 | |
| side_sign | {-1, +1} | | |
| atr_pct | [0, 0.1] | upper clip 0.1 | 10% 이상 변동성은 이상치 |
| hurst | [0, 1] | 자연 범위 | |
| cvd_ratio, ofi_norm | [-3, 3] | winsorize | z-score 수준 |
| ms_composite | [-1, 1] | 자연 범위 | |
| spread_proxy | [0, 0.05] | upper clip | |
| last_funding | [-0.003, 0.003] | clip | |
| daily_pnl_pct | [-0.05, 0.05] | clip | |
| weekly_pnl_pct | [-0.10, 0.10] | clip | |
| dd_ratio | [0, 1] | 자연 범위 | |
| coin_win_rate_5 | [0, 1] | 자연 범위 | |
| coin_avg_pnl_5 | [-0.05, 0.05] | clip | |
| coin_streak | [-1, 1] | /5 후 clip | |
| bars_since_last | [0, 1] | /max_horizon | |
| btc_return_24h | [-0.10, 0.10] | winsorize | |
| corr_btc | [-1, 1] | 자연 범위 | |
| coin one-hot | {0, 1} | | |
| intercept | 1.0 | 상수 | |

---

## 4. Reward Design (대칭)

```python
def compute_reward(action: int, pnl_pct_net: float) -> float:
    """
    ACCEPT 시: 실현 PnL.
    REJECT 시: counterfactual PnL의 부호 반전.
    스케일 동일 = bandit 학습 안정적.
    """
    return pnl_pct_net * 100  # both accept and reject use same scale


# accept된 거래:
#   reward = realized_pnl_pct * 100
#   예: +1.2% 수익 → reward = +1.2

# reject된 시그널 (counterfactual):
#   reward = -counterfactual_pnl_pct * 100
#   예: 진입했으면 +2.0% 였을 것 → reward = -2.0 (기회 놓침)
#   예: 진입했으면 -1.5% 였을 것 → reward = +1.5 (손실 회피)
```

**왜 단순한가:** 초기 단계에서 복잡한 보상(속도 보너스, DD 페널티 등)은
학습 불안정의 원인. PnL% × 100으로 통일하고, v4.4에서 필요 시 보강.

### 4.1 Counterfactual 계산

```python
def compute_counterfactual(
    df_future: pd.DataFrame,  # 시그널 이후 max_horizon bars
    entry_price: float,
    side: str,
    sl_price: float,
    tp_price: float,
    max_horizon: int = 18,
    cost_pct: float = 0.002,  # 왕복 비용 추정
) -> float:
    """시그널을 ACCEPT했다면의 가상 PnL."""
    close = df_future["close"].values
    for i in range(min(len(close), max_horizon)):
        price = close[i]
        if side == "BUY":
            if price >= tp_price:
                return (tp_price - entry_price) / entry_price - cost_pct
            if price <= sl_price:
                return (sl_price - entry_price) / entry_price - cost_pct
        else:
            if price <= tp_price:
                return (entry_price - tp_price) / entry_price - cost_pct
            if price >= sl_price:
                return (entry_price - sl_price) / entry_price - cost_pct
    # TTL expiry
    final = close[min(len(close)-1, max_horizon-1)]
    if side == "BUY":
        return (final - entry_price) / entry_price - cost_pct
    return (entry_price - final) / entry_price - cost_pct
```

---

## 5. Portfolio-Level Ranking

### 5.1 문제

현재: per-coin 독립 판단 → 5코인 동시 ACCEPT 가능 → 과다 노출.

### 5.2 해결: Cycle-Level Ranking

```python
async def _generate_signals(self):
    """Per-cycle: 모든 코인 후보를 수집 → RL score로 ranking → top_k 진입."""
    candidates = []

    for coin in COINS:
        if self.pos_manager.has_position(coin):
            continue
        if coin not in self.models or coin not in self.featured_data:
            continue

        df = self.featured_data[coin]
        regime = detect_regime(df)
        if regime in self._get_blocked(coin):
            continue

        pred = predict_2stage(self.models[coin], df,
                              self.coin_cfgs[coin].get("stage1_threshold", 0.50))
        if pred.side == "HOLD":
            continue

        state = build_rl_state(df, pred, coin, self)
        action, rl_score = self.rl_gate.score(state)

        candidates.append({
            "coin": coin, "pred": pred, "state": state,
            "action": action, "rl_score": rl_score,
            "regime": regime,
        })

    # ── Ranking ──
    # REJECT 후보 제거
    accepted = [c for c in candidates if c["action"] != 0]

    # rl_score 내림차순 정렬
    accepted.sort(key=lambda c: c["rl_score"], reverse=True)

    # 동시 신규 진입 제한
    max_new = max(1, 3 - self.pos_manager.count())  # 총 오픈 3개 이하
    taken = accepted[:max_new]

    # 실행
    for c in taken:
        await self._execute_entry(c)

    # 모든 후보 (accept+reject) 로깅
    for c in candidates:
        self.signal_logger.log(c)
```

### 5.3 rl_score 정의

```python
def score(self, state: np.ndarray) -> tuple[int, float]:
    """Returns (action_idx, expected_utility)."""
    action = self.select_action(state)
    # score = UCB of selected action (exploration 제외, exploitation만)
    A_inv = self._A_inv[action]
    theta = A_inv @ self.b[action]
    rl_score = float(state @ theta)  # expected reward (no UCB bonus)
    return action, rl_score
```

---

## 6. LinUCB 실전 구현

```python
class LinUCB:
    """
    Linear Upper Confidence Bound — production-grade.

    개선사항 (v1 대비):
    - Sherman-Morrison inverse update (O(d²) vs O(d³))
    - Forgetting factor γ (비정상 시장 대응)
    - Feature clipping + intercept term
    - Action space: [REJECT, ACCEPT_0.75, ACCEPT_1.00, ACCEPT_1.25]
    """

    def __init__(
        self,
        state_dim: int = 30,      # 29 features + 1 intercept
        n_actions: int = 4,
        alpha: float = 1.0,       # exploration parameter
        gamma: float = 0.995,     # forgetting factor (최근 데이터 우선)
    ):
        self.d = state_dim
        self.K = n_actions
        self.alpha = alpha
        self.gamma = gamma

        # Per-action: A_inv (Sherman-Morrison), b
        self._A_inv = [np.eye(state_dim) for _ in range(n_actions)]
        self.b = [np.zeros(state_dim) for _ in range(n_actions)]
        self.n_updates = [0] * n_actions

    def select_action(self, state: np.ndarray) -> int:
        """UCB-based action selection."""
        best_action = 0
        best_ucb = -np.inf
        for a in range(self.K):
            theta = self._A_inv[a] @ self.b[a]
            exploit = float(state @ theta)
            explore = self.alpha * np.sqrt(float(state @ self._A_inv[a] @ state))
            ucb = exploit + explore
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = a
        return best_action

    def score(self, state: np.ndarray) -> tuple[int, float]:
        """Action + exploitation score (for ranking)."""
        action = self.select_action(state)
        theta = self._A_inv[action] @ self.b[action]
        return action, float(state @ theta)

    def update(self, state: np.ndarray, action: int, reward: float):
        """Sherman-Morrison rank-1 update with forgetting."""
        x = state.reshape(-1, 1)  # (d, 1)

        # Forgetting: decay old information
        if self.gamma < 1.0:
            self._A_inv[action] /= self.gamma
            self.b[action] *= self.gamma

        # Sherman-Morrison: (A + xx^T)^{-1} = A^{-1} - (A^{-1}xx^TA^{-1})/(1+x^TA^{-1}x)
        Ax = self._A_inv[action] @ x           # (d, 1)
        denom = 1.0 + float(x.T @ Ax)          # scalar
        self._A_inv[action] -= (Ax @ Ax.T) / denom

        self.b[action] += reward * state
        self.n_updates[action] += 1

    def get_theta(self, action: int) -> np.ndarray:
        """Interpretable: feature importance for this action."""
        return self._A_inv[action] @ self.b[action]

    def save(self, path: str):
        import joblib
        joblib.dump({
            "A_inv": self._A_inv, "b": self.b,
            "n_updates": self.n_updates,
            "d": self.d, "K": self.K,
            "alpha": self.alpha, "gamma": self.gamma,
        }, path, compress=3)

    def load(self, path: str):
        import joblib
        d = joblib.load(path)
        self._A_inv = d["A_inv"]; self.b = d["b"]
        self.n_updates = d["n_updates"]
        self.alpha = d["alpha"]; self.gamma = d["gamma"]
```

---

## 7. Fallback 정책 (min_accept_rate 대체)

```python
# RL이 지나치게 보수적이면 v4.2 baseline으로 fallback

class RLGate:
    def __init__(self, bandit: LinUCB, warmup: int = 200, lookback: int = 100):
        self.bandit = bandit
        self.warmup = warmup
        self.lookback = lookback
        self.recent_decisions = deque(maxlen=lookback)
        self.total_signals = 0

    def decide(self, state: np.ndarray) -> tuple[int, float]:
        self.total_signals += 1

        # Warmup: 데이터 부족 시 항상 ACCEPT_1.00
        if self.total_signals < self.warmup:
            return 2, 0.0  # ACCEPT_1.00

        # Accept rate 체크
        if len(self.recent_decisions) >= 50:
            recent_accepts = sum(1 for d in self.recent_decisions if d > 0)
            accept_rate = recent_accepts / len(self.recent_decisions)

            if accept_rate < 0.25:
                # RL 일시 비활성화 → v4.2 baseline (항상 ACCEPT_1.00)
                self.recent_decisions.append(2)
                return 2, 0.0  # fallback

        action, score = self.bandit.score(state)
        self.recent_decisions.append(action)
        return action, score
```

**v1과의 차이:**
- ~~강제 accept~~ → **v4.2 fallback** (RL 자체를 끔)
- RL이 복구되면 (accept rate > 25%) 자동으로 다시 RL 모드

---

## 8. Signal Logger (Phase 1 구현체)

```python
# src/rl/signal_logger.py

@dataclass
class SignalRecord:
    ts: str                    # UTC ISO
    coin: str
    side: str                  # BUY/SELL
    regime: str

    # State vector
    state: list[float]         # 30-dim

    # Prediction
    p_trade: float
    p_direction: float
    confidence: float

    # RL decision
    action: int                # 0=REJECT, 1=ACCEPT_0.75, 2=ACCEPT_1.00, 3=ACCEPT_1.25
    rl_score: float
    sizing_mult: float         # SIZING_MAP[action]

    # Barriers (for counterfactual)
    entry_price: float
    sl_price: float
    tp_price: float

    # Downstream
    risk_gate_passed: bool     # pre_trade_gate 통과 여부
    executed: bool             # 실제 주문 실행 여부

    # Result (나중에 채워짐)
    result_pnl_pct: float | None = None
    result_exit_reason: str | None = None
    result_bars_held: int | None = None
    result_reward: float | None = None
    counterfactual: bool = False  # reject된 시그널의 가상 결과 여부
```

### 8.1 Downstream rejection 처리

```python
# RL이 ACCEPT했지만 risk_gate에서 reject된 경우:
# → signal_log에는 기록하되, bandit 학습에서 제외

record.risk_gate_passed = False
record.executed = False
# offline_train에서: skip if not executed and action != REJECT
```

---

## 9. Phase별 실행 계획

### Phase 1: Signal Logger (즉시, v4.2 파이프라인 변경 최소)

```
생성:
  src/rl/__init__.py
  src/rl/state_builder.py      # build_rl_state() → 30-dim array
  src/rl/signal_logger.py      # SignalLogger.log() / update_result()

수정:
  run_live_bot_v2.py:
    _process_coin(): state 구성 + log 호출 (5줄)
    _close_position(): result 업데이트 (3줄)
    _generate_signals(): candidate 수집 구조로 변경 (ranking 준비)
```

**완료 조건:** v4.2 그대로 운용, signal_log.jsonl 축적 시작.

### Phase 2: Offline 학습 (200+ signals 후)

```
생성:
  src/rl/bandit.py             # LinUCB 구현
  src/rl/offline_train.py      # signal_log → 학습 → 저장

실행:
  python -m src.rl.offline_train \
    --input data/reports/live_trading_v2/signal_log.jsonl \
    --output data/models/rl/linucb_v1.joblib \
    --alpha 1.0 --gamma 0.995
```

**완료 조건:** 학습된 theta 벡터의 부호/크기가 직관적으로 해석 가능.
예: `theta[trade_margin] > 0` (margin 클수록 accept 선호).

### Phase 3: Shadow Mode (2주)

```
생성:
  src/rl/rl_gate.py            # RLGate.decide() — fallback 포함

수정:
  run_live_bot_v2.py:
    _process_coin(): rl_gate.decide() 호출 → 로그에만 기록
    실제 action은 항상 ACCEPT_1.00 (v4.2 동일)

관찰 지표:
  1. Conditional lift:
     mean(PnL | RL accept) - mean(PnL | all signals)
  2. Calibration by bucket:
     rl_score 상위/중간/하위 20% 별 실제 PnL
  3. Regime별 accept quality:
     TREND_UP/DOWN, RANGE_HIGH/LOW 별 성과
  4. Reject quality:
     RL REJECT 시그널의 counterfactual PnL < 0 비율
```

**Phase 3 합격 기준:**
- `lift > 0` (RL accept 시그널이 전체보다 나음)
- `reject counterfactual PnL < 0` 비율 > 55% (절반 이상 손실 회피)
- Calibration 단조 (score 높을수록 PnL 높음)

### Phase 4: Live Integration (Phase 3 합격 시)

```
수정:
  run_live_bot_v2.py:
    _generate_signals(): candidate ranking + top_k 실행
    rl_gate.decide() 결과를 실제 적용

config/frozen_params_v4_3.yaml:
  rl:
    enabled: true
    alpha: 1.0
    gamma: 0.995
    warmup_signals: 200
    max_new_per_cycle: 2       # 동시 신규 진입 제한
    model_path: "data/models/rl/linucb_v1.joblib"
```

---

## 10. v4.3 scope 외 (v4.4로 연기)

| 기능 | 이유 |
|------|------|
| Exit RL (trailing stop, 부분 청산) | State/reward 설계 별도 필요 |
| Multi-action sizing (6+ actions) | 500+ signals 필요, Phase 4 안정화 후 |
| Deep RL (DQN/PPO) | 데이터 10,000건+ 필요 |
| Cross-coin correlation sizing | Portfolio RL 별도 설계 필요 |

---

## 11. 파일 구조 (최종)

```
src/rl/
  __init__.py
  state_builder.py      # build_rl_state(df, pred, coin, bot) → np.array(30,)
  signal_logger.py      # SignalLogger: log(), update_result()
  bandit.py             # LinUCB: select_action(), update(), save(), load()
  rl_gate.py            # RLGate: decide() with warmup + fallback
  offline_train.py      # CLI 학습 스크립트
  counterfactual.py     # compute_counterfactual() — reject PnL 추정
```
