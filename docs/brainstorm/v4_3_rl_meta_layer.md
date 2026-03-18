# v4.3 Brainstorm: RL Meta-Strategy Layer

## 1. Current v4.2 Architecture (상세)

### 1.1 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                    2h CYCLE (run_live_bot_v2.py)                │
│                                                                 │
│  Step 1: Data                                                   │
│    Binance ccxt 4h OHLCV (500 bars) ──────────────────────┐     │
│    7 coins parallel (asyncio.gather)                       │     │
│         DOT, ADA, XRP, SOL, LINK + BTC, ETH               │     │
│                                                            ▼     │
│  Step 1b: Feature Engineering                                    │
│    ┌──────────────────────────────────────────────────┐          │
│    │ Raw OHLCV (5 cols)                               │          │
│    │  → add_technical_indicators()     +38 cols       │          │
│    │  → _add_decomposition(period=42)  +5 cols        │          │
│    │  → add_signal_features()          +79 cols       │          │
│    │  → add_microstructure_rollup()    +71 cols       │          │
│    │  → _add_cross_asset_correlation() +2 cols        │          │
│    │  → feature_policy filter          -1~10 cols     │          │
│    │  = ~221 cols per coin                            │          │
│    └──────────────────────────────────────────────────┘          │
│                            │                                     │
│  Step 2: Train (daily)     │                                     │
│    per coin:               │                                     │
│    ┌───────────────────────┴──────────────────────┐              │
│    │ create_labels_triple_barrier(k_u=3.0,k_l=0.6)│              │
│    │  → y_3c: DOWN=0, HOLD=1, UP=2               │              │
│    │                                              │              │
│    │ select_features() → ~216 numeric cols        │              │
│    │ mi_select(top 120) → feature_columns         │              │
│    │                                              │              │
│    │ Stage 1: y_s1 = (y_3c != HOLD)  Trade/NoTrade│              │
│    │ Stage 2: y_s2 = (y_3c == UP)    Long/Short   │              │
│    │          (Trade samples only)                 │              │
│    │                                              │              │
│    │ Build model combo per coin:                   │              │
│    │   DOT: ET(70%) + TabM(30%)                   │              │
│    │   ADA: ET(50%) + CatBoost(50%)               │              │
│    │   XRP: ET(50%) + CatBoost(50%)               │              │
│    │   SOL: ET(50%) + CatBoost(50%)               │              │
│    │   LINK: ET(50%) + XGBoost(50%)               │              │
│    │                                              │              │
│    │ CV: TimeSeriesSplit(3, gap=12) on S1          │              │
│    │ save_combo() → joblib artifact                │              │
│    └──────────────────────────────────────────────┘              │
│                            │                                     │
│  Step 3: TTL increment     │ bars_held++ for all open positions  │
│                            │                                     │
│  Step 4: Signal Generation │                                     │
│    per coin (if no open position):                               │
│    ┌───────────────────────┴──────────────────────┐              │
│    │ ① detect_regime(df)                          │              │
│    │   → TREND_UP/DOWN, RANGE_HIGH/LOW, UNKNOWN   │              │
│    │   blocked? → skip                            │              │
│    │                                 ◄── RL GATE A│              │
│    │ ② predict_2stage(combo, df, threshold)       │              │
│    │   → PredictionResult(side, p_trade,          │              │
│    │                      p_direction, confidence)│              │
│    │   HOLD? → skip                               │              │
│    │                                 ◄── RL GATE B│              │
│    │ ③ fetch_ticker() → entry_price, spread_bps   │              │
│    │ ④ compute_barriers(entry, atr, side)         │              │
│    │   → sl_price, tp_price  (R:R = k_u/k_l = 5) │              │
│    │ ⑤ fetch_funding_rate()                       │              │
│    │ ⑥ pre_trade_gate() → 9 gates                 │              │
│    │   rejected? → skip                           │              │
│    │                                 ◄── RL GATE C│              │
│    │ ⑦ apply_micro_sizing(qty, cvd, ofi)          │              │
│    │ ⑧ round_qty/price → place order              │              │
│    └──────────────────────────────────────────────┘              │
│                                                                  │
│  Step 5: Update equity                                           │
│                                                                  │
│  Background (30s polling):                                       │
│    SlTpMonitor._check_all_positions()                            │
│      → SL_HIT / TP_HIT / TIME_STOP → _close_position()         │
│                                                ◄── RL GATE D    │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Decision Point별 가용 데이터

```
GATE A (Regime 후, 예측 전):
  ├── df: 500 bars × 221 features
  ├── regime: str (5 classes)
  ├── coin_cfg: {stage1_threshold, model_combo, blocked_regimes, ...}
  ├── equity.current, dd_tracker.{daily_pnl, weekly_pnl, killed}
  ├── pos_manager.count(), pos_manager.all_positions()
  └── ledger: 전체 거래 이력 (SQLite)

GATE B (예측 후, 리스크 전):
  ├── GATE A의 모든 정보 +
  ├── pred.side: "BUY" / "SELL"
  ├── pred.p_trade: [0, 1]  (S1 확률)
  ├── pred.p_direction: [0, 1]  (S2 확률)
  └── pred.confidence: p_trade × p_direction

GATE C (리스크 승인 후, 주문 전):
  ├── GATE B의 모든 정보 +
  ├── entry_price, sl_price, tp_price
  ├── atr, spread_bps, funding_rate
  ├── check.sizing.{qty, notional, risk_usdt}
  └── micro sizing 전

GATE D (포지션 종료 시):
  ├── pos: {coin, side, entry_price, qty, sl, tp, bars_held}
  ├── current_price, unrealized_pnl_pct
  └── exit_reason: SL_HIT / TP_HIT / TIME_STOP
```

### 1.3 현재 한계

| 한계 | 현상 | 원인 |
|------|------|------|
| 맹목적 진입 | ML 시그널이면 무조건 진입 | 최근 성과 반영 없음 |
| 코인간 우선순위 없음 | 5코인 동시 시그널 시 전부 진입 | 포트폴리오 수준 의사결정 없음 |
| 고정된 threshold | 시장 상황과 무관하게 동일 기준 | adaptive threshold 없음 |
| exit 룰 고정 | SL/TP/TTL만 사용 | trailing stop, 부분 청산 없음 |

---

## 2. RL Meta-Layer 설계

### 2.1 설계 원칙

1. **ML 예측은 그대로 유지** -- RL은 "언제 믿을지"만 결정
2. **최소 침습** -- 기존 파이프라인 수정 최소화, gate만 삽입
3. **데이터 효율** -- 거래 빈도가 낮으므로 (일 5~15건) simple model 사용
4. **안전 우선** -- RL이 리스크를 증가시킬 수 없음 (sizing ≤ 1.5x)
5. **해석 가능** -- 왜 accept/reject했는지 설명 가능해야 함

### 2.2 선택: GATE B + GATE C 조합

```
GATE A (regime 후): 불필요 -- 이미 규칙 기반으로 충분
GATE B (예측 후):   RL이 accept/reject 결정     ← 핵심
GATE C (리스크 후): RL이 sizing multiplier 결정  ← 보조
GATE D (exit 시):   v4.4로 연기 (복잡도 높음)
```

### 2.3 State Vector (20차원)

```python
@dataclass
class RLState:
    # ── Signal quality (3) ──
    p_trade: float          # S1 확률 [0, 1]
    p_direction: float      # S2 확률 [0, 1]
    confidence: float       # p_trade × p_direction [0, 1]

    # ── Market regime (4) ──
    regime_trend: float     # 1.0 if TREND_UP/DOWN, else 0.0
    regime_up: float        # 1.0 if TREND_UP, else 0.0
    atr_pct: float          # atr_14 / close (변동성 수준)
    hurst: float            # Hurst exponent (추세 지속성)

    # ── Microstructure (3) ──
    cvd_ratio: float        # cvd_ratio_6 (단기 매수/매도 압력)
    ofi_norm: float         # ofi_sum_3 normalized
    ms_composite: float     # ms_composite [-1, 1]

    # ── Portfolio state (4) ──
    open_positions: float   # 현재 오픈 수 / 5 (정규화)
    daily_pnl_pct: float    # 당일 누적 PnL / equity
    weekly_pnl_pct: float   # 주간 누적 PnL / equity
    dd_ratio: float         # current DD / daily_limit (0~1, 1이면 kill switch 직전)

    # ── Coin history (4) ──
    coin_win_rate_5: float  # 해당 코인 최근 5건 승률
    coin_avg_pnl_5: float   # 해당 코인 최근 5건 평균 PnL%
    coin_streak: float      # 연속 승(+)/패(-) 수, ±5 clamp 후 /5
    bars_since_last: float  # 마지막 거래 후 경과 bars / max_horizon

    # ── Cross-market (2) ──
    btc_return_24h: float   # BTC 24h return (시장 방향)
    corr_btc: float         # 해당 코인-BTC 상관계수
```

### 2.4 Action Space

**GATE B: Accept/Reject (binary)**
```python
gate_b_action: int  # 0 = REJECT, 1 = ACCEPT
```

**GATE C: Sizing Multiplier (연속, 이산화)**
```python
gate_c_action: int  # index into [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]
# 0.0 = reject (override), 1.0 = 그대로, 1.5 = 1.5배 확대
```

**최종 action = GATE B × GATE C:**
```
B=REJECT → qty = 0 (진입 안 함)
B=ACCEPT, C=0.75 → qty = base_qty × 0.75
B=ACCEPT, C=1.5  → qty = base_qty × 1.5 (max)
```

### 2.5 Reward Design

```python
def compute_reward(trade_result: dict) -> float:
    """
    trade_result = {
        "pnl_pct": float,      # 수수료 후 수익률
        "exit_reason": str,     # SL_HIT, TP_HIT, TIME_STOP
        "bars_held": int,
        "sizing_mult": float,   # RL이 적용한 배수
    }
    """
    pnl = trade_result["pnl_pct"]

    # 기본 보상: 실현 PnL
    reward = pnl * 100  # scale to [-10, +30] range roughly

    # TP 히트 보너스 (빨리 맞추면 추가 보상)
    if trade_result["exit_reason"] == "TP_HIT":
        speed_bonus = max(0, 1.0 - trade_result["bars_held"] / 18) * 0.5
        reward += speed_bonus

    # SL 히트 시 연속 패배 페널티
    # (이미 pnl이 음수이므로 추가 페널티는 최소화)

    # 과도한 sizing에 대한 리스크 페널티
    if trade_result["sizing_mult"] > 1.0:
        risk_penalty = (trade_result["sizing_mult"] - 1.0) * abs(pnl) * 0.5
        reward -= risk_penalty

    return reward


def compute_reject_reward(signal_state: dict) -> float:
    """
    REJECT한 경우: counterfactual reward.
    시그널 발생 후 실제 가격 변동으로 "진입했으면 어땠을지" 추정.
    """
    # 시그널 발생 시점부터 max_horizon까지의 가격 변동 관찰
    # Triple Barrier 시뮬레이션으로 가상 PnL 계산
    hypothetical_pnl = signal_state["counterfactual_pnl"]

    if hypothetical_pnl > 0:
        # 수익 기회를 놓쳤으면 음의 보상 (기회비용)
        return -hypothetical_pnl * 50  # 놓친 것에 대한 페널티
    else:
        # 손실을 피했으면 양의 보상
        return abs(hypothetical_pnl) * 50  # 피한 것에 대한 보상
```

### 2.6 알고리즘: LinUCB Contextual Bandit

```python
class LinUCB:
    """
    Linear Upper Confidence Bound for Contextual Bandits.

    각 action a에 대해:
      p_a = x^T θ_a + α √(x^T A_a^{-1} x)

    장점:
    - 샘플 효율 높음 (수백 건으로 학습 가능)
    - 해석 가능 (θ 계수 = feature importance)
    - Exploration/exploitation 자동 밸런싱 (α)
    - 비정상 환경에서도 작동 (UCB bonus)

    state_dim: 20 (RLState)
    n_actions_b: 2 (ACCEPT/REJECT)
    n_actions_c: 6 ([0.0, 0.5, 0.75, 1.0, 1.25, 1.5])
    α: exploration parameter (0.5 ~ 2.0)
    """

    def __init__(self, state_dim: int, n_actions: int, alpha: float = 1.0):
        self.d = state_dim
        self.K = n_actions
        self.alpha = alpha
        # Per-action parameters
        self.A = [np.eye(state_dim) for _ in range(n_actions)]  # d×d
        self.b = [np.zeros(state_dim) for _ in range(n_actions)]  # d×1
        self.theta = [np.zeros(state_dim) for _ in range(n_actions)]

    def select_action(self, state: np.ndarray) -> int:
        """UCB-based action selection."""
        scores = []
        for a in range(self.K):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            ucb = state @ theta + self.alpha * np.sqrt(state @ A_inv @ state)
            scores.append(ucb)
        return int(np.argmax(scores))

    def update(self, state: np.ndarray, action: int, reward: float):
        """Update parameters after observing reward."""
        self.A[action] += np.outer(state, state)
        self.b[action] += reward * state
```

### 2.7 Counterfactual 추정 (REJECT된 시그널)

```
시그널 발생 시점 t에서 REJECT된 경우:
  1. (t, entry_price, side, sl_price, tp_price)를 기록
  2. t+1 ~ t+max_horizon 동안의 가격을 관찰
  3. Triple Barrier 로직으로 가상 exit 결정
  4. 가상 PnL 계산 → reject_reward에 사용

저장 구조:
  signal_log.jsonl:
    {"ts", "coin", "side", "p_trade", "p_direction", "confidence",
     "regime", "state_vector": [...], "action": "ACCEPT"/"REJECT",
     "sizing_mult": float,
     "entry_price", "sl_price", "tp_price",
     "result": null}  # 나중에 채워짐

  결과 업데이트 (18 bars = 72h 후):
    {"ts", ..., "result": {
      "exit_reason": "TP_HIT",
      "exit_price": 5.25,
      "pnl_pct": 0.034,
      "bars_held": 8,
      "reward": 3.9
    }}
```

---

## 3. 구현 계획

### Phase 1: Signal Logger (1일)

**목적:** 모든 시그널 발생 + state vector + 결과를 기록하여 RL 학습 데이터 축적.

```
src/rl/
  state_builder.py    # RLState 구성: df + ledger + pos_manager → 20-dim vector
  signal_logger.py    # 시그널 발생/결과 JSONL 기록 + counterfactual 추적
```

**수정 범위:**
- `run_live_bot_v2.py`의 `_process_coin()`에 logger 호출 추가 (3줄)
- `_close_position()`에 result 업데이트 호출 추가 (2줄)

**Phase 1 완료 조건:** v4.2 그대로 운용하면서 signal_log.jsonl에 데이터 축적.

### Phase 2: Offline 학습 (Phase 1 후 2~4주)

**목적:** 축적된 데이터로 LinUCB 학습.

```
src/rl/
  bandit.py           # LinUCB 학습/추론/직렬화
  offline_train.py    # signal_log.jsonl → LinUCB 학습 스크립트
```

**필요 데이터량:** 최소 200 시그널 (accept+reject 합산).
- 5코인 × 12 cycles/day × 30% signal rate ≈ 18 signals/day
- 200건 / 18 = ~11일

**학습 절차:**
1. signal_log.jsonl 로드
2. 결과가 확정된 건만 필터 (counterfactual 포함)
3. state → action → reward 트리플렛 구성
4. LinUCB.update() 반복
5. 학습된 A, b 매트릭스 저장

### Phase 3: Shadow Mode (2주)

**목적:** RL gate 추천값 기록하되 실제 거래에는 반영하지 않음.

```
src/rl/
  rl_gate.py          # run_live_bot에 삽입할 gate 인터페이스
```

**수정 범위:**
- `_process_coin()`에서 RL gate 호출 → 로그에만 기록
- 실제 action은 항상 ACCEPT (v4.2 동일)
- 비교: "RL이 REJECT했는데 실제 결과 어땠나?"

**Phase 3 완료 조건:** RL gate 추가 시 net EV 개선 확인.

### Phase 4: Live Integration (Phase 3 검증 후)

**수정 범위:**
- `_process_coin()`에서 RL gate 결과를 실제 적용
- sizing_multiplier를 `apply_micro_sizing()` 후에 곱셈
- config에 `rl.enabled: true/false` 토글

```python
# _process_coin() 수정 (Phase 4):
pred = predict_2stage(combo, df, s1_thresh)
if pred.side == "HOLD":
    return

# ── RL Gate ──
state = build_rl_state(df, pred, coin, self)
gate_b = self.rl_gate.decide_accept(state)       # ACCEPT/REJECT
gate_c = self.rl_gate.decide_sizing(state)        # [0.0 ~ 1.5]
self.signal_logger.log(coin, pred, state, gate_b, gate_c)

if not gate_b:
    logger.info(f"[RL] {coin}: REJECTED (state={state.summary()})")
    return

# ... risk gate, sizing ...
qty = qty * gate_c  # RL sizing adjustment
```

---

## 4. 파일 구조 (최종)

```
src/rl/
  __init__.py
  state_builder.py      # build_rl_state(df, pred, coin, bot) → RLState → np.array
  signal_logger.py      # SignalLogger: log(), update_result(), counterfactual_check()
  bandit.py             # LinUCB: select_action(), update(), save(), load()
  rl_gate.py            # RLGate: decide_accept(), decide_sizing() — bot에 삽입
  offline_train.py      # CLI: python -m src.rl.offline_train --input signal_log.jsonl

config/frozen_params_v4_3.yaml:
  rl:
    enabled: false           # Phase 1-3: false, Phase 4: true
    alpha: 1.0               # UCB exploration parameter
    sizing_levels: [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]
    min_accept_rate: 0.30    # 안전장치: 최소 30% accept
    max_sizing: 1.5          # sizing multiplier cap
    warmup_signals: 200      # 이 수 이하면 항상 accept
    model_path: "data/models/rl/linucb_v1.joblib"
    signal_log: "data/reports/live_trading_v2/signal_log.jsonl"
```

---

## 5. 리스크 관리

| 리스크 | 대응 |
|--------|------|
| RL이 전부 REJECT | `min_accept_rate: 0.30` 강제 (최근 100건 기준) |
| 과적합 | LinUCB는 선형 → 과적합 위험 낮음 + rolling window |
| Regime shift | 최근 50 signals만 사용 (오래된 데이터 감쇠) |
| Sizing 과도 확대 | cap 1.5x, risk_engine은 RL 뒤에서 재검증 |
| Counterfactual 오차 | 보수적 추정 (슬리피지/비용 반영) |
| 데이터 부족 초기 | `warmup_signals: 200` 이전은 항상 ACCEPT |

---

## 6. 성공 지표

Phase 3 (Shadow Mode) 종료 시 측정:

| 지표 | v4.2 baseline | v4.3 target |
|------|:---:|:---:|
| Signal accept rate | 100% | 50~70% (선별) |
| Accepted 평균 PnL | X% | > X+0.1% |
| Rejected 평균 PnL | N/A | < 0% (피한 게 맞았나) |
| Portfolio Sharpe | S | > S (동일 기간) |
| Max DD | D% | ≤ D% |
| 거래 수 | N건 | ≥ 0.3N건 (min_accept_rate) |
