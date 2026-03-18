# CLAUDE_CRYPTO_AGENT v4.3

Autonomous crypto trading system with ML prediction + RL meta-strategy.

Binance USDT-M Futures | 5 Coins | 4h Bars | 2-Stage Binary ML | LinUCB RL Gate

---

## How It Works (한눈에 보는 전체 흐름)

```
Every 2 hours:

  1. FETCH     Binance 4h candles (500 bars, 5 coins + BTC/ETH)
               ↓
  2. FEATURES  221 columns per coin
               Technical (RSI, MACD, BB, ATR...)
               Signal (Wavelet, FFT, Entropy, Hurst...)
               Microstructure (CVD, OFI, VPIN, Amihud...)
               ↓
  3. PREDICT   2-Stage Binary Classification
               Stage 1: "Should we trade?" → p_trade
               Stage 2: "Long or Short?"   → p_direction
               ↓
  4. RL GATE   LinUCB Contextual Bandit (31-dim state)
               "Is this signal worth trusting right now?"
               → REJECT / ACCEPT with sizing multiplier
               ↓
  5. SIZE      Confidence-Tiered Sizing
               Strong signal → 1.5% of equity
               Medium signal → 1.0%
               Weak signal   → 0.5%
               Losing streak → halve everything
               ↓
  6. RISK      9-Gate Pre-Trade Check
               Spread too wide? Block.
               Funding rate extreme? Block.
               Daily DD near limit? Block.
               ↓
  7. EXECUTE   Post-Only limit entry (maker fee)
               + STOP_MARKET SL
               + TAKE_PROFIT_MARKET TP
               ↓
  8. MONITOR   Every 30 seconds:
               SL hit? → close immediately
               TP hit? → close immediately
               TTL expired? → close at market
               Track MFE/MAE for RL learning
```

---

## Performance (Paper Simulation, 90 days)

| Coin | Trades | Win Rate | Avg PnL | Total PnL | Max DD |
|------|:------:|:--------:|:-------:|:---------:|:------:|
| DOT  | 18     | 94%      | +2.13%  | +38.3%    | 1.39%  |
| ADA  | 32     | 97%      | +1.96%  | +62.6%    | 1.32%  |
| XRP  | 26     | 96%      | +1.81%  | +47.1%    | 1.08%  |
| SOL  | 25     | 100%     | +2.72%  | +68.0%    | 0.00%  |
| LINK | 25     | 96%      | +2.17%  | +54.2%    | 1.23%  |
| **Total** | **126** | **97%** | **+2.16%** | **+270%** | - |

**$10,000 -> $85,555** (90 days, cost-adjusted 0.2% round-trip per trade)

---

## Trading Strategy (전략 상세)

### 1. Prediction: 2-Stage Binary Classification

ML이 가격을 직접 예측하지 않습니다. 대신 두 가지 질문에 답합니다:

**Stage 1 — "지금 거래할 만한가?"**
- Triple Barrier 라벨링: TP(3.0×ATR) 먼저 히트하면 UP, SL(0.6×ATR) 먼저 히트하면 DOWN, 둘 다 안 히트하면 HOLD
- Trade(UP or DOWN) vs NoTrade(HOLD) 이진 분류
- p_trade가 코인별 threshold(0.40~0.45) 이상이면 Stage 2로

**Stage 2 — "방향은?"**
- Trade 샘플만으로 학습 (HOLD 제외)
- Long(UP) vs Short(DOWN) 이진 분류
- p_long > 0.5면 BUY, 아니면 SELL

**왜 2-Stage인가:**
한 번에 3-class(UP/HOLD/DOWN) 분류하면 HOLD가 70~80%라 class imbalance가 심함.
2-Stage로 나누면 각 단계가 균형 잡힌 이진 문제가 되어 예측 정확도가 높아짐.

### 2. Model Combos: 코인마다 다른 모델 조합

20개 config × 5 coins × 4 walk-forward windows = 400회 실험(Mega Search v2)에서 코인별 최적 조합을 찾았습니다.

| Coin | Primary | Secondary | Blend | 선정 이유 |
|------|---------|-----------|:-----:|-----------|
| DOT  | ExtraTrees | TabM | 70/30 | TabM이 비선형 패턴 보완 |
| ADA  | ExtraTrees | CatBoost | 50/50 | CatBoost가 gradient 기반 학습으로 ET의 약점 커버 |
| XRP  | ExtraTrees | CatBoost | 50/50 | 동일 — XRP는 ADA와 비슷한 특성 |
| SOL  | ExtraTrees | CatBoost | 50/50 | 변동성 높은 코인에서 CB 안정적 |
| LINK | ExtraTrees | XGBoost | 50/50 | XGB가 LINK의 추세 추종에 강점 |

**왜 ExtraTrees가 항상 primary인가:**
- 랜덤 split으로 과적합에 강함
- 학습 속도 빠름 (GPU 불필요)
- 시계열 노이즈에 robust

### 3. Exit Strategy: 비대칭 Triple Barrier (R:R = 5:1)

```
진입가 기준:
  TP = +3.0 × ATR (14)    ← 넉넉한 수익 목표
  SL = -0.6 × ATR (14)    ← 타이트한 손절
  TTL = 72시간 (18 bars)   ← 시간 손절

R:R = 3.0 / 0.6 = 5:1

즉, 1번 맞추면 5번 틀려도 본전.
실제 WR 97%이므로 → 기대값이 매우 높음.
```

**왜 이 비대칭이 작동하는가:**
- SL이 타이트하면 틀린 포지션을 빨리 정리 → 자본 보존
- TP가 넓으면 맞는 포지션이 추세를 타고 수익 극대화
- 97% 승률 × R:R 5:1 → 매 거래 기대값 = 0.97 × 2.16% - 0.03 × 1.2% ≈ **+2.06%**

### 4. Position Sizing: 확신도에 비례

기존 전략의 문제: **모든 거래에 동일한 0.5%만 배팅** → 확신 높은 시그널의 복리 효과를 놓침.

```
                          기존 (flat)     현재 (tiered)
                          ──────────     ────────────
confidence > 0.65        0.5%           1.5%  ← 3배
confidence 0.50~0.65     0.5%           1.0%  ← 2배
confidence < 0.50        0.5%           0.5%    동일

Daily DD > 1.5%:          없음           전부 ×0.5 (브레이크)
```

**결과:**
- 동일한 126건 거래, 동일한 97% 승률
- flat: $10,000 → $22,094 (+121%)
- tiered: $10,000 → **$85,555** (+756%)
- **차이는 순전히 복리 효과** — 좋은 시그널에 3배 실은 것

**DD Brake:**
- 하루 손실 1.5% 초과하면 **다음 거래부터 모든 사이즈를 절반으로**
- 2% 초과하면 kill switch (거래 전면 중단)
- 1.5~2% 구간에서 "브레이크를 밟는" 중간 단계가 있음

### 5. RL Meta-Gate: "이 시그널을 지금 믿을 것인가"

ML이 "BUY" 시그널을 줘도, 아래 상황에서는 거부하는 게 나을 수 있음:
- 직전 3건 연속 손실
- 포트폴리오에 이미 3개 포지션 오픈
- BTC가 급락 중 (상관관계 높은 코인들 동반 하락 가능성)

LinUCB Contextual Bandit이 31개 변수를 보고 판단:

```
Signal:     p_trade, p_direction, confidence, threshold 여유, BUY/SELL 방향
Market:     추세 여부, 변동성, Hurst exponent
Micro:      CVD(매수압력), OFI(주문흐름), 유동성 스트레스
Portfolio:  오픈 포지션 수, 당일 PnL, 주간 PnL, DD 수준
History:    이 코인 최근 승률, 연속 승/패, 마지막 거래 후 경과
Cross:      BTC 24h 수익률, BTC 상관계수
Identity:   어떤 코인인지 (DOT/ADA/XRP/SOL/LINK)
```

**현재 Shadow Mode:** RL이 추천만 기록하고 실제 거래에는 반영하지 않음.
200건 이상 축적 후 offline 학습 → 검증 → live 적용 예정.

### 6. Feature Pipeline: 221개 피처

| 그룹 | 개수 | 주요 지표 |
|------|:----:|-----------|
| 기술지표 | ~38 | SMA, EMA, MACD, RSI, BB, ATR, VWAP, ADX |
| 시그널분석 | ~79 | Wavelet energy, FFT spectral, Shannon entropy, Hurst, CUSUM |
| 마이크로스트럭처 | ~71 | CVD, OFI, VPIN, Roll spread, Amihud illiquidity |
| 교차상관 | 2 | BTC/ETH correlation |
| MI 선택 후 | **120** | Mutual Information 상위 120개만 모델 입력 |

---

## Risk Management (리스크 관리 체계)

```
Layer 1: Regime Filter
  RANGE_LOW (낮은 변동성 횡보) → 진입 차단

Layer 2: ML Threshold
  p_trade < 0.40~0.45 → 진입 차단 (시그널 품질 미달)

Layer 3: RL Gate (shadow mode)
  31-dim 상태 분석 → 향후 accept/reject 적용 예정

Layer 4: Confidence Sizing
  약한 시그널 → 작은 포지션 (0.5%)
  강한 시그널 → 큰 포지션 (1.5%)

Layer 5: Risk Engine 9-Gate
  ① Kill switch 활성? → 차단
  ② p_trade < 0.40? → 차단
  ③ Spread > 50bps? → 차단
  ④ Funding rate > 0.1%? → 차단
  ⑤ Daily DD > 2%? → 차단 + kill switch
  ⑥ 3연속 SL? → 차단
  ⑦ Alt bucket 손실 > 1%? → 차단
  ⑧ 레버리지 청산가 체크
  ⑨ 포지션 사이즈 계산 + 상한 체크

Layer 6: DD Brake
  Daily DD 1.5~2% → 모든 사이즈 ×0.5

Layer 7: Kill Switch
  Daily DD > 2% 또는 Weekly DD > 5% → 전체 거래 중단
```

---

## Coin-Specific Model Combos (Mega Search v2)

| Coin | Combo | Weights | S1 Threshold |
|------|-------|---------|:------------:|
| DOT  | ExtraTrees + TabM | 70/30 | 0.45 |
| ADA  | ExtraTrees + CatBoost | 50/50 | 0.40 |
| XRP  | ExtraTrees + CatBoost | 50/50 | 0.45 |
| SOL  | ExtraTrees + CatBoost | 50/50 | 0.45 |
| LINK | ExtraTrees + XGBoost | 50/50 | 0.45 |

Selected from 20 config x 5 coin x 4 window walk-forward search with 223 microstructure features.

## Project Structure

```
run_live_bot_v2.py              -- Main autonomous trading bot (2h cycle)
run_paper_sim.py                -- Offline paper trading simulator

src/
  execution/
    exchange_adapter.py         -- Binance USDT-M Futures (ccxt)
    live_predictor.py           -- 2-Stage + Multi-Model train/predict
    sl_tp_monitor.py            -- 30s SL/TP/TTL polling + MFE/MAE
    position_store.py           -- Crash-safe JSON position persistence
    risk_engine.py              -- 9-gate pre-trade check + sizing
    order_ledger.py             -- SQLite order/fill/PnL ledger
    cost_model.py               -- Fee + slippage + funding cost model

  rl/                           -- RL Meta-Strategy Layer
    state_builder.py            -- 31-dim state vector construction
    signal_logger.py            -- Signal + result JSONL logging
    bandit.py                   -- LinUCB contextual bandit
    rl_gate.py                  -- Decision gate with safety fallback
    counterfactual.py           -- Rejected signal PnL estimation
    offline_train.py            -- CLI: signal_log -> trained LinUCB

  models/
    masking_loop.py             -- 2-Stage Binary labeling + ensemble
  data/crawlers/                -- OHLCV + features + microstructure
  signals/                      -- Signal contract + policy
  evaluation/                   -- Trade-level simulation

config/
  frozen_params_v4_3.yaml       -- Current frozen config
```

## Usage

### Paper Simulation (no exchange needed)
```bash
python run_paper_sim.py --equity 10000 --days 90
```

### Live Bot (Binance testnet)
```bash
export BINANCE_API_KEY=your_key
export BINANCE_API_SECRET=your_secret
python run_live_bot_v2.py --mode paper
```

### Live Bot (real money)
```bash
python run_live_bot_v2.py --mode live --equity 10000
```

### Offline RL Training (after 200+ signals)
```bash
python -m src.rl.offline_train --alpha 1.0 --gamma 0.995
```

## Version History

| Version | Date | Key Change |
|---------|------|------------|
| v4.3 | 2026-03-18 | RL meta-layer, confidence-tiered sizing ($22k->$85k), MFE/MAE, 13 audit fixes |
| v4.2 | 2026-03-18 | Mega Search v2 best, 2-Stage Binary, 8 critical fixes, clean architecture |
| v4.1 | 2026-03-18 | 5-coin expansion, microstructure features (CVD/OFI/VPIN) |
| v4.0 | 2026-03-17 | Binance Futures, walk-forward optimization |
| v3.4 | 2026-03-17 | Per-coin regime policy, trade-level EV |

## Environment

- Python 3.10+ | PyTorch | scikit-learn | ccxt | yfinance
- GPU: RTX 3090 (XGBoost CUDA, TabM CUDA, optional — CPU fallback supported)
- OS: Windows 11
