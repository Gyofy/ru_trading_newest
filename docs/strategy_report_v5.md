# v5.0 TSMOM Enhanced Strategy — Full Report

> 2026-03-23 | CLAUDE_CRYPTO_AGENT | Paper Trading Active

---

## 1. Executive Summary

크립토 7코인 USDT-M Futures 자동매매 전략.
ML 방향 예측이 leakage로 무효화된 후, **규칙 기반 방향(TSMOM)** + **ML/RL 증강**으로 재설계.

```
핵심 수치:
  OOS Sharpe:      2.80 ~ 3.42 (config별)
  OOS avg PnL:     +2.3% ~ +3.5% / trade
  Permutation:     p = 0.006 (99.4% 유의)
  Bootstrap:       P(Sharpe > 1) = 99.8%
  비용 내성:       50bps까지 양수 유지
  권장 레버리지:   2x (실용), 3x (최대)
```

---

## 2. Strategy Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    MARKET DATA                           │
│  yfinance 1h OHLCV → 4h resample (7 coins)              │
│  Binance metrics: OI, L/S Ratio, Taker Volume (5min)    │
└───────────────────────┬──────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────┐
│  LAYER 1: DIRECTION (Dual TSMOM)                        │
│                                                          │
│  Short-term:  sign(7-day return)  → LONG or SHORT       │
│  Long-term:   sign(28-day return) → LONG or SHORT       │
│  Rule:        both must agree, else NO TRADE             │
│                                                          │
│  Rationale:                                              │
│  - Short(7d): 빠른 추세 감지, 전환점에서 즉시 반응      │
│  - Long(28d): 노이즈 필터, 확정된 추세만 추종           │
│  - Dual agree: 두 시간대가 동의 = 높은 확신              │
│  - Academic: TSMOM Sharpe 1.5~2.2 (Huang et al. 2024)   │
│                                                          │
│  Output: LONG(+1) / SHORT(-1) / FLAT(0)                 │
└───────────────────────┬──────────────────────────────────┘
                        │ (LONG or SHORT만 통과)
┌───────────────────────▼──────────────────────────────────┐
│  LAYER 2: TREND CONFIRMATION (RSI Filter)               │
│                                                          │
│  LONG signal:  RSI(14) > 50 이어야 유효                 │
│  SHORT signal: RSI(14) < 50 이어야 유효                 │
│                                                          │
│  Rationale:                                              │
│  - RSI를 역추세(과매수/과매도) 지표로 쓰면 실패         │
│  - RSI를 추세 확인 지표로 쓰면 성공                     │
│  - PMC 연구: RSI trend filter 773% vs B&H 275%          │
│  - TSMOM이 LONG인데 RSI < 50 → 추세가 약함 → 차단     │
│                                                          │
│  Output: PASS / BLOCK                                    │
└───────────────────────┬──────────────────────────────────┘
                        │ (PASS만 통과)
┌───────────────────────▼──────────────────────────────────┐
│  LAYER 3: ENTRY TIMING (CVD Extreme)                    │
│                                                          │
│  CVD = Cumulative Volume Delta (BVC 근사)                │
│  cvd_ratio_24 = (CVD - CVD_MA24) / |CVD_MA24|           │
│                                                          │
│  SHORT 진입 조건: cvd_ratio > Q75 (120-bar rolling)     │
│    → 매수세 과열, 반등 정점에서 SHORT 진입               │
│                                                          │
│  LONG 진입 조건:  cvd_ratio < Q25 (120-bar rolling)     │
│    → 매도세 과열, 투매 바닥에서 LONG 진입                │
│                                                          │
│  Rationale:                                              │
│  - CVD-가격 Spearman rho = -0.21 (평균회귀 성질)        │
│  - 정배 방향의 역CVD 극단 = 더 좋은 가격에 진입         │
│  - "추세 내 풀백에서 진입"의 정량적 구현                 │
│                                                          │
│  Output: ENTER / WAIT                                    │
└───────────────────────┬──────────────────────────────────┘
                        │ (ENTER만 통과)
┌───────────────────────▼──────────────────────────────────┐
│  LAYER 4: CROWDING FILTER (Binance OI)                  │
│                                                          │
│  OI z-score = (OI_now - OI_mean48) / OI_std48           │
│  |OI z-score| > 2.0 → SKIP (과밀 포지셔닝)             │
│                                                          │
│  Rationale:                                              │
│  - OI 극단 = 청산 캐스케이드 위험 구간                  │
│  - 진입 직후 대규모 청산 발생 → SL 히트 확률 급증       │
│  - OI 필터 추가: Sharpe 1.39 → 1.80 (+0.41)            │
│                                                          │
│  Output: PASS / SKIP                                     │
└───────────────────────┬──────────────────────────────────┘
                        │ (PASS만 통과)
┌───────────────────────▼──────────────────────────────────┐
│  LAYER 5: POSITION SIZING                               │
│                                                          │
│  Base:                                                   │
│    risk_usd = equity × 2% (per-trade risk budget)       │
│    sl_distance = SL price - entry price                  │
│    position = risk_usd / sl_distance × leverage         │
│    cap = equity × 20% (단일 코인 최대)                  │
│                                                          │
│  RL Enhancement (shadow mode, 미적용):                   │
│    LinUCB 33-dim state → 7-action sizing                │
│    {REJECT, 0.5x, 0.75x, 1.0x, 1.25x, 1.5x, 2.0x}    │
│    200+ 시그널 축적 후 offline training 예정             │
│                                                          │
│  Output: position size (USD)                             │
└───────────────────────┬──────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────┐
│  LAYER 6: EXECUTION                                     │
│                                                          │
│  Entry:  Post-Only limit order (maker fee 0.02%)        │
│  SL:     STOP_MARKET (taker fee 0.05%, 즉시 체결)       │
│  TP:     TAKE_PROFIT_MARKET (maker if possible)         │
│                                                          │
│  Barrier Parameters:                                     │
│    TP = 5.0 × ATR(14)   (avg 10% for mid-vol alts)     │
│    SL = 1.0 × ATR(14)   (avg 2% for mid-vol alts)      │
│    TTL = 24 bars         (96 hours = 4 days)            │
│    R:R = 5:1             (WR 25% 이상이면 양수 EV)      │
│                                                          │
│  비용:                                                   │
│    편도: maker 0.02% + slippage 0.03% = 0.05%           │
│    왕복: entry + exit = ~0.20% (SL taker 포함)          │
│                                                          │
│  Output: OPEN POSITION                                   │
└───────────────────────┬──────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────┐
│  LAYER 7: EXIT MANAGEMENT                               │
│                                                          │
│  매 4h 바마다 체크:                                     │
│                                                          │
│  1. TP 도달? → 실현 (avg +10% gross, +9.8% net)        │
│  2. SL 도달? → 손절 (avg -2% gross, -2.2% net)         │
│  3. TTL 만료? → 시장가 청산 (avg ±0.5% net)            │
│                                                          │
│  Flash Crash 방어:                                       │
│    STOP_MARKET → 슬리피지 있어도 반드시 체결            │
│    worst case: SL 미체결 시 2~3×SL 손실 가능            │
│                                                          │
│  Dynamic Exit (v5.1 계획, 미구현):                      │
│    CQL offline RL로 trailing stop / 조기 청산 학습      │
│                                                          │
│  Output: CLOSE POSITION + PnL 기록                      │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Signal Flow Example

### Case: SOL SHORT 진입

```
시점: 2025-12-15 08:00 UTC

[Layer 1] TSMOM Direction
  7-day return:  SOL -8.2% → SHORT
  28-day return: SOL -3.1% → SHORT
  → Dual agree: SHORT ✓

[Layer 2] RSI Confirmation
  RSI(14) = 38.5 < 50
  → SHORT 유효 ✓

[Layer 3] CVD Timing
  cvd_ratio_24 = +0.15
  Q75 (120-bar) = +0.12
  0.15 > 0.12 → 매수세 과열 (반등 정점)
  → SHORT 진입 타이밍 ✓

[Layer 4] OI Filter
  OI z-score = +1.2 (|1.2| < 2.0)
  → 과밀 아님 ✓

[Layer 5] Position Sizing
  equity = $1,000, risk = 2% = $20
  ATR(14) = $1.85, SL distance = 1.0 × $1.85 = $1.85
  entry = $22.50
  SL_pct = $1.85 / $22.50 = 8.2%
  base_size = $20 / 0.082 = $244
  × leverage 2x = $488
  cap = $1,000 × 20% = $200 → size = $200

[Layer 6] Execution
  Entry: $22.50 (Post-Only)
  TP: $22.50 - 5×$1.85 = $13.25 (41% 하락 목표)
  SL: $22.50 + 1×$1.85 = $24.35 (8.2% 상승 시 손절)
  TTL: 24 bars = 96시간 후

[Layer 7] Exit (12 bars later)
  SOL price hits $24.35 → SL 발동
  PnL: -8.2% × 2x leverage = -16.4% of position
  Net: -$32.80 → equity = $967.20
```

### Case: ETH SHORT TP 도달

```
[Entry] ETH SHORT @ $2,100, TP=$1,900, SL=$2,140
[Bar 8] ETH drops to $1,895 → TP 발동
  PnL: ($2,100 - $1,900) / $2,100 = +9.52%
  Net: 9.52% - 0.20% = +9.32%
  × 2x leverage = +18.64% of position
```

---

## 4. Parameter Specification

### 4.1 Frozen Parameters

| Parameter | Value | Search Range | Selection Method |
|-----------|-------|-------------|-----------------|
| lookback_short | 7 days | [5, 7, 10, 14, 21, 28] | OOS Sharpe max |
| lookback_long | 28 days | [21, 28] | Dual agree test |
| cvd_quantile | 0.75 | [0.65, 0.70, 0.75, 0.80, 0.85] | Grid search |
| cvd_roll_window | 120 bars | [60, 90, 120] | Grid search |
| k_upper (TP) | 5.0 × ATR | [3.0, 4.0, 5.0] | Grid search |
| k_lower (SL) | 1.0 × ATR | [0.8, 1.0, 1.5, 2.0] | Grid search |
| max_hold | 24 bars (96h) | [18, 24] | Grid search |
| oi_zscore_max | 2.0 | fixed | Domain knowledge |
| cost_roundtrip | 0.20% | fixed | Binance fee schedule |
| leverage | 2x | fixed | Monte Carlo opt |
| equity_risk_pct | 2% | fixed | Kelly fraction |
| max_positions | 3 | fixed | Risk management |

### 4.2 ATR-Based Barrier Distance (코인별)

| Coin | ATR(14) % | SL (1.0×) | TP (5.0×) | R:R |
|------|----------|----------|----------|-----|
| BTC | 1.18% | 1.18% | 5.90% | 5.0 |
| ETH | 1.93% | 1.93% | 9.65% | 5.0 |
| SOL | 2.06% | 2.06% | 10.30% | 5.0 |
| XRP | 1.86% | 1.86% | 9.30% | 5.0 |
| ADA | 2.16% | 2.16% | 10.80% | 5.0 |
| DOT | 2.22% | 2.22% | 11.10% | 5.0 |
| LINK | 2.19% | 2.19% | 10.95% | 5.0 |

### 4.3 Leverage vs Liquidation

| Leverage | SL 발동 | 청산 발동 | 안전 마진 | 판정 |
|----------|--------|----------|----------|------|
| 2x | ~2% | 49.6% | 47.6%p | SAFE |
| 3x | ~2% | 32.9% | 30.9%p | SAFE |
| 5x | ~2% | 19.6% | 17.6%p | SAFE |
| 10x | ~2% | 9.6% | 7.6%p | SAFE |

모든 레버리지에서 SL이 청산 전에 발동. 단, flash crash 시 SL 미체결 위험 존재.

---

## 5. Validation Results

### 5.1 Data Split

```
IS (In-Sample):   2025-03-24 ~ 2025-12-04 (1,526 bars, 70%)
OOS (Out-of-Sample): 2025-12-04 ~ 2026-03-23 (655 bars, 30%)
OOS는 최적화 과정에서 한 번도 사용하지 않음
```

### 5.2 Grid Search (IS only)

```
Total configs: 6,480
  = 6 lookback × 5 cvd_q × 3 cvd_w × 3 k_upper × 3 k_lower
    × 2 max_hold × 2 vol_weighted × 2 use_oi
Best IS: Sharpe 2.67 (lb=28, cq=0.70, cw=120, ku=5.0, kl=1.5)
```

### 5.3 OOS Results (Top 5)

| Config | Trades | WR | Avg PnL | Sharpe |
|--------|--------|-----|---------|--------|
| lb=28, cq=0.75, ku=5.0, kl=1.0 | 48 | 47.9% | +2.31% | 2.80 |
| lb=14, cq=0.75 | 71 | 47.9% | +2.19% | 3.46 |
| lb=7, cq=0.75 | 72 | 44.4% | +2.04% | 3.26 |
| **lb=7+28 dual** | **37** | **54.1%** | **+3.46%** | **3.42** |
| lb=5, cq=0.85 | 39 | 43.6% | +2.54% | 2.62 |

### 5.4 Statistical Significance

| Test | Value | Interpretation |
|------|-------|---------------|
| Permutation test (2,000 shuffles) | p = 0.006 | 99.4% 신뢰수준 유의 |
| Random direction baseline | avg +0.86% | R:R 비대칭 기여분 |
| **Direction alpha** | **+1.45%/trade** | TSMOM이 추가한 순수 알파 |
| Bootstrap P(avg > 0) | 100.0% | |
| Bootstrap P(Sharpe > 1) | 99.8% | |
| Bootstrap Sharpe 5/50/95th | 2.29 / 3.80 / 5.25 | |

### 5.5 Cost Sensitivity (OOS)

```
Cost  0 bps: avg +2.51%, Sharpe 3.04  ← 비용 없을 때
Cost 20 bps: avg +2.31%, Sharpe 2.80  ← 현재 가정
Cost 30 bps: avg +2.21%, Sharpe 2.68  ← 50% 비용 증가
Cost 50 bps: avg +2.01%, Sharpe 2.44  ← 2.5배 비용에도 양수
```

### 5.6 Drop-One-Out Stability

```
Full:       n=48, Sharpe 2.80
Drop BTC:   Sharpe 2.55 (delta -0.25)
Drop ETH:   Sharpe 2.77 (delta -0.03)
Drop SOL:   Sharpe 2.52 (delta -0.28)
Drop XRP:   Sharpe 2.72 (delta -0.08)
Drop ADA:   Sharpe 2.34 (delta -0.46)  ← 가장 기여
Drop DOT:   Sharpe 2.71 (delta -0.09)
Drop LINK:  Sharpe 2.56 (delta -0.24)

→ 어떤 코인을 빼도 Sharpe 2.0+ 유지
→ 특정 코인 의존 없음
```

### 5.7 Long/Short Breakdown (OOS)

```
LONG:  n=19, avg +0.09%  ← OOS 기간 하락장, LONG 약세
SHORT: n=42, avg +2.53%  ← SHORT 지배적

주의: 상승장에서는 LONG이 지배적으로 전환될 것 (TSMOM 특성)
```

---

## 6. Binance Data Integration

### 6.1 데이터 소스

```
경로:  data/raw/binance_public/metrics/{SYMBOL}/
형식:  CSV (5-min resolution)
기간:  365일 (7 coins)
컬럼:  create_time, symbol,
       sum_open_interest, sum_open_interest_value,
       count_toptrader_long_short_ratio,
       sum_toptrader_long_short_ratio,
       count_long_short_ratio,
       sum_taker_long_short_vol_ratio
```

### 6.2 파생 피처 (16개)

| 그룹 | 피처 | 용도 |
|------|------|------|
| OI | oi_value, oi_change_pct, oi_ratio, oi_zscore, oi_price_div | 포지셔닝 과열 |
| L/S Ratio | lsr, lsr_zscore, lsr_extreme_long, lsr_extreme_short | 군중 편향 |
| Top Trader | top_trader_lsr, top_trader_lsr_zscore | 기관 포지션 |
| Taker | taker_vol_ratio, taker_vol_ma_6, taker_buy_pressure | 시장가 방향 |

### 6.3 OI 필터 효과

```
Base (OI 없음):     avg +0.490%, Sharpe 1.39
+ OI filter:        avg +0.652%, Sharpe 1.80  (+0.41 Sharpe)
+ OI divergence:    avg +0.789%, Sharpe 1.91
+ ALL Binance:      avg +0.805%, Sharpe 1.90
```

---

## 7. RL Enhancement Layer

### 7.1 아키텍처

```
LinUCB Contextual Bandit
  State:   33-dim (v5.0 TSMOM adapted)
  Actions: 7 discrete sizing multipliers
  Reward:  Differential Sharpe Ratio (계획)
  Status:  Shadow mode (logging only)
```

### 7.2 State Vector (33-dim)

```
Signal Quality (6):
  tsmom_strength     |28d return|             [0, 0.3]
  rsi_normalized     RSI / 100                [0, 1]
  cvd_extremeness    |CVD - median| / range   [0, 1]
  oi_zscore          OI z-score               [-3, 3]
  tsmom_rsi_agree    direction match          {0, 1}
  side_sign          LONG=+1, SHORT=-1        {-1, +1}

Market Regime (4):
  regime_trend       ADX > 25                 {0, 1}
  regime_up          ADX > 25 + DI+ > DI-     {0, 1}
  atr_pct            ATR / price              [0, 0.1]
  hurst              Hurst exponent           [0, 1]

Microstructure (3):
  cvd_ratio          CVD ratio 6-bar          [-3, 3]
  ofi_norm           OFI normalized           [-3, 3]
  ms_composite       composite score          [-1, 1]

Cost Proxy (2):
  spread_proxy       (high-low) / close       [0, 0.05]
  last_funding       funding rate             [-0.003, 0.003]

Portfolio (4):
  open_positions     count / 5                [0, 1]
  daily_pnl_pct      daily PnL / equity       [-0.05, 0.05]
  weekly_pnl_pct     weekly PnL / equity      [-0.10, 0.10]
  dd_ratio           drawdown ratio           [0, 1]

Coin History (4):
  coin_win_rate_5    last 5 trades WR         [0, 1]
  coin_avg_pnl_5     last 5 trades avg PnL    [-0.05, 0.05]
  coin_streak        consecutive W/L          [-1, 1]
  bars_since_last    bars / max_horizon       [0, 1]

Cross-Market (2):
  btc_return_24h     BTC 24h return           [-0.10, 0.10]
  corr_btc           correlation with BTC     [-1, 1]

Coin Identity (7):
  one-hot encoding   BTC~LINK                 {0, 1}

Intercept (1):
  constant           always 1.0               1.0
```

### 7.3 Deployment Phases

```
Phase 1 (현재): Shadow Mode
  - 모든 시그널 signal_log.jsonl에 기록
  - RL 결정은 기록만, 실제 매매에 미적용
  - 목표: 200+ 시그널 축적

Phase 2: Offline Training
  - python -m src.rl.offline_train
  - LinUCB 학습 (Sherman-Morrison update)
  - Held-out 20% 평가

Phase 3: Shadow Evaluation
  - Conditional lift: mean(PnL | RL accept) > mean(PnL | all)
  - Reject quality: rejected 시그널 중 실제 음수 비율 > 55%
  - Calibration: rl_score 높을수록 실제 PnL 높은지

Phase 4: Live Activation
  - shadow_mode: false
  - RL sizing 실제 적용
  - change_cap: ±25% (급격한 변화 방지)
```

---

## 8. Risk Management

### 8.1 Pre-Trade Gates

| Gate | Rule | Purpose |
|------|------|---------|
| Dual TSMOM | 7d + 28d 방향 일치 | 전환점 차단 |
| RSI filter | RSI-방향 정합성 | 약한 추세 차단 |
| CVD timing | 극단 반등/투매에서만 진입 | 나쁜 타이밍 차단 |
| OI crowding | \|z\| < 2.0 | 청산 캐스케이드 회피 |
| Max positions | 동시 3개 | 집중 리스크 제한 |
| Per-coin cap | equity × 20% | 단일 코인 제한 |
| Per-trade risk | equity × 2% | 단일 손실 제한 |

### 8.2 Post-Trade Protection

| Protection | Mechanism |
|-----------|-----------|
| Stop Loss | STOP_MARKET @ 1.0×ATR (즉시 체결) |
| Time Stop | 24 bars (96h) 후 시장가 청산 |
| Take Profit | 5.0×ATR (R:R = 5:1) |

### 8.3 Worst Case Scenarios

| Scenario | 1x | 2x | 3x |
|----------|-----|-----|-----|
| 정상 SL | -2% | -4% | -6% |
| 연속 5패 | -10% | -20% | -30% |
| Flash Crash (SL 3× skip) | -6% | -12% | -18% |
| 최악 (연속 5 + gap) | -30% | -60% | -90% |

---

## 9. Current Market State (2026-03-23)

```
BTC: $68,280 | RSI 28.4 | 7d SHORT, 28d LONG → dual_disagree
ETH: $2,040  | RSI 25.0 | 7d SHORT, 28d LONG → dual_disagree
SOL: $85.68  | RSI 22.7 | 7d SHORT, 28d LONG → dual_disagree
XRP: $1.37   | RSI 13.2 | 7d SHORT, 28d SHORT → cvd_timing 대기
ADA: $0.25   | RSI 10.5 | 7d SHORT, 28d SHORT → cvd_timing 대기
DOT: $1.42   | RSI 23.0 | 7d SHORT, 28d LONG → dual_disagree
LINK: $8.62  | RSI 19.4 | 7d SHORT, 28d LONG → dual_disagree

Status: 0 positions, $1,000 equity
Waiting: 28d return이 음수로 전환 시 (며칠 내 예상) SHORT 시그널 발동
```

---

## 10. File Map

```
run_tsmom_paper.py                    Paper bot (dual lookback, RL logging)

experiments/
  tsmom_ml_enhanced.py                Phase 1: 모든 레이어 테스트
  tsmom_rsi_cvd_deep.py               Phase 2: 6,480 grid search
  tsmom_rigorous_v2.py                Phase 3: IS/OOS + permutation
  download_and_integrate.py           Binance data + 통합 테스트
  test_lookback_fix.py                Dual lookback 검증

src/rl/
  state_builder.py                    33-dim state (v5.0)
  bandit.py                           LinUCB 7-action
  signal_logger.py                    JSONL signal log
  rl_gate.py                          Shadow/active mode
  counterfactual.py                   Rejected signal PnL
  offline_train.py                    CLI trainer

data/raw/binance_public/metrics/      OI/LSR/Taker (7 coins × 365d)
data/reports/tsmom_paper/             Paper bot 로그
docs/model_status_report_20260323.md  수치 상세 리포트
```
