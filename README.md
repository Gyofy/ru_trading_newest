# CLAUDE_CRYPTO_AGENT

Autonomous crypto trading system — Dual TSMOM direction + CVD timing + OI crowding filter + RL sizing.

**Binance USDT-M Futures | 10 coins | Dual TSMOM(7d+28d)+RSI+CVD+OI | OOS Sharpe 4.03 | Paper Trading Active**

---

## Current Status (2026-03-23)

### v5.1 TSMOM Enhanced Strategy

Feature leakage discovery (2026-03-20) invalidated all ML direction predictions. v5.1 uses **rule-based dual TSMOM direction** + 10-coin universe + compact RL sizing.

```
Direction:  Dual TSMOM (7-day + 28-day must agree)
Filter:     RSI > 50 (LONG valid) / RSI < 50 (SHORT valid)
Timing:     CVD Q75 extreme (counter-direction overextension)
Crowding:   OI z-score < 2.0 (Binance metrics)
Barrier:    TP = 5 x ATR, SL = 1.0 x ATR, TTL = 24 bars (96h)
Universe:   10 coins (BTC ETH SOL XRP ADA DOT LINK DOGE AVAX BNB)
Leverage:   2x (paper), max 3x (live)
```

### Validation Results

| Metric | IS (70%, 9mo) | OOS (30%, 3.5mo) |
|--------|---------------|-------------------|
| Configs tested | 6,480 grid | Top-20 IS |
| Best Sharpe | 2.67 | **4.03** (10 coins) |
| Best avg PnL | +1.12%/trade | **+2.85%/trade** |
| Positive configs | many | **20/20 (100%)** |
| Permutation test | — | **p = 0.006** |
| Bootstrap P(Sharpe>1) | — | **99.8%** |

| Test | Result |
|------|--------|
| Permutation test | **p = 0.006** (statistically significant) |
| Bootstrap P(Sharpe > 1) | **99.8%** |
| Cost sensitivity | Positive up to **50 bps** roundtrip |
| Drop-one-out | All coins removable, min Sharpe 2.34 |
| Optimal leverage | **3x** (Monte Carlo, P(MDD>50%) < 8%) |

### OOS Performance by Coin (v5.1, 10 coins)

| Coin | Trades | WR | Avg PnL | Sharpe |
|------|--------|-----|---------|--------|
| BTC | 4 | 75.0% | +5.34% | 1.82 |
| ETH | 5 | 80.0% | +4.67% | 2.41 |
| SOL | 4 | 50.0% | +2.89% | 0.89 |
| XRP | 4 | 50.0% | +1.65% | 0.55 |
| ADA | 2 | 100% | +6.13% | - |
| DOT | 7 | 28.6% | -0.60% | -0.20 |
| LINK | 4 | 50.0% | +2.63% | 0.82 |
| DOGE | 13 | 46.2% | +2.17% | 1.69 |
| AVAX | 11 | 45.5% | +2.48% | 1.48 |
| BNB | 6 | 50.0% | +1.58% | 0.77 |

---

## Architecture (v5.1)

```
run_tsmom_paper.py              Paper bot (ACTIVE, 10 coins, RL logging)
run_monitor.py                  Dashboard + healthcheck + auto-restart
run_live_bot_v2.py              v4.3 ML bot (SUSPENDED)
run_btc_spike_paper.py          v4.4 BTC spike bot (SUPERSEDED)

src/
  data/crawlers/
    crypto_ohlcv.py             OHLCV + technical indicators (causal)
    signal_features.py          Wavelet/FFT/entropy (leakage fixed)
    microstructure_rollup.py    CVD/OFI/VPIN/Roll Spread/Amihud (BVC)
    binance_public_data_downloader.py   Binance metrics downloader
  execution/
    exchange_adapter.py         Binance USDT-M Futures (ccxt)
    live_predictor.py           2-Stage Binary (v4.3, suspended)
    sl_tp_monitor.py            SL/TP polling
    risk_engine.py              9-gate pre-trade check
    position_store.py           Crash-safe persistence
    cost_model.py               Fee + slippage + funding
  models/
    masking_loop.py             Triple Barrier labeling
    regime_filter.py            4-state regime (TREND_UP/DOWN, RANGE_LOW/HIGH)
  strategy/
    tsmom_core.py               Shared signal gen, backtest, metrics (v5.1)
  rl/
    bandit.py                   LinUCB 7-action (v5.1)
    state_builder.py            33-dim full / 7-dim compact state (v5.1)
    signal_logger.py            JSONL signal logging + counterfactual
    rl_gate.py                  Shadow/active mode with safety
    counterfactual.py           Rejected signal PnL estimation
    offline_train.py            CLI training pipeline
  signals/
    contract.py                 Signal dataclass
    policy.py                   SignalPolicy regime-aware filtering

experiments/
  tsmom_ml_enhanced.py          Phase 1: TSMOM base + filters
  tsmom_rsi_cvd_deep.py         Phase 2: 6,480 config grid search
  tsmom_rigorous_v2.py          Phase 3: IS/OOS split + permutation test
  download_and_integrate.py     Binance data download + integration
  v5_1_full_upgrade.py          Phase 4: 10 coins + trailing + trend score + RL
  rl_state_analysis.py          PCA + feature importance (33→6 dim reduction)

config/
  frozen_params_v4_3.yaml       Config (v4.3, RL section reused)

data/
  raw/binance_public/metrics/   OI/LSR/Taker (7 coins x 365d)
  reports/tsmom_paper/          Paper bot logs + signal_log
  reports/tsmom_*.csv           Backtest results
```

---

## Strategy Design (v5.0)

### Layer 1: Direction (Rule-Based)
```
TSMOM: sign(28-day return) -> LONG or SHORT
- Volume-weighted variant available
- No ML prediction (S2 = random after leakage fix)
- Academic basis: Sharpe 1.5-2.2 (Huang et al. 2024)
```

### Layer 2: Quality Filter (Rule + Optional ML)
```
RSI > 50 confirms LONG, RSI < 50 confirms SHORT
- RSI as trend filter, NOT overbought/oversold
- PMC study: RSI trend filter 773% vs B&H 275%
```

### Layer 3: Entry Timing (CVD Extreme)
```
SHORT signal + CVD > Q75 (buy-side overextended) = enter SHORT
LONG signal + CVD < Q25 (sell-side overextended) = enter LONG
- Spearman rho = -0.21 (CVD mean reversion)
- Counter-direction overextension = better fill price
```

### Layer 4: Crowding Filter (Binance OI)
```
|OI z-score| > 2.0 = skip (crowded positioning)
- Avoids entering at extreme OI where liquidation cascades likely
- Data: Binance public metrics (5-min, resampled to 4h)
```

### Layer 5: RL Enhancement (Shadow Mode)
```
LinUCB contextual bandit, 33-dim state, 7 actions
- Actions: REJECT, 0.5x, 0.75x, 1.0x, 1.25x, 1.5x, 2.0x
- State: TSMOM strength + RSI + CVD + OI + regime + microstructure
- Training: offline from signal_log.jsonl (200+ signals needed)
- Status: shadow mode (logging only, not applied)
```

---

## Development History

### v4.3-1m Live Session (2026-03-24) — 실거래 첫 가동

**배경**: v5.1 TSMOM paper bot과 별도로, v4.3 ML bot(`run_live_bot_v2.py`)을 Binance USDT-M Futures에 실거래 가동.

**왜 SOL 단일 종목인가?**

GitHub 원본에는 5코인(DOT/ADA/XRP/SOL/LINK) 다중 종목이 설정되어 있었으나, `run_live_bot_v2.py` line 62에서 아래와 같이 변경됨:

```python
COINS = ["SOL"]  # SOL 단일 운영 (2026-03-19)
```

**이유**: 2026-03-19 실거래 전환 직전 진행한 coin-by-coin backtest에서 SOL만 손익분기에 근접한 결과를 보였고, 나머지 4개 코인(DOT/ADA/XRP/LINK)은 수수료 대비 기대수익이 마이너스였음. 실자본 66 USDT로 다중 종목 분산 시 코인당 포지션이 너무 작아 수수료 비중이 과다해지는 문제도 있었음.

**오늘 수행한 작업 (2026-03-24)**:

| 항목 | 내용 |
|------|------|
| Post-Only 폴백 | -5022 거부 시 일반 limit 주문으로 자동 재시도 (exchange_adapter.py) |
| Binance 공개 데이터 통합 | OI/L/S/Taker/Funding 피처를 compute_features()에 연결 |
| datetime dtype 버그 | ms vs us 불일치 → `datetime64[ns]` 통일 |
| merge_asof 버그 | `reset_index().rename({"index": "_ts"})` → `pd.DataFrame({"_ts": ...})` |
| 일일 손실 한도 | 6% → **10%** |
| 포지션 크기 | 자본의 150% 상한 → **자본의 10%** |
| --yes 플래그 | 백그라운드 실행 시 확인 프롬프트 우회 |
| 종목 확장 | SOL 단일 → **BTC, ETH, SOL, XRP, ADA, DOT, LINK (7코인)** |

**첫 체결**:
```
SOL SELL FILLED @ 89.67 | SL=89.83 TP=89.11
equity: 66 USDT → 65.96 USDT
```

**현재 상태**: 실거래 중지 (2026-03-24 기록 보존)

---

### v5.1 (2026-03-23) — TSMOM Enhanced (Current)
- Universe expansion: 7 → 10 coins (Sharpe 3.42 → 4.03)
- Dual TSMOM (7d+28d) for trend transition handling
- Compact LinUCB (6-dim) trained, lift +0.85%p
- tsmom_core.py: shared strategy module (anti-spaghetti)
- run_monitor.py: dashboard + healthcheck

### v5.0 (2026-03-23) — TSMOM Base
- Pivot from ML direction to rule-based TSMOM
- 6,480 config grid search + rigorous IS/OOS validation
- Permutation test p=0.006, Bootstrap P(Sharpe>1)=99.8%
- Binance OI/LSR/Taker data integrated
- RL state_builder adapted for TSMOM inputs

### v4.4 (2026-03-20) — BTC Spike
- Feature leakage discovered (STL, Ichimoku, SVD)
- 15,120 exhaustive search: only BTC spike survived
- Paper bot: avg +0.11%/trade (weak, superseded by v5.0)

### v4.3 (2026-03-18) — ML 2-Stage (INVALIDATED)
- 8 coins, Mega Search v3, RL meta-layer
- All results based on leaked features

### v4.0-4.2 (2026-03-17) — ML Evolution (INVALIDATED)
- 2-Stage Binary, Triple Barrier, walk-forward
- Invalidated by leakage discovery

---

## Paper Bot & Monitoring

```bash
# Start paper trading (v5.1, 10 coins)
nohup python -X utf8 run_tsmom_paper.py > data/reports/tsmom_paper/nohup.log 2>&1 &

# Monitor dashboard (equity, positions, trades, RL status)
python run_monitor.py

# Healthcheck (returns exit code 0/1)
python run_monitor.py --check

# Auto-restart if stopped
python run_monitor.py --restart
```

Files:
- `data/reports/tsmom_paper/state.json` — bot state (equity, positions)
- `data/reports/tsmom_paper/trades.jsonl` — trade history
- `data/reports/tsmom_paper/signal_log.jsonl` — RL signal log
- `data/reports/tsmom_paper/bot.log` — execution log

---

## Data Pipeline

### OHLCV (yfinance)
- 10 coins: BTC, ETH, SOL, XRP, ADA, DOGE, AVAX, DOT, LINK, BNB
- 1h fetch -> 4h resample
- 365 days history

### Binance Public Metrics
```bash
python src/data/crawlers/binance_public_data_downloader.py \
  --types funding_rate open_interest long_short_ratio \
  --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT ADAUSDT DOTUSDT LINKUSDT \
  --days 365
```

Data includes: OI, Long/Short Ratio, Top Trader Ratio, Taker Buy/Sell Volume (5-min resolution)

---

## Reports & Documentation

| Report | Path | Content |
|--------|------|---------|
| **Performance Summary** | `docs/performance_summary_v5_1.md` | Full WR, Sharpe, PnL, risk analysis |
| **Strategy Report** | `docs/strategy_report_v5.md` | Architecture, signal flow, parameters |
| **Model Status** | `docs/model_status_report_20260323.md` | Validation details, RL spec |
| **Binance Data Spec** | `docs/binance_public_data_spec.md` | Data types, columns, download guide |

---

## Environment

- **Python** 3.10+ | **Exchange** Binance USDT-M Futures (ccxt)
- **ML** scikit-learn, CatBoost, XGBoost | **RL** LinUCB (custom)
- **Data** yfinance (paper) + Binance public data | **OS** Windows 11

```bash
pip install ccxt yfinance scikit-learn catboost xgboost ta pandas numpy joblib pyyaml requests tqdm
```
