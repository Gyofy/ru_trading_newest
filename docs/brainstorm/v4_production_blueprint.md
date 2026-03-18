# v4 Production Blueprint

Date: 2026-03-17
Status: APPROVED -- implementation priority defined

## Implementation Priority (confirmed)

1. **Sample Weighting** (uniqueness + recency + capped quality)
2. **Probability Calibration** (Isotonic/Platt -> Brier score monitoring)
3. **Constrained Optuna** (multi-objective: NetEV + MDD + concentration + cost)
4. **Regime-aware Stacking** (regime one-hot + interaction in meta-learner)
5. **Hurst Bucket Barrier** (3-bucket: low/mid/high -> TP/SL/TL adjustment)
6. **FracDiff Ablation** (experiment branch, 5 core features only)

## 1. Sample Weighting

### Formula
```
w_i = clip(w_uniq^a * w_recency^b * w_quality^g, 0.5, 3.0)
```

### Components
- **w_uniq**: uniqueness -- penalize overlapping barrier labels
  - For each sample, count how many other samples' barrier windows overlap
  - w_uniq = 1 / overlap_count
- **w_recency**: exponential decay, half-life 30-45 days
  - w_recency = exp(-lambda * days_ago)
- **w_quality**: capped quality (NOT raw PnL magnitude)
  - strong_positive: 1.5
  - weak_positive: 1.0
  - neutral: 0.8
  - hard_negative: 1.2 (learn from mistakes, but don't overweight)

### WARNING
- Do NOT use raw return magnitude as weight
- High-R:R strategy + return weighting = overfitting to rare tail winners

## 2. Probability Calibration

### Method
- Platt Scaling (sigmoid) for small samples
- Isotonic Regression if n > 500
- Cross-validate calibration on TimeSeriesSplit folds

### Monitoring
- Brier score per coin per regime
- Reliability diagram (calibration curve)
- Flag if predicted P(trade)=0.6 but actual hit rate is 0.3

### Impact on Threshold
- After calibration, threshold meaning changes
- Bounded adaptive threshold: DOT [0.48, 0.54], ADA [0.50, 0.56]
- Update weekly, not per-bar

## 3. Constrained Optuna

### Objective
```
maximize: Net EV (trade-level)
subject to:
  MDD < 3%
  cost_share < 20%
  2-week trade_count >= 15
  top-1 trade contribution < 40%
  top-3 trades contribution < 65%
```

### Implementation
- Optuna TPE sampler
- Pruning with MedianPruner
- Walk-forward CV as inner loop
- Per-coin optimization

## 4. Regime-aware Stacking

### Approach (NOT full MoE -- too few samples)
- Add to meta-learner input:
  - regime_onehot (4 features)
  - model_prob x regime interactions
- Keep per-coin separate meta-learners
- Strong regularization (L2)

### Why not full MoE
- ~2000 bars per coin, ~30-50 trades
- Expert subdivision -> sample starvation
- Gating instability across folds

## 5. Hurst Bucket Barrier

### 3 buckets (NOT continuous mapping)
```
DOT:
  low  (H < 0.45): TP=2.4, SL=0.6, TL=14
  mid  (0.45-0.60): TP=3.0, SL=0.6, TL=18  (current)
  high (H > 0.60): TP=3.4, SL=0.6, TL=22

ADA:
  low  (H < 0.45): TP=2.2, SL=0.8, TL=14
  mid  (0.45-0.60): TP=3.0, SL=0.8, TL=18  (current)
  high (H > 0.60): TP=3.3, SL=0.8, TL=22
```

### WARNING
- Do NOT optimize bucket boundaries with Optuna (overfit risk)
- Keep 3 buckets only, validate on frozen OOS

## 6. FracDiff Ablation

### Scope: experiment branch only
- 5 features: close, vwap, obv, realized_vol, trend_slope
- Compare: raw vs fracdiff (d ~ 0.3-0.5)
- Metric: MI stability across folds + OOS trade-level EV

## Key Principles

1. **Low degrees of freedom** -- fewer knobs, harder to overfit
2. **Constrained optimization** -- smooth equity > max return
3. **Stability-first** -- plateau threshold > peak threshold
4. **Capped weighting** -- no extreme sample influence
5. **Bucket policy > continuous** -- for barrier params
