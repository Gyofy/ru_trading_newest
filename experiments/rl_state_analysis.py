"""RL State Vector Analysis: PCA + Feature Importance + Redundancy Check.

33-dim state vector가 200 시그널로 학습 가능한지 검증.
- PCA: 실효 차원 확인 (33dim 중 몇 개가 실제로 의미?)
- 상관관계: 중복 피처 식별
- Feature Importance: 어떤 피처가 PnL과 관련?
- 차원 축소 제안: 200 시그널에 적합한 state 크기
"""

import sys, os, warnings
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from src.rl.state_builder import build_rl_state, STATE_DIM, STATE_NAMES, COIN_INDEX

# ══════════════════════════════════════════════════════════
# 1. Generate synthetic state vectors from historical data
# ══════════════════════════════════════════════════════════

def generate_historical_states():
    """Generate state vectors + PnLs from backtest trades."""
    from experiments.tsmom_rigorous_v2 import load_all, split_is_oos, generate_signals, backtest, COINS_MAP

    print("Loading data...")
    data = load_all()
    _, oos_data = split_is_oos(data)

    states, pnls, coins_list = [], [], []

    for coin in COINS_MAP:
        if coin not in oos_data:
            continue
        df = oos_data[coin]
        sig = generate_signals(df, lb=7, vw=False, cq=0.75, cw=120, use_oi=True)
        trade_pnls, sides, entry_bars = backtest(df, sig, ku=5.0, kl=1.0, mh=24)

        # Build state for each trade entry
        for i, (pnl, side, bar_idx) in enumerate(zip(trade_pnls, sides, entry_bars)):
            if bar_idx >= len(df) or bar_idx < 30:
                continue

            df_slice = df.iloc[:bar_idx + 1]
            close = df_slice["close"].iloc[-1]

            # Compute TSMOM strength
            lb_bars = 7 * 6
            if len(df_slice) > lb_bars:
                tsmom_str = abs(df_slice["close"].pct_change(lb_bars).iloc[-1])
            else:
                tsmom_str = 0.0

            rsi = df_slice["rsi_14"].iloc[-1] if "rsi_14" in df_slice.columns else 50.0

            # CVD extremeness
            if "cvd_ratio_24" in df_slice.columns:
                cvd = df_slice["cvd_ratio_24"].iloc[-1]
                cvd_ext = min(abs(cvd), 1.0)
            else:
                cvd_ext = 0.0

            # OI
            oi_z = df_slice["oi_zscore"].iloc[-1] if "oi_zscore" in df_slice.columns else 0.0

            try:
                state = build_rl_state(
                    df=df_slice,
                    pred_side="BUY" if side == 1 else "SELL",
                    coin=coin,
                    equity=1000.0, daily_pnl=0.0, weekly_pnl=0.0, dd_ratio=0.0,
                    open_count=0, coin_win_rate_5=0.5, coin_avg_pnl_5=0.0,
                    coin_streak=0, bars_since_last=10,
                    tsmom_strength=tsmom_str if not np.isnan(tsmom_str) else 0.0,
                    rsi_value=rsi if not np.isnan(rsi) else 50.0,
                    cvd_extremeness=cvd_ext,
                    oi_zscore=oi_z if not np.isnan(oi_z) else 0.0,
                    tsmom_rsi_agree=True,
                )
                states.append(state)
                pnls.append(pnl)
                coins_list.append(coin)
            except Exception as e:
                continue

    X = np.array(states)
    y = np.array(pnls)
    print(f"\nGenerated {len(X)} state vectors, {STATE_DIM} dims")
    return X, y, coins_list


# ══════════════════════════════════════════════════════════
# 2. PCA Analysis
# ══════════════════════════════════════════════════════════

def pca_analysis(X):
    print("\n" + "=" * 70)
    print("PCA ANALYSIS: Effective Dimensionality")
    print("=" * 70)

    # Remove constant columns (one-hot intercept)
    non_const = np.std(X, axis=0) > 1e-10
    X_clean = X[:, non_const]
    active_names = [n for n, nc in zip(STATE_NAMES, non_const) if nc]
    print(f"  Active features (non-constant): {X_clean.shape[1]} / {X.shape[1]}")

    # Standardize
    means = X_clean.mean(axis=0)
    stds = X_clean.std(axis=0)
    stds[stds == 0] = 1
    X_std = (X_clean - means) / stds

    pca = PCA()
    pca.fit(X_std)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_90 = np.argmax(cumvar >= 0.90) + 1
    n_95 = np.argmax(cumvar >= 0.95) + 1
    n_99 = np.argmax(cumvar >= 0.99) + 1

    print(f"\n  Explained variance:")
    for i in range(min(15, len(pca.explained_variance_ratio_))):
        bar = "#" * int(pca.explained_variance_ratio_[i] * 100)
        print(f"    PC{i+1:2d}: {pca.explained_variance_ratio_[i]:6.2%} (cum {cumvar[i]:6.2%}) {bar}")

    print(f"\n  Components for 90% variance: {n_90}")
    print(f"  Components for 95% variance: {n_95}")
    print(f"  Components for 99% variance: {n_99}")
    print(f"  Effective dimensionality: ~{n_90}-{n_95}")

    # Rule of thumb: need ~10x samples per dimension for LinUCB
    print(f"\n  With {n_90} effective dims, need ~{n_90 * 10} samples (have {len(X)})")
    if len(X) >= n_90 * 10:
        print(f"  -> SUFFICIENT for LinUCB with PCA({n_90})")
    else:
        print(f"  -> INSUFFICIENT without PCA. Reduce to {len(X) // 10} dims max")

    return pca, active_names, non_const, n_90, n_95


# ══════════════════════════════════════════════════════════
# 3. Correlation / Redundancy
# ══════════════════════════════════════════════════════════

def redundancy_analysis(X):
    print("\n" + "=" * 70)
    print("REDUNDANCY ANALYSIS: Highly Correlated Feature Pairs")
    print("=" * 70)

    non_const = np.std(X, axis=0) > 1e-10
    X_clean = X[:, non_const]
    active_names = [n for n, nc in zip(STATE_NAMES, non_const) if nc]

    corr = np.corrcoef(X_clean.T)

    # Find pairs with |corr| > 0.7
    high_corr = []
    for i in range(len(active_names)):
        for j in range(i + 1, len(active_names)):
            c = corr[i, j]
            if abs(c) > 0.7:
                high_corr.append((active_names[i], active_names[j], c))

    high_corr.sort(key=lambda x: abs(x[2]), reverse=True)

    if high_corr:
        print(f"\n  Pairs with |correlation| > 0.7:")
        for f1, f2, c in high_corr[:15]:
            print(f"    {f1:25s} <-> {f2:25s} : r={c:+.3f}")
        print(f"\n  Total high-corr pairs: {len(high_corr)}")
        print(f"  Candidates for removal (keep one from each pair):")
        remove_candidates = set()
        for f1, f2, c in high_corr:
            remove_candidates.add(f2)  # remove second of pair
        for f in sorted(remove_candidates):
            print(f"    - {f}")
    else:
        print("  No highly correlated pairs found (all |r| < 0.7)")

    return high_corr


# ══════════════════════════════════════════════════════════
# 4. Feature Importance for PnL
# ══════════════════════════════════════════════════════════

def feature_importance(X, y):
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE: Which state dims predict PnL?")
    print("=" * 70)

    non_const = np.std(X, axis=0) > 1e-10
    X_clean = X[:, non_const]
    active_names = [n for n, nc in zip(STATE_NAMES, non_const) if nc]

    # ExtraTrees regression
    model = ExtraTreesRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_clean, y)

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    print(f"\n  Top features predicting trade PnL:")
    for rank, i in enumerate(idx[:15]):
        bar = "#" * int(importances[i] * 200)
        print(f"    {rank+1:2d}. {active_names[i]:25s} imp={importances[i]:.4f} {bar}")

    # Bottom features (candidates for removal)
    print(f"\n  Bottom features (low importance, removal candidates):")
    for rank, i in enumerate(idx[-10:]):
        print(f"    {active_names[i]:25s} imp={importances[i]:.4f}")

    # Recommended state dimensions
    cumulative_imp = np.cumsum(importances[idx])
    n_80 = np.argmax(cumulative_imp >= 0.80) + 1
    n_90 = np.argmax(cumulative_imp >= 0.90) + 1

    print(f"\n  Features for 80% importance: {n_80}")
    print(f"  Features for 90% importance: {n_90}")

    return importances, active_names


# ══════════════════════════════════════════════════════════
# 5. Recommended Reduced State
# ══════════════════════════════════════════════════════════

def recommend_reduction(pca_n90, importances, active_names, n_samples, high_corr):
    print("\n" + "=" * 70)
    print("RECOMMENDATION: Optimal State Vector Size")
    print("=" * 70)

    max_dims = n_samples // 10  # rule of thumb
    print(f"\n  Samples available: {n_samples}")
    print(f"  Current dims: {STATE_DIM}")
    print(f"  PCA effective: {pca_n90}")
    print(f"  Max dims (10x rule): {max_dims}")

    # Identify top features by importance
    idx = np.argsort(importances)[::-1]
    top_features = [active_names[i] for i in idx[:max_dims]]

    # Remove redundant from top
    remove = set()
    for f1, f2, c in high_corr:
        if f2 in top_features and f1 in top_features:
            remove.add(f2)

    final = [f for f in top_features if f not in remove][:max_dims]

    print(f"\n  Recommended state ({len(final)} dims):")
    for i, f in enumerate(final):
        print(f"    {i+1:2d}. {f}")

    print(f"\n  Removed (redundant or low importance):")
    removed = [n for n in active_names if n not in final]
    for f in removed:
        reason = "redundant" if f in remove else "low importance"
        print(f"    - {f} ({reason})")

    return final


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    X, y, coins = generate_historical_states()

    if len(X) < 20:
        print("Not enough data for analysis")
        sys.exit(1)

    pca, active_names, non_const, n_90, n_95 = pca_analysis(X)
    high_corr = redundancy_analysis(X)
    importances, feat_names = feature_importance(X, y)
    recommended = recommend_reduction(n_90, importances, feat_names, len(X), high_corr)

    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Current state: {STATE_DIM} dims")
    print(f"  Effective (PCA 90%): {n_90} dims")
    print(f"  Recommended: {len(recommended)} dims")
    print(f"  Data sufficiency: {'OK' if len(X) >= n_90 * 10 else 'NEED MORE DATA'}")
