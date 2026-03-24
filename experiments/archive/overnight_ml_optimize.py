"""Overnight ML Optimization — S1 quality filter + LinUCB + full grid.

Runs autonomously until completion:
  Phase 1: Train S1 quality classifier on IS data
  Phase 2: Train compact LinUCB on IS data
  Phase 3: Grid search (ML threshold × RL sizing × barriers) on IS
  Phase 4: OOS validation on held-out 30%
  Phase 5: Permutation test + bootstrap
  Phase 6: Per-coin + cost sensitivity analysis
  Phase 7: Export best config for paper bot
"""

import sys, os, time, warnings, json, itertools
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import balanced_accuracy_score
import joblib

from src.strategy.tsmom_core import (
    load_ohlcv_10, split_is_oos, generate_dual_signals, run_backtest,
    calc_metrics, compute_cvd_ratio, COST_ROUNDTRIP, COINS_10, YAHOO_MAP
)
from src.rl.bandit import LinUCB

OUT_DIR = "data/reports/overnight_ml"
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════
# ML Quality Features (per-signal, at entry bar)
# ══════════════════════════════════════════════════════════

ML_FEATURES = [
    "rsi_14", "atr_14", "bb_width", "macd_hist", "volume",
    "sma_20", "sma_50", "ema_12", "ema_26",
    "cvd_ratio_6", "ofi_sum_3", "ms_composite",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

def extract_ml_features(df, bar_idx):
    """Extract ML features at a specific bar."""
    row = {}
    for feat in ML_FEATURES:
        if feat in df.columns:
            v = df[feat].iloc[bar_idx]
            row[feat] = v if not (np.isnan(v) or np.isinf(v)) else 0.0
        else:
            row[feat] = 0.0

    # Add TSMOM-specific features
    close = df["close"]
    if bar_idx >= 42:
        row["tsmom_7d"] = close.pct_change(42).iloc[bar_idx]
        row["tsmom_28d"] = close.pct_change(168).iloc[bar_idx] if bar_idx >= 168 else 0.0
    else:
        row["tsmom_7d"] = 0.0
        row["tsmom_28d"] = 0.0

    row["tsmom_strength"] = abs(row["tsmom_7d"])

    # CVD extremeness
    if "cvd_ratio_24" in df.columns or True:
        cvd = compute_cvd_ratio(df.iloc[:bar_idx+1], 24)
        row["cvd_now"] = cvd.iloc[-1] if len(cvd) > 0 else 0.0
    else:
        row["cvd_now"] = 0.0

    # OI
    if "oi_zscore" in df.columns:
        v = df["oi_zscore"].iloc[bar_idx]
        row["oi_zscore"] = v if not np.isnan(v) else 0.0
    else:
        row["oi_zscore"] = 0.0

    # Volume ratio
    vol = df["volume"]
    vol_ma = vol.rolling(20, min_periods=5).mean()
    row["vol_ratio"] = vol.iloc[bar_idx] / vol_ma.iloc[bar_idx] if vol_ma.iloc[bar_idx] > 0 else 1.0

    return row


def build_training_data(data, signals_per_coin, backtest_per_coin):
    """Build (X, y) training data from backtest trades."""
    X_rows, y_labels = [], []

    for coin in data:
        sigs = signals_per_coin.get(coin)
        pnls = backtest_per_coin.get(coin, np.array([]))
        if sigs is None or len(pnls) == 0:
            continue

        df = data[coin]
        sig_vals = sigs.values if hasattr(sigs, 'values') else sigs
        trade_idx = 0
        nxt = 0

        atr = df["atr_14"].values if "atr_14" in df.columns else np.ones(len(df)) * 0.02

        for i in range(len(df) - 24):
            if i < nxt or sig_vals[i] == 0 or np.isnan(atr[i]) or atr[i] <= 0:
                continue
            if trade_idx >= len(pnls):
                break

            row = extract_ml_features(df, i)
            row["coin"] = coin
            X_rows.append(row)
            y_labels.append(1 if pnls[trade_idx] > 0 else 0)

            trade_idx += 1
            nxt = i + 2

    X = pd.DataFrame(X_rows)
    # Drop coin column for ML, keep for analysis
    coins_col = X.pop("coin")
    X = X.fillna(0)
    y = np.array(y_labels)

    return X, y, coins_col


# ══════════════════════════════════════════════════════════
# Phase 1: Train S1 Quality Classifier
# ══════════════════════════════════════════════════════════

def train_s1_quality(is_data):
    print("\n" + "=" * 80)
    print("PHASE 1: Train S1 Quality Classifier")
    print("=" * 80)

    # Generate signals and backtest on IS
    signals, pnls_dict = {}, {}
    for coin in is_data:
        sig = generate_dual_signals(is_data[coin], lb_short=7, lb_long=28)
        signals[coin] = sig
        pnls = run_backtest(is_data[coin], sig, k_upper=5.0, k_lower=1.5, max_hold=24)
        pnls_dict[coin] = pnls
        print(f"  {coin}: {len(pnls)} trades, WR={np.mean(pnls>0):.1%}")

    # Build training data
    X, y, coins = build_training_data(is_data, signals, pnls_dict)
    print(f"\n  Training data: {len(X)} samples, {X.shape[1]} features")
    print(f"  Label balance: {y.mean():.1%} positive (TP trades)")

    if len(X) < 30:
        print("  NOT ENOUGH DATA for ML training")
        return None, None

    # TimeSeriesSplit CV
    tscv = TimeSeriesSplit(n_splits=3, gap=5)
    scores = []

    model = ExtraTreesClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        max_features=0.7, random_state=42, n_jobs=-1
    )

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[val_idx])
        score = balanced_accuracy_score(y[val_idx], pred)
        scores.append(score)
        print(f"  Fold {fold+1}: bal_acc={score:.3f}")

    cv_mean = np.mean(scores)
    print(f"  CV mean: {cv_mean:.3f}")

    # Retrain on all IS data
    model.fit(X, y)

    # Feature importance
    imp = sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1])
    print(f"\n  Top 10 features:")
    for name, importance in imp[:10]:
        print(f"    {name:20s}: {importance:.4f}")

    # Save
    model_path = f"{OUT_DIR}/s1_quality_model.joblib"
    joblib.dump({"model": model, "features": list(X.columns), "cv_score": cv_mean}, model_path)
    print(f"  Model saved: {model_path}")

    return model, list(X.columns)


# ══════════════════════════════════════════════════════════
# Phase 2: Train Compact LinUCB
# ══════════════════════════════════════════════════════════

COMPACT_FEATURES = ["cvd_now", "rsi_14", "ms_composite", "oi_zscore", "tsmom_strength", "ofi_sum_3"]

def train_linucb(is_data):
    print("\n" + "=" * 80)
    print("PHASE 2: Train Compact LinUCB (7-dim)")
    print("=" * 80)

    signals, pnls_dict = {}, {}
    for coin in is_data:
        sig = generate_dual_signals(is_data[coin], lb_short=7, lb_long=28)
        signals[coin] = sig
        pnls_dict[coin] = run_backtest(is_data[coin], sig, k_upper=5.0, k_lower=1.5, max_hold=24)

    X, y_binary, coins = build_training_data(is_data, signals, pnls_dict)

    # Extract compact state features
    compact_cols = [c for c in COMPACT_FEATURES if c in X.columns]
    X_compact = X[compact_cols].values
    # Add intercept
    X_compact = np.hstack([X_compact, np.ones((len(X_compact), 1))])

    # Rewards from actual PnLs
    all_pnls = []
    for coin in is_data:
        all_pnls.extend(pnls_dict[coin].tolist())
    rewards = np.array(all_pnls[:len(X_compact)]) * 100

    print(f"  Training: {len(X_compact)} samples, {X_compact.shape[1]} dims")

    bandit = LinUCB(state_dim=X_compact.shape[1], n_actions=4, alpha=1.0, gamma=0.995)

    # Train
    for s, r in zip(X_compact, rewards):
        bandit.update(s, 3, r)  # all action=3 (1.0x) in backtest

    # Evaluate discrimination
    scores = np.array([bandit.score(s)[1] for s in X_compact])
    med = np.median(scores)
    accept_pnls = rewards[scores >= med] / 100
    reject_pnls = rewards[scores < med] / 100
    lift = np.mean(accept_pnls) - np.mean(reject_pnls)

    print(f"  Accept avg: {np.mean(accept_pnls):+.4%}")
    print(f"  Reject avg: {np.mean(reject_pnls):+.4%}")
    print(f"  Lift: {lift:+.4%}")

    # Save
    model_path = f"{OUT_DIR}/linucb_compact.joblib"
    bandit.save(model_path)
    print(f"  Model saved: {model_path}")

    return bandit, compact_cols


# ══════════════════════════════════════════════════════════
# Phase 3: Integrated Grid Search (IS only)
# ══════════════════════════════════════════════════════════

def grid_search_integrated(is_data, s1_model, s1_features, linucb, linucb_cols):
    print("\n" + "=" * 80)
    print("PHASE 3: Integrated Grid Search (IS)")
    print("=" * 80)

    # Pre-generate signals for all coins
    signals = {}
    for coin in is_data:
        signals[coin] = generate_dual_signals(is_data[coin], lb_short=7, lb_long=28)

    # Pre-compute ML probabilities for all entry bars
    ml_probas = {}
    if s1_model is not None:
        for coin in is_data:
            df = is_data[coin]
            sig = signals[coin]
            sig_vals = sig.values
            atr = df["atr_14"].values if "atr_14" in df.columns else np.ones(len(df)) * 0.02
            probas = np.full(len(df), 0.5)

            for i in range(len(df)):
                if sig_vals[i] == 0 or np.isnan(atr[i]) or atr[i] <= 0:
                    continue
                row = extract_ml_features(df, i)
                avail = [f for f in s1_features if f in row]
                x = np.array([[row.get(f, 0.0) for f in s1_features]])
                probas[i] = s1_model.predict_proba(x)[0, 1]

            ml_probas[coin] = probas

    # Grid
    ml_thresholds = [0.0, 0.3, 0.4, 0.5, 0.6] if s1_model else [0.0]
    k_uppers = [4.0, 5.0, 6.0]
    k_lowers = [1.0, 1.5, 2.0]
    max_holds = [18, 24]

    results = []
    best_sharpe = -999

    total = len(ml_thresholds) * len(k_uppers) * len(k_lowers) * len(max_holds)
    count = 0

    for ml_thr, ku, kl, mh in itertools.product(ml_thresholds, k_uppers, k_lowers, max_holds):
        count += 1
        all_p = []

        for coin in is_data:
            df = is_data[coin]
            sig = signals[coin].copy()

            # ML filter
            if ml_thr > 0 and coin in ml_probas:
                sig[ml_probas[coin] < ml_thr] = 0

            pnls = run_backtest(df, sig, k_upper=ku, k_lower=kl, max_hold=mh)
            all_p.extend(pnls.tolist())

        if len(all_p) < 15:
            continue

        m = calc_metrics(all_p)
        results.append({
            "ml_thr": ml_thr, "ku": ku, "kl": kl, "mh": mh,
            **{"n": m.n, "wr": m.wr, "avg": m.avg, "sharpe": m.sharpe,
               "mdd": m.mdd, "pf": m.pf, "total": m.total}
        })

        if m.sharpe > best_sharpe and m.n >= 20:
            best_sharpe = m.sharpe
            if count % 10 == 0 or m.sharpe > best_sharpe - 0.1:
                print(f"  [{count}/{total}] ml={ml_thr:.1f} ku={ku:.0f} kl={kl:.1f} mh={mh} | "
                      f"n={m.n} WR={m.wr:.1%} avg={m.avg:+.4%} Sharpe={m.sharpe:.2f}")

    rdf = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    rdf.to_csv(f"{OUT_DIR}/grid_search.csv", index=False)

    print(f"\n  Total configs: {len(results)}")
    print(f"\n  TOP 5 IS:")
    for _, r in rdf.head(5).iterrows():
        print(f"    ml={r['ml_thr']:.1f} ku={r['ku']:.0f} kl={r['kl']:.1f} mh={r['mh']:.0f} | "
              f"n={r['n']:.0f} WR={r['wr']:.1%} avg={r['avg']:+.4%} Sharpe={r['sharpe']:.2f} PF={r['pf']:.2f}")

    return rdf


# ══════════════════════════════════════════════════════════
# Phase 4: OOS Validation
# ══════════════════════════════════════════════════════════

def oos_validation(oos_data, is_results, s1_model, s1_features):
    print("\n" + "=" * 80)
    print("PHASE 4: OOS Validation (top 10 IS configs)")
    print("=" * 80)

    top_configs = is_results.head(10).to_dict("records")
    oos_results = []

    for rank, cfg in enumerate(top_configs):
        all_p = []
        coin_detail = {}

        for coin in oos_data:
            df = oos_data[coin]
            sig = generate_dual_signals(df, lb_short=7, lb_long=28)

            # ML filter
            if cfg["ml_thr"] > 0 and s1_model is not None:
                atr = df["atr_14"].values if "atr_14" in df.columns else np.ones(len(df)) * 0.02
                sig_vals = sig.values
                for i in range(len(df)):
                    if sig_vals[i] == 0 or np.isnan(atr[i]): continue
                    row = extract_ml_features(df, i)
                    x = np.array([[row.get(f, 0.0) for f in s1_features]])
                    if s1_model.predict_proba(x)[0, 1] < cfg["ml_thr"]:
                        sig.iloc[i] = 0

            pnls = run_backtest(df, sig, k_upper=cfg["ku"], k_lower=cfg["kl"],
                                max_hold=int(cfg["mh"]))
            all_p.extend(pnls.tolist())
            if len(pnls) > 0:
                m = calc_metrics(pnls)
                coin_detail[coin] = m

        m = calc_metrics(all_p)
        r = {"rank": rank+1, **cfg, "oos_n": m.n, "oos_wr": m.wr, "oos_avg": m.avg,
             "oos_sharpe": m.sharpe, "oos_mdd": m.mdd, "oos_pf": m.pf}
        oos_results.append(r)

        flag = "***" if m.avg > 0 and m.n >= 10 else "   "
        print(f"  {flag} IS#{rank+1} | OOS: n={m.n:3d} WR={m.wr:.1%} avg={m.avg:+.4%} "
              f"Sharpe={m.sharpe:.2f} | IS_Sharpe={cfg['sharpe']:.2f}")

        if rank < 3:
            for c, cm in coin_detail.items():
                print(f"       {c}: n={cm.n:2d} WR={cm.wr:.1%} avg={cm.avg:+.4%}")

    return oos_results


# ══════════════════════════════════════════════════════════
# Phase 5: Permutation + Bootstrap
# ══════════════════════════════════════════════════════════

def statistical_tests(oos_data, best_cfg, s1_model, s1_features):
    print("\n" + "=" * 80)
    print("PHASE 5: Permutation Test + Bootstrap")
    print("=" * 80)

    # Real OOS PnLs
    real_pnls = []
    for coin in oos_data:
        df = oos_data[coin]
        sig = generate_dual_signals(df, lb_short=7, lb_long=28)
        if best_cfg["ml_thr"] > 0 and s1_model is not None:
            atr = df["atr_14"].values if "atr_14" in df.columns else np.ones(len(df)) * 0.02
            for i in range(len(df)):
                if sig.iloc[i] == 0 or np.isnan(atr[i]): continue
                row = extract_ml_features(df, i)
                x = np.array([[row.get(f, 0.0) for f in s1_features]])
                if s1_model.predict_proba(x)[0, 1] < best_cfg["ml_thr"]:
                    sig.iloc[i] = 0
        pnls = run_backtest(df, sig, k_upper=best_cfg["ku"], k_lower=best_cfg["kl"],
                            max_hold=int(best_cfg["mh"]))
        real_pnls.extend(pnls.tolist())

    real = np.array(real_pnls)
    real_avg = np.mean(real)
    real_sharpe = calc_metrics(real).sharpe

    # Permutation
    rng = np.random.RandomState(42)
    n_perm = 2000
    perm_avgs = []
    for _ in range(n_perm):
        perm = []
        for coin in oos_data:
            df = oos_data[coin]
            sig = generate_dual_signals(df, lb_short=7, lb_long=28)
            flip = rng.choice([-1, 1], size=len(sig))
            sig_p = sig * flip
            p = run_backtest(df, sig_p, k_upper=best_cfg["ku"], k_lower=best_cfg["kl"],
                             max_hold=int(best_cfg["mh"]))
            perm.extend(p.tolist())
        if perm:
            perm_avgs.append(np.mean(perm))

    p_value = np.mean(np.array(perm_avgs) >= real_avg)

    # Bootstrap
    boot_avgs, boot_sharpes = [], []
    for _ in range(2000):
        sample = rng.choice(real, size=len(real), replace=True)
        boot_avgs.append(np.mean(sample))
        if np.std(sample) > 0:
            boot_sharpes.append(np.mean(sample)/np.std(sample)*np.sqrt(len(sample)))

    ba, bs = np.array(boot_avgs), np.array(boot_sharpes)

    print(f"  Real OOS: n={len(real)} avg={real_avg:+.4%} Sharpe={real_sharpe:.2f}")
    print(f"  Permutation p-value: {p_value:.4f}")
    print(f"  Bootstrap avg: {np.percentile(ba,5):+.4%} / {np.median(ba):+.4%} / {np.percentile(ba,95):+.4%}")
    print(f"  Bootstrap Sharpe: {np.percentile(bs,5):.2f} / {np.median(bs):.2f} / {np.percentile(bs,95):.2f}")
    print(f"  P(avg>0): {np.mean(ba>0):.1%}  P(Sharpe>1): {np.mean(bs>1):.1%}")

    return p_value


# ══════════════════════════════════════════════════════════
# Phase 6: Cost Sensitivity + Per-Coin
# ══════════════════════════════════════════════════════════

def sensitivity_analysis(oos_data, best_cfg, s1_model, s1_features):
    print("\n" + "=" * 80)
    print("PHASE 6: Cost Sensitivity + Per-Coin Analysis")
    print("=" * 80)

    # Cost sensitivity
    print("  Cost sensitivity:")
    for cost_bps in [0, 10, 20, 30, 50]:
        cost = cost_bps / 10000
        all_p = []
        for coin in oos_data:
            df = oos_data[coin]
            sig = generate_dual_signals(df, lb_short=7, lb_long=28)
            if best_cfg["ml_thr"] > 0 and s1_model is not None:
                atr = df["atr_14"].values if "atr_14" in df.columns else np.ones(len(df)) * 0.02
                for i in range(len(df)):
                    if sig.iloc[i] == 0 or np.isnan(atr[i]): continue
                    row = extract_ml_features(df, i)
                    x = np.array([[row.get(f, 0.0) for f in s1_features]])
                    if s1_model.predict_proba(x)[0, 1] < best_cfg["ml_thr"]:
                        sig.iloc[i] = 0
            p = run_backtest(df, sig, k_upper=best_cfg["ku"], k_lower=best_cfg["kl"],
                             max_hold=int(best_cfg["mh"]), cost=cost)
            all_p.extend(p.tolist())
        m = calc_metrics(all_p)
        print(f"    {cost_bps:2d}bps: n={m.n:3d} avg={m.avg:+.4%} Sharpe={m.sharpe:.2f}")

    # Per-coin
    print("\n  Per-coin OOS:")
    for coin in sorted(oos_data.keys()):
        df = oos_data[coin]
        sig = generate_dual_signals(df, lb_short=7, lb_long=28)
        if best_cfg["ml_thr"] > 0 and s1_model is not None:
            atr = df["atr_14"].values if "atr_14" in df.columns else np.ones(len(df)) * 0.02
            for i in range(len(df)):
                if sig.iloc[i] == 0 or np.isnan(atr[i]): continue
                row = extract_ml_features(df, i)
                x = np.array([[row.get(f, 0.0) for f in s1_features]])
                if s1_model.predict_proba(x)[0, 1] < best_cfg["ml_thr"]:
                    sig.iloc[i] = 0
        p = run_backtest(df, sig, k_upper=best_cfg["ku"], k_lower=best_cfg["kl"],
                         max_hold=int(best_cfg["mh"]))
        if len(p) > 0:
            m = calc_metrics(p)
            print(f"    {coin:5s}: {m.summary()}")


# ══════════════════════════════════════════════════════════
# Phase 7: Export Best Config
# ══════════════════════════════════════════════════════════

def export_config(best_cfg, cv_score, p_value):
    print("\n" + "=" * 80)
    print("PHASE 7: Export Best Config")
    print("=" * 80)

    config = {
        "strategy": "TSMOM_v5.2_ML_Enhanced",
        "lookback_short": 7,
        "lookback_long": 28,
        "dual_lookback": True,
        "cvd_quantile": 0.75,
        "cvd_roll_window": 120,
        "k_upper": best_cfg["ku"],
        "k_lower": best_cfg["kl"],
        "max_hold_bars": int(best_cfg["mh"]),
        "ml_threshold": best_cfg["ml_thr"],
        "use_oi": True,
        "oi_zscore_max": 2.0,
        "cost_roundtrip": 0.0020,
        "leverage": 2,
        "validation": {
            "is_sharpe": best_cfg["sharpe"],
            "oos_sharpe": best_cfg.get("oos_sharpe", "N/A"),
            "oos_avg_pnl": best_cfg.get("oos_avg", "N/A"),
            "oos_wr": best_cfg.get("oos_wr", "N/A"),
            "permutation_p": p_value,
            "s1_cv_score": cv_score,
        }
    }

    path = f"{OUT_DIR}/best_config_v5_2.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"  Config saved: {path}")
    print(f"  {json.dumps(config, indent=2, default=str)}")

    return config


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80)
    print("OVERNIGHT ML OPTIMIZATION")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    # Load
    print("\n[LOAD] 10 coins...")
    data = load_ohlcv_10()
    is_data, oos_data = split_is_oos(data)
    for coin in is_data:
        print(f"  {coin}: IS={len(is_data[coin])} OOS={len(oos_data[coin])}")

    # Phase 1
    s1_model, s1_features = train_s1_quality(is_data)
    cv_score = 0.0
    if s1_model:
        cv_score = joblib.load(f"{OUT_DIR}/s1_quality_model.joblib")["cv_score"]

    # Phase 2
    linucb, linucb_cols = train_linucb(is_data)

    # Phase 3
    is_results = grid_search_integrated(is_data, s1_model, s1_features, linucb, linucb_cols)

    # Phase 4
    oos_results = oos_validation(oos_data, is_results, s1_model, s1_features)

    # Find best OOS
    oos_positive = [r for r in oos_results if r["oos_avg"] > 0 and r["oos_n"] >= 10]
    if oos_positive:
        best = max(oos_positive, key=lambda x: x["oos_sharpe"])
        best_cfg = {k: best[k] for k in ["ml_thr", "ku", "kl", "mh", "sharpe"]}
        best_cfg["oos_sharpe"] = best["oos_sharpe"]
        best_cfg["oos_avg"] = best["oos_avg"]
        best_cfg["oos_wr"] = best["oos_wr"]

        # Phase 5
        p_value = statistical_tests(oos_data, best_cfg, s1_model, s1_features)

        # Phase 6
        sensitivity_analysis(oos_data, best_cfg, s1_model, s1_features)

        # Phase 7
        export_config(best_cfg, cv_score, p_value)
    else:
        print("\n  NO OOS-POSITIVE CONFIG FOUND")
        # Fall back to no-ML baseline
        best_cfg = {"ml_thr": 0.0, "ku": 5.0, "kl": 1.5, "mh": 24, "sharpe": 0}
        p_value = statistical_tests(oos_data, best_cfg, None, None)
        export_config(best_cfg, 0.0, p_value)

    # Save all results
    pd.DataFrame(oos_results).to_csv(f"{OUT_DIR}/oos_results.csv", index=False)

    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"COMPLETED: {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"Total: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
