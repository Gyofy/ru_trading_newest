"""Strategy Diagnosis -- 4-step immediate improvements.

Step 1: Feature importance decomposition (MI top 80 -> actual top 10)
Step 2: Regime filter activation (UNKNOWN -> 4-state, per-regime EV)
Step 3: S1 threshold stability test (0.40~0.65 performance curve)
Step 4: Direction bias measurement (LONG/SHORT ratio + warning)

Frozen params. OOS 42 days. Trade-level simulation.
"""

import sys
sys.path.insert(0, "C:/Users/RJ/Desktop/CLAUDE_CRYPTO_AGENT")
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import json
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

from src.data.crawlers.crypto_ohlcv import fetch_all_top10
from src.data.crawlers.macro_commodity_crawler import crawl_all_macro_data
from src.models.masking_loop import (
    create_labels_triple_barrier, compute_extended_metrics,
    LABEL_MAP, HORIZONS, STAGE1_NAMES, STAGE2_NAMES,
)
from src.models.enhanced_ensemble import EnhancedEnsemble
from src.models.regime_filter import RegimeFilter, Regime4
from src.evaluation.trade_level_ev import compute_trade_level_ev
from src.execution.cost_model import CostModel, FeeSchedule, FundingConfig, MissFillConfig
from src.utils.config import bar_minutes as cfg_bar_minutes
from src.utils.feature_policy import is_excluded_feature, is_blocked_regime
from sklearn.feature_selection import mutual_info_classif

REPORT_DIR = Path("data/reports/strategy_diagnosis")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BM = cfg_bar_minutes()
MAX_HORIZON = max(HORIZONS)
PURGE_BARS = MAX_HORIZON * 2
EMBARGO_BARS = 6
OOS_DAYS = 42
RISK_FRAC = 0.005
N_JOBS = 6

ACTIVE_COINS = ["XRP", "DOT", "ADA"]

COST_MODEL = CostModel(
    fee_schedule=FeeSchedule(
        maker_fee=0.0002, taker_fee=0.00055,
        slippage_entry=0.0003, slippage_exit_limit=0.0001,
        slippage_exit_market=0.0005,
    ),
    funding_config=FundingConfig(interval_hours=8.0, default_rate=0.0001),
    miss_fill_config=MissFillConfig(reject_prob=0.15, missed_ev_pct=0.0015),
)

FROZEN_PARAMS = {
    "XRP": {
        "k_upper": 3.0, "k_lower": 0.6, "stage1_threshold": 0.6,
        "max_features": 120, "num_leaves": 47, "learning_rate": 0.02,
        "n_estimators": 100, "max_depth_tree": 6, "subsample": 0.8,
        "colsample": 0.6, "min_child_samples": 30,
    },
    "ADA": {
        "k_upper": 3.0, "k_lower": 0.6, "stage1_threshold": 0.5,
        "max_features": 120, "num_leaves": 15, "learning_rate": 0.02,
        "n_estimators": 400, "max_depth_tree": 8, "subsample": 0.7,
        "colsample": 0.9, "min_child_samples": 5,
    },
    "DOT": {
        "k_upper": 3.0, "k_lower": 0.6, "stage1_threshold": 0.5,
        "max_features": 80, "num_leaves": 47, "learning_rate": 0.1,
        "n_estimators": 300, "max_depth_tree": 8, "subsample": 0.7,
        "colsample": 0.8, "min_child_samples": 10,
    },
}

REGIME_FILTER = RegimeFilter()


# ==================== Data ====================

def fetch_data():
    print(f"\n{'='*70}")
    print(f"  DATA COLLECTION")
    print(f"{'='*70}")
    ohlcv = fetch_all_top10("365d", "1h")
    first_coin = list(ohlcv.values())[0]
    macro = crawl_all_macro_data(first_coin.index)
    macro_aligned = macro.get("aligned", pd.DataFrame())

    feature_data = {}
    for coin in ACTIVE_COINS:
        if coin not in ohlcv:
            continue
        df = ohlcv[coin].copy()
        if len(macro_aligned) > 0:
            for col in macro_aligned.columns:
                df[col] = macro_aligned[col].reindex(df.index).ffill().bfill().fillna(0)
        feature_data[coin] = df
        print(f"  {coin}: {len(df)} bars, {len(df.columns)} features")
    return feature_data


def split_oos(df):
    idx = df.index
    oos_start = idx[-1] - timedelta(days=OOS_DAYS)
    purge_td = timedelta(hours=(PURGE_BARS + EMBARGO_BARS) * (BM // 60))
    train_end = oos_start - purge_td
    return df[idx <= train_end], df[idx >= oos_start]


def prepare_features(train_df, params):
    """Label + feature select on train. Return feature_cols."""
    h = HORIZONS[-1]
    h_label = f"label_{h*BM}min"

    labeled = create_labels_triple_barrier(
        train_df.copy(), h,
        k_upper_override=params["k_upper"],
        k_lower_override=params["k_lower"], verbose=False)

    if h_label not in labeled.columns:
        return None, None, None

    exclude = {"label", "future_return", "open", "high", "low", "close", "volume"}
    for hh in HORIZONS:
        exclude.add(f"label_{hh*BM}min")
        exclude.add(f"return_{hh*BM}min")
    leak_kw = ["future", "target", "label_", "return_", "fwd_", "forward_"]

    feature_cols = [c for c in labeled.columns
                    if c not in exclude
                    and not any(kw in c.lower() for kw in leak_kw)
                    and labeled[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]

    clean = labeled.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    X = clean[feature_cols].fillna(0).values
    y = clean[h_label].fillna(1).values.astype(int)

    return feature_cols, X, y


def train_models(X, y, feature_cols, params, max_features=None):
    """Train S1 + S2 models. Return models + selected feature indices."""
    # MI selection
    if max_features and len(feature_cols) > max_features:
        n_mi = min(2000, len(X))
        mi = mutual_info_classif(X[:n_mi], y[:n_mi],
                                 discrete_features=False, random_state=42, n_neighbors=5)
        top_idx = np.argsort(mi)[-max_features:]
        selected_cols = [feature_cols[i] for i in sorted(top_idx)]
        mi_scores = {feature_cols[i]: mi[i] for i in range(len(feature_cols))}
        X_sel = X[:, sorted(top_idx)]
    else:
        selected_cols = feature_cols
        mi_scores = {}
        X_sel = X
        n_mi = min(2000, len(X))
        mi = mutual_info_classif(X[:n_mi], y[:n_mi],
                                 discrete_features=False, random_state=42, n_neighbors=5)
        mi_scores = {feature_cols[i]: mi[i] for i in range(len(feature_cols))}

    # S1
    y_s1 = (y != LABEL_MAP["HOLD"]).astype(int)
    if len(np.unique(y_s1)) < 2:
        return None, None, None, None

    s1_counts = np.bincount(y_s1, minlength=2)
    s1_sw = np.where(s1_counts > 0, len(y_s1) / (2 * s1_counts + 1e-10), 1.0)[y_s1]
    s1 = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
    s1.fit(X_sel, y_s1, sample_weight=s1_sw)

    # S2
    trade_mask = y != LABEL_MAP["HOLD"]
    s2 = None
    if trade_mask.sum() >= 30:
        y_s2 = (y[trade_mask] == LABEL_MAP["UP"]).astype(int)
        if len(np.unique(y_s2)) >= 2:
            s2_counts = np.bincount(y_s2, minlength=2)
            s2_sw = np.where(s2_counts > 0, len(y_s2) / (2 * s2_counts + 1e-10), 1.0)[y_s2]
            s2 = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
            s2.fit(X_sel[trade_mask], y_s2, sample_weight=s2_sw)

    return s1, s2, selected_cols, mi_scores


def predict_oos(s1_model, s2_model, oos_df, selected_cols, threshold):
    """Predict on OOS data."""
    oos_clean = oos_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    X_oos = oos_clean[selected_cols].fillna(0).values

    s1_probs = s1_model.predict_proba(X_oos)
    s1_pred = (s1_probs[:, 1] >= threshold).astype(int)

    s2_pred = np.zeros(len(X_oos), dtype=int)
    s2_prob = np.full(len(X_oos), 0.5)
    if s2_model is not None:
        s2_probs = s2_model.predict_proba(X_oos)
        s2_pred = np.argmax(s2_probs, axis=1)
        s2_prob = s2_probs[:, 1]

    return {
        "s1_pred": s1_pred, "s1_prob": s1_probs[:, 1],
        "s2_pred": s2_pred, "s2_prob": s2_prob,
    }


# ==================== Step 1: Feature Importance ====================

def step1_feature_importance(feature_data):
    print(f"\n{'='*70}")
    print(f"  STEP 1: FEATURE IMPORTANCE DECOMPOSITION")
    print(f"{'='*70}")

    results = {}
    for coin in ACTIVE_COINS:
        if coin not in feature_data:
            continue
        params = FROZEN_PARAMS[coin]
        train_df, _ = split_oos(feature_data[coin])
        feature_cols, X, y = prepare_features(train_df, params)
        if feature_cols is None:
            continue

        # Full MI scores (all features, not just top N)
        n_mi = min(2000, len(X))
        mi = mutual_info_classif(X[:n_mi], y[:n_mi],
                                 discrete_features=False, random_state=42, n_neighbors=5)
        mi_ranked = sorted(zip(feature_cols, mi), key=lambda x: -x[1])

        top10 = mi_ranked[:10]
        bottom10 = mi_ranked[-10:]

        print(f"\n  {coin} -- Top 10 features (MI score):")
        for i, (name, score) in enumerate(top10):
            print(f"    {i+1:2d}. {name:40s} MI={score:.4f}")

        print(f"\n  {coin} -- Bottom 10 features (likely noise):")
        for i, (name, score) in enumerate(bottom10):
            print(f"    {i+1:2d}. {name:40s} MI={score:.6f}")

        # Feature category breakdown
        categories = {
            "technical": ["rsi", "macd", "bb_", "ema_", "sma_", "atr", "adx", "stoch",
                         "obv", "mfi", "cmf", "ichimoku", "cci", "williams"],
            "wavelet": ["wavelet", "dwt_"],
            "fft": ["fft_"],
            "hilbert": ["hilbert", "inst_"],
            "entropy": ["entropy", "sample_ent"],
            "hurst_acf": ["hurst", "acf_"],
            "microstructure": ["amihud", "kyle", "spread", "vpin", "roll_"],
            "cusum": ["cusum"],
            "multi_tf": ["mtf_", "_12h", "_24h", "_48h"],
            "macro": ["gold", "vix", "dxy", "us10y", "sp500", "fear", "tvl"],
        }

        cat_mi = {}
        for cat, keywords in categories.items():
            cat_features = [(n, s) for n, s in mi_ranked
                            if any(kw in n.lower() for kw in keywords)]
            if cat_features:
                avg_mi = np.mean([s for _, s in cat_features])
                cat_mi[cat] = {"count": len(cat_features), "avg_mi": round(avg_mi, 4),
                               "total_mi": round(sum(s for _, s in cat_features), 4)}

        print(f"\n  {coin} -- Category MI breakdown:")
        for cat, info in sorted(cat_mi.items(), key=lambda x: -x[1]["avg_mi"]):
            print(f"    {cat:20s}: {info['count']:3d} features, avg_MI={info['avg_mi']:.4f}, "
                  f"total_MI={info['total_mi']:.4f}")

        results[coin] = {
            "top10": [(n, round(s, 4)) for n, s in top10],
            "bottom10": [(n, round(s, 6)) for n, s in bottom10],
            "category_mi": cat_mi,
            "total_features": len(feature_cols),
        }

    return results


# ==================== Step 2: Regime Filter ====================

def step2_regime_analysis(feature_data):
    print(f"\n{'='*70}")
    print(f"  STEP 2: REGIME FILTER ACTIVATION")
    print(f"{'='*70}")

    results = {}
    for coin in ACTIVE_COINS:
        if coin not in feature_data:
            continue
        params = FROZEN_PARAMS[coin]
        train_df, oos_df = split_oos(feature_data[coin])

        # Classify regimes on OOS
        try:
            regimes = REGIME_FILTER.classify_series(oos_df)
        except Exception as e:
            print(f"  {coin}: Regime classification failed: {e}")
            # Manual regime calculation
            regimes = _manual_regime(oos_df)

        regime_counts = Counter(regimes) if regimes is not None else {}
        print(f"\n  {coin} OOS regime distribution:")
        for r, cnt in sorted(regime_counts.items(), key=lambda x: -x[1]):
            print(f"    {str(r):20s}: {cnt} bars ({cnt/len(oos_df)*100:.1f}%)")

        # Train model + predict OOS
        feature_cols, X, y = prepare_features(train_df, params)
        if feature_cols is None:
            continue

        s1, s2, sel_cols, _ = train_models(X, y, feature_cols, params, params["max_features"])
        if s1 is None:
            continue

        preds = predict_oos(s1, s2, oos_df, sel_cols, params["stage1_threshold"])

        # Per-regime trade-level EV
        regime_ev = {}
        if regimes is not None and len(regimes) == len(oos_df):
            for regime_val in set(regimes):
                mask = np.array([r == regime_val for r in regimes])
                if mask.sum() < 5:
                    continue

                regime_oos = oos_df[mask]
                regime_preds = {
                    "s1_pred": preds["s1_pred"][mask],
                    "s1_prob": preds["s1_prob"][mask],
                    "s2_pred": preds["s2_pred"][mask],
                    "s2_prob": preds["s2_prob"][mask],
                }

                ev = compute_trade_level_ev(
                    regime_oos, regime_preds["s1_pred"], regime_preds["s1_prob"],
                    regime_preds["s2_pred"], regime_preds["s2_prob"],
                    k_upper=params["k_upper"], k_lower=params["k_lower"],
                    risk_frac=RISK_FRAC, cost_model=COST_MODEL,
                )

                regime_ev[str(regime_val)] = {
                    "bars": int(mask.sum()),
                    "trades": ev["trade_count"],
                    "avg_net_pnl": ev["avg_net_pnl"],
                    "total_net_pnl": ev["total_net_pnl"],
                    "win_rate": ev["win_rate"],
                }

                print(f"    {str(regime_val):20s}: {ev['trade_count']} trades, "
                      f"avg={ev['avg_net_pnl']:+.4%}, win={ev['win_rate']:.1%}")

        results[coin] = {"regime_distribution": dict(regime_counts), "regime_ev": regime_ev}

    return results


def _manual_regime(df):
    """Fallback regime classification when RegimeFilter fails."""
    close = df["close"].values
    n = len(close)
    regimes = []

    for i in range(n):
        lookback = min(i + 1, 42)
        segment = close[max(0, i - lookback + 1):i + 1]

        if len(segment) < 10:
            regimes.append("UNKNOWN")
            continue

        # Simple regime: EMA trend + volatility
        ema_fast = pd.Series(segment).ewm(span=10).mean().iloc[-1]
        ema_slow = pd.Series(segment).ewm(span=30).mean().iloc[-1]
        ret_std = np.std(np.diff(segment) / segment[:-1]) if len(segment) > 1 else 0
        median_std = 0.02  # typical 4h crypto vol

        if ema_fast > ema_slow * 1.005:
            if ret_std > median_std:
                regimes.append("TREND_UP")
            else:
                regimes.append("TREND_UP")
        elif ema_fast < ema_slow * 0.995:
            if ret_std > median_std:
                regimes.append("TREND_DOWN")
            else:
                regimes.append("TREND_DOWN")
        else:
            if ret_std > median_std:
                regimes.append("RANGE_HIGH")
            else:
                regimes.append("RANGE_LOW")

    return regimes


# ==================== Step 3: Threshold Stability ====================

def step3_threshold_stability(feature_data):
    print(f"\n{'='*70}")
    print(f"  STEP 3: S1 THRESHOLD STABILITY TEST")
    print(f"{'='*70}")

    thresholds = [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65]
    results = {}

    for coin in ACTIVE_COINS:
        if coin not in feature_data:
            continue
        params = FROZEN_PARAMS[coin]
        train_df, oos_df = split_oos(feature_data[coin])

        feature_cols, X, y = prepare_features(train_df, params)
        if feature_cols is None:
            continue

        s1, s2, sel_cols, _ = train_models(X, y, feature_cols, params, params["max_features"])
        if s1 is None:
            continue

        print(f"\n  {coin} -- Threshold sweep:")
        print(f"  {'th':>6s} | {'trades':>6s} | {'avg_pnl':>10s} | {'total_pnl':>10s} | "
              f"{'win%':>6s} | {'L/S':>8s} | {'verdict':>10s}")
        print(f"  {'-'*70}")

        coin_results = []
        for th in thresholds:
            preds = predict_oos(s1, s2, oos_df, sel_cols, th)

            ev = compute_trade_level_ev(
                oos_df, preds["s1_pred"], preds["s1_prob"],
                preds["s2_pred"], preds["s2_prob"],
                k_upper=params["k_upper"], k_lower=params["k_lower"],
                risk_frac=RISK_FRAC, cost_model=COST_MODEL,
            )

            # Direction ratio
            if ev["trade_count"] > 0:
                long_count = sum(1 for i in range(len(preds["s1_pred"]))
                                 if preds["s1_pred"][i] == 1 and preds["s2_pred"][i] == 1)
                short_count = sum(1 for i in range(len(preds["s1_pred"]))
                                  if preds["s1_pred"][i] == 1 and preds["s2_pred"][i] == 0)
                ls_ratio = f"{long_count}L/{short_count}S"
            else:
                ls_ratio = "-"

            marker = " <-- frozen" if abs(th - params["stage1_threshold"]) < 0.01 else ""
            verdict = "PASS" if ev["avg_net_pnl"] > 0 and ev["trade_count"] >= 5 else \
                      "few" if ev["trade_count"] < 5 else "FAIL"

            print(f"  {th:6.2f} | {ev['trade_count']:6d} | {ev['avg_net_pnl']:+10.4%} | "
                  f"{ev['total_net_pnl']:+10.4%} | {ev['win_rate']:6.1%} | "
                  f"{ls_ratio:>8s} | {verdict:>10s}{marker}")

            coin_results.append({
                "threshold": th,
                "trade_count": ev["trade_count"],
                "avg_net_pnl": ev["avg_net_pnl"],
                "total_net_pnl": ev["total_net_pnl"],
                "win_rate": ev["win_rate"],
                "ls_ratio": ls_ratio,
                "is_frozen": abs(th - params["stage1_threshold"]) < 0.01,
            })

        results[coin] = coin_results

    return results


# ==================== Step 4: Direction Bias ====================

def step4_direction_bias(feature_data):
    print(f"\n{'='*70}")
    print(f"  STEP 4: DIRECTION BIAS MEASUREMENT")
    print(f"{'='*70}")

    results = {}
    for coin in ACTIVE_COINS:
        if coin not in feature_data:
            continue
        params = FROZEN_PARAMS[coin]
        train_df, oos_df = split_oos(feature_data[coin])

        feature_cols, X, y = prepare_features(train_df, params)
        if feature_cols is None:
            continue

        s1, s2, sel_cols, _ = train_models(X, y, feature_cols, params, params["max_features"])
        if s1 is None:
            continue

        preds = predict_oos(s1, s2, oos_df, sel_cols, params["stage1_threshold"])

        # Signal direction analysis
        trade_mask = preds["s1_pred"] == 1
        n_signals = trade_mask.sum()

        if n_signals == 0:
            print(f"\n  {coin}: 0 signals (threshold too high)")
            results[coin] = {"signals": 0, "warning": "NO_SIGNALS"}
            continue

        long_mask = trade_mask & (preds["s2_pred"] == 1)
        short_mask = trade_mask & (preds["s2_pred"] == 0)
        n_long = long_mask.sum()
        n_short = short_mask.sum()
        bias_ratio = n_long / (n_short + 1e-10)

        # Market direction in OOS
        oos_close = oos_df["close"].values
        market_return = (oos_close[-1] - oos_close[0]) / oos_close[0]
        market_dir = "UP" if market_return > 0.02 else "DOWN" if market_return < -0.02 else "FLAT"

        # Time-windowed bias (split OOS into 3 periods)
        n_bars = len(oos_df)
        period_size = n_bars // 3
        period_bias = []
        for p in range(3):
            start = p * period_size
            end = (p + 1) * period_size if p < 2 else n_bars
            p_long = long_mask[start:end].sum()
            p_short = short_mask[start:end].sum()
            p_total = trade_mask[start:end].sum()
            period_bias.append({
                "period": f"P{p+1}",
                "bars": end - start,
                "signals": int(p_total),
                "long": int(p_long),
                "short": int(p_short),
                "bias": round(p_long / (p_short + 1e-10), 2),
            })

        # Warning logic
        warning = None
        if bias_ratio > 3.0:
            warning = f"STRONG LONG BIAS ({bias_ratio:.1f}x)"
        elif bias_ratio < 0.33:
            warning = f"STRONG SHORT BIAS ({1/bias_ratio:.1f}x)"
        elif bias_ratio > 2.0:
            warning = f"MODERATE LONG BIAS ({bias_ratio:.1f}x)"
        elif bias_ratio < 0.5:
            warning = f"MODERATE SHORT BIAS ({1/bias_ratio:.1f}x)"

        print(f"\n  {coin}:")
        print(f"    Signals: {n_signals} (Long: {n_long}, Short: {n_short})")
        print(f"    Bias ratio: {bias_ratio:.2f} (1.0 = balanced)")
        print(f"    Market return OOS: {market_return:+.1%} ({market_dir})")
        if warning:
            print(f"    [WARNING] {warning}")
        else:
            print(f"    Direction: BALANCED")

        print(f"    Time-windowed bias:")
        for pb in period_bias:
            print(f"      {pb['period']}: {pb['signals']} signals "
                  f"(L:{pb['long']} S:{pb['short']} bias:{pb['bias']})")

        results[coin] = {
            "signals": int(n_signals),
            "long": int(n_long),
            "short": int(n_short),
            "bias_ratio": round(bias_ratio, 2),
            "market_return": round(market_return, 4),
            "market_direction": market_dir,
            "warning": warning,
            "period_bias": period_bias,
        }

    return results


# ==================== MAIN ====================

def main():
    start = datetime.now()
    print(f"\n{'='*70}")
    print(f"  STRATEGY DIAGNOSIS -- 4 Steps")
    print(f"  {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    feature_data = fetch_data()

    step1_results = step1_feature_importance(feature_data)
    step2_results = step2_regime_analysis(feature_data)
    step3_results = step3_threshold_stability(feature_data)
    step4_results = step4_direction_bias(feature_data)

    # Save all
    full_report = {
        "generated_at": datetime.now().isoformat(),
        "step1_feature_importance": step1_results,
        "step2_regime_analysis": step2_results,
        "step3_threshold_stability": step3_results,
        "step4_direction_bias": step4_results,
    }

    with open(REPORT_DIR / "strategy_diagnosis.json", "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    elapsed = (datetime.now() - start).total_seconds() / 60
    print(f"\n{'='*70}")
    print(f"  DIAGNOSIS COMPLETE ({elapsed:.1f} min)")
    print(f"  Report: {REPORT_DIR}/strategy_diagnosis.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
