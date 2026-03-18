"""TabPFN A/B Test -- 기존 7-model ensemble vs 7-model+TabPFN(8th) vs TabPFN standalone.

v3.4 파라미터 동결. 모델 구조만 비교.
ARM A: 기존 EnhancedEnsemble (7-model + stacking)
ARM B: EnhancedEnsemble + TabPFN (8th model, soft voting)
ARM C: TabPFN standalone

OOS: 8주, trade-level barrier simulation.
"""

import sys
sys.path.insert(0, "C:/Users/RJ/Desktop/CLAUDE_CRYPTO_AGENT")
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import json, yaml, warnings, time, traceback
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

from tabpfn import TabPFNClassifier
from src.data.crawlers.crypto_ohlcv import fetch_all_top10
from src.data.crawlers.macro_commodity_crawler import crawl_all_macro_data
from src.models.masking_loop import create_labels_triple_barrier, LABEL_MAP, HORIZONS
from src.models.enhanced_ensemble import EnhancedEnsemble
from src.evaluation.trade_level_ev import compute_trade_level_ev
from src.execution.cost_model import CostModel, FeeSchedule, FundingConfig, MissFillConfig
from src.utils.config import bar_minutes as cfg_bar_minutes
from sklearn.feature_selection import mutual_info_classif

REPORT_DIR = Path("experiments/tabpfn_test/results")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open("config/frozen_params_v3_4.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

COMMON = CFG["common"]
COINS = CFG["coins"]
CC = CFG["cost_model"]
EXK = CFG["excluded_feature_keywords"]
BM = cfg_bar_minutes()
MH = COMMON["max_horizon"]
RF = COMMON["risk_frac"]
N_JOBS = 6
OOS_DAYS = 56

CM = CostModel(
    fee_schedule=FeeSchedule(maker_fee=CC["maker_fee"], taker_fee=CC["taker_fee"],
        slippage_entry=CC["slippage_entry"], slippage_exit_limit=CC["slippage_exit_limit"],
        slippage_exit_market=CC["slippage_exit_market"]),
    funding_config=FundingConfig(interval_hours=CC["funding_interval_hours"],
        default_rate=CC["funding_default_rate"]),
    miss_fill_config=MissFillConfig(reject_prob=CC["miss_fill_reject_prob"],
        missed_ev_pct=CC["miss_fill_missed_ev"]))


def ck(coin, key):
    return COINS[coin].get(f"{key}_override", COMMON[key])

def isex(c):
    return any(kw in c.lower() for kw in EXK)


def prepare_data():
    """Fetch and prepare data."""
    print("  Fetching data...")
    ohlcv = fetch_all_top10("365d", "1h")
    first = list(ohlcv.values())[0]
    macro = crawl_all_macro_data(first.index)
    ma = macro.get("aligned", pd.DataFrame())

    data = {}
    for coin in COINS:
        if coin not in ohlcv: continue
        df = ohlcv[coin].copy()
        if len(ma) > 0:
            for col in ma.columns:
                df[col] = ma[col].reindex(df.index).ffill().bfill().fillna(0)
        data[coin] = df
    return data


def prepare_features(train_df, coin):
    """Label + feature select."""
    params = COINS[coin]
    h = HORIZONS[-1]
    hl = f"label_{h*BM}min"
    ku, kl = ck(coin, "k_upper"), ck(coin, "k_lower")

    labeled = create_labels_triple_barrier(
        train_df.copy(), h, k_upper_override=ku, k_lower_override=kl, verbose=False)
    if hl not in labeled.columns:
        return None, None, None, None

    exclude = {"label", "future_return", "open", "high", "low", "close", "volume"}
    for hh in HORIZONS:
        exclude.add(f"label_{hh*BM}min"); exclude.add(f"return_{hh*BM}min")
    lk = ["future", "target", "label_", "return_", "fwd_", "forward_"]
    fcols = [c for c in labeled.columns
             if c not in exclude and not any(k in c.lower() for k in lk)
             and not isex(c) and labeled[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]

    mf = params["max_features"]
    clean = labeled.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    X = clean[fcols].fillna(0).values
    y = clean[hl].fillna(1).values.astype(int)

    if len(fcols) > mf:
        mi = mutual_info_classif(X[:min(2000, len(X))], y[:min(2000, len(X))],
                                 discrete_features=False, random_state=42, n_neighbors=5)
        top = np.argsort(mi)[-mf:]
        fcols = [fcols[i] for i in sorted(top)]
        X = clean[fcols].fillna(0).values

    return fcols, X, y, clean


def evaluate_arm(oos_df, s1_pred, s2_pred, coin, label):
    """Trade-level EV."""
    params = COINS[coin]
    ku, kl = ck(coin, "k_upper"), ck(coin, "k_lower")
    blocked = COINS[coin].get("blocked_regimes_override", CFG["blocked_regimes"])

    # Simple regime for blocking
    n = len(oos_df)
    regime_mask = np.ones(n, dtype=bool)
    close = oos_df["close"].values
    for i in range(n):
        if i < 42: continue
        seg = close[max(0, i-41):i+1]
        ef = pd.Series(seg).ewm(span=10).mean().iloc[-1]
        es = pd.Series(seg).ewm(span=30).mean().iloc[-1]
        rs = np.std(np.diff(seg)/seg[:-1]) if len(seg)>1 else 0
        if abs(ef/es - 1) < 0.005 and rs < 0.02:
            regime_mask[i] = False  # RANGE_LOW
        if "UNKNOWN" in blocked and i < 42:
            regime_mask[i] = False

    # Apply regime block to s1_pred
    s1_filtered = s1_pred.copy()
    s1_filtered[~regime_mask] = 0

    ev = compute_trade_level_ev(
        oos_df, s1_filtered, np.ones(n) * 0.5,
        s2_pred, np.ones(n) * 0.5,
        k_upper=ku, k_lower=kl, max_hold=MH,
        risk_frac=RF, cost_model=CM)

    return {
        "label": label, "coin": coin,
        "trades": ev["trade_count"],
        "avg_pnl": ev["avg_net_pnl"],
        "total_pnl": ev["total_net_pnl"],
        "win_rate": ev["win_rate"],
        "max_dd": ev["max_dd"],
        "score": ev["score"],
    }


def main():
    start = datetime.now()
    print(f"\n{'='*70}")
    print(f"  TabPFN A/B TEST")
    print(f"  {start.strftime('%Y-%m-%d %H:%M')}")
    print(f"  ARM A: EnhancedEnsemble (7-model + stacking)")
    print(f"  ARM B: Ensemble + TabPFN (soft voting)")
    print(f"  ARM C: TabPFN standalone")
    print(f"{'='*70}")

    data = prepare_data()
    results = {}

    for coin in COINS:
        if coin not in data: continue

        print(f"\n{'='*50}")
        print(f"  {coin}")
        print(f"{'='*50}")

        df = data[coin]
        idx = df.index; end = idx[-1]
        oos_start = end - timedelta(days=OOS_DAYS)
        purge = timedelta(hours=(MH*2+6)*(BM//60))
        train_df = df[idx <= oos_start - purge]
        oos_df = df[idx >= oos_start]

        params = COINS[coin]
        fcols, X_train, y_train, clean = prepare_features(train_df, coin)
        if fcols is None: continue

        oc = oos_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        X_oos = oc[fcols].fillna(0).values

        # Stage 1 labels
        y_s1 = (y_train != LABEL_MAP["HOLD"]).astype(int)
        if len(np.unique(y_s1)) < 2: continue
        s1c = np.bincount(y_s1, minlength=2)
        s1w = np.where(s1c > 0, len(y_s1) / (2 * s1c + 1e-10), 1.0)[y_s1]

        # Stage 2 labels
        tm = y_train != LABEL_MAP["HOLD"]
        y_s2 = (y_train[tm] == LABEL_MAP["UP"]).astype(int)
        s2c = np.bincount(y_s2, minlength=2)
        s2w = np.where(s2c > 0, len(y_s2) / (2 * s2c + 1e-10), 1.0)[y_s2]

        # ==================== ARM A: Baseline ====================
        print(f"\n  ARM A: EnhancedEnsemble (baseline)")
        t0 = time.time()

        s1_a = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
        s1_a.fit(X_train, y_s1, sample_weight=s1w)
        s1p_a = s1_a.predict_proba(X_oos)
        s1pred_a = (s1p_a[:, 1] >= params["stage1_threshold"]).astype(int)

        s2_a = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
        s2_a.fit(X_train[tm], y_s2, sample_weight=s2w)
        s2p_a = s2_a.predict_proba(X_oos)
        s2pred_a = np.argmax(s2p_a, axis=1)

        t_a = time.time() - t0
        ev_a = evaluate_arm(oos_df, s1pred_a, s2pred_a, coin, "A_ensemble")
        print(f"    Time: {t_a:.1f}s | {ev_a['trades']}T avg={ev_a['avg_pnl']:+.4%} "
              f"total={ev_a['total_pnl']:+.4%} dd={ev_a['max_dd']:.4%}")

        # ==================== ARM C: TabPFN Standalone ====================
        print(f"\n  ARM C: TabPFN standalone")
        t0 = time.time()

        try:
            # TabPFN has sample limit -- subsample if needed
            max_train = min(len(X_train), 10000)
            if max_train < len(X_train):
                idx_sub = np.random.RandomState(42).choice(len(X_train), max_train, replace=False)
                idx_sub.sort()
                X_t, y_t_s1, w_t = X_train[idx_sub], y_s1[idx_sub], s1w[idx_sub]
            else:
                X_t, y_t_s1, w_t = X_train, y_s1, s1w

            # S1
            tpfn_s1 = TabPFNClassifier(device="cuda", n_estimators=8)
            tpfn_s1.fit(X_t, y_t_s1)
            s1p_c = tpfn_s1.predict_proba(X_oos)
            s1pred_c = (s1p_c[:, 1] >= params["stage1_threshold"]).astype(int)

            # S2
            X_t_s2 = X_t[y_t_s1 != 0] if max_train < len(X_train) else X_train[tm]
            y_t_s2 = (y_train[tm] == LABEL_MAP["UP"]).astype(int)
            if len(X_t_s2) != len(y_t_s2):
                # Subsample s2 aligned
                tm_sub = y_t_s1 != 0
                X_t_s2 = X_t[tm_sub]
                y_t_s2_sub = (y_t_s1[tm_sub] == 0).astype(int)  # placeholder
                # Better: use original tm
                X_t_s2 = X_train[tm][:max_train]
                y_t_s2 = y_s2[:max_train]
                if len(X_t_s2) > len(y_t_s2):
                    X_t_s2 = X_t_s2[:len(y_t_s2)]

            tpfn_s2 = TabPFNClassifier(device="cuda", n_estimators=8)
            tpfn_s2.fit(X_t_s2, y_t_s2)
            s2p_c = tpfn_s2.predict_proba(X_oos)
            s2pred_c = np.argmax(s2p_c, axis=1)

            t_c = time.time() - t0
            ev_c = evaluate_arm(oos_df, s1pred_c, s2pred_c, coin, "C_tabpfn")
            print(f"    Time: {t_c:.1f}s | {ev_c['trades']}T avg={ev_c['avg_pnl']:+.4%} "
                  f"total={ev_c['total_pnl']:+.4%} dd={ev_c['max_dd']:.4%}")

        except Exception as e:
            print(f"    TabPFN FAILED: {e}")
            traceback.print_exc()
            ev_c = {"label": "C_tabpfn", "coin": coin, "trades": 0, "error": str(e)}
            s1p_c = None
            t_c = 0

        # ==================== ARM B: Ensemble + TabPFN ====================
        print(f"\n  ARM B: Ensemble + TabPFN (soft voting)")

        if s1p_c is not None:
            # Blend: 70% ensemble + 30% TabPFN
            blend_ratio = 0.7
            s1p_b = blend_ratio * s1p_a + (1 - blend_ratio) * s1p_c
            s1pred_b = (s1p_b[:, 1] >= params["stage1_threshold"]).astype(int)

            s2p_b = blend_ratio * s2p_a + (1 - blend_ratio) * s2p_c
            s2pred_b = np.argmax(s2p_b, axis=1)

            ev_b = evaluate_arm(oos_df, s1pred_b, s2pred_b, coin, "B_blend")
            print(f"    Blend(70/30) | {ev_b['trades']}T avg={ev_b['avg_pnl']:+.4%} "
                  f"total={ev_b['total_pnl']:+.4%} dd={ev_b['max_dd']:.4%}")

            # Also try 50/50
            s1p_b5 = 0.5 * s1p_a + 0.5 * s1p_c
            s1pred_b5 = (s1p_b5[:, 1] >= params["stage1_threshold"]).astype(int)
            s2p_b5 = 0.5 * s2p_a + 0.5 * s2p_c
            s2pred_b5 = np.argmax(s2p_b5, axis=1)
            ev_b5 = evaluate_arm(oos_df, s1pred_b5, s2pred_b5, coin, "B_50_50")
            print(f"    Blend(50/50) | {ev_b5['trades']}T avg={ev_b5['avg_pnl']:+.4%} "
                  f"total={ev_b5['total_pnl']:+.4%} dd={ev_b5['max_dd']:.4%}")
        else:
            ev_b = ev_a.copy()
            ev_b["label"] = "B_blend_fallback"
            ev_b5 = ev_a.copy()

        # Signal agreement
        agree_s1 = (s1pred_a == s1pred_c).mean() if s1p_c is not None else 0
        agree_s2 = (s2pred_a == s2pred_c).mean() if s1p_c is not None else 0
        print(f"\n  Signal agreement: S1 {agree_s1:.1%}, S2 {agree_s2:.1%}")

        results[coin] = {
            "A_ensemble": ev_a,
            "B_blend_70_30": ev_b,
            "B_blend_50_50": ev_b5,
            "C_tabpfn": ev_c,
            "agreement_s1": round(agree_s1, 4),
            "agreement_s2": round(agree_s2, 4),
            "time_ensemble": round(t_a, 1),
            "time_tabpfn": round(t_c, 1),
        }

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Coin':>5s} | {'ARM':>12s} | {'Trades':>6s} | {'Avg PnL':>10s} | "
          f"{'Total':>10s} | {'MDD':>8s}")
    print(f"  {'-'*65}")

    for coin in COINS:
        r = results.get(coin, {})
        for key in ["A_ensemble", "B_blend_70_30", "B_blend_50_50", "C_tabpfn"]:
            m = r.get(key, {})
            if m.get("trades", 0) > 0:
                print(f"  {coin:>5s} | {key:>12s} | {m['trades']:>6d} | "
                      f"{m['avg_pnl']:>+10.4%} | {m['total_pnl']:>+10.4%} | "
                      f"{m.get('max_dd',0):>8.4%}")
            elif "error" in m:
                print(f"  {coin:>5s} | {key:>12s} | ERROR: {m['error'][:30]}")

    with open(REPORT_DIR / "tabpfn_ab_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = (datetime.now() - start).total_seconds() / 60
    print(f"\n  Completed in {elapsed:.1f} min")
    print(f"  Report: {REPORT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
