"""TabPFN ADA Multi-OOS -- 6 windows for robust validation."""

import sys
sys.path.insert(0, "C:/Users/RJ/Desktop/CLAUDE_CRYPTO_AGENT")
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import json, yaml, warnings, time
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
CKPT = "tabpfn_v2_cls/tabpfn-v2-classifier.ckpt"

with open("config/frozen_params_v3_4.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)
COMMON = CFG["common"]; COINS = CFG["coins"]; CC = CFG["cost_model"]
EXK = CFG["excluded_feature_keywords"]
BM = cfg_bar_minutes(); MH = COMMON["max_horizon"]; RF = COMMON["risk_frac"]; N_JOBS = 6
COIN = "ADA"; PARAMS = COINS[COIN]
KU = COMMON["k_upper"]; KL = PARAMS.get("k_lower_override", COMMON["k_lower"])
BLOCKED = PARAMS.get("blocked_regimes_override", CFG["blocked_regimes"])

CM = CostModel(
    fee_schedule=FeeSchedule(maker_fee=CC["maker_fee"], taker_fee=CC["taker_fee"],
        slippage_entry=CC["slippage_entry"], slippage_exit_limit=CC["slippage_exit_limit"],
        slippage_exit_market=CC["slippage_exit_market"]),
    funding_config=FundingConfig(interval_hours=CC["funding_interval_hours"],
        default_rate=CC["funding_default_rate"]),
    miss_fill_config=MissFillConfig(reject_prob=CC["miss_fill_reject_prob"],
        missed_ev_pct=CC["miss_fill_missed_ev"]))

WINDOWS = [
    {"name": "W1", "end_offset": 42, "days": 28},
    {"name": "W2", "end_offset": 35, "days": 28},
    {"name": "W3", "end_offset": 28, "days": 28},
    {"name": "W4", "end_offset": 21, "days": 28},
    {"name": "W5", "end_offset": 14, "days": 28},
    {"name": "W6", "end_offset": 0,  "days": 28},
]

def isex(c): return any(kw in c.lower() for kw in EXK)

def evaluate(oos_df, s1p, s2p):
    n = len(oos_df); close = oos_df["close"].values
    rm = np.ones(n, dtype=bool)
    for i in range(n):
        if i < 42:
            if "UNKNOWN" in BLOCKED: rm[i]=False
            continue
        seg = close[max(0,i-41):i+1]
        ef = pd.Series(seg).ewm(span=10).mean().iloc[-1]
        es = pd.Series(seg).ewm(span=30).mean().iloc[-1]
        rs = np.std(np.diff(seg)/seg[:-1]) if len(seg)>1 else 0
        if abs(ef/es-1)<0.005 and rs<0.02: rm[i]=False
    s1f = s1p.copy(); s1f[~rm]=0
    return compute_trade_level_ev(oos_df, s1f, np.ones(n)*0.5, s2p, np.ones(n)*0.5,
        k_upper=KU, k_lower=KL, max_hold=MH, risk_frac=RF, cost_model=CM)

def main():
    start = datetime.now()
    print(f"\n{'='*70}")
    print(f"  TabPFN ADA Multi-OOS (6 windows)")
    print(f"  {start.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")

    print("\n  Fetching data...")
    ohlcv = fetch_all_top10("365d", "1h")
    first = list(ohlcv.values())[0]
    macro = crawl_all_macro_data(first.index)
    ma = macro.get("aligned", pd.DataFrame())
    df = ohlcv[COIN].copy()
    if len(ma) > 0:
        for col in ma.columns:
            df[col] = ma[col].reindex(df.index).ffill().bfill().fillna(0)

    idx = df.index; data_end = idx[-1]
    purge_td = timedelta(hours=(MH*2+6)*(BM//60))
    results = []

    for w in WINDOWS:
        oos_end = data_end - timedelta(days=w["end_offset"])
        oos_start = oos_end - timedelta(days=w["days"])
        train_end = oos_start - purge_td
        train_df = df[idx <= train_end]
        oos_df = df[(idx >= oos_start) & (idx <= oos_end)]

        print(f"\n  --- {w['name']}: {str(oos_start)[:10]} ~ {str(oos_end)[:10]} ({len(oos_df)} bars) ---")

        h = HORIZONS[-1]; hl = f"label_{h*BM}min"
        labeled = create_labels_triple_barrier(
            train_df.copy(), h, k_upper_override=KU, k_lower_override=KL, verbose=False)
        if hl not in labeled.columns: continue

        exclude = {"label","future_return","open","high","low","close","volume"}
        for hh in HORIZONS: exclude.add(f"label_{hh*BM}min"); exclude.add(f"return_{hh*BM}min")
        lk = ["future","target","label_","return_","fwd_","forward_"]
        fcols = [c for c in labeled.columns if c not in exclude and not any(k in c.lower() for k in lk)
                 and not isex(c) and labeled[c].dtype in [np.float64,np.float32,np.int64,np.int32,float,int]]
        mf = PARAMS["max_features"]
        clean = labeled.replace([np.inf,-np.inf], np.nan).ffill().bfill()
        X = clean[fcols].fillna(0).values
        y = clean[hl].fillna(1).values.astype(int)
        if len(fcols) > mf:
            mi = mutual_info_classif(X[:min(2000,len(X))], y[:min(2000,len(X))],
                                     discrete_features=False, random_state=42, n_neighbors=5)
            top = np.argsort(mi)[-mf:]; fcols = [fcols[i] for i in sorted(top)]
            X = clean[fcols].fillna(0).values

        oc = oos_df.replace([np.inf,-np.inf], np.nan).ffill().bfill()
        Xo = oc[fcols].fillna(0).values
        y_s1 = (y != LABEL_MAP["HOLD"]).astype(int)
        s1c = np.bincount(y_s1, minlength=2)
        s1w = np.where(s1c>0, len(y_s1)/(2*s1c+1e-10), 1.0)[y_s1]
        tm = y != LABEL_MAP["HOLD"]
        y_s2 = (y[tm] == LABEL_MAP["UP"]).astype(int)

        # Baseline
        s1a = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
        s1a.fit(X, y_s1, sample_weight=s1w)
        s1pa = s1a.predict_proba(Xo)
        s1preda = (s1pa[:,1] >= PARAMS["stage1_threshold"]).astype(int)
        s2a = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
        s2c = np.bincount(y_s2, minlength=2)
        s2w = np.where(s2c>0, len(y_s2)/(2*s2c+1e-10), 1.0)[y_s2]
        s2a.fit(X[tm], y_s2, sample_weight=s2w)
        s2pa = s2a.predict_proba(Xo)
        s2preda = np.argmax(s2pa, axis=1)
        ev_a = evaluate(oos_df, s1preda, s2preda)

        # TabPFN standalone
        tpfn_s1 = TabPFNClassifier(model_path=CKPT, device="cuda", n_estimators=8)
        tpfn_s1.fit(X, y_s1)
        s1pp = tpfn_s1.predict_proba(Xo)
        s1predp = (s1pp[:,1] >= PARAMS["stage1_threshold"]).astype(int)
        tpfn_s2 = TabPFNClassifier(model_path=CKPT, device="cuda", n_estimators=8)
        tpfn_s2.fit(X[tm], y_s2)
        s2pp = tpfn_s2.predict_proba(Xo)
        s2predp = np.argmax(s2pp, axis=1)
        ev_p = evaluate(oos_df, s1predp, s2predp)

        # Blend 50/50
        s1pb = 0.5*s1pa + 0.5*s1pp
        s1predb = (s1pb[:,1] >= PARAMS["stage1_threshold"]).astype(int)
        s2pb = 0.5*s2pa + 0.5*s2pp
        s2predb = np.argmax(s2pb, axis=1)
        ev_b = evaluate(oos_df, s1predb, s2predb)

        # Blend 70/30
        s1pb7 = 0.7*s1pa + 0.3*s1pp
        s1predb7 = (s1pb7[:,1] >= PARAMS["stage1_threshold"]).astype(int)
        s2pb7 = 0.7*s2pa + 0.3*s2pp
        s2predb7 = np.argmax(s2pb7, axis=1)
        ev_b7 = evaluate(oos_df, s1predb7, s2predb7)

        best_alt = max(
            [("TabPFN", ev_p), ("Blend50", ev_b), ("Blend70", ev_b7)],
            key=lambda x: x[1]["avg_net_pnl"])
        winner = best_alt[0] if best_alt[1]["avg_net_pnl"] > ev_a["avg_net_pnl"] else "BASELINE"

        print(f"    Baseline:  {ev_a['trade_count']}T avg={ev_a['avg_net_pnl']:+.4%} total={ev_a['total_net_pnl']:+.4%}")
        print(f"    TabPFN:    {ev_p['trade_count']}T avg={ev_p['avg_net_pnl']:+.4%} total={ev_p['total_net_pnl']:+.4%}")
        print(f"    Blend50:   {ev_b['trade_count']}T avg={ev_b['avg_net_pnl']:+.4%} total={ev_b['total_net_pnl']:+.4%}")
        print(f"    Blend70:   {ev_b7['trade_count']}T avg={ev_b7['avg_net_pnl']:+.4%} total={ev_b7['total_net_pnl']:+.4%}")
        print(f"    Winner: {winner}")

        results.append({
            "window": w["name"],
            "period": f"{str(oos_start)[:10]} ~ {str(oos_end)[:10]}",
            "baseline": {"trades": ev_a["trade_count"], "avg": ev_a["avg_net_pnl"], "total": ev_a["total_net_pnl"]},
            "tabpfn": {"trades": ev_p["trade_count"], "avg": ev_p["avg_net_pnl"], "total": ev_p["total_net_pnl"]},
            "blend50": {"trades": ev_b["trade_count"], "avg": ev_b["avg_net_pnl"], "total": ev_b["total_net_pnl"]},
            "blend70": {"trades": ev_b7["trade_count"], "avg": ev_b7["avg_net_pnl"], "total": ev_b7["total_net_pnl"]},
            "winner": winner,
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"  MULTI-OOS SUMMARY (ADA - TabPFN)")
    print(f"{'='*70}")
    print(f"  {'Win':>4s} | {'Baseline':>12s} | {'TabPFN':>12s} | {'Blend50':>12s} | {'Blend70':>12s} | Winner")
    print(f"  {'-'*75}")

    scores = {"BASELINE": 0, "TabPFN": 0, "Blend50": 0, "Blend70": 0}
    for r in results:
        print(f"  {r['window']:>4s} | {r['baseline']['avg']:>+12.4%} | {r['tabpfn']['avg']:>+12.4%} | "
              f"{r['blend50']['avg']:>+12.4%} | {r['blend70']['avg']:>+12.4%} | {r['winner']}")
        scores[r["winner"]] += 1

    print(f"\n  Scores: {scores}")
    best = max(scores, key=scores.get)
    print(f"  Overall winner: {best} ({scores[best]}/6)")

    with open(REPORT_DIR / "tabpfn_ada_multi_oos.json", "w") as f:
        json.dump({"results": results, "scores": scores}, f, indent=2, default=str)

    elapsed = (datetime.now() - start).total_seconds() / 60
    print(f"\n  Completed in {elapsed:.1f} min")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
