"""TabPFN A/B Test -- local weights.

ARM A: EnhancedEnsemble (7-model, baseline)
ARM B: TabPFN standalone (local .ckpt)
ARM C: Ensemble(70%) + TabPFN(30%) blend
ARM D: Ensemble(50%) + TabPFN(50%) blend
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
TABPFN_CKPT = "tabpfn_v2_cls/tabpfn-v2-classifier.ckpt"

with open("config/frozen_params_v3_4.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

COMMON = CFG["common"]; COINS = CFG["coins"]; CC = CFG["cost_model"]
EXK = CFG["excluded_feature_keywords"]
BM = cfg_bar_minutes(); MH = COMMON["max_horizon"]; RF = COMMON["risk_frac"]; N_JOBS = 6
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

def evaluate(oos_df, s1_pred, s2_pred, coin):
    ku, kl = ck(coin,"k_upper"), ck(coin,"k_lower")
    blocked = COINS[coin].get("blocked_regimes_override", CFG["blocked_regimes"])
    n = len(oos_df); close = oos_df["close"].values
    rm = np.ones(n, dtype=bool)
    for i in range(n):
        if i < 42:
            if "UNKNOWN" in blocked: rm[i]=False
            continue
        seg = close[max(0,i-41):i+1]
        ef = pd.Series(seg).ewm(span=10).mean().iloc[-1]
        es = pd.Series(seg).ewm(span=30).mean().iloc[-1]
        rs = np.std(np.diff(seg)/seg[:-1]) if len(seg)>1 else 0
        if abs(ef/es-1)<0.005 and rs<0.02: rm[i]=False
    s1f = s1_pred.copy(); s1f[~rm]=0
    return compute_trade_level_ev(oos_df, s1f, np.ones(n)*0.5, s2_pred, np.ones(n)*0.5,
        k_upper=ku, k_lower=kl, max_hold=MH, risk_frac=RF, cost_model=CM)

def main():
    start = datetime.now()
    print(f"\n{'='*70}")
    print(f"  TabPFN A/B TEST (local weights)")
    print(f"  {start.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")

    print("\n  Fetching data...")
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

    results = {}

    for coin in COINS:
        if coin not in data: continue
        print(f"\n{'='*50}")
        print(f"  {coin}")
        print(f"{'='*50}")

        df = data[coin]; idx = df.index; end = idx[-1]
        oos_start = end - timedelta(days=OOS_DAYS)
        purge = timedelta(hours=(MH*2+6)*(BM//60))
        train_df = df[idx <= oos_start - purge]
        oos_df = df[idx >= oos_start]

        params = COINS[coin]
        h = HORIZONS[-1]; hl = f"label_{h*BM}min"
        ku, kl = ck(coin,"k_upper"), ck(coin,"k_lower")

        labeled = create_labels_triple_barrier(
            train_df.copy(), h, k_upper_override=ku, k_lower_override=kl, verbose=False)
        if hl not in labeled.columns: continue

        exclude = {"label","future_return","open","high","low","close","volume"}
        for hh in HORIZONS: exclude.add(f"label_{hh*BM}min"); exclude.add(f"return_{hh*BM}min")
        lk = ["future","target","label_","return_","fwd_","forward_"]
        fcols = [c for c in labeled.columns if c not in exclude
                 and not any(k in c.lower() for k in lk) and not isex(c)
                 and labeled[c].dtype in [np.float64,np.float32,np.int64,np.int32,float,int]]
        mf = params["max_features"]
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
        s2c = np.bincount(y_s2, minlength=2)
        s2w = np.where(s2c>0, len(y_s2)/(2*s2c+1e-10), 1.0)[y_s2]

        # ARM A: Baseline
        print(f"\n  ARM A: EnhancedEnsemble")
        t0 = time.time()
        s1a = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
        s1a.fit(X, y_s1, sample_weight=s1w)
        s1pa = s1a.predict_proba(Xo)
        s1preda = (s1pa[:,1] >= params["stage1_threshold"]).astype(int)

        s2a = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
        s2a.fit(X[tm], y_s2, sample_weight=s2w)
        s2pa = s2a.predict_proba(Xo)
        s2preda = np.argmax(s2pa, axis=1)
        ta = time.time()-t0

        ev_a = evaluate(oos_df, s1preda, s2preda, coin)
        print(f"    {ta:.1f}s | {ev_a['trade_count']}T avg={ev_a['avg_net_pnl']:+.4%} "
              f"total={ev_a['total_net_pnl']:+.4%} dd={ev_a['max_dd']:.4%}")

        # ARM B: TabPFN standalone
        print(f"\n  ARM B: TabPFN standalone")
        t0 = time.time()
        try:
            tpfn_s1 = TabPFNClassifier(model_path=TABPFN_CKPT, device="cuda", n_estimators=8)
            tpfn_s1.fit(X, y_s1)
            s1pp = tpfn_s1.predict_proba(Xo)
            s1predp = (s1pp[:,1] >= params["stage1_threshold"]).astype(int)

            tpfn_s2 = TabPFNClassifier(model_path=TABPFN_CKPT, device="cuda", n_estimators=8)
            tpfn_s2.fit(X[tm], y_s2)
            s2pp = tpfn_s2.predict_proba(Xo)
            s2predp = np.argmax(s2pp, axis=1)
            tp = time.time()-t0

            ev_p = evaluate(oos_df, s1predp, s2predp, coin)
            print(f"    {tp:.1f}s | {ev_p['trade_count']}T avg={ev_p['avg_net_pnl']:+.4%} "
                  f"total={ev_p['total_net_pnl']:+.4%} dd={ev_p['max_dd']:.4%}")

            # ARM C: Blend 70/30
            s1pb7 = 0.7*s1pa + 0.3*s1pp
            s1predb7 = (s1pb7[:,1] >= params["stage1_threshold"]).astype(int)
            s2pb7 = 0.7*s2pa + 0.3*s2pp
            s2predb7 = np.argmax(s2pb7, axis=1)
            ev_b7 = evaluate(oos_df, s1predb7, s2predb7, coin)
            print(f"\n  ARM C: Blend 70/30")
            print(f"    {ev_b7['trade_count']}T avg={ev_b7['avg_net_pnl']:+.4%} "
                  f"total={ev_b7['total_net_pnl']:+.4%} dd={ev_b7['max_dd']:.4%}")

            # ARM D: Blend 50/50
            s1pb5 = 0.5*s1pa + 0.5*s1pp
            s1predb5 = (s1pb5[:,1] >= params["stage1_threshold"]).astype(int)
            s2pb5 = 0.5*s2pa + 0.5*s2pp
            s2predb5 = np.argmax(s2pb5, axis=1)
            ev_b5 = evaluate(oos_df, s1predb5, s2predb5, coin)
            print(f"\n  ARM D: Blend 50/50")
            print(f"    {ev_b5['trade_count']}T avg={ev_b5['avg_net_pnl']:+.4%} "
                  f"total={ev_b5['total_net_pnl']:+.4%} dd={ev_b5['max_dd']:.4%}")

            agree_s1 = (s1preda == s1predp).mean()
            agree_s2 = (s2preda == s2predp).mean()
            print(f"\n  Agreement: S1 {agree_s1:.1%}, S2 {agree_s2:.1%}")

            results[coin] = {
                "A_ensemble": {"trades": ev_a["trade_count"], "avg": ev_a["avg_net_pnl"], "total": ev_a["total_net_pnl"], "dd": ev_a["max_dd"]},
                "B_tabpfn": {"trades": ev_p["trade_count"], "avg": ev_p["avg_net_pnl"], "total": ev_p["total_net_pnl"], "dd": ev_p["max_dd"]},
                "C_blend70": {"trades": ev_b7["trade_count"], "avg": ev_b7["avg_net_pnl"], "total": ev_b7["total_net_pnl"], "dd": ev_b7["max_dd"]},
                "D_blend50": {"trades": ev_b5["trade_count"], "avg": ev_b5["avg_net_pnl"], "total": ev_b5["total_net_pnl"], "dd": ev_b5["max_dd"]},
                "agreement_s1": round(agree_s1, 4),
                "time_ensemble": round(ta, 1), "time_tabpfn": round(tp, 1),
            }
        except Exception as e:
            print(f"    FAILED: {e}")
            traceback.print_exc()
            results[coin] = {"A_ensemble": {"trades": ev_a["trade_count"], "avg": ev_a["avg_net_pnl"], "total": ev_a["total_net_pnl"]}, "error": str(e)}

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Coin':>5s} | {'ARM':>12s} | {'Trades':>6s} | {'Avg PnL':>10s} | {'Total':>10s} | {'MDD':>8s}")
    print(f"  {'-'*65}")
    for coin in COINS:
        r = results.get(coin, {})
        for key in ["A_ensemble", "B_tabpfn", "C_blend70", "D_blend50"]:
            m = r.get(key, {})
            if "trades" in m and m["trades"] > 0:
                print(f"  {coin:>5s} | {key:>12s} | {m['trades']:>6d} | {m['avg']:>+10.4%} | "
                      f"{m['total']:>+10.4%} | {m.get('dd',0):>8.4%}")

    with open(REPORT_DIR / "tabpfn_local_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = (datetime.now() - start).total_seconds() / 60
    print(f"\n  Completed in {elapsed:.1f} min")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
