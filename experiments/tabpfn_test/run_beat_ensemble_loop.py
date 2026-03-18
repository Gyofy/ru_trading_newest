"""Beat 7-model Ensemble Loop -- runs until 2026-03-18 10:30.

Tries different model combinations to beat baseline on BOTH DOT and ADA.
Each round tests a different configuration and logs results.

Configs to try:
- TabPFN standalone / blend ratios (10-90% in 10% steps)
- TabM k=16/32/64 / blend ratios
- TabPFN + TabM combined
- TabPFN n_estimators variations (4, 8, 16)
- TabM + different architectures (tabm, tabm-mini)
- Ensemble subset (LGB only) + TabPFN/TabM
"""

import sys
sys.path.insert(0, "C:/Users/RJ/Desktop/CLAUDE_CRYPTO_AGENT")
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import json, yaml, warnings, time, traceback
import numpy as np, pandas as pd
import torch, torch.nn as nn
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from tabpfn import TabPFNClassifier
from tabm import TabM
from src.data.crawlers.crypto_ohlcv import fetch_all_top10
from src.data.crawlers.macro_commodity_crawler import crawl_all_macro_data
from src.models.masking_loop import create_labels_triple_barrier, LABEL_MAP, HORIZONS
from src.models.enhanced_ensemble import EnhancedEnsemble
from src.evaluation.trade_level_ev import compute_trade_level_ev
from src.execution.cost_model import CostModel, FeeSchedule, FundingConfig, MissFillConfig
from src.utils.config import bar_minutes as cfg_bar_minutes

REPORT_DIR = Path("experiments/tabpfn_test/results")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = REPORT_DIR / "beat_ensemble_log.jsonl"
CKPT = "tabpfn_v2_cls/tabpfn-v2-classifier.ckpt"
DEADLINE = datetime(2026, 3, 18, 10, 30, 0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("config/frozen_params_v3_4.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)
COMMON = CFG["common"]; CC = CFG["cost_model"]

# XRP도 포함 (threshold 0.46으로 테스트)
COINS_CFG = {
    **CFG["coins"],
    "XRP": {
        "stage1_threshold": 0.46, "max_features": 120,
        "num_leaves": 47, "learning_rate": 0.02,
        "n_estimators": 100, "max_depth_tree": 6,
        "subsample": 0.8, "colsample": 0.6,
        "min_child_samples": 30,
        "blocked_regimes_override": ["RANGE_LOW"],
    },
}
EXK = CFG["excluded_feature_keywords"]
BM = cfg_bar_minutes(); MH = COMMON["max_horizon"]; RF = COMMON["risk_frac"]; N_JOBS = 6

CM = CostModel(
    fee_schedule=FeeSchedule(maker_fee=CC["maker_fee"], taker_fee=CC["taker_fee"],
        slippage_entry=CC["slippage_entry"], slippage_exit_limit=CC["slippage_exit_limit"],
        slippage_exit_market=CC["slippage_exit_market"]),
    funding_config=FundingConfig(interval_hours=CC["funding_interval_hours"],
        default_rate=CC["funding_default_rate"]),
    miss_fill_config=MissFillConfig(reject_prob=CC["miss_fill_reject_prob"],
        missed_ev_pct=CC["miss_fill_missed_ev"]))

OOS_WINDOWS = [
    {"end_offset": 42, "days": 28},
    {"end_offset": 28, "days": 28},
    {"end_offset": 14, "days": 28},
    {"end_offset": 0,  "days": 28},
]


def ck(coin, key):
    return COINS_CFG[coin].get(f"{key}_override", COMMON[key])
def isex(c):
    return any(kw in c.lower() for kw in EXK)


class TabMWrap:
    def __init__(self, nf, k=32, d=128, nb=3, ep=200):
        self.nf=nf; self.k=k; self.d=d; self.nb=nb; self.ep=ep
        self.scaler=StandardScaler(); self.model=None

    def fit(self, X, y, sample_weight=None):
        Xs = self.scaler.fit_transform(X).astype(np.float32)
        self.model = TabM.make(n_num_features=self.nf, cat_cardinalities=[], d_out=2,
            arch_type="tabm", k=self.k, d_block=self.d, n_blocks=self.nb, dropout=0.1).to(DEVICE)
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.ep)
        Xt=torch.tensor(Xs, device=DEVICE); yt=torch.tensor(y, dtype=torch.long, device=DEVICE)
        wt = torch.tensor(sample_weight, dtype=torch.float32, device=DEVICE) if sample_weight is not None else None
        self.model.train(); bl=float("inf"); bs=None; ni=0
        for ep in range(self.ep):
            perm=torch.randperm(len(Xt), device=DEVICE); el=0; nb=0
            for i in range(0, len(Xt), 256):
                ix=perm[i:i+256]; out=self.model(x_num=Xt[ix], x_cat=None).mean(dim=1)
                if wt is not None: loss=(nn.functional.cross_entropy(out, yt[ix], reduction="none")*wt[ix]).mean()
                else: loss=nn.functional.cross_entropy(out, yt[ix])
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
                opt.step(); el+=loss.item(); nb+=1
            sch.step(); al=el/max(nb,1)
            if al<bl: bl=al; ni=0; bs={k:v.clone() for k,v in self.model.state_dict().items()}
            else: ni+=1
            if ni>=20: break
        if bs: self.model.load_state_dict(bs)
        return self

    def predict_proba(self, X):
        Xs=self.scaler.transform(X).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            out=self.model(x_num=torch.tensor(Xs, device=DEVICE), x_cat=None).mean(dim=1)
            return torch.softmax(out, dim=1).cpu().numpy()


def evaluate(oos_df, s1p, s2p, coin):
    ku,kl = ck(coin,"k_upper"), ck(coin,"k_lower")
    blocked = COINS_CFG[coin].get("blocked_regimes_override", CFG["blocked_regimes"])
    n=len(oos_df); close=oos_df["close"].values; rm=np.ones(n, dtype=bool)
    for i in range(n):
        if i<42:
            if "UNKNOWN" in blocked: rm[i]=False
            continue
        seg=close[max(0,i-41):i+1]
        ef=pd.Series(seg).ewm(span=10).mean().iloc[-1]
        es=pd.Series(seg).ewm(span=30).mean().iloc[-1]
        rs=np.std(np.diff(seg)/seg[:-1]) if len(seg)>1 else 0
        if abs(ef/es-1)<0.005 and rs<0.02: rm[i]=False
    s1f=s1p.copy(); s1f[~rm]=0
    return compute_trade_level_ev(oos_df, s1f, np.ones(n)*0.5, s2p, np.ones(n)*0.5,
        k_upper=ku, k_lower=kl, max_hold=MH, risk_frac=RF, cost_model=CM)


def run_multi_oos(coin, df, fcols, X, y, config):
    """Run one config across 4 OOS windows. Returns avg of avg_pnl."""
    idx = df.index; data_end = idx[-1]
    purge_td = timedelta(hours=(MH*2+6)*(BM//60))
    params = COINS_CFG[coin]

    y_s1 = (y != LABEL_MAP["HOLD"]).astype(int)
    s1c = np.bincount(y_s1, minlength=2)
    s1w = np.where(s1c>0, len(y_s1)/(2*s1c+1e-10), 1.0)[y_s1]
    tm = y != LABEL_MAP["HOLD"]
    y_s2 = (y[tm] == LABEL_MAP["UP"]).astype(int)
    s2c = np.bincount(y_s2, minlength=2)
    s2w = np.where(s2c>0, len(y_s2)/(2*s2c+1e-10), 1.0)[y_s2]

    window_results = []

    for w in OOS_WINDOWS:
        oos_end = data_end - timedelta(days=w["end_offset"])
        oos_start = oos_end - timedelta(days=w["days"])
        train_end = oos_start - purge_td

        train_mask = idx <= train_end
        oos_mask = (idx >= oos_start) & (idx <= oos_end)
        X_tr = X[train_mask[:len(X)]] if len(train_mask) > len(X) else X
        y_s1_tr = y_s1[:len(X_tr)]
        s1w_tr = s1w[:len(X_tr)]
        tm_tr = tm[:len(X_tr)]
        y_s2_tr = y_s2[:tm_tr.sum()]

        oos_df_w = df[oos_mask]
        if len(oos_df_w) < 10: continue
        oc = oos_df_w.replace([np.inf,-np.inf], np.nan).ffill().bfill()
        Xo = oc[fcols].fillna(0).values

        # Baseline
        s1a = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
        s1a.fit(X_tr, y_s1_tr, sample_weight=s1w_tr)
        s1pa = s1a.predict_proba(Xo)
        s2a = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
        s2w_tr = np.where(s2c>0, len(y_s2_tr)/(2*s2c+1e-10), 1.0)[y_s2_tr] if len(y_s2_tr) > 0 else None
        s2a.fit(X_tr[tm_tr], y_s2_tr, sample_weight=s2w_tr)
        s2pa = s2a.predict_proba(Xo)

        # Alternative model based on config
        try:
            if config["model"] == "tabpfn":
                alt_s1 = TabPFNClassifier(model_path=CKPT, device="cuda", n_estimators=config.get("n_est", 8))
                alt_s1.fit(X_tr, y_s1_tr)
                s1p_alt = alt_s1.predict_proba(Xo)
                alt_s2 = TabPFNClassifier(model_path=CKPT, device="cuda", n_estimators=config.get("n_est", 8))
                alt_s2.fit(X_tr[tm_tr], y_s2_tr)
                s2p_alt = alt_s2.predict_proba(Xo)

            elif config["model"] == "tabm":
                alt_s1 = TabMWrap(X_tr.shape[1], k=config.get("k",32), d=config.get("d",128),
                                  nb=config.get("nb",3), ep=config.get("ep",200))
                alt_s1.fit(X_tr, y_s1_tr, sample_weight=s1w_tr)
                s1p_alt = alt_s1.predict_proba(Xo)
                alt_s2 = TabMWrap(X_tr[tm_tr].shape[1], k=config.get("k",32), d=config.get("d",128),
                                  nb=config.get("nb",3), ep=config.get("ep",200))
                alt_s2.fit(X_tr[tm_tr], y_s2_tr, sample_weight=s2w_tr)
                s2p_alt = alt_s2.predict_proba(Xo)

            elif config["model"] == "tabpfn+tabm":
                pfn_s1 = TabPFNClassifier(model_path=CKPT, device="cuda", n_estimators=8)
                pfn_s1.fit(X_tr, y_s1_tr)
                s1p_pfn = pfn_s1.predict_proba(Xo)
                tm_s1 = TabMWrap(X_tr.shape[1], k=config.get("k",32))
                tm_s1.fit(X_tr, y_s1_tr, sample_weight=s1w_tr)
                s1p_tm = tm_s1.predict_proba(Xo)
                s1p_alt = 0.5 * s1p_pfn + 0.5 * s1p_tm

                pfn_s2 = TabPFNClassifier(model_path=CKPT, device="cuda", n_estimators=8)
                pfn_s2.fit(X_tr[tm_tr], y_s2_tr)
                s2p_pfn = pfn_s2.predict_proba(Xo)
                tm_s2 = TabMWrap(X_tr[tm_tr].shape[1], k=config.get("k",32))
                tm_s2.fit(X_tr[tm_tr], y_s2_tr, sample_weight=s2w_tr)
                s2p_tm = tm_s2.predict_proba(Xo)
                s2p_alt = 0.5 * s2p_pfn + 0.5 * s2p_tm
            else:
                continue
        except Exception as e:
            window_results.append({"error": str(e)})
            continue

        # Blend
        br = config.get("blend_ratio", 0.5)
        s1p_blend = (1-br)*s1pa + br*s1p_alt
        s2p_blend = (1-br)*s2pa + br*s2p_alt
        s1pred = (s1p_blend[:,1] >= params["stage1_threshold"]).astype(int)
        s2pred = np.argmax(s2p_blend, axis=1)

        # Baseline pred
        s1pred_base = (s1pa[:,1] >= params["stage1_threshold"]).astype(int)
        s2pred_base = np.argmax(s2pa, axis=1)

        ev_base = evaluate(oos_df_w, s1pred_base, s2pred_base, coin)
        ev_blend = evaluate(oos_df_w, s1pred, s2pred, coin)

        window_results.append({
            "baseline_avg": ev_base["avg_net_pnl"],
            "blend_avg": ev_blend["avg_net_pnl"],
            "baseline_total": ev_base["total_net_pnl"],
            "blend_total": ev_blend["total_net_pnl"],
            "blend_trades": ev_blend["trade_count"],
            "win": ev_blend["avg_net_pnl"] > ev_base["avg_net_pnl"],
        })

    if not window_results or all("error" in r for r in window_results):
        return None

    valid = [r for r in window_results if "error" not in r]
    wins = sum(1 for r in valid if r["win"])
    avg_improvement = np.mean([r["blend_avg"] - r["baseline_avg"] for r in valid])

    return {
        "wins": wins,
        "total_windows": len(valid),
        "avg_improvement": round(avg_improvement, 6),
        "details": valid,
    }


def main():
    start = datetime.now()
    print(f"\n{'='*70}")
    print(f"  BEAT ENSEMBLE LOOP")
    print(f"  Deadline: {DEADLINE}")
    print(f"  {start.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")

    print("\n  Fetching data...")
    ohlcv = fetch_all_top10("365d", "1h")
    first = list(ohlcv.values())[0]
    macro = crawl_all_macro_data(first.index)
    ma = macro.get("aligned", pd.DataFrame())

    coin_data = {}
    for coin in COINS_CFG:
        if coin not in ohlcv: continue
        df = ohlcv[coin].copy()
        if len(ma) > 0:
            for col in ma.columns:
                df[col] = ma[col].reindex(df.index).ffill().bfill().fillna(0)

        params = COINS_CFG[coin]
        h = HORIZONS[-1]; hl = f"label_{h*BM}min"
        ku, kl = ck(coin,"k_upper"), ck(coin,"k_lower")
        labeled = create_labels_triple_barrier(
            df.copy(), h, k_upper_override=ku, k_lower_override=kl, verbose=False)
        if hl not in labeled.columns: continue

        exclude = {"label","future_return","open","high","low","close","volume"}
        for hh in HORIZONS: exclude.add(f"label_{hh*BM}min"); exclude.add(f"return_{hh*BM}min")
        lk = ["future","target","label_","return_","fwd_","forward_"]
        fcols = [c for c in labeled.columns if c not in exclude and not any(k in c.lower() for k in lk)
                 and not isex(c) and labeled[c].dtype in [np.float64,np.float32,np.int64,np.int32,float,int]]
        mf = params["max_features"]
        clean = labeled.replace([np.inf,-np.inf], np.nan).ffill().bfill()
        X = clean[fcols].fillna(0).values
        y = clean[hl].fillna(1).values.astype(int)
        if len(fcols) > mf:
            mi = mutual_info_classif(X[:min(2000,len(X))], y[:min(2000,len(X))],
                                     discrete_features=False, random_state=42, n_neighbors=5)
            top = np.argsort(mi)[-mf:]; fcols = [fcols[i] for i in sorted(top)]
            X = clean[fcols].fillna(0).values

        coin_data[coin] = {"df": df, "fcols": fcols, "X": X, "y": y}

    # Config generator
    configs = []
    # TabPFN blends
    for br in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        configs.append({"name": f"TabPFN_b{int(br*100)}", "model": "tabpfn", "blend_ratio": br, "n_est": 8})
    for ne in [4, 16, 32]:
        for br in [0.3, 0.5]:
            configs.append({"name": f"TabPFN_e{ne}_b{int(br*100)}", "model": "tabpfn", "blend_ratio": br, "n_est": ne})
    # TabM blends
    for k in [16, 32, 64]:
        for br in [0.2, 0.3, 0.4, 0.5]:
            configs.append({"name": f"TabM_k{k}_b{int(br*100)}", "model": "tabm", "blend_ratio": br, "k": k})
    # TabPFN+TabM combined
    for br in [0.3, 0.4, 0.5]:
        configs.append({"name": f"PFN+TabM_b{int(br*100)}", "model": "tabpfn+tabm", "blend_ratio": br, "k": 32})

    best_results = {}
    round_num = 0

    for cfg in configs:
        if datetime.now() >= DEADLINE:
            break

        round_num += 1
        remaining = (DEADLINE - datetime.now()).total_seconds() / 3600
        print(f"\n  Round {round_num}/{len(configs)} | {cfg['name']} | {remaining:.1f}h remaining")

        for coin in COINS_CFG:
            if coin not in coin_data: continue
            cd = coin_data[coin]

            try:
                result = run_multi_oos(coin, cd["df"], cd["fcols"], cd["X"], cd["y"], cfg)
            except Exception as e:
                print(f"    {coin} ERROR: {e}")
                continue

            if result is None: continue

            tag = "WIN" if result["wins"] > result["total_windows"] / 2 else "lose"
            print(f"    {coin}: {result['wins']}/{result['total_windows']} windows, "
                  f"avg_imp={result['avg_improvement']:+.4%} [{tag}]")

            key = f"{coin}_{cfg['name']}"
            entry = {"coin": coin, "config": cfg["name"], **result, "timestamp": datetime.now().isoformat()}

            # Log
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")

            # Track best
            prev_best = best_results.get(coin, {}).get("wins", 0)
            if result["wins"] > prev_best or (result["wins"] == prev_best and
                result["avg_improvement"] > best_results.get(coin, {}).get("avg_improvement", -999)):
                best_results[coin] = {**entry}
                print(f"    *** NEW BEST for {coin}: {cfg['name']} ({result['wins']}/{result['total_windows']})")

    # Final
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS ({round_num} configs tested)")
    print(f"{'='*70}")
    for coin, best in best_results.items():
        print(f"  {coin}: {best['config']} | {best['wins']}/{best['total_windows']} wins | "
              f"avg_imp={best['avg_improvement']:+.4%}")

    beaten = {c: b for c, b in best_results.items() if b["wins"] > b["total_windows"] / 2}
    if beaten:
        print(f"\n  ENSEMBLE BEATEN on: {list(beaten.keys())}")
    else:
        print(f"\n  ENSEMBLE UNBEATEN -- 7-model remains king")

    with open(REPORT_DIR / "beat_ensemble_final.json", "w") as f:
        json.dump({"best_per_coin": best_results, "total_configs": round_num,
                   "beaten_coins": list(beaten.keys())}, f, indent=2, default=str)

    print(f"\n  Completed at {datetime.now().strftime('%H:%M')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
