"""Mega Search -- 7-model decomposition + TabPFN + TabM all combinations.

Base models (from EnhancedEnsemble):
  1. LightGBM (GPU)
  2. XGBoost (GPU)
  3. CatBoost (GPU)
  4. BalancedRandomForest (CPU)
  5. ExtraTrees (CPU)
  6. HistGradientBoosting (CPU)

New models:
  7. TabPFN (foundation model)
  8. TabM (parameter-efficient MLP ensemble)

Combinations to test:
  - Each base model solo
  - Each base model + TabPFN
  - Each base model + TabM
  - Each base model + TabPFN + TabM
  - All 7 + TabPFN
  - All 7 + TabM
  - All 7 + TabPFN + TabM
  - Top-3 base + TabPFN
  - Top-3 base + TabM
  - TabPFN + TabM only (no tree)
  - Various blend ratios

3 coins x N configs x 4 OOS windows.
Runs until 2026-03-18 10:30.
"""

import sys
sys.path.insert(0, "C:/Users/RJ/Desktop/CLAUDE_CRYPTO_AGENT")
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import json, yaml, warnings, time, traceback
import numpy as np, pandas as pd
import torch, torch.nn as nn
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")

from tabpfn import TabPFNClassifier
from tabm import TabM
from src.data.crawlers.crypto_ohlcv import fetch_all_top10
from src.data.crawlers.macro_commodity_crawler import crawl_all_macro_data
from src.models.masking_loop import create_labels_triple_barrier, LABEL_MAP, HORIZONS
from src.evaluation.trade_level_ev import compute_trade_level_ev
from src.execution.cost_model import CostModel, FeeSchedule, FundingConfig, MissFillConfig
from src.utils.config import bar_minutes as cfg_bar_minutes

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    HAS_IMBLEARN = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAS_IMBLEARN = False

REPORT_DIR = Path("experiments/tabpfn_test/results")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = REPORT_DIR / "mega_search_log.jsonl"
CKPT = "tabpfn_v2_cls/tabpfn-v2-classifier.ckpt"
DEADLINE = datetime(2026, 3, 18, 10, 30, 0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with open("config/frozen_params_v3_4.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)
COMMON = CFG["common"]; CC = CFG["cost_model"]
EXK = CFG["excluded_feature_keywords"]
BM = cfg_bar_minutes(); MH = COMMON["max_horizon"]; RF = COMMON["risk_frac"]

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


# ==================== Individual Base Models ====================

def build_lgb(rs=42):
    return lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, device="gpu", gpu_use_dp=False,
        verbose=-1, is_unbalance=True, random_state=rs)

def build_xgb(rs=42):
    return xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", device="cuda",
        verbosity=0, random_state=rs, eval_metric="logloss")

def build_cb(rs=42):
    return CatBoostClassifier(iterations=150, depth=6, learning_rate=0.05,
        auto_class_weights="Balanced", task_type="GPU", verbose=0, random_seed=rs)

def build_brf(rs=42):
    if HAS_IMBLEARN:
        return BalancedRandomForestClassifier(n_estimators=200, max_depth=8,
            n_jobs=6, random_state=rs, sampling_strategy="all")
    return RandomForestClassifier(n_estimators=200, max_depth=8,
        class_weight="balanced", n_jobs=6, random_state=rs)

def build_et(rs=42):
    return ExtraTreesClassifier(n_estimators=200, max_depth=8,
        class_weight="balanced", n_jobs=6, random_state=rs)

def build_hgb(rs=42):
    return HistGradientBoostingClassifier(max_iter=200, max_depth=6,
        class_weight="balanced", random_state=rs, verbose=0)


class TabMWrap:
    def __init__(self, nf, k=32, d=128, nb=3, ep=150):
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
            if ni>=15: break
        if bs: self.model.load_state_dict(bs)
        return self
    def predict_proba(self, X):
        Xs=self.scaler.transform(X).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            out=self.model(x_num=torch.tensor(Xs, device=DEVICE), x_cat=None).mean(dim=1)
            return torch.softmax(out, dim=1).cpu().numpy()


def fit_predict_model(name, X_tr, y_tr, X_oos, sw=None):
    """Fit a single model and return predict_proba on OOS."""
    if name == "lgb":
        m = build_lgb(); m.fit(X_tr, y_tr, sample_weight=sw); return m.predict_proba(X_oos)
    elif name == "xgb":
        m = build_xgb(); m.fit(X_tr, y_tr, sample_weight=sw); return m.predict_proba(X_oos)
    elif name == "cb":
        m = build_cb(); m.fit(X_tr, y_tr, sample_weight=sw); return m.predict_proba(X_oos)
    elif name == "brf":
        m = build_brf(); m.fit(X_tr, y_tr, sample_weight=sw); return m.predict_proba(X_oos)
    elif name == "et":
        m = build_et(); m.fit(X_tr, y_tr, sample_weight=sw); return m.predict_proba(X_oos)
    elif name == "hgb":
        m = build_hgb(); m.fit(X_tr, y_tr, sample_weight=sw); return m.predict_proba(X_oos)
    elif name == "tabpfn":
        m = TabPFNClassifier(model_path=CKPT, device="cuda", n_estimators=8)
        m.fit(X_tr, y_tr); return m.predict_proba(X_oos)
    elif name == "tabm":
        m = TabMWrap(X_tr.shape[1], k=32, d=128, nb=3, ep=150)
        m.fit(X_tr, y_tr, sample_weight=sw); return m.predict_proba(X_oos)
    return None


def evaluate(oos_df, s1p, s2p, coin):
    ku,kl = ck(coin,"k_upper"), ck(coin,"k_lower")
    blocked = COINS_CFG[coin].get("blocked_regimes_override", CFG.get("blocked_regimes",["RANGE_LOW"]))
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


# ==================== Ensemble Configs ====================

def generate_configs():
    """Generate all model combination configs."""
    base_models = ["lgb", "xgb", "cb", "brf", "et", "hgb"]
    new_models = ["tabpfn", "tabm"]
    configs = []

    # 1. Each base model solo
    for m in base_models:
        configs.append({"name": m, "models": [m], "weights": [1.0]})

    # 2. New models solo
    for m in new_models:
        configs.append({"name": m, "models": [m], "weights": [1.0]})

    # 3. Each base + TabPFN (equal weight)
    for m in base_models:
        configs.append({"name": f"{m}+tabpfn", "models": [m, "tabpfn"], "weights": [0.5, 0.5]})
        configs.append({"name": f"{m}+tabpfn_70", "models": [m, "tabpfn"], "weights": [0.7, 0.3]})

    # 4. Each base + TabM
    for m in base_models:
        configs.append({"name": f"{m}+tabm", "models": [m, "tabm"], "weights": [0.5, 0.5]})
        configs.append({"name": f"{m}+tabm_70", "models": [m, "tabm"], "weights": [0.7, 0.3]})

    # 5. Each base + TabPFN + TabM
    for m in base_models:
        configs.append({"name": f"{m}+pfn+tm", "models": [m, "tabpfn", "tabm"], "weights": [0.4, 0.3, 0.3]})

    # 6. Top combos (GPU trio)
    configs.append({"name": "lgb+xgb+cb", "models": ["lgb","xgb","cb"], "weights": [0.33,0.33,0.34]})
    configs.append({"name": "lgb+xgb+cb+pfn", "models": ["lgb","xgb","cb","tabpfn"], "weights": [0.25,0.25,0.25,0.25]})
    configs.append({"name": "lgb+xgb+cb+tm", "models": ["lgb","xgb","cb","tabm"], "weights": [0.25,0.25,0.25,0.25]})
    configs.append({"name": "lgb+xgb+cb+pfn+tm", "models": ["lgb","xgb","cb","tabpfn","tabm"], "weights": [0.2,0.2,0.2,0.2,0.2]})

    # 7. TabPFN + TabM only
    configs.append({"name": "pfn+tm", "models": ["tabpfn","tabm"], "weights": [0.5,0.5]})
    configs.append({"name": "pfn+tm_70pfn", "models": ["tabpfn","tabm"], "weights": [0.7,0.3]})
    configs.append({"name": "pfn+tm_30pfn", "models": ["tabpfn","tabm"], "weights": [0.3,0.7]})

    # 8. Full 8-model
    configs.append({"name": "all8_equal", "models": base_models+new_models, "weights": [0.125]*8})
    configs.append({"name": "all8_tree_heavy", "models": base_models+new_models,
                    "weights": [0.15,0.15,0.15,0.1,0.1,0.1,0.125,0.125]})

    return configs


def run_config_multi_oos(coin, df, fcols, X_full, y_full, config):
    """Run one config across 4 OOS windows."""
    idx = df.index; data_end = idx[-1]
    purge_td = timedelta(hours=(MH*2+6)*(BM//60))
    params = COINS_CFG[coin]

    window_results = []

    for w in OOS_WINDOWS:
        if datetime.now() >= DEADLINE: break

        oos_end = data_end - timedelta(days=w["end_offset"])
        oos_start = oos_end - timedelta(days=w["days"])
        train_end = oos_start - purge_td

        train_mask = idx <= train_end
        oos_mask = (idx >= oos_start) & (idx <= oos_end)

        n_train = train_mask.sum()
        if n_train > len(X_full): n_train = len(X_full)
        X_tr = X_full[:n_train]
        y_tr = y_full[:n_train]

        oos_df = df[oos_mask]
        if len(oos_df) < 10: continue
        oc = oos_df.replace([np.inf,-np.inf], np.nan).ffill().bfill()
        Xo = oc[fcols].fillna(0).values

        y_s1 = (y_tr != LABEL_MAP["HOLD"]).astype(int)
        s1c = np.bincount(y_s1, minlength=2)
        s1w = np.where(s1c>0, len(y_s1)/(2*s1c+1e-10), 1.0)[y_s1]
        tm = y_tr != LABEL_MAP["HOLD"]
        y_s2 = (y_tr[tm] == LABEL_MAP["UP"]).astype(int)
        s2c = np.bincount(y_s2, minlength=2)
        s2w = np.where(s2c>0, len(y_s2)/(2*s2c+1e-10), 1.0)[y_s2]

        try:
            # S1: weighted average of all models in config
            s1_probs_list = []
            for mname in config["models"]:
                p = fit_predict_model(mname, X_tr, y_s1, Xo, sw=s1w)
                if p is not None:
                    s1_probs_list.append(p)

            if not s1_probs_list: continue

            weights = config["weights"][:len(s1_probs_list)]
            wsum = sum(weights)
            s1p = sum(w/wsum * p for w, p in zip(weights, s1_probs_list))
            s1pred = (s1p[:,1] >= params["stage1_threshold"]).astype(int)

            # S2
            s2_probs_list = []
            for mname in config["models"]:
                p = fit_predict_model(mname, X_tr[tm], y_s2, Xo, sw=s2w)
                if p is not None:
                    s2_probs_list.append(p)

            if not s2_probs_list: continue
            weights2 = config["weights"][:len(s2_probs_list)]
            wsum2 = sum(weights2)
            s2p = sum(w/wsum2 * p for w, p in zip(weights2, s2_probs_list))
            s2pred = np.argmax(s2p, axis=1)

            ev = evaluate(oos_df, s1pred, s2pred, coin)
            window_results.append({
                "avg": ev["avg_net_pnl"], "total": ev["total_net_pnl"],
                "trades": ev["trade_count"], "dd": ev["max_dd"],
            })

        except Exception as e:
            window_results.append({"error": str(e)[:50]})

    valid = [r for r in window_results if "error" not in r and r["trades"] > 0]
    if not valid:
        return None

    return {
        "n_windows": len(valid),
        "avg_pnl_mean": round(np.mean([r["avg"] for r in valid]), 6),
        "total_pnl_sum": round(np.sum([r["total"] for r in valid]), 6),
        "avg_trades": round(np.mean([r["trades"] for r in valid]), 1),
        "avg_dd": round(np.mean([r["dd"] for r in valid]), 6),
    }


def main():
    start = datetime.now()
    print(f"\n{'='*70}")
    print(f"  MEGA SEARCH -- Beat 7-model Ensemble")
    print(f"  Deadline: {DEADLINE}")
    print(f"  Coins: {list(COINS_CFG.keys())}")
    print(f"  {start.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")

    # Data
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
        print(f"  {coin}: {len(X)} bars, {len(fcols)} features")

    configs = generate_configs()
    print(f"\n  {len(configs)} configs x {len(COINS_CFG)} coins = {len(configs)*len(COINS_CFG)} experiments")

    best_per_coin = {}
    round_num = 0

    for cfg in configs:
        if datetime.now() >= DEADLINE: break
        round_num += 1
        remaining = (DEADLINE - datetime.now()).total_seconds() / 3600

        print(f"\n  [{round_num}/{len(configs)}] {cfg['name']} | {remaining:.1f}h left")

        for coin in COINS_CFG:
            if coin not in coin_data: continue
            if datetime.now() >= DEADLINE: break

            cd = coin_data[coin]
            t0 = time.time()

            try:
                result = run_config_multi_oos(coin, cd["df"], cd["fcols"], cd["X"], cd["y"], cfg)
            except Exception as e:
                print(f"    {coin}: ERROR {str(e)[:40]}")
                continue

            if result is None:
                print(f"    {coin}: no valid results")
                continue

            elapsed = time.time() - t0
            print(f"    {coin}: avg={result['avg_pnl_mean']:+.4%} "
                  f"total={result['total_pnl_sum']:+.4%} "
                  f"trades={result['avg_trades']:.0f} dd={result['avg_dd']:.4%} [{elapsed:.1f}s]")

            # Track best
            prev = best_per_coin.get(coin, {}).get("avg_pnl_mean", -999)
            if result["avg_pnl_mean"] > prev:
                best_per_coin[coin] = {"config": cfg["name"], **result}
                print(f"    *** NEW BEST for {coin}!")

            # Log
            entry = {"coin": coin, "config": cfg["name"], **result,
                     "time": round(elapsed, 1), "ts": datetime.now().isoformat()}
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    # Final
    print(f"\n{'='*70}")
    print(f"  MEGA SEARCH COMPLETE ({round_num} configs tested)")
    print(f"{'='*70}")

    for coin, best in best_per_coin.items():
        print(f"\n  {coin} BEST: {best['config']}")
        print(f"    avg_pnl={best['avg_pnl_mean']:+.4%} total={best['total_pnl_sum']:+.4%} "
              f"trades={best['avg_trades']:.0f} dd={best['avg_dd']:.4%}")

    with open(REPORT_DIR / "mega_search_final.json", "w") as f:
        json.dump({"best_per_coin": best_per_coin, "configs_tested": round_num,
                   "deadline": str(DEADLINE)}, f, indent=2, default=str)

    print(f"\n  Finished at {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
