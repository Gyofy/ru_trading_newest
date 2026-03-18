"""Mega Search v2 -- Microstructure features + 5 coins.

Changes from v1:
- 223 features (was 135) -- CVD, OFI, VPIN, Roll, Amihud added
- 5 coins: DOT, ADA, XRP + SOL, LINK
- Loads pre-computed data from data/microstructure/
- Same 8-model configs + ET-focused combos
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

warnings.filterwarnings("ignore")

from tabpfn import TabPFNClassifier
from tabm import TabM
from src.models.masking_loop import create_labels_triple_barrier, LABEL_MAP, HORIZONS
from src.evaluation.trade_level_ev import compute_trade_level_ev
from src.execution.cost_model import CostModel, FeeSchedule, FundingConfig, MissFillConfig
from src.utils.config import bar_minutes as cfg_bar_minutes
from src.utils.feature_policy import is_excluded_feature

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    HAS_IMBLEARN = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAS_IMBLEARN = False

REPORT_DIR = Path("experiments/tabpfn_test/results_v2")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = REPORT_DIR / "mega_search_v2_log.jsonl"
CKPT = "tabpfn_v2_cls/tabpfn-v2-classifier.ckpt"
DEADLINE = datetime(2026, 3, 18, 22, 0, 0)  # tonight
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MS_DIR = Path("data/microstructure")

BM = cfg_bar_minutes()
MH = max(HORIZONS)
RF = 0.005

# 5 coins
COINS = {
    "DOT": {"th": 0.50, "mf": 120, "k_lower": 0.6, "blocked": ["RANGE_LOW"]},
    "ADA": {"th": 0.52, "mf": 120, "k_lower": 0.8, "blocked": ["RANGE_LOW", "UNKNOWN"]},
    "XRP": {"th": 0.46, "mf": 120, "k_lower": 0.6, "blocked": ["RANGE_LOW"]},
    "SOL": {"th": 0.50, "mf": 120, "k_lower": 0.6, "blocked": ["RANGE_LOW"]},
    "LINK": {"th": 0.50, "mf": 120, "k_lower": 0.6, "blocked": ["RANGE_LOW"]},
}
K_UPPER = 3.0

CC = {"maker_fee": 0.0002, "taker_fee": 0.00055, "slippage_entry": 0.0003,
      "slippage_exit_limit": 0.0001, "slippage_exit_market": 0.0005,
      "funding_interval_hours": 8.0, "funding_default_rate": 0.0001,
      "miss_fill_reject_prob": 0.15, "miss_fill_missed_ev": 0.0015}
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


# ==================== Models ====================

def build_et(rs=42):
    return ExtraTreesClassifier(n_estimators=200, max_depth=8,
        class_weight="balanced", n_jobs=6, random_state=rs)
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
        auto_class_weights="Balanced", task_type="CPU", thread_count=6, verbose=0, random_seed=rs)
def build_brf(rs=42):
    if HAS_IMBLEARN:
        return BalancedRandomForestClassifier(n_estimators=200, max_depth=8,
            n_jobs=6, random_state=rs, sampling_strategy="all")
    return RandomForestClassifier(n_estimators=200, max_depth=8,
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
            for i in range(0, len(Xt), 128):
                ix=perm[i:i+128]; out=self.model(x_num=Xt[ix], x_cat=None).mean(dim=1)
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
        Xs=self.scaler.transform(X).astype(np.float32); self.model.eval()
        with torch.no_grad():
            out=self.model(x_num=torch.tensor(Xs, device=DEVICE), x_cat=None).mean(dim=1)
            return torch.softmax(out, dim=1).cpu().numpy()


def fit_predict(name, X_tr, y_tr, X_oos, sw=None):
    if name == "et": m=build_et(); m.fit(X_tr, y_tr, sample_weight=sw); return m.predict_proba(X_oos)
    elif name == "lgb": m=build_lgb(); m.fit(X_tr, y_tr, sample_weight=sw); return m.predict_proba(X_oos)
    elif name == "xgb": m=build_xgb(); m.fit(X_tr, y_tr, sample_weight=sw); return m.predict_proba(X_oos)
    elif name == "cb": m=build_cb(); m.fit(X_tr, y_tr, sample_weight=sw); return m.predict_proba(X_oos)
    elif name == "brf": m=build_brf(); m.fit(X_tr, y_tr, sample_weight=sw); return m.predict_proba(X_oos)
    elif name == "hgb": m=build_hgb(); m.fit(X_tr, y_tr, sample_weight=sw); return m.predict_proba(X_oos)
    elif name == "tabpfn":
        m=TabPFNClassifier(model_path=CKPT, device="cuda", n_estimators=8)
        m.fit(X_tr, y_tr); return m.predict_proba(X_oos)
    elif name == "tabm":
        m=TabMWrap(X_tr.shape[1]); m.fit(X_tr, y_tr, sample_weight=sw); return m.predict_proba(X_oos)
    return None


def evaluate(oos_df, s1p, s2p, coin):
    kl = COINS[coin].get("k_lower", 0.6)
    blocked = COINS[coin].get("blocked", ["RANGE_LOW"])
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
        k_upper=K_UPPER, k_lower=kl, max_hold=MH, risk_frac=RF, cost_model=CM)


def generate_configs():
    configs = []
    # ET-focused (v1 winner)
    configs.append({"name": "et", "models": ["et"], "weights": [1.0]})
    configs.append({"name": "et+tabpfn_70", "models": ["et","tabpfn"], "weights": [0.7,0.3]})
    configs.append({"name": "et+tabpfn_50", "models": ["et","tabpfn"], "weights": [0.5,0.5]})
    configs.append({"name": "et+tabm_70", "models": ["et","tabm"], "weights": [0.7,0.3]})
    configs.append({"name": "et+tabm_50", "models": ["et","tabm"], "weights": [0.5,0.5]})
    configs.append({"name": "et+pfn+tm", "models": ["et","tabpfn","tabm"], "weights": [0.5,0.25,0.25]})
    # Other solos
    for m in ["lgb","xgb","cb","brf","hgb","tabpfn","tabm"]:
        configs.append({"name": m, "models": [m], "weights": [1.0]})
    # Top combos
    configs.append({"name": "et+lgb", "models": ["et","lgb"], "weights": [0.5,0.5]})
    configs.append({"name": "et+xgb", "models": ["et","xgb"], "weights": [0.5,0.5]})
    configs.append({"name": "et+cb", "models": ["et","cb"], "weights": [0.5,0.5]})
    configs.append({"name": "lgb+xgb+cb", "models": ["lgb","xgb","cb"], "weights": [0.33,0.33,0.34]})
    configs.append({"name": "et+lgb+tabpfn", "models": ["et","lgb","tabpfn"], "weights": [0.4,0.3,0.3]})
    configs.append({"name": "et+lgb+tabm", "models": ["et","lgb","tabm"], "weights": [0.4,0.3,0.3]})
    configs.append({"name": "pfn+tm", "models": ["tabpfn","tabm"], "weights": [0.5,0.5]})
    return configs


def main():
    start = datetime.now()
    print(f"\n{'='*70}")
    print(f"  MEGA SEARCH v2 -- Microstructure + 5 Coins")
    print(f"  {start.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Coins: {list(COINS.keys())}")
    print(f"  Features: 223 (incl CVD, OFI, VPIN, Roll, Amihud)")
    print(f"  Deadline: {DEADLINE}")
    print(f"{'='*70}")

    # Load pre-computed data
    coin_data = {}
    for coin in COINS:
        fpath = MS_DIR / f"{coin}_4h_features.csv"
        if not fpath.exists():
            print(f"  [SKIP] {coin}: no data")
            continue
        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        print(f"  {coin}: {len(df)} bars, {len(df.columns)} features")

        # Label
        h = HORIZONS[-1]; hl = f"label_{h*BM}min"
        kl = COINS[coin].get("k_lower", 0.6)
        labeled = create_labels_triple_barrier(
            df.copy(), h, k_upper_override=K_UPPER, k_lower_override=kl, verbose=False)
        if hl not in labeled.columns:
            print(f"  [SKIP] {coin}: labeling failed")
            continue

        # Feature select
        exclude = {"label","future_return","open","high","low","close","volume"}
        for hh in HORIZONS: exclude.add(f"label_{hh*BM}min"); exclude.add(f"return_{hh*BM}min")
        lk = ["future","target","label_","return_","fwd_","forward_"]
        fcols = [c for c in labeled.columns if c not in exclude
                 and not any(k in c.lower() for k in lk)
                 and not is_excluded_feature(c)
                 and labeled[c].dtype in [np.float64,np.float32,np.int64,np.int32,float,int]]

        mf = COINS[coin]["mf"]
        clean = labeled.replace([np.inf,-np.inf], np.nan).ffill().bfill()
        X = clean[fcols].fillna(0).values
        y = clean[hl].fillna(1).values.astype(int)

        if len(fcols) > mf:
            mi = mutual_info_classif(X[:min(2000,len(X))], y[:min(2000,len(X))],
                                     discrete_features=False, random_state=42, n_neighbors=5)
            top = np.argsort(mi)[-mf:]
            fcols = [fcols[i] for i in sorted(top)]
            X = clean[fcols].fillna(0).values

        coin_data[coin] = {"df": df, "fcols": fcols, "X": X, "y": y}
        print(f"  {coin}: selected {len(fcols)} features")

    configs = generate_configs()
    print(f"\n  {len(configs)} configs x {len(coin_data)} coins")

    best = {}
    rnum = 0

    # Resume: skip already completed configs
    done_configs = set()
    if LOG_FILE.exists():
        for line in open(LOG_FILE):
            d = json.loads(line.strip())
            done_configs.add(d["config"])
            coin = d["coin"]
            if coin not in best or d["avg_pnl_mean"] > best[coin]["avg_pnl_mean"]:
                best[coin] = {"config": d["config"], **{k:v for k,v in d.items() if k not in ["coin","config","ts"]}}
        print(f"  Resuming: {len(done_configs)} configs already done, skipping")

    for cfg in configs:
        if datetime.now() >= DEADLINE: break
        if cfg["name"] in done_configs:
            rnum += 1
            continue
        rnum += 1
        remain = (DEADLINE - datetime.now()).total_seconds() / 3600
        print(f"\n  [{rnum}/{len(configs)}] {cfg['name']} | {remain:.1f}h left")

        for coin in coin_data:
            if datetime.now() >= DEADLINE: break
            cd = coin_data[coin]
            df = cd["df"]; fcols = cd["fcols"]; X = cd["X"]; y = cd["y"]
            idx = df.index; data_end = idx[-1]
            purge_td = timedelta(hours=(MH*2+6)*(BM//60))
            params = COINS[coin]

            y_s1 = (y != LABEL_MAP["HOLD"]).astype(int)
            s1c = np.bincount(y_s1, minlength=2)
            s1w = np.where(s1c>0, len(y_s1)/(2*s1c+1e-10), 1.0)[y_s1]
            tm = y != LABEL_MAP["HOLD"]
            y_s2 = (y[tm] == LABEL_MAP["UP"]).astype(int)
            s2c = np.bincount(y_s2, minlength=2)
            s2w = np.where(s2c>0, len(y_s2)/(2*s2c+1e-10), 1.0)[y_s2]

            window_results = []
            t0 = time.time()

            for w in OOS_WINDOWS:
                oos_end = data_end - timedelta(days=w["end_offset"])
                oos_start = oos_end - timedelta(days=w["days"])
                train_end = oos_start - purge_td
                n_train = (idx <= train_end).sum()
                if n_train > len(X): n_train = len(X)
                X_tr = X[:n_train]; y_s1_tr = y_s1[:n_train]; s1w_tr = s1w[:n_train]
                tm_tr = tm[:n_train]; y_s2_tr = y_s2[:tm_tr.sum()]; s2w_tr = s2w[:len(y_s2_tr)]

                oos_df = df[(idx >= oos_start) & (idx <= oos_end)]
                if len(oos_df) < 10: continue
                Xo = oos_df.replace([np.inf,-np.inf], np.nan).ffill().bfill()[fcols].fillna(0).values

                try:
                    s1_list = [fit_predict(m, X_tr, y_s1_tr, Xo, s1w_tr) for m in cfg["models"]]
                    s1_list = [p for p in s1_list if p is not None]
                    if not s1_list: continue
                    wts = cfg["weights"][:len(s1_list)]; ws = sum(wts)
                    s1p = sum(w/ws*p for w,p in zip(wts, s1_list))
                    s1pred = (s1p[:,1] >= params["th"]).astype(int)

                    s2_list = [fit_predict(m, X_tr[tm_tr], y_s2_tr, Xo, s2w_tr) for m in cfg["models"]]
                    s2_list = [p for p in s2_list if p is not None]
                    if not s2_list: continue
                    wts2 = cfg["weights"][:len(s2_list)]; ws2 = sum(wts2)
                    s2p = sum(w/ws2*p for w,p in zip(wts2, s2_list))
                    s2pred = np.argmax(s2p, axis=1)

                    ev = evaluate(oos_df, s1pred, s2pred, coin)
                    window_results.append({"avg": ev["avg_net_pnl"], "total": ev["total_net_pnl"],
                                           "trades": ev["trade_count"], "dd": ev["max_dd"]})
                except Exception as e:
                    continue

            valid = [r for r in window_results if r["trades"] > 0]
            if not valid: continue

            result = {
                "n_windows": len(valid),
                "avg_pnl_mean": round(np.mean([r["avg"] for r in valid]), 6),
                "total_pnl_sum": round(np.sum([r["total"] for r in valid]), 6),
                "avg_trades": round(np.mean([r["trades"] for r in valid]), 1),
                "avg_dd": round(np.mean([r["dd"] for r in valid]), 6),
            }

            elapsed = time.time() - t0
            print(f"    {coin}: avg={result['avg_pnl_mean']:+.4%} total={result['total_pnl_sum']:+.4%} "
                  f"trades={result['avg_trades']:.0f} [{elapsed:.1f}s]")

            prev = best.get(coin, {}).get("avg_pnl_mean", -999)
            if result["avg_pnl_mean"] > prev:
                best[coin] = {"config": cfg["name"], **result}
                print(f"    *** NEW BEST for {coin}!")

            entry = {"coin": coin, "config": cfg["name"], **result, "ts": datetime.now().isoformat()}
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    # Final
    print(f"\n{'='*70}")
    print(f"  MEGA SEARCH v2 COMPLETE ({rnum} configs)")
    print(f"{'='*70}")
    for coin in COINS:
        b = best.get(coin)
        if b:
            print(f"  {coin:>5s}: {b['config']:>20s} | avg={b['avg_pnl_mean']:+.4%} "
                  f"total={b['total_pnl_sum']:+.4%} trades={b['avg_trades']:.0f} dd={b['avg_dd']:.4%}")

    with open(REPORT_DIR / "mega_search_v2_final.json", "w") as f:
        json.dump({"best": best, "configs_tested": rnum, "coins": list(COINS.keys()),
                   "features": "223 (microstructure included)"}, f, indent=2, default=str)

    print(f"\n  Finished at {datetime.now().strftime('%H:%M')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
