"""TabM ADA Multi-OOS Validation.

ADA blend 70/30이 baseline을 2.9x 상회한 결과가 noise인지 검증.
4개 서로 다른 OOS 구간에서 반복 테스트.

Window 1: -8w ~ -4w (가장 과거)
Window 2: -6w ~ -2w
Window 3: -4w ~ now (가장 최근, 기존 테스트와 일부 겹침)
Window 4: -7w ~ -3w
"""

import sys
sys.path.insert(0, "C:/Users/RJ/Desktop/CLAUDE_CRYPTO_AGENT")
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import json, yaml, warnings, time
import numpy as np, pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

from tabm import TabM
from src.data.crawlers.crypto_ohlcv import fetch_all_top10
from src.data.crawlers.macro_commodity_crawler import crawl_all_macro_data
from src.models.masking_loop import create_labels_triple_barrier, LABEL_MAP, HORIZONS
from src.models.enhanced_ensemble import EnhancedEnsemble
from src.evaluation.trade_level_ev import compute_trade_level_ev
from src.execution.cost_model import CostModel, FeeSchedule, FundingConfig, MissFillConfig
from src.utils.config import bar_minutes as cfg_bar_minutes
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CM = CostModel(
    fee_schedule=FeeSchedule(maker_fee=CC["maker_fee"], taker_fee=CC["taker_fee"],
        slippage_entry=CC["slippage_entry"], slippage_exit_limit=CC["slippage_exit_limit"],
        slippage_exit_market=CC["slippage_exit_market"]),
    funding_config=FundingConfig(interval_hours=CC["funding_interval_hours"],
        default_rate=CC["funding_default_rate"]),
    miss_fill_config=MissFillConfig(reject_prob=CC["miss_fill_reject_prob"],
        missed_ev_pct=CC["miss_fill_missed_ev"]))

COIN = "ADA"
PARAMS = COINS[COIN]
KU = COMMON["k_upper"]
KL = PARAMS.get("k_lower_override", COMMON["k_lower"])
BLOCKED = PARAMS.get("blocked_regimes_override", CFG["blocked_regimes"])

# 4 OOS windows (28 days each)
OOS_WINDOWS = [
    {"name": "W1_oldest", "oos_end_offset": 28, "oos_days": 28},
    {"name": "W2_mid1",   "oos_end_offset": 14, "oos_days": 28},
    {"name": "W3_mid2",   "oos_end_offset": 21, "oos_days": 28},
    {"name": "W4_recent", "oos_end_offset": 0,  "oos_days": 28},
]


def isex(c):
    return any(kw in c.lower() for kw in EXK)


class TabMClassifier:
    def __init__(self, n_features, n_classes=2, k=64, d_block=128,
                 n_blocks=3, lr=1e-3, epochs=200, batch_size=256, device="cuda"):
        self.n_features = n_features
        self.n_classes = n_classes
        self.k = k; self.d_block = d_block; self.n_blocks = n_blocks
        self.lr = lr; self.epochs = epochs; self.batch_size = batch_size
        self.device = device; self.model = None
        self.scaler = StandardScaler()

    def _build(self):
        return TabM.make(
            n_num_features=self.n_features, cat_cardinalities=[], d_out=self.n_classes,
            arch_type="tabm", k=self.k, d_block=self.d_block,
            n_blocks=self.n_blocks, dropout=0.1,
        ).to(self.device)

    def fit(self, X, y, sample_weight=None):
        Xs = self.scaler.fit_transform(X).astype(np.float32)
        self.model = self._build()
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        Xt = torch.tensor(Xs, device=self.device)
        yt = torch.tensor(y, dtype=torch.long, device=self.device)
        wt = torch.tensor(sample_weight, dtype=torch.float32, device=self.device) if sample_weight is not None else None

        self.model.train()
        n = len(Xt); best_loss = float("inf"); best_state = None; no_imp = 0
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            eloss = 0; nb = 0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i+self.batch_size]
                out = self.model(x_num=Xt[idx], x_cat=None).mean(dim=1)
                if wt is not None:
                    loss = (nn.functional.cross_entropy(out, yt[idx], reduction="none") * wt[idx]).mean()
                else:
                    loss = nn.functional.cross_entropy(out, yt[idx])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step(); eloss += loss.item(); nb += 1
            sch.step()
            al = eloss / max(nb, 1)
            if al < best_loss: best_loss = al; no_imp = 0; best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else: no_imp += 1
            if no_imp >= 20: break
        if best_state: self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(X).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            out = self.model(x_num=torch.tensor(Xs, device=self.device), x_cat=None).mean(dim=1)
            return torch.softmax(out, dim=1).cpu().numpy()


def evaluate(oos_df, s1_pred, s2_pred):
    n = len(oos_df); close = oos_df["close"].values
    rm = np.ones(n, dtype=bool)
    for i in range(n):
        if i < 42: rm[i] = "UNKNOWN" not in BLOCKED; continue
        seg = close[max(0,i-41):i+1]
        ef = pd.Series(seg).ewm(span=10).mean().iloc[-1]
        es = pd.Series(seg).ewm(span=30).mean().iloc[-1]
        rs = np.std(np.diff(seg)/seg[:-1]) if len(seg)>1 else 0
        if abs(ef/es-1)<0.005 and rs<0.02: rm[i]=False
        if "UNKNOWN" in BLOCKED and i<42: rm[i]=False
    s1f = s1_pred.copy(); s1f[~rm] = 0
    ev = compute_trade_level_ev(oos_df, s1f, np.ones(n)*0.5, s2_pred, np.ones(n)*0.5,
        k_upper=KU, k_lower=KL, max_hold=MH, risk_frac=RF, cost_model=CM)
    return ev


def main():
    start = datetime.now()
    print(f"\n{'='*70}")
    print(f"  TabM ADA Multi-OOS Validation")
    print(f"  {start.strftime('%Y-%m-%d %H:%M')}")
    print(f"  4 OOS windows x (baseline + TabM blend)")
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

    for w in OOS_WINDOWS:
        wname = w["name"]
        oos_end = data_end - timedelta(days=w["oos_end_offset"])
        oos_start = oos_end - timedelta(days=w["oos_days"])
        train_end = oos_start - purge_td

        train_df = df[idx <= train_end]
        oos_df = df[(idx >= oos_start) & (idx <= oos_end)]

        print(f"\n  --- {wname}: OOS {str(oos_start)[:10]} ~ {str(oos_end)[:10]} ({len(oos_df)} bars) ---")

        # Features
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
        s2c = np.bincount(y_s2, minlength=2)
        s2w = np.where(s2c>0, len(y_s2)/(2*s2c+1e-10), 1.0)[y_s2]

        # ARM A: Baseline
        s1a = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
        s1a.fit(X, y_s1, sample_weight=s1w)
        s1pa = s1a.predict_proba(Xo)
        s1preda = (s1pa[:,1] >= PARAMS["stage1_threshold"]).astype(int)

        s2a = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
        s2a.fit(X[tm], y_s2, sample_weight=s2w)
        s2pa = s2a.predict_proba(Xo)
        s2preda = np.argmax(s2pa, axis=1)

        ev_a = evaluate(oos_df, s1preda, s2preda)
        print(f"    Baseline:    {ev_a['trade_count']}T avg={ev_a['avg_net_pnl']:+.4%} total={ev_a['total_net_pnl']:+.4%}")

        # ARM B: TabM k64 blend 70/30
        tabm_s1 = TabMClassifier(n_features=X.shape[1], k=64, d_block=128, n_blocks=3, epochs=200, device=DEVICE)
        tabm_s1.fit(X, y_s1, sample_weight=s1w)
        s1pt = tabm_s1.predict_proba(Xo)

        tabm_s2 = TabMClassifier(n_features=X[tm].shape[1], k=64, d_block=128, n_blocks=3, epochs=200, device=DEVICE)
        tabm_s2.fit(X[tm], y_s2, sample_weight=s2w)
        s2pt = tabm_s2.predict_proba(Xo)

        # Blend 70/30
        s1pb = 0.7 * s1pa + 0.3 * s1pt
        s1predb = (s1pb[:,1] >= PARAMS["stage1_threshold"]).astype(int)
        s2pb = 0.7 * s2pa + 0.3 * s2pt
        s2predb = np.argmax(s2pb, axis=1)

        ev_b = evaluate(oos_df, s1predb, s2predb)
        print(f"    TabM blend:  {ev_b['trade_count']}T avg={ev_b['avg_net_pnl']:+.4%} total={ev_b['total_net_pnl']:+.4%}")

        # TabM standalone
        s1predt = (s1pt[:,1] >= PARAMS["stage1_threshold"]).astype(int)
        s2predt = np.argmax(s2pt, axis=1)
        ev_t = evaluate(oos_df, s1predt, s2predt)
        print(f"    TabM only:   {ev_t['trade_count']}T avg={ev_t['avg_net_pnl']:+.4%} total={ev_t['total_net_pnl']:+.4%}")

        winner = "BLEND" if ev_b["avg_net_pnl"] > ev_a["avg_net_pnl"] else "BASELINE"
        print(f"    Winner: {winner}")

        results.append({
            "window": wname,
            "oos_period": f"{str(oos_start)[:10]} ~ {str(oos_end)[:10]}",
            "baseline": {"trades": ev_a["trade_count"], "avg": ev_a["avg_net_pnl"], "total": ev_a["total_net_pnl"]},
            "blend70": {"trades": ev_b["trade_count"], "avg": ev_b["avg_net_pnl"], "total": ev_b["total_net_pnl"]},
            "tabm_only": {"trades": ev_t["trade_count"], "avg": ev_t["avg_net_pnl"], "total": ev_t["total_net_pnl"]},
            "winner": winner,
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"  MULTI-OOS SUMMARY (ADA)")
    print(f"{'='*70}")
    print(f"  {'Window':>12s} | {'Baseline avg':>14s} | {'Blend avg':>14s} | {'TabM avg':>14s} | Winner")
    print(f"  {'-'*75}")

    baseline_wins = 0; blend_wins = 0
    for r in results:
        b = r["baseline"]; bl = r["blend70"]; t = r["tabm_only"]
        print(f"  {r['window']:>12s} | {b['avg']:>+14.4%} | {bl['avg']:>+14.4%} | {t['avg']:>+14.4%} | {r['winner']}")
        if r["winner"] == "BASELINE": baseline_wins += 1
        else: blend_wins += 1

    print(f"\n  Score: Baseline {baseline_wins} - Blend {blend_wins}")
    verdict = "BLEND CONFIRMED" if blend_wins >= 3 else "BASELINE HOLDS" if baseline_wins >= 3 else "INCONCLUSIVE"
    print(f"  Verdict: {verdict}")

    with open(REPORT_DIR / "tabm_ada_multi_oos.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = (datetime.now() - start).total_seconds() / 60
    print(f"\n  Completed in {elapsed:.1f} min")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
