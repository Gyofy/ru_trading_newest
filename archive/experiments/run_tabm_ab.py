"""TabM A/B Test -- 7-model ensemble vs TabM vs Ensemble+TabM blend.

TabM = Parameter-Efficient MLP Ensemble (ICLR 2025, Yandex Research)
Core idea: k implicit MLPs sharing weights via BatchEnsemble adapters (R, S, B).
Weight sharing = effective regularization for tabular data.

ARM A: EnhancedEnsemble (7-model + stacking) -- baseline
ARM B: TabM standalone (k=32 implicit MLPs)
ARM C: Ensemble(70%) + TabM(30%) soft voting
ARM D: TabM only, tuned (wider/deeper)
"""

import sys
sys.path.insert(0, "C:/Users/RJ/Desktop/CLAUDE_CRYPTO_AGENT")
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import json, yaml, warnings, time, traceback
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
OOS_DAYS = 56
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


class TabMClassifier:
    """Scikit-learn compatible wrapper for TabM."""

    def __init__(self, n_features, n_classes=2, k=32, d_block=192,
                 n_blocks=3, lr=1e-3, epochs=200, batch_size=256,
                 device="cuda", arch_type="tabm"):
        self.n_features = n_features
        self.n_classes = n_classes
        self.k = k
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.arch_type = arch_type
        self.model = None
        self.scaler = StandardScaler()

    def _build_model(self):
        model = TabM.make(
            n_num_features=self.n_features,
            cat_cardinalities=[],
            d_out=self.n_classes,
            arch_type=self.arch_type,
            k=self.k,
            d_block=self.d_block,
            n_blocks=self.n_blocks,
            dropout=0.1,
        ).to(self.device)
        return model

    def fit(self, X, y, sample_weight=None):
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)
        self.model = self._build_model()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        X_t = torch.tensor(X_scaled, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)

        if sample_weight is not None:
            w_t = torch.tensor(sample_weight, dtype=torch.float32, device=self.device)
        else:
            w_t = None

        self.model.train()
        n = len(X_t)
        best_loss = float("inf")
        patience = 20
        no_improve = 0

        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, self.batch_size):
                idx = perm[i:i+self.batch_size]
                xb = X_t[idx]
                yb = y_t[idx]

                # TabM expects (batch, features) for x_num
                out = self.model(x_num=xb, x_cat=None)  # (batch, k, n_classes)
                # Average over k ensemble members
                logits = out.mean(dim=1)  # (batch, n_classes)

                if w_t is not None:
                    wb = w_t[idx]
                    loss = nn.functional.cross_entropy(logits, yb, reduction="none")
                    loss = (loss * wb).mean()
                else:
                    loss = nn.functional.cross_entropy(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        # Load best
        if best_state:
            self.model.load_state_dict(best_state)

        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X).astype(np.float32)
        self.model.eval()
        X_t = torch.tensor(X_scaled, device=self.device)

        with torch.no_grad():
            out = self.model(x_num=X_t, x_cat=None)  # (batch, k, n_classes)
            logits = out.mean(dim=1)  # (batch, n_classes)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


def prepare_data():
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
    params = COINS[coin]
    h = HORIZONS[-1]; hl = f"label_{h*BM}min"
    ku, kl = ck(coin, "k_upper"), ck(coin, "k_lower")
    labeled = create_labels_triple_barrier(
        train_df.copy(), h, k_upper_override=ku, k_lower_override=kl, verbose=False)
    if hl not in labeled.columns: return None, None, None
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
    return fcols, X, y


def evaluate_arm(oos_df, s1_pred, s2_pred, coin, label):
    params = COINS[coin]
    ku, kl = ck(coin, "k_upper"), ck(coin, "k_lower")
    blocked = COINS[coin].get("blocked_regimes_override", CFG["blocked_regimes"])
    n = len(oos_df)
    close = oos_df["close"].values
    regime_mask = np.ones(n, dtype=bool)
    for i in range(n):
        if i < 42: continue
        seg = close[max(0, i-41):i+1]
        ef = pd.Series(seg).ewm(span=10).mean().iloc[-1]
        es = pd.Series(seg).ewm(span=30).mean().iloc[-1]
        rs = np.std(np.diff(seg)/seg[:-1]) if len(seg)>1 else 0
        if abs(ef/es - 1) < 0.005 and rs < 0.02:
            regime_mask[i] = False
        if "UNKNOWN" in blocked and i < 42:
            regime_mask[i] = False
    s1_f = s1_pred.copy(); s1_f[~regime_mask] = 0
    ev = compute_trade_level_ev(
        oos_df, s1_f, np.ones(n)*0.5, s2_pred, np.ones(n)*0.5,
        k_upper=ku, k_lower=kl, max_hold=MH, risk_frac=RF, cost_model=CM)
    return {"label": label, "coin": coin, "trades": ev["trade_count"],
            "avg_pnl": ev["avg_net_pnl"], "total_pnl": ev["total_net_pnl"],
            "win_rate": ev["win_rate"], "max_dd": ev["max_dd"], "score": ev["score"]}


def main():
    start = datetime.now()
    print(f"\n{'='*70}")
    print(f"  TabM A/B TEST (ICLR 2025)")
    print(f"  {start.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*70}")

    data = prepare_data()
    results = {}

    # TabM configs to test
    tabm_configs = [
        {"name": "TabM_k16", "k": 16, "d_block": 128, "n_blocks": 3, "epochs": 200},
        {"name": "TabM_k32", "k": 32, "d_block": 192, "n_blocks": 3, "epochs": 200},
        {"name": "TabM_k64", "k": 64, "d_block": 128, "n_blocks": 4, "epochs": 300},
    ]

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
        fcols, X_train, y_train = prepare_features(train_df, coin)
        if fcols is None: continue

        oc = oos_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        X_oos = oc[fcols].fillna(0).values

        y_s1 = (y_train != LABEL_MAP["HOLD"]).astype(int)
        if len(np.unique(y_s1)) < 2: continue
        s1c = np.bincount(y_s1, minlength=2)
        s1w = np.where(s1c > 0, len(y_s1)/(2*s1c+1e-10), 1.0)[y_s1]
        tm = y_train != LABEL_MAP["HOLD"]
        y_s2 = (y_train[tm] == LABEL_MAP["UP"]).astype(int)
        s2c = np.bincount(y_s2, minlength=2)
        s2w = np.where(s2c > 0, len(y_s2)/(2*s2c+1e-10), 1.0)[y_s2]

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
        print(f"    {t_a:.1f}s | {ev_a['trades']}T avg={ev_a['avg_pnl']:+.4%} "
              f"total={ev_a['total_pnl']:+.4%} dd={ev_a['max_dd']:.4%}")

        coin_results = {"A_ensemble": ev_a, "time_A": round(t_a, 1)}

        # ==================== ARM B/C/D: TabM configs ====================
        for cfg in tabm_configs:
            print(f"\n  ARM {cfg['name']}:")
            t0 = time.time()

            try:
                # S1
                tabm_s1 = TabMClassifier(
                    n_features=X_train.shape[1], n_classes=2,
                    k=cfg["k"], d_block=cfg["d_block"], n_blocks=cfg["n_blocks"],
                    epochs=cfg["epochs"], device=DEVICE)
                tabm_s1.fit(X_train, y_s1, sample_weight=s1w)
                s1p_t = tabm_s1.predict_proba(X_oos)
                s1pred_t = (s1p_t[:, 1] >= params["stage1_threshold"]).astype(int)

                # S2
                tabm_s2 = TabMClassifier(
                    n_features=X_train[tm].shape[1], n_classes=2,
                    k=cfg["k"], d_block=cfg["d_block"], n_blocks=cfg["n_blocks"],
                    epochs=cfg["epochs"], device=DEVICE)
                tabm_s2.fit(X_train[tm], y_s2, sample_weight=s2w)
                s2p_t = tabm_s2.predict_proba(X_oos)
                s2pred_t = np.argmax(s2p_t, axis=1)

                t_t = time.time() - t0

                # Standalone
                ev_t = evaluate_arm(oos_df, s1pred_t, s2pred_t, coin, cfg["name"])
                print(f"    Standalone: {t_t:.1f}s | {ev_t['trades']}T "
                      f"avg={ev_t['avg_pnl']:+.4%} total={ev_t['total_pnl']:+.4%} "
                      f"dd={ev_t['max_dd']:.4%}")

                # Blend 70/30
                s1p_blend = 0.7 * s1p_a + 0.3 * s1p_t
                s1pred_blend = (s1p_blend[:, 1] >= params["stage1_threshold"]).astype(int)
                s2p_blend = 0.7 * s2p_a + 0.3 * s2p_t
                s2pred_blend = np.argmax(s2p_blend, axis=1)
                ev_blend = evaluate_arm(oos_df, s1pred_blend, s2pred_blend, coin,
                                        f"{cfg['name']}_blend70")
                print(f"    Blend 70/30: {ev_blend['trades']}T "
                      f"avg={ev_blend['avg_pnl']:+.4%} total={ev_blend['total_pnl']:+.4%}")

                # Blend 50/50
                s1p_b5 = 0.5 * s1p_a + 0.5 * s1p_t
                s1pred_b5 = (s1p_b5[:, 1] >= params["stage1_threshold"]).astype(int)
                s2p_b5 = 0.5 * s2p_a + 0.5 * s2p_t
                s2pred_b5 = np.argmax(s2p_b5, axis=1)
                ev_b5 = evaluate_arm(oos_df, s1pred_b5, s2pred_b5, coin,
                                     f"{cfg['name']}_blend50")
                print(f"    Blend 50/50: {ev_b5['trades']}T "
                      f"avg={ev_b5['avg_pnl']:+.4%} total={ev_b5['total_pnl']:+.4%}")

                # Agreement
                agree = (s1pred_a == s1pred_t).mean()
                print(f"    Agreement: {agree:.1%}")

                coin_results[cfg["name"]] = ev_t
                coin_results[f"{cfg['name']}_blend70"] = ev_blend
                coin_results[f"{cfg['name']}_blend50"] = ev_b5
                coin_results[f"time_{cfg['name']}"] = round(t_t, 1)
                coin_results[f"agree_{cfg['name']}"] = round(agree, 4)

            except Exception as e:
                print(f"    FAILED: {e}")
                traceback.print_exc()
                coin_results[cfg["name"]] = {"error": str(e)}

        results[coin] = coin_results

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Coin':>5s} | {'ARM':>20s} | {'Trades':>6s} | {'Avg PnL':>10s} | "
          f"{'Total':>10s} | {'MDD':>8s}")
    print(f"  {'-'*75}")

    for coin in COINS:
        r = results.get(coin, {})
        for key, val in sorted(r.items()):
            if isinstance(val, dict) and "trades" in val and val["trades"] > 0:
                print(f"  {coin:>5s} | {key:>20s} | {val['trades']:>6d} | "
                      f"{val['avg_pnl']:>+10.4%} | {val['total_pnl']:>+10.4%} | "
                      f"{val.get('max_dd', 0):>8.4%}")

    with open(REPORT_DIR / "tabm_ab_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = (datetime.now() - start).total_seconds() / 60
    print(f"\n  Completed in {elapsed:.1f} min")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
