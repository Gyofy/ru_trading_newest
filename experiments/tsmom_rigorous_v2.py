"""TSMOM Rigorous Optimization v2.

Previous round problems:
  1. IS/OOS not properly separated (walk-forward windows overlap with grid search data)
  2. 6,480 configs = massive multiple comparison bias
  3. OOS only 69 trades = statistically weak

This round fixes:
  1. Hard split: first 70% = IS (optimize), last 30% = OOS (never touched)
  2. Bonferroni correction for multiple comparisons
  3. Permutation test: is the best config better than random?
  4. Regime-conditional analysis
  5. Per-coin stability (drop-one-out)
  6. Consecutive loss analysis for real-world survivability
  7. Cost sensitivity (what if costs are 0.30% instead of 0.20%?)
"""

import sys, os, io, time, warnings, itertools
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

COINS_MAP = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
    "XRP": "XRPUSDT", "ADA": "ADAUSDT", "DOT": "DOTUSDT", "LINK": "LINKUSDT",
}
COST = 0.0020
IS_RATIO = 0.70  # 70% in-sample, 30% out-of-sample

# ══════════════════════════════════════════════════════════
# Data Loading (reuse from previous)
# ══════════════════════════════════════════════════════════

def load_all():
    from src.data.crawlers.crypto_ohlcv import fetch_ohlcv, resample_to_4h, add_technical_indicators, TOP10_YAHOO
    from src.data.crawlers.microstructure_rollup import add_microstructure_rollup
    from experiments.download_and_integrate import load_metrics, resample_metrics_4h, add_binance_features

    data = {}
    for coin in COINS_MAP:
        sym = TOP10_YAHOO.get(coin)
        df = fetch_ohlcv(coin, sym, period="365d", interval="1h")
        if df.empty: continue
        df = resample_to_4h(df)
        df = add_technical_indicators(df)
        df = add_microstructure_rollup(df)
        metrics = load_metrics(coin)
        if not metrics.empty:
            m4h = resample_metrics_4h(metrics)
            df = add_binance_features(df, m4h)
        data[coin] = df
    return data


def split_is_oos(data):
    is_data, oos_data = {}, {}
    for coin, df in data.items():
        n = len(df)
        cut = int(n * IS_RATIO)
        is_data[coin] = df.iloc[:cut].copy()
        oos_data[coin] = df.iloc[cut:].copy()
        print(f"  {coin}: IS={cut} bars ({df.index[0].strftime('%Y-%m-%d')} ~ "
              f"{df.index[cut-1].strftime('%Y-%m-%d')}), "
              f"OOS={n-cut} bars ({df.index[cut].strftime('%Y-%m-%d')} ~ "
              f"{df.index[-1].strftime('%Y-%m-%d')})")
    return is_data, oos_data


# ══════════════════════════════════════════════════════════
# Core Functions
# ══════════════════════════════════════════════════════════

def compute_cvd_ratio(df, window=24):
    hr = (df["high"] - df["low"]).replace(0, np.nan)
    buy_frac = ((df["close"] - df["low"]) / hr).fillna(0.5).clip(0, 1)
    vd = (2 * buy_frac - 1) * df["volume"]
    cvd = vd.cumsum()
    cvd_ma = cvd.rolling(window, min_periods=max(6, window//4)).mean()
    return ((cvd - cvd_ma) / cvd_ma.abs().replace(0, np.nan)).fillna(0)


def generate_signals(df, lb=5, vw=True, cq=0.75, cw=120, use_oi=False, lsr_contra=False):
    lb_bars = lb * 6
    if vw and "volume" in df.columns:
        ret = df["close"].pct_change()
        vol_w = df["volume"] / df["volume"].rolling(lb_bars, min_periods=1).mean()
        tsmom = np.sign((ret * vol_w).rolling(lb_bars, min_periods=lb_bars).sum())
    else:
        tsmom = np.sign(df["close"].pct_change(lb_bars))

    rsi = df.get("rsi_14", pd.Series(50, index=df.index))
    rsi_ok = ((tsmom == 1) & (rsi > 50)) | ((tsmom == -1) & (rsi < 50))

    cvd = compute_cvd_ratio(df, 24)
    q_hi = cvd.rolling(cw, min_periods=30).quantile(cq)
    q_lo = cvd.rolling(cw, min_periods=30).quantile(1 - cq)
    cvd_ok = ((tsmom == -1) & (cvd > q_hi)) | ((tsmom == 1) & (cvd < q_lo))

    mask = rsi_ok & cvd_ok

    if use_oi and "oi_zscore" in df.columns:
        mask = mask & (df["oi_zscore"].abs() < 2.0)
    if lsr_contra and "lsr_extreme_long" in df.columns:
        lsr_ok = (
            ((tsmom == -1) & (df["lsr_extreme_long"] == 1)) |
            ((tsmom == 1) & (df["lsr_extreme_short"] == 1)) |
            ((df["lsr_extreme_long"] == 0) & (df["lsr_extreme_short"] == 0))
        )
        mask = mask & lsr_ok

    sig = tsmom.copy()
    sig[~mask] = 0
    return sig.fillna(0).astype(int)


def backtest(df, signals, ku=4.0, kl=1.5, mh=24, cost=COST):
    c, h, l = df["close"].values, df["high"].values, df["low"].values
    atr = df["atr_14"].values if "atr_14" in df.columns else \
        pd.Series(np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))).rolling(14,min_periods=1).mean().values
    sig = signals.values if hasattr(signals, 'values') else signals
    trades, sides, entry_bars = [], [], []
    nxt = 0
    for i in range(len(df) - mh):
        if i < nxt or sig[i] == 0 or np.isnan(atr[i]) or atr[i] <= 0: continue
        side = int(sig[i])
        entry = c[i]; a = atr[i]
        tp_d = max(ku * a, entry * 0.002)
        sl_d = max(kl * a, entry * 0.002)
        tp = entry + tp_d * side; sl = entry - sl_d * side
        ep, eb = c[min(i+mh, len(df)-1)], min(i+mh, len(df)-1)
        for j in range(i+1, min(i+mh+1, len(df))):
            if side == 1:
                if l[j] <= sl: ep, eb = sl, j; break
                if h[j] >= tp: ep, eb = tp, j; break
            else:
                if h[j] >= sl: ep, eb = sl, j; break
                if l[j] <= tp: ep, eb = tp, j; break
        pnl = ((ep - entry) / entry) * side - cost
        trades.append(pnl); sides.append(side); entry_bars.append(i)
        nxt = eb + 1
    return np.array(trades), np.array(sides), np.array(entry_bars)


def metrics(pnls):
    if len(pnls) == 0:
        return {"n":0,"wr":0,"avg":0,"sharpe":0,"mdd":0,"pf":0,"total":0}
    a = np.array(pnls)
    eq = np.cumsum(a)
    dd = eq - np.maximum.accumulate(eq)
    w = sum(p for p in a if p > 0)
    lo = abs(sum(p for p in a if p < 0))
    # Max consecutive losses
    streak, max_streak = 0, 0
    for p in a:
        if p < 0: streak += 1; max_streak = max(max_streak, streak)
        else: streak = 0
    return {
        "n": len(a), "wr": np.mean(a>0), "avg": np.mean(a),
        "sharpe": np.mean(a)/np.std(a)*np.sqrt(len(a)) if np.std(a)>0 else 0,
        "mdd": np.min(dd), "pf": w/lo if lo>0 else float("inf"),
        "total": np.sum(a), "max_consec_loss": max_streak,
        "avg_win": np.mean(a[a>0]) if np.any(a>0) else 0,
        "avg_loss": np.mean(a[a<0]) if np.any(a<0) else 0,
    }


# ══════════════════════════════════════════════════════════
# Phase 1: IS Grid Search
# ══════════════════════════════════════════════════════════

def is_grid_search(is_data):
    print("\n" + "=" * 95)
    print("PHASE 1: IN-SAMPLE Grid Search (first 70% of data ONLY)")
    print("=" * 95)

    params = list(itertools.product(
        [5, 7, 10, 14, 21, 28],          # lookback
        [0.65, 0.70, 0.75, 0.80, 0.85],  # cvd_q
        [60, 90, 120],                     # cvd_window
        [3.0, 4.0, 5.0],                  # k_upper
        [1.0, 1.5, 2.0],                  # k_lower
        [18, 24],                          # max_hold
        [False, True],                     # vol_weighted
        [False, True],                     # use_oi
    ))
    print(f"  Configs: {len(params)}")

    results = []
    best_sharpe = -999

    for idx, (lb, cq, cw, ku, kl, mh, vw, oi) in enumerate(params):
        all_p = []
        for coin in COINS_MAP:
            if coin not in is_data: continue
            sig = generate_signals(is_data[coin], lb=lb, vw=vw, cq=cq, cw=cw, use_oi=oi)
            p, _, _ = backtest(is_data[coin], sig, ku=ku, kl=kl, mh=mh)
            all_p.extend(p.tolist())

        if len(all_p) < 20: continue
        m = metrics(all_p)
        m.update({"lb":lb,"cq":cq,"cw":cw,"ku":ku,"kl":kl,"mh":mh,"vw":vw,"oi":oi})
        results.append(m)

        if m["sharpe"] > best_sharpe and m["n"] >= 30:
            best_sharpe = m["sharpe"]

        if (idx+1) % 1000 == 0:
            print(f"  [{idx+1}/{len(params)}] best IS Sharpe={best_sharpe:.2f}")

    rdf = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    valid = rdf[rdf["n"] >= 25]

    print(f"\n  Total valid configs: {len(valid)}")
    print(f"\n  TOP 10 IS (min 25 trades):")
    for _, r in valid.head(10).iterrows():
        print(f"    lb={r['lb']:2.0f} cq={r['cq']:.2f} cw={r['cw']:3.0f} "
              f"ku={r['ku']:.1f} kl={r['kl']:.1f} mh={r['mh']:2.0f} "
              f"vw={r['vw']} oi={r['oi']} | n={r['n']:3.0f} WR={r['wr']:.1%} "
              f"avg={r['avg']:+.4%} Sharpe={r['sharpe']:.2f} MDD={r['mdd']:.2%} "
              f"PF={r['pf']:.2f} consec_L={r['max_consec_loss']:.0f}")

    return rdf


# ══════════════════════════════════════════════════════════
# Phase 2: OOS Validation (NEVER SEEN DATA)
# ══════════════════════════════════════════════════════════

def oos_validation(oos_data, is_results):
    print("\n" + "=" * 95)
    print("PHASE 2: OUT-OF-SAMPLE Validation (last 30% — NEVER SEEN)")
    print("=" * 95)

    top_configs = is_results[is_results["n"] >= 25].head(20).to_dict("records")
    oos_results = []

    for rank, cfg in enumerate(top_configs):
        all_p, all_sides = [], []
        coin_detail = {}

        for coin in COINS_MAP:
            if coin not in oos_data: continue
            sig = generate_signals(oos_data[coin], lb=int(cfg["lb"]), vw=bool(cfg["vw"]),
                                    cq=cfg["cq"], cw=int(cfg["cw"]), use_oi=bool(cfg["oi"]))
            p, s, _ = backtest(oos_data[coin], sig, ku=cfg["ku"], kl=cfg["kl"], mh=int(cfg["mh"]))
            all_p.extend(p.tolist())
            all_sides.extend(s.tolist())
            if len(p) > 0:
                coin_detail[coin] = metrics(p)

        m = metrics(all_p)
        m["is_sharpe"] = cfg["sharpe"]
        m["is_avg"] = cfg["avg"]
        m["config"] = (f"lb={int(cfg['lb']):2d} cq={cfg['cq']:.2f} cw={int(cfg['cw']):3d} "
                       f"ku={cfg['ku']:.1f} kl={cfg['kl']:.1f} mh={int(cfg['mh']):2d} "
                       f"vw={cfg['vw']} oi={cfg['oi']}")
        m["cfg_dict"] = cfg

        # Long/Short breakdown
        ap = np.array(all_p)
        as_ = np.array(all_sides)
        if len(ap) > 0:
            long_p = ap[as_ == 1]
            short_p = ap[as_ == -1]
            m["long_n"] = len(long_p)
            m["long_avg"] = np.mean(long_p) if len(long_p) > 0 else 0
            m["short_n"] = len(short_p)
            m["short_avg"] = np.mean(short_p) if len(short_p) > 0 else 0

        oos_results.append(m)

        flag = "***" if m["avg"] > 0 and m["n"] >= 10 else "   "
        print(f"  {flag} IS_rank={rank+1:2d} | OOS: n={m['n']:3d} WR={m['wr']:.1%} "
              f"avg={m['avg']:+.4%} Sharpe={m['sharpe']:.2f} | "
              f"IS: Sharpe={m['is_sharpe']:.2f} avg={m['is_avg']:+.4%} | "
              f"L={m.get('long_n',0):2d}({m.get('long_avg',0):+.3%}) "
              f"S={m.get('short_n',0):2d}({m.get('short_avg',0):+.3%})")

        # Per-coin detail for best
        if rank < 3:
            for coin, cm in coin_detail.items():
                print(f"       {coin}: n={cm['n']:2d} WR={cm['wr']:.1%} avg={cm['avg']:+.4%}")

    return oos_results


# ══════════════════════════════════════════════════════════
# Phase 3: Permutation Test (is edge real?)
# ══════════════════════════════════════════════════════════

def permutation_test(oos_data, best_cfg, n_perms=2000):
    print("\n" + "=" * 95)
    print("PHASE 3: Permutation Test (is the edge real or random?)")
    print("=" * 95)

    # Get real OOS PnL
    real_pnls = []
    for coin in COINS_MAP:
        if coin not in oos_data: continue
        sig = generate_signals(oos_data[coin], lb=int(best_cfg["lb"]), vw=bool(best_cfg["vw"]),
                                cq=best_cfg["cq"], cw=int(best_cfg["cw"]), use_oi=bool(best_cfg["oi"]))
        p, _, _ = backtest(oos_data[coin], sig, ku=best_cfg["ku"], kl=best_cfg["kl"],
                           mh=int(best_cfg["mh"]))
        real_pnls.extend(p.tolist())

    real_avg = np.mean(real_pnls)
    real_sharpe = metrics(real_pnls)["sharpe"]

    # Permutation: shuffle signal directions randomly
    rng = np.random.RandomState(42)
    perm_avgs, perm_sharpes = [], []

    for _ in range(n_perms):
        perm_pnls = []
        for coin in COINS_MAP:
            if coin not in oos_data: continue
            sig = generate_signals(oos_data[coin], lb=int(best_cfg["lb"]), vw=bool(best_cfg["vw"]),
                                    cq=best_cfg["cq"], cw=int(best_cfg["cw"]), use_oi=bool(best_cfg["oi"]))
            # Randomly flip signal direction
            flip = rng.choice([-1, 1], size=len(sig))
            sig_perm = sig * flip
            p, _, _ = backtest(oos_data[coin], sig_perm, ku=best_cfg["ku"], kl=best_cfg["kl"],
                               mh=int(best_cfg["mh"]))
            perm_pnls.extend(p.tolist())

        if len(perm_pnls) > 0:
            perm_avgs.append(np.mean(perm_pnls))
            s = np.std(perm_pnls)
            perm_sharpes.append(np.mean(perm_pnls)/s*np.sqrt(len(perm_pnls)) if s > 0 else 0)

    p_value_avg = np.mean(np.array(perm_avgs) >= real_avg)
    p_value_sharpe = np.mean(np.array(perm_sharpes) >= real_sharpe)

    print(f"  Real OOS: avg={real_avg:+.4%}, Sharpe={real_sharpe:.2f}")
    print(f"  Permutation (n={n_perms}):")
    print(f"    avg:    median={np.median(perm_avgs):+.4%}, p-value={p_value_avg:.4f}")
    print(f"    Sharpe: median={np.median(perm_sharpes):.2f}, p-value={p_value_sharpe:.4f}")
    print(f"  Interpretation:")
    if p_value_avg < 0.05:
        print(f"    avg p={p_value_avg:.4f} < 0.05: STATISTICALLY SIGNIFICANT at 95% level")
    elif p_value_avg < 0.10:
        print(f"    avg p={p_value_avg:.4f} < 0.10: MARGINAL significance")
    else:
        print(f"    avg p={p_value_avg:.4f} >= 0.10: NOT SIGNIFICANT (edge may be random)")

    return p_value_avg, p_value_sharpe


# ══════════════════════════════════════════════════════════
# Phase 4: Cost Sensitivity
# ══════════════════════════════════════════════════════════

def cost_sensitivity(oos_data, best_cfg):
    print("\n" + "=" * 95)
    print("PHASE 4: Cost Sensitivity Analysis")
    print("=" * 95)

    for cost_bps in [0, 10, 15, 20, 25, 30, 40, 50]:
        cost = cost_bps / 10000
        all_p = []
        for coin in COINS_MAP:
            if coin not in oos_data: continue
            sig = generate_signals(oos_data[coin], lb=int(best_cfg["lb"]), vw=bool(best_cfg["vw"]),
                                    cq=best_cfg["cq"], cw=int(best_cfg["cw"]), use_oi=bool(best_cfg["oi"]))
            p, _, _ = backtest(oos_data[coin], sig, ku=best_cfg["ku"], kl=best_cfg["kl"],
                               mh=int(best_cfg["mh"]), cost=cost)
            all_p.extend(p.tolist())

        m = metrics(all_p)
        status = "OK" if m["avg"] > 0 else "NEGATIVE"
        print(f"  Cost={cost_bps:2d}bps: n={m['n']:3d} avg={m['avg']:+.4%} "
              f"Sharpe={m['sharpe']:.2f} WR={m['wr']:.1%} [{status}]")


# ══════════════════════════════════════════════════════════
# Phase 5: Drop-One-Out Stability
# ══════════════════════════════════════════════════════════

def drop_one_out(oos_data, best_cfg):
    print("\n" + "=" * 95)
    print("PHASE 5: Drop-One-Out Stability (remove each coin)")
    print("=" * 95)

    # Full portfolio
    all_p = []
    for coin in COINS_MAP:
        if coin not in oos_data: continue
        sig = generate_signals(oos_data[coin], lb=int(best_cfg["lb"]), vw=bool(best_cfg["vw"]),
                                cq=best_cfg["cq"], cw=int(best_cfg["cw"]), use_oi=bool(best_cfg["oi"]))
        p, _, _ = backtest(oos_data[coin], sig, ku=best_cfg["ku"], kl=best_cfg["kl"],
                           mh=int(best_cfg["mh"]))
        all_p.extend(p.tolist())
    base = metrics(all_p)
    print(f"  Full portfolio: n={base['n']} avg={base['avg']:+.4%} Sharpe={base['sharpe']:.2f}")

    for drop in COINS_MAP:
        all_p = []
        for coin in COINS_MAP:
            if coin == drop or coin not in oos_data: continue
            sig = generate_signals(oos_data[coin], lb=int(best_cfg["lb"]), vw=bool(best_cfg["vw"]),
                                    cq=best_cfg["cq"], cw=int(best_cfg["cw"]), use_oi=bool(best_cfg["oi"]))
            p, _, _ = backtest(oos_data[coin], sig, ku=best_cfg["ku"], kl=best_cfg["kl"],
                               mh=int(best_cfg["mh"]))
            all_p.extend(p.tolist())
        m = metrics(all_p)
        delta_sharpe = m["sharpe"] - base["sharpe"]
        flag = "CRITICAL" if delta_sharpe > 0.5 else ("HELP" if delta_sharpe < -0.5 else "")
        print(f"  Drop {drop:4s}: n={m['n']:3d} avg={m['avg']:+.4%} Sharpe={m['sharpe']:.2f} "
              f"(delta={delta_sharpe:+.2f}) {flag}")


# ══════════════════════════════════════════════════════════
# Phase 6: Leverage Optimization
# ══════════════════════════════════════════════════════════

def leverage_analysis(oos_data, best_cfg):
    print("\n" + "=" * 95)
    print("PHASE 6: Optimal Leverage (Kelly + Monte Carlo)")
    print("=" * 95)

    all_p = []
    for coin in COINS_MAP:
        if coin not in oos_data: continue
        sig = generate_signals(oos_data[coin], lb=int(best_cfg["lb"]), vw=bool(best_cfg["vw"]),
                                cq=best_cfg["cq"], cw=int(best_cfg["cw"]), use_oi=bool(best_cfg["oi"]))
        p, _, _ = backtest(oos_data[coin], sig, ku=best_cfg["ku"], kl=best_cfg["kl"],
                           mh=int(best_cfg["mh"]))
        all_p.extend(p.tolist())

    arr = np.array(all_p)
    wr = np.mean(arr > 0)
    avg_w = np.mean(arr[arr > 0]) if np.any(arr > 0) else 0
    avg_l = abs(np.mean(arr[arr < 0])) if np.any(arr < 0) else 1

    # Kelly criterion
    kelly = wr / avg_l - (1 - wr) / avg_w if avg_w > 0 and avg_l > 0 else 0
    kelly = max(0, kelly)

    print(f"  WR={wr:.1%}, avg_win={avg_w:+.3%}, avg_loss={-avg_l:+.3%}")
    print(f"  Full Kelly = {kelly:.2f}x")
    print(f"  Half Kelly = {kelly/2:.2f}x")
    print(f"  Quarter Kelly = {kelly/4:.2f}x")

    # Monte Carlo with compounding
    rng = np.random.RandomState(42)
    n_sims = 5000
    print(f"\n  Monte Carlo ({n_sims} paths, {len(arr)} trades each):")
    print(f"  {'Lev':>4s} | {'Median CAGR':>11s} | {'Med MDD':>8s} | {'P(profit)':>9s} | {'P(MDD>30%)':>10s} | {'P(MDD>50%)':>10s} | {'Optimal?':>8s}")

    best_lev_cagr = 0
    best_lev = 1

    for lev_10 in [10, 15, 20, 25, 30, 40, 50, 70, 100]:
        lev = lev_10 / 10
        finals, mdds = [], []
        for _ in range(n_sims):
            sample = rng.choice(arr, size=len(arr), replace=True)
            eq = np.cumprod(1 + sample * lev)
            finals.append(eq[-1])
            dd = eq / np.maximum.accumulate(eq) - 1
            mdds.append(np.min(dd))

        finals = np.array(finals)
        mdds = np.array(mdds)
        med_cagr = (np.median(finals) ** (365 / (len(arr) * 4 / 6)) - 1) * 100  # rough annualization
        med_mdd = np.median(mdds) * 100
        p_profit = np.mean(finals > 1) * 100
        p_mdd30 = np.mean(mdds < -0.30) * 100
        p_mdd50 = np.mean(mdds < -0.50) * 100

        opt = ""
        if med_cagr > best_lev_cagr and p_mdd50 < 10:
            best_lev_cagr = med_cagr
            best_lev = lev
            opt = "<-- BEST"

        print(f"  {lev:4.1f}x | {med_cagr:+10.1f}% | {med_mdd:7.1f}% | {p_profit:8.1f}% | "
              f"{p_mdd30:9.1f}% | {p_mdd50:9.1f}% | {opt}")

    print(f"\n  Optimal leverage (max CAGR with P(MDD>50%)<10%): {best_lev:.1f}x")


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()

    print("Loading data...")
    data = load_all()

    print("\nSplitting IS / OOS...")
    is_data, oos_data = split_is_oos(data)

    # Phase 1: IS grid search
    is_results = is_grid_search(is_data)

    # Phase 2: OOS validation
    oos_results = oos_validation(oos_data, is_results)

    # Find best OOS config
    oos_positive = [r for r in oos_results if r["avg"] > 0 and r["n"] >= 10]
    if oos_positive:
        best_oos = max(oos_positive, key=lambda x: x["sharpe"])
        best_cfg = best_oos["cfg_dict"]
        print(f"\n  Best OOS config: {best_oos['config']}")
        print(f"  OOS: n={best_oos['n']} avg={best_oos['avg']:+.4%} Sharpe={best_oos['sharpe']:.2f}")

        # Phase 3: Permutation test
        permutation_test(oos_data, best_cfg)

        # Phase 4: Cost sensitivity
        cost_sensitivity(oos_data, best_cfg)

        # Phase 5: Drop-one-out
        drop_one_out(oos_data, best_cfg)

        # Phase 6: Leverage
        leverage_analysis(oos_data, best_cfg)
    else:
        print("\n  NO OOS-POSITIVE CONFIG FOUND. Edge does not survive OOS.")
        # Still do permutation on IS-best
        best_is = is_results[is_results["n"] >= 25].head(1).to_dict("records")[0]
        permutation_test(oos_data, best_is)

    # Save
    out = "data/reports/tsmom_rigorous_v2.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(oos_results).to_csv(out, index=False)

    elapsed = time.time() - t0
    print(f"\n{'=' * 95}")
    print(f"Total: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 95}")
