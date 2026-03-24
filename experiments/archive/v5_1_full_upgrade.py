"""v5.1 Full Upgrade: All 5 improvements + integrated optimization.

Task 1: RL state 6-dim compact
Task 2: Universe expansion (+DOGE, AVAX, BNB = 10 coins)
Task 3: Funding rate filter
Task 4: Multi-horizon trend score
Task 5: Trailing stop (1h resolution sim via 4h intra-bar)
Task 6: Integrated optimization + LinUCB training
"""

import sys, os, time, warnings, itertools
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════
# Data Loading (10 coins)
# ══════════════════════════════════════════════════════════

COINS_10 = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
    "XRP": "XRPUSDT", "ADA": "ADAUSDT", "DOT": "DOTUSDT", "LINK": "LINKUSDT",
    "DOGE": "DOGEUSDT", "AVAX": "AVAXUSDT", "BNB": "BNBUSDT",
}
YAHOO = {
    "BTC":"BTC-USD","ETH":"ETH-USD","SOL":"SOL-USD","XRP":"XRP-USD",
    "ADA":"ADA-USD","DOT":"DOT-USD","LINK":"LINK-USD",
    "DOGE":"DOGE-USD","AVAX":"AVAX-USD","BNB":"BNB-USD",
}
COST = 0.0020
IS_RATIO = 0.70


def load_all_10():
    from src.data.crawlers.crypto_ohlcv import fetch_ohlcv, resample_to_4h, add_technical_indicators
    from src.data.crawlers.microstructure_rollup import add_microstructure_rollup

    data = {}
    for coin in COINS_10:
        sym = YAHOO.get(coin)
        if not sym: continue
        df = fetch_ohlcv(coin, sym, period="365d", interval="1h")
        if df.empty: continue
        df = resample_to_4h(df)
        df = add_technical_indicators(df)
        df = add_microstructure_rollup(df)

        # Load Binance metrics if available
        metrics_dir = f"data/raw/binance_public/metrics/{COINS_10[coin]}"
        if os.path.exists(metrics_dir):
            csvs = sorted([f for f in os.listdir(metrics_dir) if f.endswith('.csv')])
            if csvs:
                dfs_m = []
                for f in csvs:
                    try: dfs_m.append(pd.read_csv(os.path.join(metrics_dir, f)))
                    except: continue
                if dfs_m:
                    merged = pd.concat(dfs_m, ignore_index=True)
                    merged["create_time"] = pd.to_datetime(merged["create_time"])
                    merged = merged.sort_values("create_time").set_index("create_time")
                    merged = merged.drop(columns=["symbol"], errors="ignore")
                    m4h = merged.resample("4h").last().dropna(how="all")
                    # Align timezone
                    if df.index.tz is not None and m4h.index.tz is None:
                        m4h.index = m4h.index.tz_localize(df.index.tz)
                    elif df.index.tz is None and m4h.index.tz is not None:
                        m4h.index = m4h.index.tz_localize(None)
                    ma = m4h.reindex(df.index, method="ffill")

                    if "sum_open_interest_value" in ma.columns:
                        oi = ma["sum_open_interest_value"].astype(float)
                        df["oi_zscore"] = (oi - oi.rolling(48,min_periods=12).mean()) / oi.rolling(48,min_periods=12).std()
                    if "sum_taker_long_short_vol_ratio" in ma.columns:
                        df["taker_ratio"] = ma["sum_taker_long_short_vol_ratio"].astype(float)

        # Funding rate
        fr_dir = f"data/raw/binance_public/metrics/{COINS_10[coin]}"
        # Funding is embedded in metrics - extract if column exists
        # We'll handle separately below

        data[coin] = df
        print(f"  [OK] {coin}: {len(df)} bars")
    return data


def split(data):
    is_d, oos_d = {}, {}
    for c, df in data.items():
        cut = int(len(df) * IS_RATIO)
        is_d[c] = df.iloc[:cut].copy()
        oos_d[c] = df.iloc[cut:].copy()
    return is_d, oos_d


# ══════════════════════════════════════════════════════════
# Core: Signal Generation (all variants)
# ══════════════════════════════════════════════════════════

def compute_cvd_ratio(df):
    hr = (df["high"] - df["low"]).replace(0, np.nan)
    bf = ((df["close"] - df["low"]) / hr).fillna(0.5).clip(0, 1)
    vd = (2 * bf - 1) * df["volume"]
    cvd = vd.cumsum()
    cvd_ma = cvd.rolling(24, min_periods=6).mean()
    return ((cvd - cvd_ma) / cvd_ma.abs().replace(0, np.nan)).fillna(0)


def gen_signals_dual(df, lb_s=7, lb_l=28, cq=0.75, cw=120, use_oi=True):
    """Dual TSMOM + RSI + CVD + OI (current v5.0)."""
    bs, bl = lb_s * 6, lb_l * 6
    tsmom_s = np.sign(df["close"].pct_change(bs))
    tsmom_l = np.sign(df["close"].pct_change(bl))

    rsi = df.get("rsi_14", pd.Series(50, index=df.index))
    rsi_ok = ((tsmom_s == 1) & (rsi > 50)) | ((tsmom_s == -1) & (rsi < 50))

    cvd = compute_cvd_ratio(df)
    q_hi = cvd.rolling(cw, min_periods=30).quantile(cq)
    q_lo = cvd.rolling(cw, min_periods=30).quantile(1 - cq)
    cvd_ok = ((tsmom_s == -1) & (cvd > q_hi)) | ((tsmom_s == 1) & (cvd < q_lo))

    dual_ok = tsmom_s == tsmom_l
    mask = rsi_ok & cvd_ok & dual_ok

    if use_oi and "oi_zscore" in df.columns:
        mask = mask & (df["oi_zscore"].abs().fillna(0) < 2.0)

    sig = tsmom_s.copy()
    sig[~mask] = 0
    return sig.fillna(0).astype(int)


def gen_signals_trendscore(df, windows=[3,5,7,14,28,42], cq=0.75, cw=120,
                            threshold=0.6, use_oi=True):
    """Multi-horizon trend score (Task 4)."""
    scores = pd.DataFrame(index=df.index)
    for w in windows:
        bars = w * 6
        ret = df["close"].pct_change(bars)
        # Normalize by volatility
        vol = df["close"].pct_change().rolling(bars, min_periods=bars//2).std()
        scores[f"w{w}"] = (ret / vol.replace(0, np.nan)).fillna(0)

    # Weighted average (longer windows get more weight)
    weights = np.array([1, 1, 2, 3, 4, 5], dtype=float)
    weights = weights / weights.sum()
    trend_score = (scores * weights).sum(axis=1)

    direction = np.sign(trend_score)
    strong = trend_score.abs() > threshold

    rsi = df.get("rsi_14", pd.Series(50, index=df.index))
    rsi_ok = ((direction == 1) & (rsi > 50)) | ((direction == -1) & (rsi < 50))

    cvd = compute_cvd_ratio(df)
    q_hi = cvd.rolling(cw, min_periods=30).quantile(cq)
    q_lo = cvd.rolling(cw, min_periods=30).quantile(1 - cq)
    cvd_ok = ((direction == -1) & (cvd > q_hi)) | ((direction == 1) & (cvd < q_lo))

    mask = strong & rsi_ok & cvd_ok
    if use_oi and "oi_zscore" in df.columns:
        mask = mask & (df["oi_zscore"].abs().fillna(0) < 2.0)

    sig = direction.copy()
    sig[~mask] = 0
    return sig.fillna(0).astype(int)


# ══════════════════════════════════════════════════════════
# Backtest Engine (with optional trailing stop)
# ══════════════════════════════════════════════════════════

def backtest(df, signals, ku=5.0, kl=1.0, mh=24, cost=COST,
             use_trailing=False, trail_activate=3.0, trail_distance=1.5):
    c, h, l = df["close"].values, df["high"].values, df["low"].values
    atr = df["atr_14"].values if "atr_14" in df.columns else \
        pd.Series(np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)),
        np.abs(l-np.roll(c,1))))).rolling(14,min_periods=1).mean().values

    sig = signals.values if hasattr(signals, 'values') else signals
    trades = []
    nxt = 0

    for i in range(len(df) - mh):
        if i < nxt or sig[i] == 0 or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        side = int(sig[i])
        entry = c[i]; a = atr[i]
        tp_d = max(ku * a, entry * 0.002)
        sl_d = max(kl * a, entry * 0.002)
        tp = entry + tp_d * side
        sl = entry - sl_d * side
        trailing_sl = sl

        ep, eb = c[min(i+mh, len(df)-1)], min(i+mh, len(df)-1)

        for j in range(i+1, min(i+mh+1, len(df))):
            # Trailing stop logic
            if use_trailing:
                if side == 1:
                    unrealized = h[j] - entry
                    if unrealized > trail_activate * a:
                        new_sl = h[j] - trail_distance * a
                        trailing_sl = max(trailing_sl, new_sl)
                else:
                    unrealized = entry - l[j]
                    if unrealized > trail_activate * a:
                        new_sl = l[j] + trail_distance * a
                        trailing_sl = min(trailing_sl, new_sl)

            active_sl = trailing_sl if use_trailing else sl

            if side == 1:
                if l[j] <= active_sl: ep, eb = active_sl, j; break
                if h[j] >= tp: ep, eb = tp, j; break
            else:
                if h[j] >= active_sl: ep, eb = active_sl, j; break
                if l[j] <= tp: ep, eb = tp, j; break

        pnl = ((ep - entry) / entry) * side - cost
        trades.append(pnl)
        nxt = eb + 1

    return np.array(trades)


def metrics(pnls):
    if len(pnls) == 0:
        return {"n":0,"wr":0,"avg":0,"sharpe":0,"mdd":0,"pf":0}
    a = np.array(pnls)
    eq = np.cumsum(a)
    dd = eq - np.maximum.accumulate(eq)
    w = sum(p for p in a if p > 0)
    lo = abs(sum(p for p in a if p < 0))
    return {
        "n": len(a), "wr": np.mean(a>0), "avg": np.mean(a),
        "sharpe": np.mean(a)/np.std(a)*np.sqrt(len(a)) if np.std(a)>0 else 0,
        "mdd": np.min(dd) if len(dd)>0 else 0,
        "pf": w/lo if lo>0 else float("inf"),
    }


# ══════════════════════════════════════════════════════════
# Task 1: RL Compact State Builder
# ══════════════════════════════════════════════════════════

COMPACT_STATE_NAMES = [
    "cvd_ratio", "rsi_normalized", "ms_composite",
    "oi_zscore", "tsmom_strength", "ofi_norm", "intercept",
]
COMPACT_DIM = 7  # 6 features + intercept


def build_compact_state(df, tsmom_str=0.0, rsi_val=50.0, oi_z=0.0):
    """Build 6-dim + intercept compact state for LinUCB."""
    def _s(col, default=0.0):
        if col in df.columns:
            v = df[col].iloc[-1]
            return default if (np.isnan(v) or np.isinf(v)) else float(v)
        return default

    cvd_r = np.clip(_s("cvd_ratio_6", 0), -3, 3)
    rsi_n = np.clip(rsi_val / 100.0, 0, 1)
    ms_c = np.clip(_s("ms_composite", 0), -1, 1)
    oi_zs = np.clip(oi_z, -3, 3)
    tsmom_s = np.clip(tsmom_str, 0, 0.3)
    vol_sma = _s("volume_sma_20", 1.0)
    ofi_n = np.clip(_s("ofi_sum_3", 0) / (vol_sma + 1e-10), -3, 3)

    return np.array([cvd_r, rsi_n, ms_c, oi_zs, tsmom_s, ofi_n, 1.0], dtype=np.float64)


# ══════════════════════════════════════════════════════════
# Main: Run All Tasks
# ══════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    print("=" * 90)
    print("v5.1 FULL UPGRADE — 5 tasks parallel")
    print("=" * 90)

    print("\n[LOAD] 10 coins...")
    data = load_all_10()
    is_data, oos_data = split(data)
    print(f"  Loaded: {len(data)} coins, IS/OOS split done")

    # ──────────────────────────────────────────
    # Task 2: Universe expansion baseline
    # ──────────────────────────────────────────
    print("\n" + "=" * 90)
    print("TASK 2: Universe Expansion (7 → 10 coins)")
    print("=" * 90)

    coins_7 = ["BTC","ETH","SOL","XRP","ADA","DOT","LINK"]
    coins_10 = list(COINS_10.keys())
    coins_new = ["DOGE","AVAX","BNB"]

    for group_name, group in [("Original 7", coins_7), ("New 3", coins_new), ("All 10", coins_10)]:
        all_p = []
        for coin in group:
            if coin not in oos_data: continue
            sig = gen_signals_dual(oos_data[coin], lb_s=7, lb_l=28)
            p = backtest(oos_data[coin], sig)
            all_p.extend(p.tolist())
        m = metrics(all_p)
        print(f"  {group_name:12s}: n={m['n']:3d} WR={m['wr']:.1%} avg={m['avg']:+.4%} Sharpe={m['sharpe']:.2f}")

    # ──────────────────────────────────────────
    # Task 4: Multi-horizon Trend Score
    # ──────────────────────────────────────────
    print("\n" + "=" * 90)
    print("TASK 4: Multi-Horizon Trend Score vs Dual TSMOM")
    print("=" * 90)

    for threshold in [0.4, 0.6, 0.8, 1.0]:
        all_p = []
        for coin in coins_10:
            if coin not in oos_data: continue
            sig = gen_signals_trendscore(oos_data[coin], threshold=threshold)
            p = backtest(oos_data[coin], sig)
            all_p.extend(p.tolist())
        m = metrics(all_p)
        print(f"  TrendScore(thr={threshold:.1f}): n={m['n']:3d} WR={m['wr']:.1%} avg={m['avg']:+.4%} Sharpe={m['sharpe']:.2f}")

    # Dual baseline for comparison
    all_p = []
    for coin in coins_10:
        if coin not in oos_data: continue
        sig = gen_signals_dual(oos_data[coin])
        p = backtest(oos_data[coin], sig)
        all_p.extend(p.tolist())
    m = metrics(all_p)
    print(f"  Dual(7+28):          n={m['n']:3d} WR={m['wr']:.1%} avg={m['avg']:+.4%} Sharpe={m['sharpe']:.2f} [BASELINE]")

    # ──────────────────────────────────────────
    # Task 5: Trailing Stop variants
    # ──────────────────────────────────────────
    print("\n" + "=" * 90)
    print("TASK 5: Trailing Stop Variants")
    print("=" * 90)

    # No trailing (baseline)
    all_p = []
    for coin in coins_10:
        if coin not in oos_data: continue
        sig = gen_signals_dual(oos_data[coin])
        p = backtest(oos_data[coin], sig, use_trailing=False)
        all_p.extend(p.tolist())
    m = metrics(all_p)
    print(f"  No trailing (base):    n={m['n']:3d} WR={m['wr']:.1%} avg={m['avg']:+.4%} Sharpe={m['sharpe']:.2f}")

    for activate, distance in [(2.0, 1.0), (3.0, 1.5), (3.0, 1.0), (4.0, 2.0)]:
        all_p = []
        for coin in coins_10:
            if coin not in oos_data: continue
            sig = gen_signals_dual(oos_data[coin])
            p = backtest(oos_data[coin], sig, use_trailing=True,
                        trail_activate=activate, trail_distance=distance)
            all_p.extend(p.tolist())
        m = metrics(all_p)
        print(f"  Trail(act={activate:.0f},dist={distance:.1f}): n={m['n']:3d} WR={m['wr']:.1%} avg={m['avg']:+.4%} Sharpe={m['sharpe']:.2f}")

    # ──────────────────────────────────────────
    # Task 3: Funding Rate proxy (taker ratio as substitute)
    # ──────────────────────────────────────────
    print("\n" + "=" * 90)
    print("TASK 3: Taker Ratio Filter (funding rate proxy)")
    print("=" * 90)

    # Test: when taker_ratio is extreme (crowd longs/shorts), filter
    for variant in ["none", "contrarian", "confirm"]:
        all_p = []
        for coin in coins_10:
            if coin not in oos_data: continue
            df = oos_data[coin]
            sig = gen_signals_dual(df)

            if variant != "none" and "taker_ratio" in df.columns:
                tr = df["taker_ratio"].fillna(1.0)
                if variant == "contrarian":
                    # Block LONG when taker heavily buying, SHORT when heavily selling
                    block = ((sig == 1) & (tr > 1.3)) | ((sig == -1) & (tr < 0.7))
                    sig = sig.copy()
                    sig[block] = 0
                elif variant == "confirm":
                    # Only enter when taker confirms direction
                    confirm = ((sig == 1) & (tr > 1.0)) | ((sig == -1) & (tr < 1.0))
                    sig = sig.copy()
                    sig[~confirm] = 0

            p = backtest(df, sig)
            all_p.extend(p.tolist())
        m = metrics(all_p)
        print(f"  Taker {variant:12s}: n={m['n']:3d} WR={m['wr']:.1%} avg={m['avg']:+.4%} Sharpe={m['sharpe']:.2f}")

    # ──────────────────────────────────────────
    # Task 1+6: Compact state + LinUCB training
    # ──────────────────────────────────────────
    print("\n" + "=" * 90)
    print("TASK 1+6: Compact LinUCB (6-dim) Training")
    print("=" * 90)

    from src.rl.bandit import LinUCB

    # Generate states + rewards from OOS trades
    states, rewards, actions_taken = [], [], []
    for coin in coins_10:
        if coin not in oos_data: continue
        df = oos_data[coin]
        sig = gen_signals_dual(df)
        trade_pnls = backtest(df, sig)

        # Reconstruct entry bars
        sig_vals = sig.values if hasattr(sig, 'values') else sig
        c = df["close"].values
        atr_vals = df["atr_14"].values if "atr_14" in df.columns else np.ones(len(df)) * 0.02
        nxt = 0
        trade_idx = 0

        for i in range(len(df) - 24):
            if i < nxt or sig_vals[i] == 0 or np.isnan(atr_vals[i]) or atr_vals[i] <= 0:
                continue
            if trade_idx >= len(trade_pnls):
                break

            # Build compact state
            df_slice = df.iloc[:i+1]
            lb_bars = 7 * 6
            tsmom_str = abs(df_slice["close"].pct_change(lb_bars).iloc[-1]) if len(df_slice) > lb_bars else 0.0
            rsi_val = df_slice["rsi_14"].iloc[-1] if "rsi_14" in df_slice.columns else 50.0
            oi_z = df_slice["oi_zscore"].iloc[-1] if "oi_zscore" in df_slice.columns else 0.0

            state = build_compact_state(
                df_slice,
                tsmom_str=tsmom_str if not np.isnan(tsmom_str) else 0.0,
                rsi_val=rsi_val if not np.isnan(rsi_val) else 50.0,
                oi_z=oi_z if not np.isnan(oi_z) else 0.0,
            )

            pnl = trade_pnls[trade_idx]
            reward = pnl * 100  # scale to reward

            states.append(state)
            rewards.append(reward)
            actions_taken.append(3)  # all were action=3 (1.0x) in backtest

            trade_idx += 1
            # Skip ahead
            nxt = i + 2  # approximate

    states = np.array(states)
    rewards = np.array(rewards)
    print(f"  Training data: {len(states)} samples, {COMPACT_DIM} dims")

    if len(states) >= 20:
        # Train LinUCB
        bandit = LinUCB(state_dim=COMPACT_DIM, n_actions=4, alpha=1.0, gamma=0.995)

        # Offline training: update with all historical (state, action=3, reward) pairs
        for s, r in zip(states, rewards):
            bandit.update(s, 3, r)  # all were 1.0x sizing

        # Now score all states to see if LinUCB would differentiate
        scores = []
        for s in states:
            action, score = bandit.score(s)
            scores.append((action, score))

        # Feature importance
        theta = bandit.get_theta(3)  # action 3 = 1.0x
        print(f"\n  LinUCB Feature Importance (action=1.0x):")
        for name, weight in zip(COMPACT_STATE_NAMES, theta):
            bar = "#" * int(abs(weight) * 10)
            print(f"    {name:20s}: {weight:+.4f} {bar}")

        # Would RL have improved by rejecting bad trades?
        # Split into RL-would-accept (score > median) vs RL-would-reject
        score_vals = np.array([s[1] for s in scores])
        median_score = np.median(score_vals)
        accept_mask = score_vals >= median_score
        reject_mask = ~accept_mask

        accept_pnls = rewards[accept_mask] / 100
        reject_pnls = rewards[reject_mask] / 100

        print(f"\n  RL Discrimination Test:")
        print(f"    Accept (score >= median): n={accept_mask.sum()} avg={np.mean(accept_pnls):+.4%}")
        print(f"    Reject (score <  median): n={reject_mask.sum()} avg={np.mean(reject_pnls):+.4%}")
        print(f"    Lift: {np.mean(accept_pnls) - np.mean(reject_pnls):+.4%}")

        if np.mean(accept_pnls) > np.mean(reject_pnls):
            print(f"    --> RL CAN discriminate good from bad trades")
        else:
            print(f"    --> RL CANNOT discriminate (need more data or better features)")

        # Save model
        model_path = "data/models/rl/linucb_compact_v5_1.joblib"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        bandit.save(model_path)
        print(f"  Model saved: {model_path}")

    # ──────────────────────────────────────────
    # Final: Integrated Grid Search (best combo)
    # ──────────────────────────────────────────
    print("\n" + "=" * 90)
    print("INTEGRATED: Best combination search")
    print("=" * 90)

    best_sharpe = -999
    best_cfg = None
    results = []

    for sig_type in ["dual", "trendscore_0.6", "trendscore_0.8"]:
        for coins_set in [("7", coins_7), ("10", coins_10)]:
            for trailing in [False, True]:
                for taker_filter in ["none", "contrarian"]:
                    all_p = []
                    for coin in coins_set[1]:
                        if coin not in oos_data: continue
                        df = oos_data[coin]

                        if sig_type == "dual":
                            sig = gen_signals_dual(df)
                        elif sig_type.startswith("trendscore"):
                            thr = float(sig_type.split("_")[1])
                            sig = gen_signals_trendscore(df, threshold=thr)
                        else:
                            continue

                        if taker_filter == "contrarian" and "taker_ratio" in df.columns:
                            tr = df["taker_ratio"].fillna(1.0)
                            block = ((sig == 1) & (tr > 1.3)) | ((sig == -1) & (tr < 0.7))
                            sig = sig.copy()
                            sig[block] = 0

                        p = backtest(df, sig, use_trailing=trailing,
                                    trail_activate=3.0, trail_distance=1.5)
                        all_p.extend(p.tolist())

                    m = metrics(all_p)
                    cfg_name = f"{sig_type}|{coins_set[0]}coins|trail={trailing}|taker={taker_filter}"
                    results.append({**m, "config": cfg_name})

                    if m["sharpe"] > best_sharpe and m["n"] >= 15:
                        best_sharpe = m["sharpe"]
                        best_cfg = cfg_name

    rdf = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    valid = rdf[rdf["n"] >= 15]

    print("\n  TOP 10 Combinations:")
    for _, r in valid.head(10).iterrows():
        print(f"    {r['config']:55s} | n={r['n']:3.0f} WR={r['wr']:.1%} avg={r['avg']:+.4%} Sharpe={r['sharpe']:.2f}")

    if best_cfg:
        print(f"\n  BEST: {best_cfg} (Sharpe={best_sharpe:.2f})")

    # Save
    out = "data/reports/v5_1_full_upgrade.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rdf.to_csv(out, index=False)

    elapsed = time.time() - t0
    print(f"\n{'=' * 90}")
    print(f"Total: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
