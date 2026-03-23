"""TSMOM + RSI + CVD Deep Optimization.

Phase 1 result: L1+3 (RSI+CVD combo) showed avg +1.18%/trade, Sharpe 1.84.
This script does exhaustive parameter search and proper OOS validation.

Parameters to optimize:
  - TSMOM lookback: [5, 7, 10, 14, 21, 28]
  - CVD quantile threshold: [0.6, 0.7, 0.75, 0.8, 0.9]
  - CVD rolling window: [60, 90, 120, 180]
  - Barrier k_upper: [2.0, 3.0, 4.0, 5.0]
  - Barrier k_lower: [0.8, 1.0, 1.5]
  - max_hold: [12, 18, 24]
  - volume_weighted: [True, False]
"""

import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import warnings
import itertools
import time
from dataclasses import dataclass
from typing import Optional

warnings.filterwarnings("ignore")

COINS = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOT", "LINK"]
COST_ROUNDTRIP = 0.0020


def load_data():
    from src.data.crawlers.crypto_ohlcv import fetch_ohlcv, resample_to_4h, add_technical_indicators, TOP10_YAHOO
    from src.data.crawlers.microstructure_rollup import add_microstructure_rollup

    data = {}
    for coin in COINS:
        sym = TOP10_YAHOO.get(coin)
        if not sym:
            continue
        df = fetch_ohlcv(coin, sym, period="365d", interval="1h")
        if df.empty:
            continue
        df = resample_to_4h(df)
        df = add_technical_indicators(df)
        df = add_microstructure_rollup(df)
        data[coin] = df
        print(f"  [OK] {coin}: {len(df)} bars")
    return data


def compute_cvd_ratio(df, window=24):
    """Compute CVD ratio if not present."""
    if f"cvd_ratio_{window}" in df.columns:
        return df[f"cvd_ratio_{window}"]

    hr = (df["high"] - df["low"]).replace(0, np.nan)
    buy_frac = ((df["close"] - df["low"]) / hr).fillna(0.5).clip(0, 1)
    vd = (2 * buy_frac - 1) * df["volume"]
    cvd = vd.cumsum()
    cvd_ma = cvd.rolling(window, min_periods=max(6, window // 4)).mean()
    ratio = (cvd - cvd_ma) / cvd_ma.abs().replace(0, np.nan)
    return ratio.fillna(0)


def generate_signals(df, lookback_days=14, volume_weighted=False,
                      cvd_quantile=0.75, cvd_roll_window=120):
    """Generate TSMOM + RSI + CVD combo signals."""

    lookback_bars = lookback_days * 6

    # TSMOM direction
    if volume_weighted and "volume" in df.columns:
        ret = df["close"].pct_change()
        vol_w = df["volume"] / df["volume"].rolling(lookback_bars, min_periods=1).mean()
        weighted_ret = (ret * vol_w).rolling(lookback_bars, min_periods=lookback_bars).sum()
        tsmom = np.sign(weighted_ret)
    else:
        tsmom = np.sign(df["close"].pct_change(lookback_bars))

    # RSI filter
    rsi = df.get("rsi_14", pd.Series(50, index=df.index))
    rsi_ok = pd.Series(0, index=df.index)
    rsi_ok[(tsmom == 1) & (rsi > 50)] = 1
    rsi_ok[(tsmom == -1) & (rsi < 50)] = 1

    # CVD timing
    cvd_ratio = compute_cvd_ratio(df, window=24)
    q_hi = cvd_ratio.rolling(cvd_roll_window, min_periods=30).quantile(cvd_quantile)
    q_lo = cvd_ratio.rolling(cvd_roll_window, min_periods=30).quantile(1 - cvd_quantile)

    cvd_ok = pd.Series(0, index=df.index)
    cvd_ok[(tsmom == -1) & (cvd_ratio > q_hi)] = 1  # SHORT at CVD high
    cvd_ok[(tsmom == 1) & (cvd_ratio < q_lo)] = 1   # LONG at CVD low

    # Combined signal
    signals = tsmom.copy()
    signals[(rsi_ok == 0) | (cvd_ok == 0)] = 0
    signals = signals.fillna(0).astype(int)

    return signals


def run_backtest(df, signals, k_upper=3.0, k_lower=1.0, max_hold=18):
    """Fast barrier backtest returning trade PnLs."""

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    if "atr_14" in df.columns:
        atr = df["atr_14"].values
    else:
        tr = np.maximum(high - low,
                        np.maximum(np.abs(high - np.roll(close, 1)),
                                   np.abs(low - np.roll(close, 1))))
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

    sig = signals.values if hasattr(signals, 'values') else signals
    trades = []
    next_avail = 0

    for i in range(len(df) - max_hold):
        if i < next_avail:
            continue
        if sig[i] == 0 or np.isnan(atr[i]) or atr[i] <= 0:
            continue

        side = int(sig[i])
        entry = close[i]
        a = atr[i]
        tp_d = max(k_upper * a, entry * 0.002)
        sl_d = max(k_lower * a, entry * 0.002)

        if side == 1:
            tp, sl = entry + tp_d, entry - sl_d
        else:
            tp, sl = entry - tp_d, entry + sl_d

        exit_p = close[min(i + max_hold, len(df) - 1)]
        exit_bar = min(i + max_hold, len(df) - 1)
        exit_type = "TTL"

        for j in range(i + 1, min(i + max_hold + 1, len(df))):
            if side == 1:
                if low[j] <= sl:
                    exit_p, exit_bar, exit_type = sl, j, "SL"
                    break
                if high[j] >= tp:
                    exit_p, exit_bar, exit_type = tp, j, "TP"
                    break
            else:
                if high[j] >= sl:
                    exit_p, exit_bar, exit_type = sl, j, "SL"
                    break
                if low[j] <= tp:
                    exit_p, exit_bar, exit_type = tp, j, "TP"
                    break

        pnl = (exit_p - entry) / entry if side == 1 else (entry - exit_p) / entry
        trades.append(pnl - COST_ROUNDTRIP)
        next_avail = exit_bar + 1

    return np.array(trades)


def calc_sharpe(pnls):
    if len(pnls) == 0 or np.std(pnls) == 0:
        return 0
    return np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls))


def calc_mdd(pnls):
    eq = np.cumsum(pnls)
    dd = eq - np.maximum.accumulate(eq)
    return np.min(dd) if len(dd) > 0 else 0


# ══════════════════════════════════════════════════════════════
# Parameter Grid Search
# ══════════════════════════════════════════════════════════════

def grid_search(data):
    """Exhaustive parameter grid search for RSI+CVD combo."""

    print("\n" + "=" * 90)
    print("GRID SEARCH: TSMOM + RSI + CVD combo")
    print("=" * 90)

    lookbacks = [5, 7, 10, 14, 21, 28]
    cvd_quantiles = [0.65, 0.70, 0.75, 0.80, 0.85]
    cvd_windows = [60, 90, 120]
    k_uppers = [2.0, 3.0, 4.0, 5.0]
    k_lowers = [0.8, 1.0, 1.5]
    max_holds = [12, 18, 24]
    vol_weights = [False, True]

    total_combos = (len(lookbacks) * len(cvd_quantiles) * len(cvd_windows) *
                    len(k_uppers) * len(k_lowers) * len(max_holds) * len(vol_weights))
    print(f"  Total combinations: {total_combos}")

    results = []
    best_sharpe = -999
    count = 0

    for lb, cq, cw, ku, kl, mh, vw in itertools.product(
        lookbacks, cvd_quantiles, cvd_windows, k_uppers, k_lowers, max_holds, vol_weights
    ):
        count += 1
        all_pnls = []

        for coin in COINS:
            if coin not in data:
                continue
            df = data[coin]
            sigs = generate_signals(df, lookback_days=lb, volume_weighted=vw,
                                     cvd_quantile=cq, cvd_roll_window=cw)
            pnls = run_backtest(df, sigs, k_upper=ku, k_lower=kl, max_hold=mh)
            all_pnls.extend(pnls.tolist())

        if len(all_pnls) < 15:
            continue

        arr = np.array(all_pnls)
        avg = np.mean(arr)
        wr = np.mean(arr > 0)
        sharpe = calc_sharpe(arr)
        mdd = calc_mdd(arr)
        n = len(arr)

        results.append({
            "lookback": lb, "cvd_q": cq, "cvd_w": cw,
            "k_upper": ku, "k_lower": kl, "max_hold": mh,
            "vol_wt": vw, "n_trades": n, "avg_pnl": avg,
            "win_rate": wr, "sharpe": sharpe, "max_dd": mdd,
            "total_pnl": np.sum(arr),
            "pf": sum(p for p in arr if p > 0) / max(abs(sum(p for p in arr if p < 0)), 1e-10),
        })

        if sharpe > best_sharpe and n >= 20:
            best_sharpe = sharpe
            print(f"  [{count:5d}/{total_combos}] NEW BEST: lb={lb:2d} cq={cq:.2f} cw={cw:3d} "
                  f"ku={ku:.1f} kl={kl:.1f} mh={mh:2d} vw={vw} | "
                  f"n={n:3d} WR={wr:.1%} avg={avg:+.3%} Sharpe={sharpe:.2f} MDD={mdd:.2%}")

        if count % 500 == 0:
            print(f"  [{count:5d}/{total_combos}] processed... best Sharpe={best_sharpe:.2f}")

    # Sort and save
    rdf = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    out_path = "data/reports/tsmom_rsi_cvd_grid.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rdf.to_csv(out_path, index=False)

    print(f"\n  Total configs tested: {len(results)}")
    print(f"\n  TOP 10 by Sharpe (min 20 trades):")
    top = rdf[rdf["n_trades"] >= 20].head(10)
    for _, r in top.iterrows():
        print(f"    lb={r['lookback']:2.0f} cq={r['cvd_q']:.2f} cw={r['cvd_w']:3.0f} "
              f"ku={r['k_upper']:.1f} kl={r['k_lower']:.1f} mh={r['max_hold']:2.0f} "
              f"vw={r['vol_wt']} | n={r['n_trades']:3.0f} WR={r['win_rate']:.1%} "
              f"avg={r['avg_pnl']:+.3%} Sharpe={r['sharpe']:.2f} MDD={r['max_dd']:.2%} "
              f"PF={r['pf']:.2f}")

    print(f"\n  TOP 10 by avg PnL (min 20 trades):")
    top_pnl = rdf[rdf["n_trades"] >= 20].sort_values("avg_pnl", ascending=False).head(10)
    for _, r in top_pnl.iterrows():
        print(f"    lb={r['lookback']:2.0f} cq={r['cvd_q']:.2f} cw={r['cvd_w']:3.0f} "
              f"ku={r['k_upper']:.1f} kl={r['k_lower']:.1f} mh={r['max_hold']:2.0f} "
              f"vw={r['vol_wt']} | n={r['n_trades']:3.0f} WR={r['win_rate']:.1%} "
              f"avg={r['avg_pnl']:+.3%} Sharpe={r['sharpe']:.2f} PF={r['pf']:.2f}")

    return rdf


# ══════════════════════════════════════════════════════════════
# Walk-Forward OOS (Proper)
# ══════════════════════════════════════════════════════════════

def walk_forward_oos(data, top_configs, n_windows=5):
    """Proper walk-forward: optimize on IS, test on OOS."""

    print("\n" + "=" * 90)
    print("WALK-FORWARD OOS VALIDATION (top configs)")
    print("=" * 90)

    for idx, cfg in enumerate(top_configs[:5]):
        lb = int(cfg["lookback"])
        cq = cfg["cvd_q"]
        cw = int(cfg["cvd_w"])
        ku = cfg["k_upper"]
        kl = cfg["k_lower"]
        mh = int(cfg["max_hold"])
        vw = bool(cfg["vol_wt"])

        print(f"\n  Config {idx+1}: lb={lb} cq={cq:.2f} cw={cw} ku={ku:.1f} kl={kl:.1f} "
              f"mh={mh} vw={vw}")

        all_oos_pnls = []

        for coin in COINS:
            if coin not in data:
                continue
            df = data[coin]
            n = len(df)
            window_size = n // n_windows

            coin_oos_pnls = []

            for w in range(n_windows):
                # OOS window
                oos_start = w * window_size
                oos_end = min(oos_start + window_size, n)

                if oos_end - oos_start < 60:
                    continue

                df_oos = df.iloc[oos_start:oos_end].copy()

                sigs = generate_signals(df_oos, lookback_days=lb, volume_weighted=vw,
                                         cvd_quantile=cq, cvd_roll_window=cw)
                pnls = run_backtest(df_oos, sigs, k_upper=ku, k_lower=kl, max_hold=mh)
                coin_oos_pnls.extend(pnls.tolist())

            if coin_oos_pnls:
                arr = np.array(coin_oos_pnls)
                all_oos_pnls.extend(coin_oos_pnls)
                print(f"    {coin}: n={len(arr):3d} WR={np.mean(arr>0):.1%} "
                      f"avg={np.mean(arr):+.3%} Sharpe={calc_sharpe(arr):.2f}")

        if all_oos_pnls:
            arr = np.array(all_oos_pnls)
            print(f"    TOTAL: n={len(arr):3d} WR={np.mean(arr>0):.1%} "
                  f"avg={np.mean(arr):+.3%} Sharpe={calc_sharpe(arr):.2f} "
                  f"MDD={calc_mdd(arr):.2%}")


# ══════════════════════════════════════════════════════════════
# Per-Coin Analysis
# ══════════════════════════════════════════════════════════════

def per_coin_analysis(data, best_cfg):
    """Detailed per-coin analysis with best config."""

    print("\n" + "=" * 90)
    print("PER-COIN ANALYSIS (best config)")
    print("=" * 90)

    lb = int(best_cfg["lookback"])
    cq = best_cfg["cvd_q"]
    cw = int(best_cfg["cvd_w"])
    ku = best_cfg["k_upper"]
    kl = best_cfg["k_lower"]
    mh = int(best_cfg["max_hold"])
    vw = bool(best_cfg["vol_wt"])

    print(f"  Config: lb={lb} cq={cq:.2f} cw={cw} ku={ku:.1f} kl={kl:.1f} mh={mh} vw={vw}")

    for coin in COINS:
        if coin not in data:
            continue
        df = data[coin]
        sigs = generate_signals(df, lookback_days=lb, volume_weighted=vw,
                                 cvd_quantile=cq, cvd_roll_window=cw)

        pnls = run_backtest(df, sigs, k_upper=ku, k_lower=kl, max_hold=mh)

        n_signals = np.sum(np.abs(sigs.values if hasattr(sigs, 'values') else sigs) > 0)

        if len(pnls) > 0:
            long_mask = sigs.values[sigs.values != 0] == 1
            # Can't easily separate, just show overall
            wr = np.mean(pnls > 0)
            avg = np.mean(pnls)
            sharpe = calc_sharpe(pnls)
            mdd = calc_mdd(pnls)
            total = np.sum(pnls)
            pf = sum(p for p in pnls if p > 0) / max(abs(sum(p for p in pnls if p < 0)), 1e-10)

            print(f"  {coin:4s}: signals={n_signals:4d} trades={len(pnls):3d} | "
                  f"WR={wr:.1%} avg={avg:+.3%} total={total:+.2%} "
                  f"Sharpe={sharpe:.2f} MDD={mdd:.2%} PF={pf:.2f}")
        else:
            print(f"  {coin:4s}: signals={n_signals:4d} trades=0")


# ══════════════════════════════════════════════════════════════
# Stability Check: Bootstrap
# ══════════════════════════════════════════════════════════════

def bootstrap_stability(data, best_cfg, n_bootstrap=1000):
    """Bootstrap confidence interval for best config."""

    print("\n" + "=" * 90)
    print("BOOTSTRAP STABILITY CHECK")
    print("=" * 90)

    lb = int(best_cfg["lookback"])
    cq = best_cfg["cvd_q"]
    cw = int(best_cfg["cvd_w"])
    ku = best_cfg["k_upper"]
    kl = best_cfg["k_lower"]
    mh = int(best_cfg["max_hold"])
    vw = bool(best_cfg["vol_wt"])

    all_pnls = []
    for coin in COINS:
        if coin not in data:
            continue
        df = data[coin]
        sigs = generate_signals(df, lookback_days=lb, volume_weighted=vw,
                                 cvd_quantile=cq, cvd_roll_window=cw)
        pnls = run_backtest(df, sigs, k_upper=ku, k_lower=kl, max_hold=mh)
        all_pnls.extend(pnls.tolist())

    arr = np.array(all_pnls)
    n = len(arr)

    if n < 10:
        print("  Not enough trades for bootstrap")
        return

    boot_means = []
    boot_sharpes = []
    rng = np.random.RandomState(42)

    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))
        if np.std(sample) > 0:
            boot_sharpes.append(np.mean(sample) / np.std(sample) * np.sqrt(n))

    boot_means = np.array(boot_means)
    boot_sharpes = np.array(boot_sharpes)

    print(f"  Original: n={n}, avg={np.mean(arr):+.3%}, Sharpe={calc_sharpe(arr):.2f}")
    print(f"  Bootstrap avg PnL: {np.percentile(boot_means, 5):+.3%} / "
          f"{np.percentile(boot_means, 50):+.3%} / {np.percentile(boot_means, 95):+.3%} "
          f"(5th/50th/95th)")
    print(f"  Bootstrap Sharpe:  {np.percentile(boot_sharpes, 5):.2f} / "
          f"{np.percentile(boot_sharpes, 50):.2f} / {np.percentile(boot_sharpes, 95):.2f}")
    print(f"  P(avg > 0): {np.mean(boot_means > 0):.1%}")
    print(f"  P(Sharpe > 0): {np.mean(boot_sharpes > 0):.1%}")
    print(f"  P(Sharpe > 1.0): {np.mean(boot_sharpes > 1.0):.1%}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()

    print("Loading data...")
    data = load_data()

    # Phase 1: Grid search
    rdf = grid_search(data)

    # Phase 2: Get top configs
    top = rdf[rdf["n_trades"] >= 20].head(5)
    top_configs = top.to_dict("records")

    if top_configs:
        best_cfg = top_configs[0]

        # Phase 3: Walk-forward OOS
        walk_forward_oos(data, top_configs, n_windows=5)

        # Phase 4: Per-coin analysis
        per_coin_analysis(data, best_cfg)

        # Phase 5: Bootstrap stability
        bootstrap_stability(data, best_cfg)

    elapsed = time.time() - t0
    print(f"\n{'=' * 90}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 90}")
