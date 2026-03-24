"""GARCH Dynamic SL/TP — v5.2 upgrade test.

Replace fixed ATR(14) barrier with GARCH(1,1) 1-step forecast.
- High vol forecast → widen SL (avoid whipsaw)
- Low vol forecast → tighten SL (efficient capital use)

Compare: ATR(14) fixed vs GARCH adaptive vs hybrid (max of both).
"""

import sys, os, time, warnings
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from arch import arch_model

from src.strategy.tsmom_core import (
    load_ohlcv_10, split_is_oos, generate_dual_signals, calc_metrics, COST_ROUNDTRIP
)

# ══════════════════════════════════════════════════════════
# GARCH Volatility Forecaster
# ══════════════════════════════════════════════════════════

def rolling_garch_forecast(close: pd.Series, window: int = 500) -> pd.Series:
    """Rolling GARCH(1,1) 1-step ahead vol forecast.

    Returns: Series of forecasted volatility (same scale as returns).
    Falls back to realized vol if GARCH fails to converge.
    """
    returns = close.pct_change().dropna() * 100  # scale for GARCH stability

    forecasts = pd.Series(np.nan, index=returns.index)
    fallback_vol = returns.rolling(14).std()

    # Pre-compute for speed: fit every 6 bars (24h), reuse between fits
    last_fit_vol = None
    last_fit_bar = -999

    for i in range(window, len(returns)):
        # Refit every 6 bars (24h) to balance speed vs freshness
        if i - last_fit_bar >= 6 or last_fit_vol is None:
            sample = returns.iloc[max(0, i - window):i]
            try:
                model = arch_model(sample, vol='Garch', p=1, q=1,
                                   dist='t', rescale=False)
                res = model.fit(disp='off', show_warning=False)
                fc = res.forecast(horizon=1)
                vol = np.sqrt(fc.variance.values[-1, 0]) / 100  # back to decimal
                if np.isnan(vol) or vol <= 0 or vol > 0.5:
                    vol = fallback_vol.iloc[i] / 100
                last_fit_vol = vol
                last_fit_bar = i
            except Exception:
                last_fit_vol = fallback_vol.iloc[i] / 100 if not np.isnan(fallback_vol.iloc[i]) else 0.02

        forecasts.iloc[i] = last_fit_vol

    return forecasts


def add_garch_vol(df: pd.DataFrame) -> pd.DataFrame:
    """Add GARCH forecast column to dataframe."""
    df = df.copy()
    df["garch_vol"] = rolling_garch_forecast(df["close"], window=500)

    # Also compute vol regime
    if "atr_14" in df.columns:
        atr_pct = df["atr_14"] / df["close"]
        garch_ratio = df["garch_vol"] / atr_pct.replace(0, np.nan)
        df["garch_ratio"] = garch_ratio.fillna(1.0)
        # >1 = GARCH predicts higher vol than ATR average
        # <1 = GARCH predicts lower vol
    return df


# ══════════════════════════════════════════════════════════
# Dynamic Barrier Backtest
# ══════════════════════════════════════════════════════════

def backtest_dynamic(df, signals, ku=5.0, kl=1.5, mh=24, cost=COST_ROUNDTRIP,
                      mode="fixed"):
    """Backtest with different barrier modes.

    mode:
      "fixed":   SL = kl × ATR(14)          (current v5.1)
      "garch":   SL = kl × GARCH_forecast × price
      "hybrid":  SL = kl × max(ATR, GARCH×price)  (conservative)
      "adaptive": SL scales with garch_ratio
    """
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values

    atr = df["atr_14"].values if "atr_14" in df.columns else np.ones(len(df)) * 0.02
    garch = df["garch_vol"].values if "garch_vol" in df.columns else np.full(len(df), np.nan)
    garch_r = df["garch_ratio"].values if "garch_ratio" in df.columns else np.ones(len(df))

    sig = signals.values if hasattr(signals, "values") else signals
    trades = []
    exit_types = []
    nxt = 0

    for i in range(len(df) - mh):
        if i < nxt or sig[i] == 0 or np.isnan(atr[i]) or atr[i] <= 0:
            continue

        side = int(sig[i])
        entry = c[i]
        a = atr[i]

        # Dynamic barrier calculation
        if mode == "fixed":
            vol_est = a
        elif mode == "garch":
            gv = garch[i] * entry if not np.isnan(garch[i]) else a
            vol_est = max(gv, entry * 0.005)  # floor 0.5%
        elif mode == "hybrid":
            gv = garch[i] * entry if not np.isnan(garch[i]) else a
            vol_est = max(a, gv)  # take the wider one
        elif mode == "adaptive":
            # Scale ATR by GARCH ratio
            gr = garch_r[i] if not np.isnan(garch_r[i]) else 1.0
            gr = np.clip(gr, 0.5, 2.0)  # limit scaling
            vol_est = a * gr
        else:
            vol_est = a

        tp_d = max(ku * vol_est, entry * 0.002)
        sl_d = max(kl * vol_est, entry * 0.002)
        tp = entry + tp_d * side
        sl = entry - sl_d * side

        ep = c[min(i + mh, len(df) - 1)]
        eb = min(i + mh, len(df) - 1)
        et = "TTL"

        for j in range(i + 1, min(i + mh + 1, len(df))):
            if side == 1:
                if l[j] <= sl: ep, eb, et = sl, j, "SL"; break
                if h[j] >= tp: ep, eb, et = tp, j, "TP"; break
            else:
                if h[j] >= sl: ep, eb, et = sl, j, "SL"; break
                if l[j] <= tp: ep, eb, et = tp, j, "TP"; break

        pnl = ((ep - entry) / entry) * side - cost
        trades.append(pnl)
        exit_types.append(et)
        nxt = eb + 1

    return np.array(trades), exit_types


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80)
    print("GARCH Dynamic SL/TP Test")
    print("=" * 80)

    print("\n[1] Loading data (10 coins)...")
    data = load_ohlcv_10()
    _, oos_data = split_is_oos(data)

    print(f"\n[2] Computing GARCH forecasts...")
    for coin in data:
        print(f"  {coin}...", end=" ", flush=True)
        oos_data[coin] = add_garch_vol(oos_data[coin])
        gv = oos_data[coin]["garch_vol"].dropna()
        gr = oos_data[coin]["garch_ratio"].dropna()
        print(f"garch_vol mean={gv.mean():.4f}, garch_ratio mean={gr.mean():.2f}")

    print(f"\n[3] Backtest: Fixed vs GARCH vs Hybrid vs Adaptive")
    print("-" * 80)

    for mode in ["fixed", "garch", "hybrid", "adaptive"]:
        all_pnls, all_exits = [], []
        coin_results = {}

        for coin in oos_data:
            df = oos_data[coin]
            sig = generate_dual_signals(df, lb_short=7, lb_long=28)
            pnls, exits = backtest_dynamic(df, sig, ku=5.0, kl=1.5, mh=24, mode=mode)
            all_pnls.extend(pnls.tolist())
            all_exits.extend(exits)

            if len(pnls) > 0:
                m = calc_metrics(pnls)
                sl_rate = exits.count("SL") / len(exits)
                coin_results[coin] = {"avg": m.avg, "wr": m.wr, "sl_rate": sl_rate, "n": m.n}

        m = calc_metrics(all_pnls)
        sl_rate = all_exits.count("SL") / len(all_exits) if all_exits else 0
        tp_rate = all_exits.count("TP") / len(all_exits) if all_exits else 0
        ttl_rate = all_exits.count("TTL") / len(all_exits) if all_exits else 0

        print(f"\n  [{mode:8s}] {m.summary()} | SL={sl_rate:.1%} TP={tp_rate:.1%} TTL={ttl_rate:.1%}")

        # Per-coin breakdown for this mode
        for coin, cr in sorted(coin_results.items()):
            print(f"    {coin:5s}: n={cr['n']:2d} WR={cr['wr']:.1%} avg={cr['avg']:+.4%} SL_rate={cr['sl_rate']:.1%}")

    # [4] DOGE-specific analysis (the problem coin)
    print(f"\n[4] DOGE Focus: SL hit rate comparison")
    print("-" * 80)

    if "DOGE" in oos_data:
        df = oos_data["DOGE"]
        sig = generate_dual_signals(df, lb_short=7, lb_long=28)

        for mode in ["fixed", "garch", "hybrid", "adaptive"]:
            pnls, exits = backtest_dynamic(df, sig, ku=5.0, kl=1.5, mh=24, mode=mode)
            if len(pnls) > 0:
                sl_1bar = sum(1 for i, (e, p) in enumerate(zip(exits, pnls))
                             if e == "SL")
                m = calc_metrics(pnls)
                print(f"  DOGE [{mode:8s}]: n={m.n:2d} WR={m.wr:.1%} avg={m.avg:+.4%} "
                      f"SL_rate={exits.count('SL')/len(exits):.1%}")

    # [5] Vol regime filter test
    print(f"\n[5] Vol Regime Filter: skip entry when GARCH > 1.5x ATR")
    print("-" * 80)

    for threshold in [1.2, 1.5, 2.0]:
        all_p = []
        for coin in oos_data:
            df = oos_data[coin]
            sig = generate_dual_signals(df, lb_short=7, lb_long=28)

            # Block entries when GARCH predicts high vol
            if "garch_ratio" in df.columns:
                gr = df["garch_ratio"].fillna(1.0)
                sig_filtered = sig.copy()
                sig_filtered[gr > threshold] = 0
            else:
                sig_filtered = sig

            pnls, _ = backtest_dynamic(df, sig_filtered, mode="fixed")
            all_p.extend(pnls.tolist())

        m = calc_metrics(all_p)
        print(f"  garch_ratio < {threshold:.1f}: {m.summary()}")

    # Baseline (no filter)
    all_p = []
    for coin in oos_data:
        sig = generate_dual_signals(oos_data[coin], lb_short=7, lb_long=28)
        pnls, _ = backtest_dynamic(oos_data[coin], sig, mode="fixed")
        all_p.extend(pnls.tolist())
    m = calc_metrics(all_p)
    print(f"  No filter (base): {m.summary()}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"Total: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
