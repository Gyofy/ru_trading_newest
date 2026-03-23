"""Test different lookback configurations to fix zero-entry problem."""
import sys, os, warnings
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from experiments.tsmom_rigorous_v2 import generate_signals, backtest, metrics, load_all, split_is_oos, COINS_MAP

print("Loading...")
data = load_all()
is_data, oos_data = split_is_oos(data)

print("\n=== OOS Performance by Lookback ===")
for lb in [5, 7, 10, 14, 21, 28]:
    all_p = []
    for coin in COINS_MAP:
        if coin not in oos_data: continue
        sig = generate_signals(oos_data[coin], lb=lb, vw=False, cq=0.75, cw=120, use_oi=True)
        p, _, _ = backtest(oos_data[coin], sig, ku=5.0, kl=1.0, mh=24)
        all_p.extend(p.tolist())
    m = metrics(all_p)
    print(f"  lb={lb:2d}: n={m['n']:3d} WR={m['wr']:.1%} avg={m['avg']:+.4%} Sharpe={m['sharpe']:.2f} MDD={m['mdd']:.2%}")

print("\n=== Dual Lookback (short + long agree) ===")
for short_lb in [5, 7, 10]:
    for long_lb in [21, 28]:
        all_p = []
        for coin in COINS_MAP:
            if coin not in oos_data: continue
            df = oos_data[coin]
            sig_s = generate_signals(df, lb=short_lb, vw=False, cq=0.75, cw=120, use_oi=True)
            sig_l = generate_signals(df, lb=long_lb, vw=False, cq=0.75, cw=120, use_oi=True)
            combined = sig_s.copy()
            combined[sig_s != sig_l] = 0
            p, _, _ = backtest(df, combined, ku=5.0, kl=1.0, mh=24)
            all_p.extend(p.tolist())
        m = metrics(all_p)
        if m["n"] >= 5:
            print(f"  short={short_lb:2d} + long={long_lb:2d}: n={m['n']:3d} WR={m['wr']:.1%} avg={m['avg']:+.4%} Sharpe={m['sharpe']:.2f}")

print("\n=== Short Lookback + Various CVD ===")
for lb in [5, 7, 10]:
    for cq in [0.65, 0.70, 0.75, 0.85]:
        all_p = []
        for coin in COINS_MAP:
            if coin not in oos_data: continue
            sig = generate_signals(oos_data[coin], lb=lb, vw=False, cq=cq, cw=120, use_oi=True)
            p, _, _ = backtest(oos_data[coin], sig, ku=5.0, kl=1.0, mh=24)
            all_p.extend(p.tolist())
        m = metrics(all_p)
        print(f"  lb={lb} cq={cq:.2f}: n={m['n']:3d} WR={m['wr']:.1%} avg={m['avg']:+.4%} Sharpe={m['sharpe']:.2f}")

# How many signals in last 30 bars of OOS for each config?
print("\n=== Signal Frequency (last 30 bars of OOS) ===")
for lb in [5, 7, 14, 28]:
    total_sigs = 0
    for coin in COINS_MAP:
        if coin not in oos_data: continue
        df = oos_data[coin]
        sig = generate_signals(df, lb=lb, vw=False, cq=0.75, cw=120, use_oi=True)
        last30 = sig.iloc[-30:]
        total_sigs += (last30 != 0).sum()
    print(f"  lb={lb:2d}: {total_sigs} signals in last 30 bars (5 days) across 7 coins")
