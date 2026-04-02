"""HTF (15m + 1h) dual-timeframe backtest with parameter sweep.

Usage: python -m backtest.run_htf_backtest
"""

from __future__ import annotations

import json
import logging
import sys
import time
import copy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.data_loader import BacktestDataHub
from backtest.engine_htf import HTFBacktestEngine, STRATEGIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest.htf_main")


def analyze_trades(trades: list[dict], label: str = "") -> dict:
    """Quick metrics for a trade list."""
    if not trades:
        return {"n": 0, "label": label}

    df = pd.DataFrame(trades)
    n = len(df)
    wins = df[df["pnl_net"] > 0]
    wr = len(wins) / n * 100
    gross = df["pnl_gross"].sum()
    fees = df["fee"].sum()
    net = df["pnl_net"].sum()
    pf = (wins["pnl_net"].sum() / abs(df[df["pnl_net"] <= 0]["pnl_net"].sum() + 0.01)) if len(wins) > 0 else 0

    exits = df["exit_reason"].value_counts().to_dict()
    tp_rate = exits.get("TP_HIT", 0) / n * 100
    sl_rate = exits.get("SL_HIT", 0) / n * 100

    avg_bars = df["bars_held"].mean()
    avg_mfe = df["mfe_pct"].mean()
    avg_mae = df["mae_pct"].mean()

    return {
        "label": label,
        "n": n,
        "wr": round(wr, 1),
        "gross": round(gross, 2),
        "fees": round(fees, 2),
        "net": round(net, 2),
        "pf": round(pf, 2),
        "tp_rate": round(tp_rate, 1),
        "sl_rate": round(sl_rate, 1),
        "avg_bars": round(avg_bars, 0),
        "avg_mfe": round(avg_mfe, 3),
        "avg_mae": round(avg_mae, 3),
        "exits": exits,
    }


def print_results(results: list[dict]):
    header = f"{'Strategy':<22} {'N':>4} {'WR%':>6} {'Gross$':>9} {'Fee$':>7} {'Net$':>9} {'PF':>5} {'TP%':>5} {'SL%':>5} {'Bars':>5} {'MFE%':>6} {'MAE%':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r["n"] == 0:
            print(f"{r['label']:<22} {'--':>4}")
            continue
        print(
            f"{r['label']:<22} {r['n']:>4} {r['wr']:>5.1f}% "
            f"{r['gross']:>+9.2f} {r['fees']:>7.2f} {r['net']:>+9.2f} "
            f"{r['pf']:>5.2f} {r['tp_rate']:>4.1f}% {r['sl_rate']:>4.1f}% "
            f"{r['avg_bars']:>5.0f} {r['avg_mfe']:>+5.3f} {r['avg_mae']:>+6.3f}"
        )


def main():
    start = time.time()

    # Load data
    data_dir = ROOT / "data" / "raw" / "binance"
    coins = ["SOL", "XRP", "ADA", "DOT"]
    all_coins = coins + ["BTC", "ETH"]

    logger.info("Loading data...")
    hub = BacktestDataHub(data_dir, all_coins)

    # ══════════════════════════════════════════════════════════
    # Phase 1: Default parameters - all 6 strategies
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  PHASE 1: 6 HTF STRATEGIES - DEFAULT PARAMETERS (SL=1xATR, TP=2xATR)")
    print("=" * 100)

    engine = HTFBacktestEngine(hub, coins)
    trades = engine.run()

    # Per-strategy breakdown
    df_all = pd.DataFrame(trades) if trades else pd.DataFrame()
    results = []
    if not df_all.empty:
        for strat in sorted(df_all["strategy"].unique()):
            stdf = df_all[df_all["strategy"] == strat]
            results.append(analyze_trades(stdf.to_dict("records"), strat))
        results.append(analyze_trades(trades, "TOTAL"))

    print()
    print_results(results)

    # ══════════════════════════════════════════════════════════
    # Phase 2: Individual strategy sweep
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  PHASE 2: PARAMETER SWEEP (SL x TP) PER STRATEGY")
    print("=" * 100)

    sl_mults = [0.5, 0.75, 1.0, 1.5, 2.0]
    tp_mults = [1.5, 2.0, 3.0, 4.0, 5.0]

    sweep_results = []
    all_strategies = list(STRATEGIES.keys())

    for strat_name in all_strategies:
        best_net = -999999
        best_cfg = {}
        tested = 0

        for sl_m in sl_mults:
            for tp_m in tp_mults:
                if tp_m < sl_m:
                    continue  # R:R must be >= 1

                cfgs = {strat_name: {"sl_mult": sl_m, "tp_mult": tp_m}}
                eng = HTFBacktestEngine(hub, coins, strategies=cfgs, max_positions=4)
                t = eng.run(enabled_strategies=[strat_name])

                m = analyze_trades(t, f"{strat_name} SL={sl_m} TP={tp_m}")
                sweep_results.append({
                    "strategy": strat_name,
                    "sl_mult": sl_m,
                    "tp_mult": tp_m,
                    **m,
                })

                if m["net"] > best_net:
                    best_net = m["net"]
                    best_cfg = m
                    best_cfg["sl_mult"] = sl_m
                    best_cfg["tp_mult"] = tp_m

                tested += 1

        status = "PROFITABLE" if best_net > 0 else "NOT PROFITABLE"
        print(f"\n  [{strat_name}] Best of {tested} ({status}):")
        if best_cfg.get("n", 0) > 0:
            print(f"    SL={best_cfg.get('sl_mult'):.2f}x  TP={best_cfg.get('tp_mult'):.2f}x")
            print(f"    N={best_cfg['n']}  WR={best_cfg['wr']:.1f}%  Gross=${best_cfg['gross']:+.2f}  Net=${best_cfg['net']:+.2f}  PF={best_cfg['pf']:.2f}")
            print(f"    TP%={best_cfg['tp_rate']:.1f}  SL%={best_cfg['sl_rate']:.1f}  MFE={best_cfg['avg_mfe']:+.3f}%  MAE={best_cfg['avg_mae']:+.3f}%")
        else:
            print(f"    No trades generated")

    # ══════════════════════════════════════════════════════════
    # Phase 3: Best combination
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  PHASE 3: PROFITABLE STRATEGIES COMBINED")
    print("=" * 100)

    profitable = [r for r in sweep_results if r.get("net", -1) > 0 and r.get("n", 0) >= 5]
    if profitable:
        profitable.sort(key=lambda x: x["net"], reverse=True)
        print(f"\n  Found {len(profitable)} profitable configurations:\n")
        header = f"  {'Strategy':<20} {'SL':>5} {'TP':>5} {'N':>4} {'WR%':>6} {'Gross$':>9} {'Net$':>9} {'PF':>5} {'TP%':>5}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for p in profitable[:20]:
            print(
                f"  {p['strategy']:<20} {p['sl_mult']:>5.2f} {p['tp_mult']:>5.2f} "
                f"{p['n']:>4} {p['wr']:>5.1f}% {p['gross']:>+9.2f} {p['net']:>+9.2f} "
                f"{p['pf']:>5.2f} {p['tp_rate']:>4.1f}%"
            )

        # Run best combo
        best_per_strat = {}
        for p in profitable:
            s = p["strategy"]
            if s not in best_per_strat or p["net"] > best_per_strat[s]["net"]:
                best_per_strat[s] = p

        if best_per_strat:
            combo_cfgs = {}
            enabled = []
            for s, p in best_per_strat.items():
                combo_cfgs[s] = {"sl_mult": p["sl_mult"], "tp_mult": p["tp_mult"]}
                enabled.append(s)

            print(f"\n  Running combined portfolio: {enabled}")
            eng = HTFBacktestEngine(hub, coins, strategies=combo_cfgs, max_positions=6)
            combo_trades = eng.run(enabled_strategies=enabled)

            combo_results = []
            df_combo = pd.DataFrame(combo_trades) if combo_trades else pd.DataFrame()
            if not df_combo.empty:
                for strat in sorted(df_combo["strategy"].unique()):
                    stdf = df_combo[df_combo["strategy"] == strat]
                    combo_results.append(analyze_trades(stdf.to_dict("records"), strat))
                combo_results.append(analyze_trades(combo_trades, "COMBINED"))

            print()
            print_results(combo_results)
    else:
        print("\n  No profitable configurations found in sweep.")

    # ══════════════════════════════════════════════════════════
    # Save
    # ══════════════════════════════════════════════════════════
    output_dir = ROOT / "backtest" / "output"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "htf_sweep_results.json", "w") as f:
        json.dump(sweep_results, f, indent=2, default=str)

    if trades:
        with open(output_dir / "htf_trades.jsonl", "w") as f:
            for t in trades:
                f.write(json.dumps(t, default=str) + "\n")

    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed:.0f}s")
    print()


if __name__ == "__main__":
    main()
