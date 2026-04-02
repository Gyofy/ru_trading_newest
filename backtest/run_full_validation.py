"""Full validation suite: every test that can break a backtest result.

Tests:
1. Walk-Forward (70/30 temporal split)
2. First half vs Second half (regime robustness)
3. Reversed signals (edge existence proof)
4. Random entry benchmark (are signals better than random?)
5. Per-coin isolation (no single coin carrying results)
6. Parameter neighborhood (nearby params also profitable?)
7. Monte Carlo shuffle (is PnL sequence luck?)
8. Leave-one-out coin (removing any coin still profitable?)

If a strategy passes ALL tests = real edge. Otherwise = overfitting or luck.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import random
import copy
import uuid
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.data_loader import BacktestDataHub
from backtest.engine_htf import (
    _compute_rsi, compute_htf_barriers, Position,
    FEE_RATE, ENTRY_SLIP, EXIT_SLIP,
)
from backtest.run_htf_max_trades import (
    MaxTradesEngine, STRATEGIES_1H, analyze, print_table,
    eval_tsmom_1h, eval_tsmom_12h, eval_rel_strength_1h,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest.validation")

# The 3 profitable strategies with their optimal params
PROFITABLE = {
    "tsmom_1h":        (eval_tsmom_1h,        {"sl_mult": 1.5, "tp_mult": 5.0, "min_move_pct": 0.004}),
    "tsmom_12h":       (eval_tsmom_12h,        {"sl_mult": 1.5, "tp_mult": 5.0, "min_move_pct": 0.003}),
    "rel_strength_1h": (eval_rel_strength_1h,  {"sl_mult": 2.0, "tp_mult": 5.0, "min_rs_pct": 0.008}),
}

COINS_10 = ["SOL", "XRP", "ADA", "DOT", "ARB", "AVAX", "LINK", "OP", "TAO", "APT"]
ALL_COINS = COINS_10 + ["BTC", "ETH"]  # BTC/ETH as reference


class TimeLimitedEngine(MaxTradesEngine):
    """Engine that only trades within a time window."""

    def __init__(self, hub, coins, start_ts=None, end_ts=None, **kwargs):
        super().__init__(hub, coins, **kwargs)
        self._start_ts = start_ts
        self._end_ts = end_ts

    def run(self, strategies=None):
        strats = strategies or PROFITABLE
        all_ts = self._hub.get_all_1m_timestamps(self._coins[:4])

        if self._start_ts:
            all_ts = all_ts[all_ts >= self._start_ts]
        if self._end_ts:
            all_ts = all_ts[all_ts <= self._end_ts]

        warmup = 1500
        for i, ts in enumerate(all_ts):
            if i < warmup:
                continue
            self._check_exits(ts)
            if ts.minute == 0:
                self._evaluate_all(ts, strats)

        for pos in list(self._open):
            self._close(pos, "BACKTEST_END", all_ts[-1])
        return self._closed


class ReversedEngine(MaxTradesEngine):
    """Engine that reverses all signals (BUY->SELL, SELL->BUY)."""

    def _evaluate_all(self, ts, strats):
        if len(self._open) >= self._max_pos:
            return
        for name, (fn, cfg) in strats.items():
            if len(self._open) >= self._max_pos:
                break
            for coin in self._coins:
                if any(p.coin == coin and p.strategy == name for p in self._open):
                    continue
                cd_key = f"{name}:{coin}"
                if cd_key in self._cooldowns and (ts - self._cooldowns[cd_key]).total_seconds() < 3600:
                    continue
                try:
                    sig = fn(self._hub, coin, ts, cfg)
                except Exception:
                    continue
                if sig is None:
                    continue
                # REVERSE the signal
                sig["side"] = "SELL" if sig["side"] == "BUY" else "BUY"
                self._open_pos(name, coin, sig, ts, cfg)
                self._cooldowns[cd_key] = ts


class RandomEngine(MaxTradesEngine):
    """Engine with random entry direction at random times."""

    def _evaluate_all(self, ts, strats):
        if len(self._open) >= self._max_pos:
            return
        # Random: 10% chance of entry each hour per coin
        for coin in self._coins:
            if any(p.coin == coin for p in self._open):
                continue
            if random.random() > 0.10:
                continue
            side = "BUY" if random.random() > 0.5 else "SELL"
            sig = {"side": side, "extra": {"trigger": "random", "signal_strength": 0}}
            # Use first strategy's params for barriers
            name = list(strats.keys())[0]
            cfg = strats[name][1]
            self._open_pos(name, coin, sig, ts, cfg)


def quick_summary(trades, label=""):
    """One-line summary."""
    if not trades:
        return f"  {label:<35} {'NO TRADES':>10}"
    df = pd.DataFrame(trades)
    n = len(df)
    wr = len(df[df["pnl_net"] > 0]) / n * 100
    net = df["pnl_net"].sum()
    gross = df["pnl_gross"].sum()
    pf = df[df["pnl_net"] > 0]["pnl_net"].sum() / (abs(df[df["pnl_net"] <= 0]["pnl_net"].sum()) + 0.01)
    tp = df["exit_reason"].value_counts().get("TP_HIT", 0) / n * 100
    return f"  {label:<35} N={n:>4}  WR={wr:>5.1f}%  Gross=${gross:>+9.2f}  Net=${net:>+9.2f}  PF={pf:>5.2f}  TP={tp:>4.1f}%"


def main():
    start = time.time()

    data_dir = ROOT / "data" / "raw" / "binance"
    logger.info("Loading all coins...")
    hub = BacktestDataHub(data_dir, ALL_COINS + ["DOGE", "BNB", "SUI"])

    # Get time boundaries
    all_ts = hub.get_all_1m_timestamps(COINS_10[:4])
    mid_ts = all_ts[len(all_ts) // 2]
    split_70 = all_ts[int(len(all_ts) * 0.70)]

    print(f"\n  Data: {all_ts[0]} ~ {all_ts[-1]}")
    print(f"  50% split: {mid_ts}")
    print(f"  70% split: {split_70}")

    results = {}

    # ══════════════════════════════════════════════════════════
    # TEST 0: Baseline (full period, 10 coins, 3 strategies)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  TEST 0: BASELINE (full period)")
    print("=" * 90)

    eng = MaxTradesEngine(hub, COINS_10, max_pos=12)
    t0 = eng.run(strategies=PROFITABLE)
    print(quick_summary(t0, "BASELINE"))
    results["baseline"] = analyze(t0, "BASELINE")

    # ══════════════════════════════════════════════════════════
    # TEST 1: Walk-Forward (train 70% -> test 30%)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  TEST 1: WALK-FORWARD (train first 70%, test last 30%)")
    print("=" * 90)

    eng_train = TimeLimitedEngine(hub, COINS_10, end_ts=split_70, max_pos=12)
    t_train = eng_train.run(strategies=PROFITABLE)
    print(quick_summary(t_train, "TRAIN (first 70%)"))
    results["wf_train"] = analyze(t_train, "WF_TRAIN")

    eng_test = TimeLimitedEngine(hub, COINS_10, start_ts=split_70, max_pos=12)
    t_test = eng_test.run(strategies=PROFITABLE)
    print(quick_summary(t_test, "TEST (last 30%)"))
    results["wf_test"] = analyze(t_test, "WF_TEST")

    # ══════════════════════════════════════════════════════════
    # TEST 2: First Half vs Second Half
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  TEST 2: FIRST HALF vs SECOND HALF")
    print("=" * 90)

    eng_h1 = TimeLimitedEngine(hub, COINS_10, end_ts=mid_ts, max_pos=12)
    t_h1 = eng_h1.run(strategies=PROFITABLE)
    print(quick_summary(t_h1, "FIRST HALF (days 1-7)"))
    results["half_1"] = analyze(t_h1, "HALF_1")

    eng_h2 = TimeLimitedEngine(hub, COINS_10, start_ts=mid_ts, max_pos=12)
    t_h2 = eng_h2.run(strategies=PROFITABLE)
    print(quick_summary(t_h2, "SECOND HALF (days 8-14)"))
    results["half_2"] = analyze(t_h2, "HALF_2")

    # ══════════════════════════════════════════════════════════
    # TEST 3: Reversed Signals (edge existence proof)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  TEST 3: REVERSED SIGNALS (if edge exists, reversed should lose)")
    print("=" * 90)

    eng_rev = ReversedEngine(hub, COINS_10, max_pos=12)
    t_rev = eng_rev.run(strategies=PROFITABLE)
    print(quick_summary(t_rev, "REVERSED SIGNALS"))
    print(quick_summary(t0, "ORIGINAL (for comparison)"))
    results["reversed"] = analyze(t_rev, "REVERSED")

    # ══════════════════════════════════════════════════════════
    # TEST 4: Random Entry Benchmark (5 runs averaged)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  TEST 4: RANDOM ENTRY BENCHMARK (5 runs)")
    print("=" * 90)

    random_nets = []
    for i in range(5):
        random.seed(42 + i)
        eng_rnd = RandomEngine(hub, COINS_10, max_pos=12)
        t_rnd = eng_rnd.run(strategies=PROFITABLE)
        if t_rnd:
            rnet = sum(t["pnl_net"] for t in t_rnd)
            random_nets.append(rnet)
            print(quick_summary(t_rnd, f"RANDOM RUN {i+1}"))

    avg_random = np.mean(random_nets) if random_nets else 0
    print(f"\n  Random average Net: ${avg_random:+.2f}")
    print(f"  Strategy Net:       ${results['baseline'].get('net', 0):+.2f}")
    print(f"  Edge over random:   ${results['baseline'].get('net', 0) - avg_random:+.2f}")
    results["random_avg_net"] = round(avg_random, 2)

    # ══════════════════════════════════════════════════════════
    # TEST 5: Per-Strategy Isolation
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  TEST 5: PER-STRATEGY ISOLATION")
    print("=" * 90)

    for sname in PROFITABLE:
        single = {sname: PROFITABLE[sname]}
        eng_s = MaxTradesEngine(hub, COINS_10, max_pos=12)
        t_s = eng_s.run(strategies=single)
        print(quick_summary(t_s, f"ONLY {sname}"))
        results[f"solo_{sname}"] = analyze(t_s, sname)

    # ══════════════════════════════════════════════════════════
    # TEST 6: Parameter Neighborhood (nearby params profitable?)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  TEST 6: PARAMETER NEIGHBORHOOD (+-20% from optimal)")
    print("=" * 90)

    for sname, (fn, base_cfg) in PROFITABLE.items():
        neighbors_profitable = 0
        neighbors_total = 0
        for sl_delta in [-0.2, -0.1, 0, +0.1, +0.2]:
            for tp_delta in [-0.2, -0.1, 0, +0.1, +0.2]:
                test_cfg = dict(base_cfg)
                test_cfg["sl_mult"] = base_cfg["sl_mult"] * (1 + sl_delta)
                test_cfg["tp_mult"] = base_cfg["tp_mult"] * (1 + tp_delta)
                single = {sname: (fn, test_cfg)}
                eng_n = MaxTradesEngine(hub, COINS_10, max_pos=12)
                t_n = eng_n.run(strategies=single)
                if t_n:
                    net = sum(t["pnl_net"] for t in t_n)
                    if net > 0:
                        neighbors_profitable += 1
                neighbors_total += 1

        pct = neighbors_profitable / neighbors_total * 100
        status = "ROBUST" if pct >= 60 else "FRAGILE" if pct >= 30 else "OVERFITTED"
        print(f"  {sname:<22} {neighbors_profitable}/{neighbors_total} ({pct:.0f}%) neighbors profitable -> [{status}]")
        results[f"neighborhood_{sname}"] = {"profitable": neighbors_profitable, "total": neighbors_total, "pct": round(pct, 1)}

    # ══════════════════════════════════════════════════════════
    # TEST 7: Monte Carlo (shuffle trade order, check if profit is luck)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  TEST 7: MONTE CARLO (1000 shuffles of trade PnL sequence)")
    print("=" * 90)

    if t0:
        pnls = [t["pnl_net"] for t in t0]
        actual_net = sum(pnls)
        actual_mdd = _max_drawdown(pnls)

        worse_net = 0
        worse_mdd = 0
        n_sims = 1000
        sim_nets = []
        for _ in range(n_sims):
            shuffled = pnls.copy()
            random.shuffle(shuffled)
            sim_net = sum(shuffled)  # same total, different order
            sim_mdd = _max_drawdown(shuffled)
            sim_nets.append(sim_net)
            if sim_mdd <= actual_mdd:
                worse_mdd += 1

        # The total is always the same when shuffling, so check drawdown robustness
        print(f"  Actual Net PnL:  ${actual_net:+.2f}")
        print(f"  Actual Max DD:   ${actual_mdd:+.2f}")
        print(f"  MDD worse in {worse_mdd}/{n_sims} ({worse_mdd/n_sims*100:.1f}%) shuffles")
        print(f"  -> MDD robustness: {'GOOD' if worse_mdd/n_sims > 0.3 else 'FRAGILE (sequence-dependent)'}")

        # Also check: what % of random subsets of trades are profitable?
        subset_profitable = 0
        for _ in range(1000):
            subset = random.sample(pnls, max(len(pnls) // 2, 1))
            if sum(subset) > 0:
                subset_profitable += 1
        print(f"  Random 50% subsets profitable: {subset_profitable}/1000 ({subset_profitable/10:.1f}%)")
        results["mc_mdd_robustness"] = round(worse_mdd / n_sims * 100, 1)
        results["mc_subset_profitable"] = round(subset_profitable / 10, 1)

    # ══════════════════════════════════════════════════════════
    # TEST 8: Leave-One-Out Coin
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  TEST 8: LEAVE-ONE-OUT COIN (remove each coin, still profitable?)")
    print("=" * 90)

    loo_pass = 0
    for exclude_coin in COINS_10:
        loo_coins = [c for c in COINS_10 if c != exclude_coin]
        eng_loo = MaxTradesEngine(hub, loo_coins, max_pos=12)
        t_loo = eng_loo.run(strategies=PROFITABLE)
        net = sum(t["pnl_net"] for t in t_loo) if t_loo else 0
        status = "+" if net > 0 else "-"
        print(f"  Without {exclude_coin:<5}: Net=${net:>+9.2f} [{status}]")
        if net > 0:
            loo_pass += 1

    print(f"\n  Leave-one-out: {loo_pass}/{len(COINS_10)} still profitable")
    results["loo_pass"] = loo_pass
    results["loo_total"] = len(COINS_10)

    # ══════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  FINAL VERDICT: INTERSECTION OF ALL TESTS")
    print("=" * 90)

    checks = []

    # 1. Baseline profitable
    c1 = results.get("baseline", {}).get("net", 0) > 0
    checks.append(("Baseline profitable", c1))

    # 2. WF test profitable
    c2 = results.get("wf_test", {}).get("net", 0) > 0
    checks.append(("Walk-forward OOS profitable", c2))

    # 3. Both halves profitable
    c3 = results.get("half_1", {}).get("net", 0) > 0 and results.get("half_2", {}).get("net", 0) > 0
    checks.append(("Both halves profitable", c3))

    # 4. Reversed signals lose money
    c4 = results.get("reversed", {}).get("net", 0) < 0
    checks.append(("Reversed signals negative", c4))

    # 5. Better than random
    c5 = results.get("baseline", {}).get("net", 0) > results.get("random_avg_net", 0) * 1.5
    checks.append(("Beats random by 50%+", c5))

    # 6. Each strategy solo profitable
    c6 = all(results.get(f"solo_{s}", {}).get("net", 0) > 0 for s in PROFITABLE)
    checks.append(("All strategies solo profitable", c6))

    # 7. Parameter neighborhoods robust
    c7 = all(results.get(f"neighborhood_{s}", {}).get("pct", 0) >= 50 for s in PROFITABLE)
    checks.append(("All strategies param-robust (50%+)", c7))

    # 8. MC subset mostly profitable
    c8 = results.get("mc_subset_profitable", 0) >= 60
    checks.append(("MC 50%-subsets profitable 60%+", c8))

    # 9. LOO mostly passes
    c9 = results.get("loo_pass", 0) >= 8
    checks.append(("Leave-one-out 8/10+ pass", c9))

    passed = sum(1 for _, v in checks if v)
    total = len(checks)

    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  RESULT: {passed}/{total} tests passed")
    if passed == total:
        print("  >>> ALL TESTS PASSED — Edge is likely REAL <<<")
    elif passed >= 7:
        print("  >>> MOSTLY PASSED — Edge probable, proceed with caution <<<")
    elif passed >= 5:
        print("  >>> MIXED — Some edge, but fragile. More data needed <<<")
    else:
        print("  >>> FAILED — Likely overfitting or luck <<<")

    # Save
    output_dir = ROOT / "backtest" / "output"
    with open(output_dir / "full_validation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed:.0f}s\n")


def _max_drawdown(pnls):
    """Max drawdown from PnL sequence."""
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(np.min(dd))


if __name__ == "__main__":
    main()
