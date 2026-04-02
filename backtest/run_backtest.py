"""Entry point for historical backtesting of 6 strategies.

Usage:
    python -m backtest.run_backtest
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import yaml

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.data_loader import BacktestDataHub
from backtest.engine import BacktestEngine
from backtest.report import BacktestReport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest.main")


def main():
    start_time = time.time()

    # ── 1. Load config ──
    config_path = ROOT / "config" / "multi_strategy.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info(f"Config loaded: {config_path}")

    # ── 2. Load data ──
    data_dir = ROOT / "data" / "raw" / "binance"
    trade_coins = ["SOL", "XRP", "ADA", "DOT"]
    all_coins = trade_coins + ["BTC", "ETH"]  # BTC/ETH as reference

    logger.info(f"Loading data from {data_dir}")
    hub = BacktestDataHub(data_dir, all_coins)

    # ── 3. Run backtest ──
    logger.info("Starting backtest engine...")
    engine = BacktestEngine(hub, config, trade_coins)
    trades = engine.run(progress_interval=2000)

    elapsed = time.time() - start_time
    logger.info(f"Backtest completed in {elapsed:.1f}s - {len(trades)} trades")

    # ── 4. Generate report ──
    report = BacktestReport(trades, config)
    report.print_summary()

    # ── 5. Save outputs ──
    output_dir = ROOT / "backtest" / "output"
    report.save_all(output_dir)
    logger.info(f"All outputs saved to {output_dir}")

    # ── 6. Parameter sweep for profitability points ──
    if trades:
        print("\n" + "=" * 80)
        print("  PROFITABILITY EXPLORATION - Fee-Aware Parameter Sweep")
        print("=" * 80)
        run_parameter_sweep(hub, config, trade_coins)


def run_parameter_sweep(hub: BacktestDataHub, base_config: dict, coins: list[str]):
    """Sweep key parameters to find profitable configurations."""
    import copy

    results = []

    # Sweep SL and TP multipliers for each strategy
    strategies_to_test = ["cvd_extreme", "liquidation_fade", "vwap_reversion",
                          "funding_arb", "volume_impulse", "oi_divergence"]

    # Quick parameter sweep: vary SL mult and TP ratio
    sl_mults = [2.0, 3.0, 4.0, 6.0, 8.0]
    tp_rrs = [1.5, 2.0, 3.0, 4.0, 5.0]

    for strat in strategies_to_test:
        if strat not in base_config.get("strategies", {}):
            continue

        best_net = -999999
        best_params = {}
        tested = 0

        for sl_m in sl_mults:
            for tp_r in tp_rrs:
                # Skip unreasonable combos
                if tp_r < 1.0:
                    continue

                test_config = copy.deepcopy(base_config)
                # Disable all strategies except current one
                for s in test_config.get("strategies", {}):
                    test_config["strategies"][s]["enabled"] = (s == strat)

                test_config["strategies"][strat]["sl_atr_mult"] = sl_m
                extra = test_config["strategies"][strat].get("extra", {})
                # Set TP ratio key
                if strat == "liquidation_fade":
                    extra["tp_atr_mult"] = tp_r * sl_m  # keep relative
                else:
                    extra["tp_rr"] = tp_r
                test_config["strategies"][strat]["extra"] = extra

                engine = BacktestEngine(hub, test_config, coins)
                trades = engine.run(progress_interval=999999)

                if trades:
                    import pandas as pd
                    df = pd.DataFrame(trades)
                    net_pnl = df["pnl_net_usdt"].sum()
                    n = len(df)
                    wr = len(df[df["pnl_net_usdt"] > 0]) / n * 100 if n > 0 else 0
                    pf = (df[df["pnl_net_usdt"] > 0]["pnl_net_usdt"].sum() /
                          abs(df[df["pnl_net_usdt"] <= 0]["pnl_net_usdt"].sum() + 0.01))

                    results.append({
                        "strategy": strat,
                        "sl_mult": sl_m,
                        "tp_rr": tp_r,
                        "n_trades": n,
                        "net_pnl": round(net_pnl, 2),
                        "win_rate": round(wr, 1),
                        "profit_factor": round(pf, 2),
                    })

                    if net_pnl > best_net:
                        best_net = net_pnl
                        best_params = {"sl_mult": sl_m, "tp_rr": tp_r, "net_pnl": net_pnl, "n": n, "wr": wr}

                tested += 1

        if best_params:
            status = "PROFITABLE" if best_net > 0 else "NOT PROFITABLE"
            print(f"\n  [{strat}] Best config ({tested} tested): [{status}]")
            print(f"    SL mult={best_params['sl_mult']}, TP R:R={best_params.get('tp_rr', '?')}")
            print(f"    {best_params['n']} trades, WR={best_params['wr']:.1f}%, Net=${best_net:+.2f}")
        else:
            print(f"\n  [{strat}] No trades generated across all parameter combos")

    # Save sweep results
    if results:
        output_dir = ROOT / "backtest" / "output"
        with open(output_dir / "parameter_sweep.json", "w") as f:
            json.dump(results, f, indent=2)

        # Find any profitable configurations
        profitable = [r for r in results if r["net_pnl"] > 0]
        print(f"\n\n  === PROFITABLE CONFIGURATIONS: {len(profitable)} / {len(results)} ===")
        if profitable:
            profitable.sort(key=lambda x: x["net_pnl"], reverse=True)
            for p in profitable[:10]:
                print(
                    f"    {p['strategy']:<20} SL={p['sl_mult']:.1f} TP={p['tp_rr']:.1f} "
                    f"N={p['n_trades']:>4} WR={p['win_rate']:>5.1f}% "
                    f"PF={p['profit_factor']:>5.2f} Net=${p['net_pnl']:>+8.2f}"
                )
        else:
            print("    NONE - no parameter combination was net profitable after fees")
            print("    This is an honest result: the strategies may not have edge at 1m frequency")

    print()


if __name__ == "__main__":
    main()
