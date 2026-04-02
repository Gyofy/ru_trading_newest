"""BacktestReport — metrics, analysis, and honest assessment of 6 strategies."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("backtest.report")


class BacktestReport:
    """Compute and display comprehensive backtest metrics."""

    def __init__(self, trades: list[dict], config: dict):
        self.trades = trades
        self.config = config
        self.df = pd.DataFrame(trades) if trades else pd.DataFrame()

    def compute_strategy_metrics(self) -> dict[str, dict]:
        """Per-strategy metrics with fee-aware PnL."""
        if self.df.empty:
            return {}

        results = {}
        for strat in self.df["strategy"].unique():
            sdf = self.df[self.df["strategy"] == strat]
            results[strat] = self._compute_metrics(sdf, strat)

        # Overall
        results["__TOTAL__"] = self._compute_metrics(self.df, "TOTAL")
        return results

    def _compute_metrics(self, df: pd.DataFrame, name: str) -> dict:
        n = len(df)
        if n == 0:
            return {"name": name, "n_trades": 0}

        wins = df[df["pnl_net_usdt"] > 0]
        losses = df[df["pnl_net_usdt"] <= 0]
        wr = len(wins) / n * 100

        gross_pnl = df["pnl_usdt"].sum()
        total_fees = df["fee_usdt"].sum()
        net_pnl = df["pnl_net_usdt"].sum()

        avg_win = float(wins["pnl_net_usdt"].mean()) if len(wins) > 0 else 0
        avg_loss = float(losses["pnl_net_usdt"].mean()) if len(losses) > 0 else 0

        # Profit Factor
        total_wins = float(wins["pnl_net_usdt"].sum()) if len(wins) > 0 else 0
        total_losses = abs(float(losses["pnl_net_usdt"].sum())) if len(losses) > 0 else 0.01
        pf = total_wins / total_losses if total_losses > 0 else 0

        # Sharpe (per-trade, annualized assuming 1440 trades/day for 1m)
        returns = df["pnl_net_pct"].values
        avg_bars = df["bars_held"].mean()
        if np.std(returns) > 0 and avg_bars > 0:
            trades_per_day = 1440 / avg_bars
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(trades_per_day * 252))
        else:
            sharpe = 0.0

        # Drawdown (cumulative PnL series)
        cum_pnl = df["pnl_net_usdt"].cumsum()
        running_max = cum_pnl.cummax()
        drawdown = cum_pnl - running_max
        max_dd = float(drawdown.min())

        # Exit reason breakdown
        exit_counts = df["exit_reason"].value_counts().to_dict()

        # MFE/MAE analysis
        avg_mfe = float(df["mfe_pct"].mean()) * 100
        avg_mae = float(df["mae_pct"].mean()) * 100

        # Per-coin breakdown
        coin_pnl = df.groupby("coin")["pnl_net_usdt"].agg(["count", "sum", "mean"])

        # SL/TP distance analysis
        avg_sl_dist = float(df["sl_distance_pct"].mean())
        avg_tp_dist = float(df["tp_distance_pct"].mean())

        return {
            "name": name,
            "n_trades": n,
            "win_rate": round(wr, 1),
            "gross_pnl": round(gross_pnl, 2),
            "total_fees": round(total_fees, 2),
            "net_pnl": round(net_pnl, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(pf, 2),
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(max_dd, 2),
            "avg_bars_held": round(float(avg_bars), 1),
            "avg_mfe_pct": round(avg_mfe, 3),
            "avg_mae_pct": round(avg_mae, 3),
            "avg_sl_dist_pct": round(avg_sl_dist, 3),
            "avg_tp_dist_pct": round(avg_tp_dist, 3),
            "exit_reasons": exit_counts,
            "coin_pnl": {
                coin: {"count": int(row["count"]), "pnl": round(row["sum"], 2)}
                for coin, row in coin_pnl.iterrows()
            },
        }

    def fee_impact_analysis(self) -> dict:
        """How much do fees eat into profitability?"""
        if self.df.empty:
            return {}

        result = {}
        for strat in list(self.df["strategy"].unique()) + ["__TOTAL__"]:
            sdf = self.df if strat == "__TOTAL__" else self.df[self.df["strategy"] == strat]

            gross = sdf["pnl_usdt"].sum()
            fees = sdf["fee_usdt"].sum()
            net = sdf["pnl_net_usdt"].sum()

            # WR with and without fees
            wr_gross = len(sdf[sdf["pnl_usdt"] > 0]) / len(sdf) * 100 if len(sdf) > 0 else 0
            wr_net = len(sdf[sdf["pnl_net_usdt"] > 0]) / len(sdf) * 100 if len(sdf) > 0 else 0

            # Break-even analysis: minimum WR needed given avg win/loss
            wins = sdf[sdf["pnl_usdt"] > 0]
            losses = sdf[sdf["pnl_usdt"] <= 0]
            avg_gross_win = float(wins["pnl_usdt"].mean()) if len(wins) > 0 else 0
            avg_gross_loss = abs(float(losses["pnl_usdt"].mean())) if len(losses) > 0 else 0

            if avg_gross_win + avg_gross_loss > 0:
                be_wr = avg_gross_loss / (avg_gross_win + avg_gross_loss) * 100
            else:
                be_wr = 50.0

            result[strat] = {
                "gross_pnl": round(gross, 2),
                "total_fees": round(fees, 2),
                "net_pnl": round(net, 2),
                "fee_pct_of_gross": round(fees / abs(gross) * 100, 1) if abs(gross) > 0.01 else 0,
                "wr_gross": round(wr_gross, 1),
                "wr_net": round(wr_net, 1),
                "wr_drop_from_fees": round(wr_gross - wr_net, 1),
                "breakeven_wr": round(be_wr, 1),
            }

        return result

    def data_leakage_check(self) -> list[str]:
        """Verify no data leakage in backtest results."""
        warnings = []

        if self.df.empty:
            warnings.append("No trades to check")
            return warnings

        # Check 1: Entry before exit
        for _, t in self.df.iterrows():
            if t["ts_entry"] >= t["ts_exit"]:
                warnings.append(f"Trade {t['trade_id']}: entry >= exit timestamp")

        # Check 2: SL/TP hit should match exit price
        sl_trades = self.df[self.df["exit_reason"] == "SL_HIT"]
        for _, t in sl_trades.iterrows():
            expected = t["sl_price"]
            # Account for slippage
            if abs(t["exit_price"] - expected) / expected > 0.002:  # 0.2% tolerance
                warnings.append(
                    f"Trade {t['trade_id']}: SL exit price {t['exit_price']:.4f} "
                    f"far from SL {expected:.4f}"
                )

        # Check 3: Unrealistic win rates
        for strat in self.df["strategy"].unique():
            sdf = self.df[self.df["strategy"] == strat]
            if len(sdf) >= 10:
                wr = len(sdf[sdf["pnl_net_usdt"] > 0]) / len(sdf) * 100
                if wr > 80:
                    warnings.append(
                        f"{strat}: WR {wr:.1f}% suspiciously high — check for lookahead"
                    )

        # Check 4: Average bars held should be > 0
        if (self.df["bars_held"] == 0).any():
            warnings.append("Some trades have 0 bars held — immediate exit")

        # Check 5: MFE should always be >= 0
        if (self.df["mfe_pct"] < -0.001).any():
            warnings.append("Negative MFE detected — logic error in tracking")

        if not warnings:
            warnings.append("PASS: No data leakage indicators found")

        return warnings

    def parameter_sensitivity(self) -> dict:
        """Analyze if tighter/looser params would improve results."""
        if self.df.empty:
            return {}

        result = {}
        for strat in self.df["strategy"].unique():
            sdf = self.df[self.df["strategy"] == strat]
            if len(sdf) < 10:
                continue

            analysis = {}

            # SL analysis: what % of trades were stopped out that later went profitable?
            sl_trades = sdf[sdf["exit_reason"] == "SL_HIT"]
            if len(sl_trades) > 0:
                # MFE > SL dist means trade went profitable before SL hit
                sl_premature = sl_trades[sl_trades["mfe_pct"] > sl_trades["sl_distance_pct"] / 100]
                analysis["sl_premature_pct"] = round(len(sl_premature) / len(sl_trades) * 100, 1)
                analysis["sl_avg_mae"] = round(float(sl_trades["mae_pct"].mean()) * 100, 3)

            # TP analysis: MFE distribution for TP hits
            tp_trades = sdf[sdf["exit_reason"] == "TP_HIT"]
            if len(tp_trades) > 0:
                analysis["tp_avg_mfe"] = round(float(tp_trades["mfe_pct"].mean()) * 100, 3)
                analysis["tp_could_have_more"] = round(
                    float((tp_trades["mfe_pct"] > tp_trades["tp_distance_pct"] / 100).mean()) * 100, 1
                )

            # Time stop analysis
            ttl_trades = sdf[sdf["exit_reason"] == "TIME_STOP"]
            if len(ttl_trades) > 0:
                ttl_net = ttl_trades["pnl_net_usdt"].mean()
                analysis["ttl_count"] = len(ttl_trades)
                analysis["ttl_avg_pnl"] = round(float(ttl_net), 2)

            # Signal strength vs outcome
            if "signal_strength" in sdf.columns:
                strength = sdf["signal_strength"]
                if strength.std() > 0:
                    median_str = float(strength.median())
                    high_str = sdf[strength > median_str]
                    low_str = sdf[strength <= median_str]
                    analysis["high_strength_wr"] = round(
                        len(high_str[high_str["pnl_net_usdt"] > 0]) / max(len(high_str), 1) * 100, 1
                    )
                    analysis["low_strength_wr"] = round(
                        len(low_str[low_str["pnl_net_usdt"] > 0]) / max(len(low_str), 1) * 100, 1
                    )

            result[strat] = analysis

        return result

    def print_summary(self):
        """Print comprehensive console summary."""
        metrics = self.compute_strategy_metrics()
        fee_analysis = self.fee_impact_analysis()
        leakage = self.data_leakage_check()
        sensitivity = self.parameter_sensitivity()

        print("\n" + "=" * 80)
        print("  BACKTEST REPORT - 6 Strategy Historical Replay")
        print("=" * 80)

        if self.df.empty:
            print("\n  NO TRADES GENERATED\n")
            return

        print(f"\n  Period: {self.df['ts_entry'].min()[:10]} ~ {self.df['ts_exit'].max()[:10]}")
        print(f"  Total Trades: {len(self.df)}")
        print()

        # Strategy summary table
        header = f"{'Strategy':<20} {'Trades':>6} {'WR%':>6} {'Gross$':>9} {'Fees$':>8} {'Net$':>9} {'PF':>5} {'MDD$':>8} {'AvgBars':>7}"
        print(header)
        print("-" * len(header))

        for name in sorted(metrics.keys()):
            if name == "__TOTAL__":
                continue
            m = metrics[name]
            print(
                f"{m['name']:<20} {m['n_trades']:>6} {m['win_rate']:>5.1f}% "
                f"{m['gross_pnl']:>+9.2f} {m['total_fees']:>8.2f} {m['net_pnl']:>+9.2f} "
                f"{m['profit_factor']:>5.2f} {m['max_drawdown']:>8.2f} {m['avg_bars_held']:>7.1f}"
            )

        if "__TOTAL__" in metrics:
            m = metrics["__TOTAL__"]
            print("-" * len(header))
            print(
                f"{'TOTAL':<20} {m['n_trades']:>6} {m['win_rate']:>5.1f}% "
                f"{m['gross_pnl']:>+9.2f} {m['total_fees']:>8.2f} {m['net_pnl']:>+9.2f} "
                f"{m['profit_factor']:>5.2f} {m['max_drawdown']:>8.2f} {m['avg_bars_held']:>7.1f}"
            )

        # Fee impact
        print("\n" + "=" * 80)
        print("  FEE IMPACT ANALYSIS")
        print("=" * 80)
        header2 = f"{'Strategy':<20} {'Gross$':>9} {'Fees$':>8} {'Net$':>9} {'Fee/Gross%':>10} {'WR(gross)':>9} {'WR(net)':>8} {'BE WR':>6}"
        print(header2)
        print("-" * len(header2))
        for name in sorted(fee_analysis.keys()):
            if name == "__TOTAL__":
                continue
            f = fee_analysis[name]
            print(
                f"{name:<20} {f['gross_pnl']:>+9.2f} {f['total_fees']:>8.2f} {f['net_pnl']:>+9.2f} "
                f"{f['fee_pct_of_gross']:>9.1f}% {f['wr_gross']:>8.1f}% {f['wr_net']:>7.1f}% {f['breakeven_wr']:>5.1f}%"
            )
        if "__TOTAL__" in fee_analysis:
            f = fee_analysis["__TOTAL__"]
            print("-" * len(header2))
            print(
                f"{'TOTAL':<20} {f['gross_pnl']:>+9.2f} {f['total_fees']:>8.2f} {f['net_pnl']:>+9.2f} "
                f"{f['fee_pct_of_gross']:>9.1f}% {f['wr_gross']:>8.1f}% {f['wr_net']:>7.1f}% {f['breakeven_wr']:>5.1f}%"
            )

        # Exit reason breakdown
        print("\n" + "=" * 80)
        print("  EXIT REASON BREAKDOWN")
        print("=" * 80)
        for name in sorted(metrics.keys()):
            if name == "__TOTAL__":
                continue
            m = metrics[name]
            exits = m.get("exit_reasons", {})
            parts = [f"{k}:{v}" for k, v in sorted(exits.items())]
            print(f"  {name:<20} {', '.join(parts)}")

        # Per-coin breakdown
        print("\n" + "=" * 80)
        print("  PER-COIN BREAKDOWN")
        print("=" * 80)
        coin_totals = defaultdict(lambda: {"count": 0, "pnl": 0})
        for name, m in metrics.items():
            if name == "__TOTAL__":
                continue
            for coin, data in m.get("coin_pnl", {}).items():
                coin_totals[coin]["count"] += data["count"]
                coin_totals[coin]["pnl"] += data["pnl"]
        for coin in sorted(coin_totals.keys()):
            d = coin_totals[coin]
            print(f"  {coin:<6} {d['count']:>5} trades  Net PnL: ${d['pnl']:>+8.2f}")

        # Parameter sensitivity
        print("\n" + "=" * 80)
        print("  PARAMETER SENSITIVITY ANALYSIS")
        print("=" * 80)
        for name, analysis in sorted(sensitivity.items()):
            print(f"\n  [{name}]")
            if "sl_premature_pct" in analysis:
                print(f"    SL premature (went profit before SL): {analysis['sl_premature_pct']:.1f}%")
            if "tp_could_have_more" in analysis:
                print(f"    TP could have captured more (MFE > TP dist): {analysis['tp_could_have_more']:.1f}%")
            if "ttl_count" in analysis:
                print(f"    Time stops: {analysis['ttl_count']} trades, avg PnL: ${analysis['ttl_avg_pnl']:.2f}")
            if "high_strength_wr" in analysis:
                print(f"    Signal strength → WR: high={analysis['high_strength_wr']:.1f}%, low={analysis['low_strength_wr']:.1f}%")

        # Data leakage check
        print("\n" + "=" * 80)
        print("  DATA LEAKAGE CHECK")
        print("=" * 80)
        for w in leakage:
            print(f"  {'[OK]' if 'PASS' in w else '[WARN]'} {w}")

        # MFE/MAE analysis for optimization hints
        print("\n" + "=" * 80)
        print("  MFE/MAE OPTIMIZATION HINTS")
        print("=" * 80)
        for strat in sorted(metrics.keys()):
            if strat == "__TOTAL__":
                continue
            m = metrics[strat]
            if m["n_trades"] < 5:
                continue
            sdf = self.df[self.df["strategy"] == strat]

            # Percentile analysis
            mfe_vals = sdf["mfe_pct"].values * 100
            mae_vals = sdf["mae_pct"].values * 100

            print(f"\n  [{strat}] (n={m['n_trades']})")
            print(f"    MFE (max favorable): P25={np.percentile(mfe_vals, 25):.3f}% P50={np.percentile(mfe_vals, 50):.3f}% P75={np.percentile(mfe_vals, 75):.3f}%")
            print(f"    MAE (max adverse):   P25={np.percentile(mae_vals, 25):.3f}% P50={np.percentile(mae_vals, 50):.3f}% P75={np.percentile(mae_vals, 75):.3f}%")
            print(f"    Current SL dist:     {m['avg_sl_dist_pct']:.3f}%")
            print(f"    Current TP dist:     {m['avg_tp_dist_pct']:.3f}%")

            # Suggestion
            mae_p75 = np.percentile(np.abs(mae_vals), 75)
            suggested_sl = mae_p75 * 1.2
            mfe_p50 = np.percentile(mfe_vals, 50)
            suggested_tp = mfe_p50 * 0.9 if mfe_p50 > 0 else m["avg_tp_dist_pct"]

            print(f"    Suggested SL:        {suggested_sl:.3f}% (P75 MAE × 1.2)")
            print(f"    Suggested TP:        {suggested_tp:.3f}% (P50 MFE × 0.9)")

        print("\n" + "=" * 80)
        print()

    def save_all(self, output_dir: Path):
        """Save all outputs to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Trade log (JSONL format compatible with StrategySolver)
        with open(output_dir / "trade_context_backtest.jsonl", "w") as f:
            for t in self.trades:
                f.write(json.dumps(t, default=str) + "\n")

        # Metrics
        metrics = self.compute_strategy_metrics()
        with open(output_dir / "metrics_summary.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        # Fee analysis
        fee_analysis = self.fee_impact_analysis()
        with open(output_dir / "fee_analysis.json", "w") as f:
            json.dump(fee_analysis, f, indent=2, default=str)

        # Leakage check
        leakage = self.data_leakage_check()
        with open(output_dir / "leakage_check.json", "w") as f:
            json.dump(leakage, f, indent=2)

        # Sensitivity
        sensitivity = self.parameter_sensitivity()
        with open(output_dir / "parameter_sensitivity.json", "w") as f:
            json.dump(sensitivity, f, indent=2, default=str)

        # Equity curve CSV
        if not self.df.empty:
            eq = self.df[["ts_exit", "strategy", "pnl_net_usdt"]].copy()
            eq.sort_values("ts_exit", inplace=True)
            eq["cumulative_pnl"] = eq["pnl_net_usdt"].cumsum()
            eq.to_csv(output_dir / "equity_curve.csv", index=False)

        logger.info(f"Reports saved to {output_dir}")
