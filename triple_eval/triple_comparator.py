"""TripleComparator — 3가지 모드(paper/backtest/demo) 전략 성과 비교."""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("triple_comparator")


class TripleComparator:
    """Compare strategy performance across paper, backtest, and demo modes."""

    MODES = ("paper", "backtest", "demo")

    def __init__(
        self,
        state_dirs: dict[str, str],
        discord_webhook: str = "",
        lookback_hours: int = 24,
        initial_equity: float = 5000.0,
    ) -> None:
        self.state_dirs = {k: Path(v) for k, v in state_dirs.items()}
        self.discord_webhook = discord_webhook
        self.lookback_hours = lookback_hours
        self.initial_equity = initial_equity

    def _load_all_trades(self, mode: str) -> list[dict]:
        """Load ALL closed trades from trade_context.jsonl (no time filter) — for equity calc."""
        state_dir = self.state_dirs.get(mode)
        if state_dir is None:
            return []
        jsonl_path = state_dir / "trade_context.jsonl"
        if not jsonl_path.exists():
            return []
        trades: list[dict] = []
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("exit_reason"):  # 청산된 거래만
                        trades.append(record)
        except Exception as e:
            logger.error(f"[Comparator] Failed to load all trades for {mode}: {e}")
        return trades

    def _compute_equity(self, mode: str) -> float:
        """초기 자본 + 전체 누적 PnL = 현재 추정 자본."""
        all_trades = self._load_all_trades(mode)
        cumulative_pnl = sum(
            float(t.get("pnl_net_usdt", t.get("pnl_usdt", 0)))
            for t in all_trades
        )
        return self.initial_equity + cumulative_pnl

    def _load_trades(self, mode: str) -> list[dict]:
        """Load trades from trade_context.jsonl filtered by lookback_hours."""
        state_dir = self.state_dirs.get(mode)
        if state_dir is None:
            logger.warning(f"[Comparator] No state_dir for mode={mode}")
            return []

        jsonl_path = state_dir / "trade_context.jsonl"
        if not jsonl_path.exists():
            logger.info(f"[Comparator] {jsonl_path} not found — no trades for {mode}")
            return []

        cutoff_ts = time.time() - self.lookback_hours * 3600
        trades: list[dict] = []

        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Parse timestamp from record
                    ts_str = record.get("exit_ts") or record.get("ts") or ""
                    if ts_str:
                        try:
                            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            record_ts = dt.timestamp()
                        except Exception:
                            record_ts = cutoff_ts  # include if can't parse
                    else:
                        record_ts = cutoff_ts

                    if record_ts >= cutoff_ts:
                        trades.append(record)
        except Exception as e:
            logger.error(f"[Comparator] Failed to load {jsonl_path}: {e}")

        return trades

    def compute_metrics(self, trades: list[dict]) -> dict:
        """Compute aggregate metrics from trade list."""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl_usdt": 0.0,
                "avg_pnl_usdt": 0.0,
                "profit_factor": 0.0,
                "sl_rate": 0.0,
                "tp_rate": 0.0,
                "ttl_rate": 0.0,
                "avg_mfe_pct": 0.0,
                "avg_mae_pct": 0.0,
                "avg_bars_held": 0.0,
                "per_strategy": {},
            }

        total = len(trades)
        pnl_list = [float(t.get("pnl_net_usdt", t.get("pnl_usdt", 0))) for t in trades]
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p <= 0]
        win_rate = len(wins) / total if total > 0 else 0.0
        total_pnl = sum(pnl_list)
        avg_pnl = total_pnl / total if total > 0 else 0.0

        gross_wins = sum(wins)
        gross_losses = abs(sum(losses))
        profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else float("inf")

        reasons = [str(t.get("exit_reason", "")).upper() for t in trades]
        sl_rate = reasons.count("SL_HIT") / total if total > 0 else 0.0
        tp_rate = reasons.count("TP_HIT") / total if total > 0 else 0.0
        ttl_count = sum(1 for r in reasons if "TTL" in r or "MAX_BARS" in r)
        ttl_rate = ttl_count / total if total > 0 else 0.0

        mfe_list = [float(t.get("mfe_pct", 0)) for t in trades]
        mae_list = [abs(float(t.get("mae_pct", 0))) for t in trades]
        bars_list = [float(t.get("bars_held", 0)) for t in trades]

        avg_mfe = sum(mfe_list) / len(mfe_list) if mfe_list else 0.0
        avg_mae = sum(mae_list) / len(mae_list) if mae_list else 0.0
        avg_bars = sum(bars_list) / len(bars_list) if bars_list else 0.0

        per_strategy = self._per_strategy_metrics(trades)

        return {
            "total_trades": total,
            "win_rate": round(win_rate, 4),
            "total_pnl_usdt": round(total_pnl, 4),
            "avg_pnl_usdt": round(avg_pnl, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else 999.0,
            "sl_rate": round(sl_rate, 4),
            "tp_rate": round(tp_rate, 4),
            "ttl_rate": round(ttl_rate, 4),
            "avg_mfe_pct": round(avg_mfe * 100, 4),
            "avg_mae_pct": round(avg_mae * 100, 4),
            "avg_bars_held": round(avg_bars, 2),
            "per_strategy": per_strategy,
        }

    def _per_strategy_metrics(self, trades: list[dict]) -> dict:
        """Compute per-strategy metrics."""
        strategy_trades: dict[str, list[dict]] = {}
        for t in trades:
            strat = str(t.get("strategy", "unknown"))
            strategy_trades.setdefault(strat, []).append(t)

        result = {}
        for strat, strat_trades in strategy_trades.items():
            pnl_list = [float(t.get("pnl_net_usdt", t.get("pnl_usdt", 0))) for t in strat_trades]
            total = len(pnl_list)
            wins = [p for p in pnl_list if p > 0]
            losses = [p for p in pnl_list if p <= 0]
            win_rate = len(wins) / total if total > 0 else 0.0
            total_pnl = sum(pnl_list)
            gross_wins = sum(wins)
            gross_losses = abs(sum(losses))
            pf = (gross_wins / gross_losses) if gross_losses > 0 else float("inf")

            result[strat] = {
                "total_trades": total,
                "win_rate": round(win_rate, 4),
                "total_pnl_usdt": round(total_pnl, 4),
                "profit_factor": round(pf, 4) if pf != float("inf") else 999.0,
            }

        return result

    def format_discord(
        self,
        all_metrics: dict[str, dict],
        lookback_hours: int,
        equity_map: dict[str, float] | None = None,
    ) -> str:
        """Format metrics as a Discord table with 3 columns."""
        paper = all_metrics.get("paper", {})
        backtest = all_metrics.get("backtest", {})
        demo = all_metrics.get("demo", {})
        equity_map = equity_map or {}

        def fmt_val(metrics: dict, key: str, fmt: str = "") -> str:
            v = metrics.get(key, 0)
            if v is None:
                return "N/A"
            if fmt == "pct":
                return f"{v * 100:.1f}%"
            elif fmt == "pct_direct":
                return f"{v:.1f}%"
            elif fmt == "pnl":
                sign = "+" if v >= 0 else ""
                return f"{sign}${v:.1f}"
            elif fmt == "float2":
                return f"{v:.2f}"
            elif fmt == "int":
                return str(int(v))
            return str(v)

        def fmt_equity(mode: str) -> str:
            eq = equity_map.get(mode, self.initial_equity)
            ret = (eq - self.initial_equity) / self.initial_equity * 100
            sign = "+" if ret >= 0 else ""
            return f"${eq:.0f}({sign}{ret:.2f}%)"

        sep = "─" * 62
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lh_str = "전체" if lookback_hours >= 9999 else f"최근 {lookback_hours}h"
        lines = [
            f"📊 Triple Evaluation — {now_str}",
            f"초기 자본: ${self.initial_equity:,.0f}  |  조회 구간: {lh_str}",
            f"```",
            sep,
            f"{'지표':<22} {'Paper':>12} {'Backtest':>12} {'Demo':>12}",
            sep,
            f"{'💰 현재 자본(수익률)':<22} {fmt_equity('paper'):>12} {fmt_equity('backtest'):>12} {fmt_equity('demo'):>12}",
            sep,
            f"{'총 거래수':<22} {fmt_val(paper, 'total_trades', 'int'):>12} {fmt_val(backtest, 'total_trades', 'int'):>12} {fmt_val(demo, 'total_trades', 'int'):>12}",
            f"{'승률':<22} {fmt_val(paper, 'win_rate', 'pct'):>12} {fmt_val(backtest, 'win_rate', 'pct'):>12} {fmt_val(demo, 'win_rate', 'pct'):>12}",
            f"{'구간 PnL':<22} {fmt_val(paper, 'total_pnl_usdt', 'pnl'):>12} {fmt_val(backtest, 'total_pnl_usdt', 'pnl'):>12} {fmt_val(demo, 'total_pnl_usdt', 'pnl'):>12}",
            f"{'수익 팩터':<22} {fmt_val(paper, 'profit_factor', 'float2'):>12} {fmt_val(backtest, 'profit_factor', 'float2'):>12} {fmt_val(demo, 'profit_factor', 'float2'):>12}",
            f"{'SL 비율':<22} {fmt_val(paper, 'sl_rate', 'pct'):>12} {fmt_val(backtest, 'sl_rate', 'pct'):>12} {fmt_val(demo, 'sl_rate', 'pct'):>12}",
            f"{'TP 비율':<22} {fmt_val(paper, 'tp_rate', 'pct'):>12} {fmt_val(backtest, 'tp_rate', 'pct'):>12} {fmt_val(demo, 'tp_rate', 'pct'):>12}",
            f"{'평균 MFE':<22} {fmt_val(paper, 'avg_mfe_pct', 'pct_direct'):>12} {fmt_val(backtest, 'avg_mfe_pct', 'pct_direct'):>12} {fmt_val(demo, 'avg_mfe_pct', 'pct_direct'):>12}",
            sep,
            "전략별 Best/Worst:",
        ]

        for mode_name, metrics in [("Paper", paper), ("Backtest", backtest), ("Demo", demo)]:
            per = metrics.get("per_strategy", {})
            if not per:
                lines.append(f"  {mode_name}: (데이터 없음)")
                continue
            sorted_strats = sorted(per.items(), key=lambda x: x[1].get("total_pnl_usdt", 0))
            worst_strat, worst_data = sorted_strats[0]
            best_strat, best_data = sorted_strats[-1]
            best_pnl = best_data.get("total_pnl_usdt", 0)
            worst_pnl = worst_data.get("total_pnl_usdt", 0)
            sign_b = "+" if best_pnl >= 0 else ""
            sign_w = "+" if worst_pnl >= 0 else ""
            lines.append(
                f"  {mode_name}: Best={best_strat}({sign_b}${best_pnl:.1f})"
                f" | Worst={worst_strat}({sign_w}${worst_pnl:.1f})"
            )

        lines.append("```")
        return "\n".join(lines)

    def run_and_post(self) -> dict:
        """Run comparison across all modes and post to Discord."""
        all_metrics: dict[str, dict] = {}
        equity_map: dict[str, float] = {}

        for mode in self.MODES:
            trades = self._load_trades(mode)
            metrics = self.compute_metrics(trades)
            all_metrics[mode] = metrics
            equity_map[mode] = self._compute_equity(mode)
            ret_pct = (equity_map[mode] - self.initial_equity) / self.initial_equity * 100
            logger.info(
                f"[Comparator] {mode}: {metrics['total_trades']} trades "
                f"WR={metrics['win_rate']:.1%} PnL=${metrics['total_pnl_usdt']:.2f} "
                f"equity=${equity_map[mode]:.2f}({ret_pct:+.2f}%)"
            )

        msg = self.format_discord(all_metrics, self.lookback_hours, equity_map)

        # 항상 터미널/로그에 출력 (Discord 설정 여부 무관)
        print("\n" + msg, flush=True)
        logger.info("[Comparator] 30분 성적 리포트:\n" + msg)

        if self.discord_webhook:
            try:
                self._post_discord(msg)
            except Exception as e:
                logger.error(f"[Comparator] Discord post failed: {e}")

        return all_metrics

    def save_report(self, all_metrics: dict, report_path: Path) -> None:
        """Save metrics to JSON report."""
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lookback_hours": self.lookback_hours,
            "metrics": all_metrics,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"[Comparator] Report saved to {report_path}")

    def _post_discord(self, message: str) -> None:
        """Post message to Discord webhook."""
        if not self.discord_webhook:
            return
        # Discord embed description limit: 4096 chars
        if len(message) > 4000:
            message = message[:3997] + "..."
        payload = json.dumps({
            "embeds": [{
                "title": "Triple Evaluation Report",
                "description": message,
                "color": 3447003,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }]
        }).encode("utf-8")
        req = urllib.request.Request(
            self.discord_webhook,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "DiscordBot/1.0"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
