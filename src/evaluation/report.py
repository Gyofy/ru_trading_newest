"""Evaluation Report — walk-forward 결과를 리포트로 변환.

JSON + 콘솔 출력 + (향후) Markdown 지원.

Usage:
    reporter = EvalReporter(output_dir="data/reports/evaluation")
    reporter.print_summary(wf_result)
    reporter.save_json(wf_result, tag="hot_scanner_v1")
"""

from __future__ import annotations

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

from .walk_forward import WalkForwardResult, TradeResult


class EvalReporter:
    """Walk-forward 결과 리포트 생성."""

    def __init__(self, output_dir: str = "data/reports/evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def print_summary(self, result: WalkForwardResult, title: str = "") -> None:
        """콘솔에 요약 출력."""
        s = result.summary()
        if "error" in s:
            print(f"[Eval] No trades to evaluate (signals: {s['total_signals']})")
            return

        header = f"  WALK-FORWARD EVALUATION{f': {title}' if title else ''}"
        print()
        print("=" * 70)
        print(header)
        print("=" * 70)

        print(f"\n  Signal Coverage")
        print(f"    Total signals:      {s['total_signals']}")
        print(f"    Actionable:         {s['actionable_signals']} ({s['coverage']:.1%})")
        print(f"    Trades executed:    {s['n_trades']}")
        print(f"    Windows:            {s['n_windows']}")

        print(f"\n  Performance (fee-adjusted)")
        print(f"    Win rate:           {s['win_rate']:.1%}")
        print(f"    Total return:       {s['total_return_pct']:+.2f}%")
        print(f"    Avg return/trade:   {s['avg_return_pct']:+.3f}%")
        print(f"    Median return:      {s['median_return_pct']:+.3f}%")
        print(f"    Expectancy:         {s['expectancy_per_trade']:+.3f}%")
        print(f"    Profit factor:      {s['profit_factor']:.2f}")

        print(f"\n  Risk")
        print(f"    Max drawdown:       {s['max_drawdown_pct']:+.2f}%")
        print(f"    Avg win:            {s['avg_win']:+.3f}%")
        print(f"    Avg loss:           {s['avg_loss']:+.3f}%")
        print(f"    Avg MFE:            {s['max_favorable_avg']:+.2f}%")
        print(f"    Avg MAE:            {s['max_adverse_avg']:+.2f}%")

        print(f"\n  Exit Analysis")
        print(f"    TP hit rate:        {s['tp_hit_rate']:.1%}")
        print(f"    SL hit rate:        {s['sl_hit_rate']:.1%}")
        print(f"    Expiry rate:        {s['expiry_rate']:.1%}")
        print(f"    Avg hold bars:      {s['avg_hold_bars']:.1f}")
        print(f"    Turnover/window:    {s['turnover_per_window']:.1f}")

        # Regime breakdown
        if s.get("regime_pnl"):
            print(f"\n  By Regime")
            for regime, stats in sorted(s["regime_pnl"].items()):
                print(f"    {regime:10s}  n={stats['n']:3d}  "
                      f"avg={stats['avg_ret']:+.3f}%  total={stats['total_ret']:+.2f}%")

        # Coin breakdown
        if s.get("coin_pnl"):
            print(f"\n  By Coin")
            coins_sorted = sorted(s["coin_pnl"].items(),
                                  key=lambda x: x[1]["total_ret"], reverse=True)
            for coin, stats in coins_sorted:
                print(f"    {coin:8s}  n={stats['n']:3d}  "
                      f"avg={stats['avg_ret']:+.3f}%  total={stats['total_ret']:+.2f}%")

        # Window trend
        if s.get("window_returns") and len(s["window_returns"]) > 1:
            wr = s["window_returns"]
            print(f"\n  Window Trend")
            print(f"    Returns: {' → '.join(f'{r:+.2f}%' for r in wr)}")
            # 추세 방향
            if len(wr) >= 3:
                first_half = np.mean(wr[:len(wr)//2])
                second_half = np.mean(wr[len(wr)//2:])
                trend = "improving" if second_half > first_half else "degrading"
                print(f"    Trend:   {trend} (1st half {first_half:+.2f}% → 2nd half {second_half:+.2f}%)")

        print()
        print("=" * 70)

    def save_json(
        self,
        result: WalkForwardResult,
        tag: str = "eval",
        extra_meta: dict | None = None,
    ) -> Path:
        """JSON 파일로 저장."""
        s = result.summary()
        now = datetime.now()

        report = {
            "generated_at": now.isoformat(),
            "tag": tag,
            "summary": _make_serializable(s),
            "trades": [
                {
                    "signal_id": t.signal.signal_id,
                    "symbol": t.signal.symbol,
                    "action": t.signal.action.value,
                    "regime": t.signal.regime.value,
                    "ts": t.signal.ts.isoformat(),
                    "pred_return": t.signal.pred_return,
                    "p_up": t.signal.p_up,
                    "confidence": t.signal.confidence,
                    "size": t.signal.size,
                    "tp": t.signal.take_profit,
                    "sl": t.signal.stop_loss,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "forward_return_pct": round(t.forward_return_pct, 4),
                    "net_return_pct": round(t.net_return_pct, 4),
                    "hit_tp": t.hit_tp,
                    "hit_sl": t.hit_sl,
                    "expired": t.expired,
                    "hold_bars": t.hold_bars,
                    "max_favorable": round(t.max_favorable, 4),
                    "max_adverse": round(t.max_adverse, 4),
                    "is_win": t.is_win,
                }
                for t in result.all_trades
            ],
        }

        if extra_meta:
            report["meta"] = extra_meta

        fname = f"{tag}_{now.strftime('%Y%m%d_%H%M%S')}.json"
        fpath = self.output_dir / fname
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"[Eval] Report saved: {fpath}")
        return fpath


def _make_serializable(obj):
    """numpy 타입을 JSON 직렬화 가능하게 변환."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 6)
    if isinstance(obj, np.ndarray):
        return [_make_serializable(v) for v in obj.tolist()]
    if isinstance(obj, float):
        if np.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return round(obj, 6)
    return obj
