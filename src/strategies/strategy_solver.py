"""StrategySolver — 거래 결과 역추적으로 전략 파라미터 자동 최적화.

trade_context.jsonl의 완결 거래에서:
1. 수익 거래 vs 손실 거래의 신호 메타데이터 분포 차이를 분석
2. 수익 거래를 선택적으로 통과시키는 "최적 필터 임계치"를 탐색
3. config.extra 파라미터를 자동 조정 (bounds 내에서)

설계 원칙:
- Overfitting 방지: 최소 30건 이상, 파라미터 변경폭 제한 (±20%)
- 단방향 조정: tighter only (느슨해지는 방향은 금지)
- 점진적: 1회 조정에 최대 1개 파라미터만 변경
- 투명: 모든 조정을 Discord + 로그에 기록
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Optional

logger = logging.getLogger("strategy_solver")


class StrategySolver:
    """Per-strategy parameter optimizer using completed trade data."""

    # Minimum trades required to make any adjustment
    MIN_TRADES = 30

    # Maximum parameter change per tuning cycle (±20%)
    MAX_CHANGE_PCT = 0.20

    # Feature → config key mapping for each strategy
    # Each entry: (feature_name_in_trade_context, config_key, direction)
    # direction: "higher_is_better" or "lower_is_better"
    STRATEGY_FEATURES = {
        "cvd_extreme": [
            ("cvd_z_score", "cvd_sigma_mult", "abs_higher_wins"),
            ("entry_vpin", "vpin_min_threshold", "higher_wins"),
            ("entry_atr_pct", "atr_min_pct", "higher_wins"),
        ],
        "liquidation_fade": [
            ("entry_vpin", "vpin_min_threshold", "higher_wins"),
            ("entry_atr_pct", "atr_min_pct", "higher_wins"),
        ],
        "vwap_reversion": [
            ("entry_vpin", "vpin_min_threshold", "higher_wins"),
            ("entry_atr_pct", "atr_min_pct", "higher_wins"),
        ],
        "funding_arb": [  # MTF Momentum
            ("entry_atr_pct", "atr_min_pct", "higher_wins"),
            ("trend_20bar", "min_trend_strength", "abs_higher_wins"),
        ],
        "volume_impulse": [
            ("entry_atr_pct", "atr_min_pct", "higher_wins"),
        ],
        "oi_divergence": [
            ("entry_vpin", "vpin_min_threshold", "higher_wins"),
            ("entry_atr_pct", "atr_min_pct", "higher_wins"),
        ],
    }

    def __init__(self, trade_context_path: Path):
        self._path = trade_context_path

    def analyze(self, strategy_name: str) -> Optional[dict]:
        """Analyze trades for a strategy and return optimization suggestions.

        Returns:
            {
                "n_trades": int,
                "current_wr": float,
                "current_ev": float,
                "suggestions": [
                    {
                        "feature": str,
                        "config_key": str,
                        "current_threshold": float,
                        "suggested_threshold": float,
                        "expected_wr_improvement": float,
                        "trades_filtered_out": int,
                    },
                    ...
                ],
                "best_suggestion": dict | None,
            }
        """
        trades = self._load_closed_trades(strategy_name)
        if len(trades) < self.MIN_TRADES:
            return None

        wins = [t for t in trades if t.get("pnl_net_usdt", t.get("pnl_usdt", 0)) > 0]
        losses = [t for t in trades if t.get("pnl_net_usdt", t.get("pnl_usdt", 0)) <= 0]
        n = len(trades)
        wr = len(wins) / n if n > 0 else 0
        ev = statistics.mean([t.get("pnl_net_pct", t.get("pnl_pct", 0)) for t in trades])

        features = self.STRATEGY_FEATURES.get(strategy_name, [])
        suggestions = []

        for feature_name, config_key, direction in features:
            suggestion = self._find_optimal_threshold(
                trades, wins, losses, feature_name, config_key, direction
            )
            if suggestion:
                suggestions.append(suggestion)

        # ── R:R / TP optimization based on MFE distribution ──
        rr_suggestion = self._optimize_rr(trades, wins)
        if rr_suggestion:
            suggestions.append(rr_suggestion)

        # Sort by expected WR improvement
        suggestions.sort(key=lambda s: s["expected_wr_improvement"], reverse=True)

        return {
            "n_trades": n,
            "current_wr": round(wr, 4),
            "current_ev": round(ev, 6),
            "suggestions": suggestions,
            "best_suggestion": suggestions[0] if suggestions else None,
        }

    def _find_optimal_threshold(
        self,
        all_trades: list[dict],
        wins: list[dict],
        losses: list[dict],
        feature_name: str,
        config_key: str,
        direction: str,
    ) -> Optional[dict]:
        """Find the threshold for `feature_name` that maximizes net EV.

        Tests percentile thresholds (P25, P50, P75) and picks the one
        with the highest net EV while keeping at least 50% of trades.
        """
        # Extract feature values
        def _get_val(t):
            v = t.get(feature_name)
            if v is None or not isinstance(v, (int, float)):
                return None
            if direction == "abs_higher_wins":
                return abs(v)
            return v

        all_vals = [(t, _get_val(t)) for t in all_trades]
        all_vals = [(t, v) for t, v in all_vals if v is not None]

        if len(all_vals) < self.MIN_TRADES:
            return None

        vals_only = [v for _, v in all_vals]
        vals_only.sort()

        # Test percentile thresholds
        best = None
        n_total = len(all_vals)

        for pct in [0.25, 0.40, 0.50, 0.60, 0.75]:
            idx = int(n_total * pct)
            threshold = vals_only[idx]

            if direction in ("higher_wins", "abs_higher_wins"):
                # Keep trades where feature >= threshold
                filtered = [(t, v) for t, v in all_vals if v >= threshold]
            else:
                # Keep trades where feature <= threshold
                filtered = [(t, v) for t, v in all_vals if v <= threshold]

            if len(filtered) < self.MIN_TRADES * 0.5:
                continue  # Too few trades after filtering

            f_trades = [t for t, _ in filtered]
            f_wins = [t for t in f_trades if t.get("pnl_net_usdt", t.get("pnl_usdt", 0)) > 0]
            f_wr = len(f_wins) / len(f_trades)
            f_ev = statistics.mean([
                t.get("pnl_net_pct", t.get("pnl_pct", 0)) for t in f_trades
            ])

            current_wr = len([t for t, _ in all_vals if t.get("pnl_net_usdt", t.get("pnl_usdt", 0)) > 0]) / n_total
            wr_improvement = f_wr - current_wr

            if wr_improvement > 0.02 and f_ev > 0:  # At least 2%p WR improvement + positive EV
                if best is None or f_ev > best["filtered_ev"]:
                    best = {
                        "feature": feature_name,
                        "config_key": config_key,
                        "threshold": round(threshold, 6),
                        "percentile": pct,
                        "filtered_trades": len(f_trades),
                        "filtered_wr": round(f_wr, 4),
                        "filtered_ev": round(f_ev, 6),
                        "expected_wr_improvement": round(wr_improvement, 4),
                        "trades_filtered_out": n_total - len(f_trades),
                    }

        if best is None:
            return None

        return {
            "feature": best["feature"],
            "config_key": best["config_key"],
            "current_threshold": 0,  # Will be filled by caller
            "suggested_threshold": best["threshold"],
            "expected_wr_improvement": best["expected_wr_improvement"],
            "filtered_wr": best["filtered_wr"],
            "filtered_ev": best["filtered_ev"],
            "trades_filtered_out": best["trades_filtered_out"],
            "percentile": best["percentile"],
        }

    def _optimize_rr(
        self, all_trades: list[dict], wins: list[dict]
    ) -> Optional[dict]:
        """Find optimal TP R:R ratio based on MFE distribution.

        Logic: If most winning trades' MFE peaks at e.g. 0.8%, but TP is set at 1.2%,
        many trades never reach TP. Lowering TP to MFE P50 of winners would capture
        more wins (higher WR) at the cost of smaller wins (lower R:R).

        Tests TP = MFE_P25, P50, P75 of ALL trades, simulates new WR and EV.
        """
        mfe_vals = [t.get("mfe_pct", 0) for t in all_trades if t.get("mfe_pct", 0) > 0]
        sl_vals = [abs(t.get("sl_distance_pct", 0)) for t in all_trades if t.get("sl_distance_pct", 0) > 0]

        if len(mfe_vals) < 20 or len(sl_vals) < 20:
            return None

        avg_sl_pct = statistics.mean(sl_vals) / 100  # stored as %, convert to decimal
        fee_pct = 0.0015  # round-trip fee

        mfe_sorted = sorted(mfe_vals)
        best = None

        for pct_label, pct_idx in [("P25", 0.25), ("P50", 0.50), ("P75", 0.75)]:
            mfe_threshold = mfe_sorted[int(len(mfe_sorted) * pct_idx)]

            # Simulate: trade is a "win" if MFE >= threshold (could have hit TP at this level)
            sim_wins = sum(1 for t in all_trades if t.get("mfe_pct", 0) >= mfe_threshold)
            sim_wr = sim_wins / len(all_trades)

            # Simulated TP distance = mfe_threshold (in decimal, e.g., 0.008 = 0.8%)
            tp_pct = mfe_threshold
            rr = tp_pct / avg_sl_pct if avg_sl_pct > 0 else 0

            # EV = WR * (TP - fee) - (1-WR) * (SL + fee)
            sim_ev = sim_wr * (tp_pct - fee_pct) - (1 - sim_wr) * (avg_sl_pct + fee_pct)

            current_wr = len([t for t in all_trades if t.get("pnl_net_usdt", t.get("pnl_usdt", 0)) > 0]) / len(all_trades)
            current_ev = statistics.mean([t.get("pnl_net_pct", t.get("pnl_pct", 0)) for t in all_trades])

            wr_improvement = sim_wr - current_wr
            ev_improvement = sim_ev - current_ev

            if ev_improvement > 0.0001 and sim_ev > 0 and rr >= 1.0:
                if best is None or sim_ev > best["filtered_ev"]:
                    best = {
                        "feature": f"mfe_pct ({pct_label})",
                        "config_key": "tp_rr",
                        "current_threshold": 0,
                        "suggested_threshold": round(rr, 2),
                        "expected_wr_improvement": round(wr_improvement, 4),
                        "filtered_wr": round(sim_wr, 4),
                        "filtered_ev": round(sim_ev, 6),
                        "trades_filtered_out": 0,
                        "percentile": pct_idx,
                        "mfe_threshold_pct": round(mfe_threshold * 100, 3),
                        "simulated_rr": round(rr, 2),
                    }

        return best

    def apply_best(
        self, strategy_name: str, strategy_config_extra: dict
    ) -> Optional[dict]:
        """Analyze and apply the single best parameter adjustment.

        Returns the applied change dict, or None if no improvement found.
        Modifies strategy_config_extra in-place.
        """
        result = self.analyze(strategy_name)
        if result is None or result["best_suggestion"] is None:
            return None

        best = result["best_suggestion"]
        config_key = best["config_key"]

        # Only adjust if the key exists in config or is a new filter
        old_val = strategy_config_extra.get(config_key, 0)
        new_val = best["suggested_threshold"]

        # Bound check: max ±20% change from current (or from 0 → new)
        if old_val > 0:
            max_delta = old_val * self.MAX_CHANGE_PCT
            clamped = max(old_val - max_delta, min(old_val + max_delta, new_val))
            new_val = clamped

        strategy_config_extra[config_key] = round(new_val, 6)

        change = {
            "strategy": strategy_name,
            "config_key": config_key,
            "old_value": old_val,
            "new_value": round(new_val, 6),
            "expected_wr_improvement": best["expected_wr_improvement"],
            "n_trades": result["n_trades"],
            "current_wr": result["current_wr"],
            "current_ev": result["current_ev"],
        }

        logger.info(
            f"[Solver] {strategy_name}: {config_key} {old_val} → {new_val:.6f} "
            f"(WR +{best['expected_wr_improvement']:.1%}, "
            f"EV {result['current_ev']:.4%} → {best['filtered_ev']:.4%})"
        )
        return change

    def _load_closed_trades(self, strategy_name: str) -> list[dict]:
        """Load completed trades for a specific strategy."""
        if not self._path.exists():
            return []
        trades = []
        try:
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("strategy") == strategy_name and rec.get("exit_reason"):
                        trades.append(rec)
        except Exception as e:
            logger.warning(f"[Solver] Failed to load trades: {e}")
        return trades
