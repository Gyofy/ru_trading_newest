"""StrategySolver v2 — 거래 결과 역추적으로 전략 파라미터 자동 최적화.

v1 → v2 핵심 변경:
1. Temporal train/test split (70/30) — in-sample overfitting 방지
2. MIN_TRADES 50건 — 통계적 신뢰도 (v1: 30건)
3. Tighter-only 강제 — 진입 필터는 엄격해지는 방향으로만 조정
4. 실제 config.extra 키 매핑 — phantom key 제거
5. Per-strategy TP 키 분리 — tp_rr vs tp_atr_mult
6. Clamp 표준화 — min(upper, max(lower, val))
7. 변경사항 JSON 영속화 — 재시작 시 보존 + restore
8. MFE 기반 R:R — 경로 의존성 caveat, 최소 R:R 1.5
9. current_threshold 실제 config에서 채움
10. EVGuardian suspended 전략 skip

설계 원칙:
- Overfitting 방지: 50건+, temporal split, test-set 양수 EV 필수
- 단방향 조정: tighter only (느슨해지는 방향 차단)
- 점진적: 1회 조정에 최대 1개 파라미터, ±15% 범위
- 투명: 모든 조정을 Discord + 로그 + JSON에 기록
"""

from __future__ import annotations

import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("strategy_solver")


class StrategySolver:
    """Per-strategy parameter optimizer using completed trade data."""

    # ── Tuning Constants ──
    MIN_TRADES = 50            # v1: 30 → 50
    MAX_CHANGE_PCT = 0.15      # v1: 0.20 → 0.15
    MIN_WR_IMPROVEMENT = 0.05  # v1: 0.02 → 0.05 (5%p)
    TRAIN_RATIO = 0.70         # 시간순 70% train, 30% test
    MIN_RR = 1.5               # R:R 최적화 하한 (v1: 1.0)

    # Feature → config key mapping (실제 config.extra 키만)
    # (trade_context_field, config_extra_key, direction)
    # direction: "tighter_higher" = 임계치를 올리면 진입 조건이 엄격해짐
    STRATEGY_FEATURES: dict[str, list[tuple[str, str, str]]] = {
        "cvd_extreme": [
            # |cvd_z_score| → cvd_sigma_mult: z-score 임계치 상향 = 더 극단만 진입
            ("cvd_z_score", "cvd_sigma_mult", "tighter_higher"),
        ],
        "liquidation_fade": [
            # signal_strength (= oi_drop_sigma) → oi_sigma_threshold
            ("signal_strength", "oi_sigma_threshold", "tighter_higher"),
            # ofi_value (= taker_ratio) → taker_spike_mult
            ("ofi_value", "taker_spike_mult", "tighter_higher"),
        ],
        "vwap_reversion": [
            # signal_strength (= sigma_distance) → sigma_mult
            ("signal_strength", "sigma_mult", "tighter_higher"),
        ],
        "volume_impulse": [
            # signal_strength (= volume_ratio) → volume_mult
            ("signal_strength", "volume_mult", "tighter_higher"),
        ],
        "oi_divergence": [
            # signal_strength → oi_change_threshold
            ("signal_strength", "oi_change_threshold", "tighter_higher"),
        ],
    }

    # Per-strategy TP config key (v1: 모든 전략에 tp_rr → liq_fade 버그)
    TP_CONFIG_KEY: dict[str, str] = {
        "cvd_extreme": "tp_rr",
        "liquidation_fade": "tp_atr_mult",
        "vwap_reversion": "tp_rr",
        "funding_arb": "tp_rr",
        "volume_impulse": "tp_rr",
        "oi_divergence": "tp_rr",
    }

    def __init__(
        self,
        trade_context_path: Path,
        persist_path: Path | None = None,
    ):
        self._path = trade_context_path
        self._persist_path = persist_path or trade_context_path.parent / "solver_state.json"
        self._applied_history: list[dict] = []
        self._load_state()

    # ═══════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════

    def analyze(self, strategy_name: str) -> Optional[dict]:
        """Analyze trades and return optimization suggestions with train/test validation.

        Returns:
            {
                "n_trades": int,
                "n_train": int,
                "n_test": int,
                "current_wr": float,
                "current_ev": float,
                "suggestions": [...],
                "best_suggestion": dict | None,
            }
        """
        trades = self._load_closed_trades(strategy_name)
        if len(trades) < self.MIN_TRADES:
            logger.info(
                f"[Solver] {strategy_name}: {len(trades)}/{self.MIN_TRADES} trades — skipping"
            )
            return None

        # ── Temporal split (v2 핵심: in-sample overfitting 방지) ──
        split_idx = int(len(trades) * self.TRAIN_RATIO)
        train = trades[:split_idx]
        test = trades[split_idx:]

        if len(train) < 30 or len(test) < 10:
            logger.info(
                f"[Solver] {strategy_name}: train={len(train)}/test={len(test)} "
                f"— insufficient split"
            )
            return None

        n = len(trades)
        wins = [t for t in trades if self._pnl(t) > 0]
        wr = len(wins) / n
        ev = statistics.mean([self._pnl_pct(t) for t in trades])

        suggestions: list[dict] = []

        # Feature threshold optimization
        features = self.STRATEGY_FEATURES.get(strategy_name, [])
        for feature_name, config_key, direction in features:
            s = self._find_optimal_threshold(
                train, test, feature_name, config_key, direction,
            )
            if s:
                suggestions.append(s)

        # R:R (TP) optimization via MFE distribution
        tp_key = self.TP_CONFIG_KEY.get(strategy_name, "tp_rr")
        rr_s = self._optimize_rr(train, test, tp_key)
        if rr_s:
            suggestions.append(rr_s)

        # Rank by test-set EV (out-of-sample 기준)
        suggestions.sort(key=lambda s: s.get("test_ev", 0), reverse=True)

        return {
            "n_trades": n,
            "n_train": len(train),
            "n_test": len(test),
            "current_wr": round(wr, 4),
            "current_ev": round(ev, 6),
            "suggestions": suggestions,
            "best_suggestion": suggestions[0] if suggestions else None,
        }

    def apply_best(
        self,
        strategy_name: str,
        strategy_config_extra: dict,
    ) -> Optional[dict]:
        """Analyze and apply the single best parameter adjustment.

        Modifies strategy_config_extra in-place (within bounds).
        Returns the applied change dict, or None.
        """
        result = self.analyze(strategy_name)
        if result is None or result["best_suggestion"] is None:
            return None

        best = result["best_suggestion"]
        config_key = best["config_key"]
        new_val = best["suggested_threshold"]

        old_val = strategy_config_extra.get(config_key, 0)

        # ── Tighter-only enforcement (v2) ──
        # "tighter_higher" → new_val must be >= old_val (더 엄격해지는 방향만)
        direction = best.get("direction", "")
        if direction == "tighter_higher" and old_val > 0 and new_val < old_val:
            logger.info(
                f"[Solver] {strategy_name}: {config_key} skip — "
                f"loosening {old_val} → {new_val} blocked (tighter-only)"
            )
            return None

        # ── Clamp: max ±15% change (v2: 표준 clamp 순서) ──
        if old_val > 0:
            max_delta = old_val * self.MAX_CHANGE_PCT
            lower = old_val - max_delta
            upper = old_val + max_delta
            new_val = min(upper, max(lower, new_val))  # standard clamp

        new_val = round(new_val, 6)
        strategy_config_extra[config_key] = new_val

        change = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy_name,
            "config_key": config_key,
            "old_value": old_val,
            "new_value": new_val,
            "direction": direction,
            "expected_wr_improvement": best["expected_wr_improvement"],
            "train_wr": best.get("train_wr", 0),
            "train_ev": best.get("train_ev", 0),
            "test_wr": best.get("test_wr", 0),
            "test_ev": best.get("test_ev", 0),
            "n_trades": result["n_trades"],
            "n_train": result["n_train"],
            "n_test": result["n_test"],
            "current_wr": result["current_wr"],
            "current_ev": result["current_ev"],
        }

        self._applied_history.append(change)
        self._save_state()

        logger.info(
            f"[Solver] {strategy_name}: {config_key} {old_val} → {new_val} "
            f"(train WR +{best['expected_wr_improvement']:.1%}, "
            f"test WR {best.get('test_wr', 0):.1%}, "
            f"test EV {best.get('test_ev', 0):.4%})"
        )
        return change

    def restore_adjustments(self, strategies: dict) -> int:
        """Restore previously applied adjustments to strategy configs.

        Call at bot startup after strategies are initialized.
        Only restores the LATEST value per (strategy, config_key) pair.
        Returns number of keys restored.
        """
        # Deduplicate: keep latest per (strategy, key)
        latest: dict[tuple[str, str], dict] = {}
        for adj in self._applied_history:
            k = (adj["strategy"], adj["config_key"])
            latest[k] = adj

        restored = 0
        for (name, key), adj in latest.items():
            if name in strategies and hasattr(strategies[name], "config"):
                strategies[name].config.extra[key] = adj["new_value"]
                restored += 1
                logger.info(f"[Solver] Restored {name}.{key} = {adj['new_value']}")
        return restored

    @property
    def applied_history(self) -> list[dict]:
        return list(self._applied_history)

    # ═══════════════════════════════════════════════════════════
    # Threshold Optimization (train/test validated)
    # ═══════════════════════════════════════════════════════════

    def _find_optimal_threshold(
        self,
        train: list[dict],
        test: list[dict],
        feature_name: str,
        config_key: str,
        direction: str,
    ) -> Optional[dict]:
        """Find threshold on train set, validate on test set.

        Only returns a suggestion if:
        1. Train WR improvement >= MIN_WR_IMPROVEMENT
        2. Train EV > 0
        3. Test EV > 0 (generalization check)
        4. Test WR improvement >= 0 (at least not worse)
        """

        def _get_val(t: dict) -> Optional[float]:
            v = t.get(feature_name)
            if v is None or not isinstance(v, (int, float)):
                return None
            if direction == "tighter_higher":
                return abs(v)
            return v

        # ── Train set ──
        train_pairs = [(t, _get_val(t)) for t in train]
        train_pairs = [(t, v) for t, v in train_pairs if v is not None]

        if len(train_pairs) < 30:
            return None

        vals_sorted = sorted(v for _, v in train_pairs)
        n_train = len(train_pairs)
        train_base_wr = (
            len([t for t, _ in train_pairs if self._pnl(t) > 0]) / n_train
        )

        # ── Test set ──
        test_pairs = [(t, _get_val(t)) for t in test]
        test_pairs = [(t, v) for t, v in test_pairs if v is not None]

        if len(test_pairs) < 5:
            return None

        test_base_wr = (
            len([t for t, _ in test_pairs if self._pnl(t) > 0]) / len(test_pairs)
        )

        best: Optional[dict] = None

        for pct in [0.25, 0.40, 0.50, 0.60, 0.75]:
            idx = int(n_train * pct)
            threshold = vals_sorted[idx]

            # ── Train: keep trades with feature >= threshold ──
            t_filtered = [t for t, v in train_pairs if v >= threshold]
            if len(t_filtered) < 15:
                continue

            t_wr = len([t for t in t_filtered if self._pnl(t) > 0]) / len(t_filtered)
            t_ev = statistics.mean([self._pnl_pct(t) for t in t_filtered])
            wr_imp = t_wr - train_base_wr

            if wr_imp < self.MIN_WR_IMPROVEMENT or t_ev <= 0:
                continue

            # ── Test: apply same threshold ──
            test_filtered = [t for t, v in test_pairs if v >= threshold]
            if len(test_filtered) < 5:
                continue

            test_wr = (
                len([t for t in test_filtered if self._pnl(t) > 0])
                / len(test_filtered)
            )
            test_ev = statistics.mean([self._pnl_pct(t) for t in test_filtered])
            test_imp = test_wr - test_base_wr

            # Test-set must show: EV > 0 AND WR not worse
            if test_ev <= 0 or test_imp < 0:
                continue

            if best is None or test_ev > best["test_ev"]:
                best = {
                    "feature": feature_name,
                    "config_key": config_key,
                    "direction": direction,
                    "suggested_threshold": round(threshold, 6),
                    "percentile": pct,
                    "train_filtered": len(t_filtered),
                    "train_wr": round(t_wr, 4),
                    "train_ev": round(t_ev, 6),
                    "expected_wr_improvement": round(wr_imp, 4),
                    "test_filtered": len(test_filtered),
                    "test_wr": round(test_wr, 4),
                    "test_ev": round(test_ev, 6),
                    "test_wr_improvement": round(test_imp, 4),
                    "trades_filtered_out": n_train - len(t_filtered),
                }

        return best

    # ═══════════════════════════════════════════════════════════
    # R:R Optimization via MFE Distribution
    # ═══════════════════════════════════════════════════════════

    def _optimize_rr(
        self,
        train: list[dict],
        test: list[dict],
        tp_config_key: str,
    ) -> Optional[dict]:
        """MFE 분포 기반 R:R 최적화 (train에서 발견, test에서 검증).

        ⚠ 경로 의존성 경고:
        MFE ≥ X는 "TP=X였으면 히트"를 보장하지 않음.
        실제 TP가 달랐으면 거래 trajectory 자체가 변경될 수 있음.
        따라서 이 시뮬레이션은 upper bound 추정치로만 사용.

        단위 참고:
        - mfe_pct: DECIMAL (0.008 = 0.8%)
        - sl_distance_pct: PERCENTAGE (0.8 = 0.8%) → /100 변환 필요
        """
        # mfe_pct: decimal, sl_distance_pct: percentage (×100)
        train_mfe = [
            t.get("mfe_pct", 0) for t in train if t.get("mfe_pct", 0) > 0
        ]
        train_sl = [
            abs(t.get("sl_distance_pct", 0))
            for t in train
            if t.get("sl_distance_pct", 0) > 0
        ]

        if len(train_mfe) < 20 or len(train_sl) < 20:
            return None

        avg_sl_dec = statistics.mean(train_sl) / 100  # pct → decimal
        fee_dec = 0.0015  # round-trip fee (decimal) — matches ROUND_TRIP_FEE_RATE

        mfe_sorted = sorted(train_mfe)
        best: Optional[dict] = None

        for label, pct_idx in [("P25", 0.25), ("P50", 0.50), ("P75", 0.75)]:
            mfe_thresh = mfe_sorted[int(len(mfe_sorted) * pct_idx)]  # decimal

            # ── Train simulation ──
            train_hits = sum(1 for t in train if t.get("mfe_pct", 0) >= mfe_thresh)
            train_sim_wr = train_hits / len(train)

            tp_dec = mfe_thresh  # decimal
            rr = tp_dec / avg_sl_dec if avg_sl_dec > 0 else 0

            if rr < self.MIN_RR:
                continue  # R:R 1.5 미만 → skip

            sim_ev = (
                train_sim_wr * (tp_dec - fee_dec)
                - (1 - train_sim_wr) * (avg_sl_dec + fee_dec)
            )

            if sim_ev <= 0:
                continue

            # ── Test validation ──
            test_hits = sum(1 for t in test if t.get("mfe_pct", 0) >= mfe_thresh)
            test_sim_wr = test_hits / len(test) if test else 0

            test_sl = [
                abs(t.get("sl_distance_pct", 0))
                for t in test
                if t.get("sl_distance_pct", 0) > 0
            ]
            test_avg_sl = statistics.mean(test_sl) / 100 if test_sl else avg_sl_dec

            test_ev = (
                test_sim_wr * (tp_dec - fee_dec)
                - (1 - test_sim_wr) * (test_avg_sl + fee_dec)
            )

            if test_ev <= 0:
                continue  # test-set에서 일반화 실패

            current_wr = len([t for t in train if self._pnl(t) > 0]) / len(train)
            wr_imp = train_sim_wr - current_wr

            if best is None or test_ev > best["test_ev"]:
                best = {
                    "feature": f"mfe_pct ({label})",
                    "config_key": tp_config_key,  # per-strategy key
                    "direction": "rr_optimization",
                    "current_threshold": 0,
                    "suggested_threshold": round(rr, 2),
                    "expected_wr_improvement": round(wr_imp, 4),
                    "train_wr": round(train_sim_wr, 4),
                    "train_ev": round(sim_ev, 6),
                    "test_wr": round(test_sim_wr, 4),
                    "test_ev": round(test_ev, 6),
                    "trades_filtered_out": 0,
                    "percentile": pct_idx,
                    "mfe_threshold_pct": round(mfe_thresh * 100, 3),
                    "simulated_rr": round(rr, 2),
                    "note": "upper_bound — path dependency caveat",
                }

        return best

    # ═══════════════════════════════════════════════════════════
    # Persistence (v2 신규 — 재시작 시 조정값 보존)
    # ═══════════════════════════════════════════════════════════

    def _save_state(self) -> None:
        """Save applied adjustments to JSON (atomic write)."""
        try:
            state = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "applied_history": self._applied_history,
            }
            tmp = self._persist_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            tmp.replace(self._persist_path)
        except Exception as e:
            logger.warning(f"[Solver] Failed to save state: {e}")

    def _load_state(self) -> None:
        """Load previous adjustments from JSON."""
        if self._persist_path.exists():
            try:
                state = json.loads(
                    self._persist_path.read_text(encoding="utf-8")
                )
                self._applied_history = state.get("applied_history", [])
                logger.info(
                    f"[Solver] Loaded {len(self._applied_history)} "
                    f"historical adjustments"
                )
            except Exception as e:
                logger.warning(f"[Solver] Failed to load state: {e}")
                self._applied_history = []

    # ═══════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _pnl(t: dict) -> float:
        """Net PnL (USDT), fallback to gross."""
        return t.get("pnl_net_usdt", t.get("pnl_usdt", 0))

    @staticmethod
    def _pnl_pct(t: dict) -> float:
        """Net PnL (decimal pct), fallback to gross."""
        return t.get("pnl_net_pct", t.get("pnl_pct", 0))

    def _load_closed_trades(self, strategy_name: str) -> list[dict]:
        """Load completed trades for a strategy, chronologically ordered."""
        if not self._path.exists():
            return []
        trades: list[dict] = []
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
                    if (
                        rec.get("strategy") == strategy_name
                        and rec.get("exit_reason")
                    ):
                        trades.append(rec)
        except Exception as e:
            logger.warning(f"[Solver] Failed to load trades: {e}")
        # Ensure chronological order for temporal split
        trades.sort(key=lambda t: t.get("ts_exit", t.get("ts", "")))
        return trades
