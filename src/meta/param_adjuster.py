"""ParamAdjuster — DrawdownThrottle 발동 시 전략 파라미터 자동 미세조정.
EVGridSearch  — FeeEVGate 차단 시 EV > 0 달성 파라미터 그리드 서치.

실패 패턴 분석 → config.extra를 in-memory로 수정 (재시작 없이 즉시 반영).
YAML 파일은 건드리지 않음 — 다음 재시작 시 원래 값 복원 가능하도록.

분석 지표:
  - EARLY_EXIT_NO_MFE 비율: 시그널 품질 문제 → 진입 기준 강화
  - SL_HIT 비율 + MAE: SL 거리 문제 → SL 확대
  - MFE/MAE 비율: 방향성 문제 → 전략별 핵심 파라미터 조정
  - WR: 전반적 성과 → 복합 조정

EVGridSearch 탐색 공간:
  sl_atr_mult × (tp_rr | tp_atr_mult) 전 조합
  EV = WR × tp_pct - (1-WR) × sl_pct - fee > 0 조건 충족 조합 중 최소 변화량 선택
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger("param_adjuster")

# ── 조정 가능한 파라미터별 (최솟값, 최댓값, 단계 배율) ──────────────────
# 단계 배율: 조건 충족 시 현재값에 곱함 (>1 = 상향, <1 = 하향)
_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    # 시그널 강도 기준 (높을수록 진입 어려움)
    "sigma_mult":            (1.0, 3.5),
    "cvd_quantile":          (0.85, 0.99),
    "cvd_sigma_mult":        (1.0, 3.0),
    "oi_sigma_threshold":    (0.3, 2.5),
    "taker_spike_mult":      (0.5, 2.5),
    "volume_mult":           (1.5, 5.0),
    "min_body_pct":          (0.0003, 0.006),
    "min_price_move_pct":    (0.0005, 0.006),
    "oi_change_threshold":   (0.001, 0.010),
    # 거래량 감소 기준 (낮을수록 더 강한 감소 요구)
    "vol_decay_threshold":   (0.50, 0.90),
    # 손절/익절 (sl 넓힐수록 안전, tp_rr 높일수록 손익비↑)
    "sl_atr_mult":           (2.0, 8.0),
    "tp_rr":                 (1.5, 6.0),
    "tp_atr_mult":           (2.0, 12.0),
}

_MIN_SAMPLE = 5  # 조정에 필요한 최소 완결 거래 수

# ── EVGridSearch 탐색 그리드 ────────────────────────────────────────────────
_SL_GRID: list[float]     = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]
_TP_RR_GRID: list[float]  = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
_TP_ATR_GRID: list[float] = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]


class EVGridSearch:
    """FeeEVGate 차단 시 EV > 0 달성 파라미터 그리드 서치.

    EV = WR × tp_pct - (1-WR) × actual_sl_pct - fee

    actual_sl_pct = max(sl_atr_mult × atr_pct, sl_floor_pct)  ← base.py SL floor 반영
    tp_pct (tp_rr 전략)  = sl_atr_mult × atr_pct × tp_rr     ← floor 미적용 (TP는 ATR 기반)
    tp_pct (tp_atr 전략) = tp_atr_mult × atr_pct

    로그 태그:
      [EV_GRID]      — 서치 요약 + 각 조합 결과 (WARNING/INFO)
      [PARAM_CHANGE] — 실제 적용된 파라미터 변경 (WARNING, grep 추적용)
    """

    COOLDOWN_SEC = 1800  # 30분: 동일 전략 중복 서치 방지

    def __init__(self) -> None:
        self._last_search: dict[str, float] = {}
        self.apply_count: dict[str, int] = {}  # strategy → EV 그리드 서치 적용 횟수

    def search_and_apply(
        self,
        strategy_obj,
        fee_ev_gate,
        atr_pct: float,
        sl_floor_pct: float = 0.015,
    ) -> tuple[dict, str]:
        """EV > 0 달성 파라미터 서치 + in-memory 적용.

        Args:
            strategy_obj: StrategyBase 인스턴스 (config.extra 직접 수정)
            fee_ev_gate: FeeEVGate 인스턴스 (WR·fee 통계 접근)
            atr_pct: 현재 ATR / price (소수점, e.g. 0.012 = 1.2%)
            sl_floor_pct: base.py SL_FLOOR_PCT (default 0.015 = 1.5% 데이터수집모드)

        Returns:
            (applied_params, summary_text)
            applied_params: {"param": (old_val, new_val), ...}
        """
        strategy = strategy_obj.name
        now = time.time()

        # ── 쿨다운 체크 ──
        remaining = self.COOLDOWN_SEC - (now - self._last_search.get(strategy, 0.0))
        if remaining > 0:
            return {}, f"쿨다운 {int(remaining)}초 남음 — 서치 생략"
        self._last_search[strategy] = now

        # ── WR / fee 통계 ──
        stats = fee_ev_gate.get_strategy_stats(strategy)
        if not stats:
            return {}, "FeeEV 통계 없음 — 서치 생략"

        wr = stats["wr"]
        fee = fee_ev_gate._fee_rate
        extra = strategy_obj.config.extra

        # 현재 파라미터 읽기
        cur_sl     = float(extra.get("sl_atr_mult", 4.0))
        has_tp_rr  = "tp_rr" in extra
        has_tp_atr = "tp_atr_mult" in extra
        cur_tp_rr  = float(extra.get("tp_rr", 3.0))
        cur_tp_atr = float(extra.get("tp_atr_mult", 6.0))

        sl_lo,  sl_hi  = _PARAM_BOUNDS.get("sl_atr_mult",  (2.0,  8.0))
        rr_lo,  rr_hi  = _PARAM_BOUNDS.get("tp_rr",        (1.5,  6.0))
        atr_lo, atr_hi = _PARAM_BOUNDS.get("tp_atr_mult",  (2.0, 12.0))

        # ── 그리드 탐색 ──
        results: list[dict] = []

        for sl_mult in _SL_GRID:
            if not (sl_lo <= sl_mult <= sl_hi):
                continue
            raw_sl_pct   = sl_mult * atr_pct
            # base.py와 동일한 SL floor 적용 (실제 EV 계산에 사용)
            actual_sl_pct = max(raw_sl_pct, sl_floor_pct)
            # TP는 ATR 기반으로 floor 미적용 (base.py 동작과 동일)
            raw_tp_base = raw_sl_pct  # tp_rr 기반 전략용 베이스

            if has_tp_rr:
                for tp_rr in _TP_RR_GRID:
                    if not (rr_lo <= tp_rr <= rr_hi):
                        continue
                    tp_pct = raw_tp_base * tp_rr
                    ev = wr * tp_pct - (1 - wr) * actual_sl_pct - fee
                    results.append({
                        "sl_atr_mult": sl_mult,
                        "tp_rr": tp_rr,
                        "tp_atr_mult": None,
                        "sl_pct": actual_sl_pct,
                        "tp_pct": tp_pct,
                        "ev": ev,
                        "sl_floored": actual_sl_pct > raw_sl_pct,
                    })

            if has_tp_atr:
                for tp_atr in _TP_ATR_GRID:
                    if not (atr_lo <= tp_atr <= atr_hi):
                        continue
                    tp_pct = tp_atr * atr_pct
                    ev = wr * tp_pct - (1 - wr) * actual_sl_pct - fee
                    results.append({
                        "sl_atr_mult": sl_mult,
                        "tp_rr": None,
                        "tp_atr_mult": tp_atr,
                        "sl_pct": actual_sl_pct,
                        "tp_pct": tp_pct,
                        "ev": ev,
                        "sl_floored": actual_sl_pct > raw_sl_pct,
                    })

        n_total    = len(results)
        n_positive = sum(1 for r in results if r["ev"] > 0)

        # ── 전체 결과 로그 (summary WARNING, 개별 INFO) ──
        logger.warning(
            f"[EV_GRID] strategy={strategy} wr={wr:.1%} fee={fee:.3%} "
            f"atr_pct={atr_pct:.4%} | combos={n_total} positive={n_positive}/{n_total}"
        )
        for r in results:
            tp_label = (
                f"tp_rr={r['tp_rr']:.1f}"
                if r["tp_rr"] is not None
                else f"tp_atr={r['tp_atr_mult']:.1f}"
            )
            mark = "✓" if r["ev"] > 0 else "✗"
            logger.info(
                f"[EV_GRID] {strategy} sl_mult={r['sl_atr_mult']:.1f} {tp_label}"
                f" sl={r['sl_pct']:.4%} tp={r['tp_pct']:.4%}"
                f" EV={r['ev']:+.4%} {mark}"
            )

        # ── 해 없음 ──
        positive = [r for r in results if r["ev"] > 0]
        if not positive:
            logger.warning(
                f"[EV_GRID] NO_SOLUTION strategy={strategy} wr={wr:.1%} "
                f"— _PARAM_BOUNDS 내 EV > 0 달성 불가 (combos={n_total})"
            )
            return {}, f"EV > 0 달성 불가 (WR={wr:.1%}, combos={n_total})"

        # ── 최적 선택: 현재값과 최소 거리 (tie-break: 높은 EV) ──
        def _score(r: dict) -> tuple:
            sl_d = abs(r["sl_atr_mult"] - cur_sl)
            tp_d = (
                abs(r["tp_rr"] - cur_tp_rr)
                if r["tp_rr"] is not None
                else abs(r["tp_atr_mult"] - cur_tp_atr)
            )
            return (sl_d + tp_d, -r["ev"])

        best = min(positive, key=_score)

        # ── 적용 ──
        applied: dict[str, tuple] = {}

        if abs(best["sl_atr_mult"] - cur_sl) > 0.05:
            strategy_obj.config.extra["sl_atr_mult"] = best["sl_atr_mult"]
            applied["sl_atr_mult"] = (cur_sl, best["sl_atr_mult"])
            logger.warning(
                f"[PARAM_CHANGE] strategy={strategy} param=sl_atr_mult "
                f"old={cur_sl} new={best['sl_atr_mult']} "
                f"reason=ev_grid_search ev={best['ev']:+.4%} "
                f"wr={wr:.2%} n={stats['n']}"
            )

        if best["tp_rr"] is not None and abs(best["tp_rr"] - cur_tp_rr) > 0.05:
            strategy_obj.config.extra["tp_rr"] = best["tp_rr"]
            applied["tp_rr"] = (cur_tp_rr, best["tp_rr"])
            logger.warning(
                f"[PARAM_CHANGE] strategy={strategy} param=tp_rr "
                f"old={cur_tp_rr} new={best['tp_rr']} "
                f"reason=ev_grid_search ev={best['ev']:+.4%} "
                f"wr={wr:.2%} n={stats['n']}"
            )

        if best["tp_atr_mult"] is not None and abs(best["tp_atr_mult"] - cur_tp_atr) > 0.05:
            strategy_obj.config.extra["tp_atr_mult"] = best["tp_atr_mult"]
            applied["tp_atr_mult"] = (cur_tp_atr, best["tp_atr_mult"])
            logger.warning(
                f"[PARAM_CHANGE] strategy={strategy} param=tp_atr_mult "
                f"old={cur_tp_atr} new={best['tp_atr_mult']} "
                f"reason=ev_grid_search ev={best['ev']:+.4%} "
                f"wr={wr:.2%} n={stats['n']}"
            )

        if applied:
            self.apply_count[strategy] = self.apply_count.get(strategy, 0) + 1
            changes_str = " ".join(f"{k}:{v[0]}→{v[1]}" for k, v in applied.items())
            logger.warning(
                f"[EV_GRID] APPLIED strategy={strategy} {changes_str} "
                f"EV={best['ev']:+.4%} (누적 조정 {self.apply_count[strategy]}회)"
            )
        else:
            logger.info(
                f"[EV_GRID] NO_CHANGE strategy={strategy} "
                f"현재 파라미터가 이미 최적 (best EV={best['ev']:+.4%})"
            )

        tp_desc = (
            f"tp_rr={best['tp_rr']}"
            if best["tp_rr"] is not None
            else f"tp_atr_mult={best['tp_atr_mult']}"
        )
        summary = (
            f"[EV GridSearch] {strategy}: WR={wr:.1%} | "
            f"combos={n_total} positive={n_positive}\n"
            f"Best: sl_atr_mult={best['sl_atr_mult']} {tp_desc} "
            f"EV={best['ev']:+.4%}"
        )
        return applied, summary


class ParamAdjuster:
    """전략 실패 패턴 분석 → 파라미터 in-memory 미세조정."""

    def __init__(self, trade_context_path: Path):
        self._path = trade_context_path
        self.apply_count: dict[str, int] = {}  # strategy → DrawdownThrottle 파라미터 조정 횟수

    # ── 공개 인터페이스 ─────────────────────────────────────────────────────

    def adjust(
        self,
        strategy_obj,
        n_recent: int = 15,
    ) -> tuple[dict, str]:
        """DrawdownThrottle PAUSE 시 호출.

        Args:
            strategy_obj: StrategyBase 인스턴스 (config.extra를 직접 수정)
            n_recent: 분석할 최근 거래 수

        Returns:
            (changed_params, summary_text)
            changed_params: {"param": (old_val, new_val), ...}
        """
        strategy_name = strategy_obj.name
        analysis = self._analyze(strategy_name, n_recent)

        if analysis["n"] < _MIN_SAMPLE:
            logger.info(
                f"[ParamAdjuster] {strategy_name}: "
                f"데이터 {analysis['n']}건 < 최소 {_MIN_SAMPLE}건 — 조정 생략"
            )
            return {}, f"데이터 부족 ({analysis['n']}건) — 조정 생략"

        changes = self._compute_changes(strategy_name, analysis, strategy_obj.config.extra)

        if not changes:
            return {}, "변경 없음 — 현재 파라미터 유지"

        # in-memory 적용 + 트러블슈팅용 구조화 로그
        applied = {}
        for param, (old_val, new_val) in changes.items():
            strategy_obj.config.extra[param] = new_val
            applied[param] = (old_val, new_val)
            # [PARAM_CHANGE] 태그: grep으로 즉시 추적 가능
            logger.warning(
                f"[PARAM_CHANGE] strategy={strategy_name} "
                f"param={param} old={old_val} new={new_val} "
                f"reason=drawdown_throttle_pause "
                f"wr={analysis.get('wr', 0):.2%} "
                f"n={analysis.get('n', 0)}"
            )

        self.apply_count[strategy_name] = self.apply_count.get(strategy_name, 0) + 1
        summary = self._format_summary(strategy_name, analysis, applied)
        return applied, summary

    # ── 분석 ────────────────────────────────────────────────────────────────

    def _analyze(self, strategy: str, n_recent: int) -> dict:
        """trade_context.jsonl에서 최근 N건 완결 거래 분석."""
        records = self._load_closed(strategy, n_recent)
        n = len(records)
        if n == 0:
            return {"n": 0}

        wins = [r for r in records if r.get("pnl_net_pct", 0) > 0]
        losses = [r for r in records if r.get("pnl_net_pct", 0) <= 0]

        exit_counts: dict[str, int] = defaultdict(int)
        for r in records:
            exit_counts[r.get("exit_reason", "UNKNOWN")] += 1

        mfe_vals = [r.get("mfe_pct", 0.0) for r in records]
        mae_vals = [abs(r.get("mae_pct", 0.0)) for r in records]
        avg_mfe = sum(mfe_vals) / n
        avg_mae = sum(mae_vals) / n

        # EARLY_EXIT_NO_MFE: MFE=0% 인 건 (방향 틀림)
        zero_mfe_exits = sum(
            1 for r in records
            if "EARLY" in r.get("exit_reason", "") and r.get("mfe_pct", 0) <= 0.0001
        )

        return {
            "n": n,
            "wr": len(wins) / n,
            "avg_mfe": avg_mfe,
            "avg_mae": avg_mae,
            "mfe_mae_ratio": avg_mfe / avg_mae if avg_mae > 0 else 999.0,
            "exit_counts": dict(exit_counts),
            "early_exit_pct": exit_counts.get("EARLY_EXIT_NO_MFE", 0) / n,
            "sl_hit_pct": exit_counts.get("SL_HIT", 0) / n,
            "tp_hit_pct": exit_counts.get("TP_HIT", 0) / n,
            "zero_mfe_pct": zero_mfe_exits / n,
        }

    def _load_closed(self, strategy: str, n: int) -> list[dict]:
        """strategy의 완결 거래 최근 N건 반환."""
        if not self._path.exists():
            return []
        records = []
        try:
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("strategy") == strategy and r.get("exit_reason"):
                        records.append(r)
        except Exception as e:
            logger.warning(f"[ParamAdjuster] 파일 읽기 실패: {e}")
        return records[-n:]

    # ── 조정 계산 ───────────────────────────────────────────────────────────

    def _compute_changes(
        self,
        strategy: str,
        analysis: dict,
        current_extra: dict,
    ) -> dict[str, tuple[float, float]]:
        """실패 패턴 → 파라미터 변경 dict 반환.

        Returns: {"param": (old_val, new_val)}
        """
        changes: dict[str, tuple[float, float]] = {}
        wr = analysis["wr"]
        early_pct = analysis["early_exit_pct"]
        zero_mfe_pct = analysis["zero_mfe_pct"]
        sl_pct = analysis["sl_hit_pct"]
        mfe_mae = analysis["mfe_mae_ratio"]

        def _adjust(param: str, factor: float) -> None:
            """현재 extra에 있는 param을 factor 배만큼 조정, bounds 적용."""
            if param not in current_extra:
                return
            old = float(current_extra[param])
            lo, hi = _PARAM_BOUNDS.get(param, (0.0, 1e9))
            new = max(lo, min(hi, round(old * factor, 6)))
            if abs(new - old) / (abs(old) + 1e-9) > 0.01:  # 1% 이상 변화 시만 기록
                changes[param] = (old, new)

        # ── 패턴 1: 시그널 품질 나쁨 (MFE=0% 조기 청산 많음) ──
        # → 진입 기준 강화 (시그널 강도 파라미터 상향)
        if zero_mfe_pct >= 0.6:
            factor = 1.25 if zero_mfe_pct >= 0.8 else 1.15
            for p in ("sigma_mult", "cvd_sigma_mult", "cvd_quantile",
                      "volume_mult", "min_body_pct", "min_price_move_pct"):
                _adjust(p, factor)
            # vol_decay는 하향 (더 강한 거래량 감소 요구)
            _adjust("vol_decay_threshold", 1 / factor)
            _adjust("oi_sigma_threshold", factor)
            _adjust("taker_spike_mult", factor)

        # ── 패턴 2: SL 자주 맞음 + MAE가 MFE 압도 ──
        # → SL 확대 (더 넉넉하게)
        if sl_pct >= 0.40 and mfe_mae < 0.5:
            _adjust("sl_atr_mult", 1.20)

        # ── 패턴 3: 전체 WR 매우 낮음 ──
        # → 진입 기준 복합 강화
        if wr < 0.20:
            for p in ("sigma_mult", "cvd_sigma_mult", "volume_mult",
                      "min_body_pct", "oi_sigma_threshold", "taker_spike_mult"):
                _adjust(p, 1.15)
            _adjust("vol_decay_threshold", 0.88)

        # ── 패턴 4: MFE > MAE 이지만 TP를 못 잡음 (조기 청산) ──
        # → tp_rr 하향 (TP를 더 가깝게 → 실현율 상승)
        if analysis["tp_hit_pct"] < 0.15 and analysis["avg_mfe"] > 0.001:
            _adjust("tp_rr", 0.85)
            _adjust("tp_atr_mult", 0.85)

        return changes

    # ── 리포트 ──────────────────────────────────────────────────────────────

    def _format_summary(
        self,
        strategy: str,
        analysis: dict,
        applied: dict,
    ) -> str:
        n = analysis["n"]
        wr = analysis["wr"]
        lines = [
            f"**전략:** `{strategy}` | 분석 {n}건 | WR {wr:.0%}",
            f"조기청산(MFE=0) {analysis['zero_mfe_pct']:.0%} | "
            f"SL {analysis['sl_hit_pct']:.0%} | TP {analysis['tp_hit_pct']:.0%}",
            f"avg MFE {analysis['avg_mfe']:.3%} | avg MAE {analysis['avg_mae']:.3%}",
            "",
            "**파라미터 조정:**",
        ]
        for param, (old, new) in applied.items():
            direction = "▲" if new > old else "▼"
            lines.append(f"  {direction} `{param}`: `{old}` → `{new}`")
        return "\n".join(lines)
