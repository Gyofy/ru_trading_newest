"""EVGuardian — F1 기대수익(EV) 자동 차단 + F5 일일 수수료 예산 관리.

F1 — Expected Value Guard:
  trade_context.jsonl을 읽어 전략별 EV를 계산한다.
  EV = tp_hit_rate × avg_tp_pct − (1 − tp_hit_rate) × avg_sl_pct − avg_fee_drag_pct
  MIN_SAMPLE건 이상 데이터가 있을 때만 판단한다.
  EV < EV_THRESHOLD 이면 해당 전략 suspended → tick() 호출 차단.
  EV가 회복되면 자동으로 resumed.

F5 — Daily Fee Budget Guard:
  전략별 하루 수수료 누적을 추적한다.
  전체(All-strategy) 합계가 fee_budget_daily_usdt 초과 시 신규 진입 차단.
  UTC 자정에 자동 리셋.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ev_guardian")


class EVGuardian:
    """전략 생존 여부를 실시간으로 판단하는 가드.

    run_multi_strategy.py의 _strategy_loop()에서 tick() 호출 전에 check_entry_allowed()를
    호출해 진입 허가 여부를 확인한다.
    거래 종료 시마다 record_fee()를 호출해 일일 수수료를 누적한다.
    """

    # ── F1 상수 ────────────────────────────────────────────
    MIN_SAMPLE: int = 10          # EV 판단에 필요한 최소 완결 거래 수
    EV_THRESHOLD: float = -0.0015  # EV < -0.15% → 전략 차단
    EV_RESUME_THRESHOLD: float = 0.0  # EV ≥ 0% 회복 시 재개

    def __init__(
        self,
        jsonl_path: Path,
        initial_equity: float,
        fee_budget_pct: float = 0.005,   # 일일 수수료 한도: 잔고의 0.5%
        report_path: Optional[Path] = None,
    ):
        self._jsonl_path = jsonl_path
        self._fee_budget_daily = initial_equity * fee_budget_pct
        self._report_path = report_path

        # F1: per-strategy EV 통계 {name: {ev, suspended, n_trades, reason, ...}}
        self._ev_stats: dict[str, dict] = {}
        self._last_eval_ts: float = 0.0

        # F5: 일일 수수료 누적 {strategy: usdt}
        self._fee_day: str = ""              # 현재 UTC 날짜 문자열
        self._daily_fees: dict[str, float] = {}  # strategy → 오늘 수수료 합계
        self._daily_fee_total: float = 0.0   # 전체 오늘 수수료 합계

        self._lock = asyncio.Lock()

    # ── F1: EV 평가 ───────────────────────────────────────

    def evaluate(self) -> dict[str, dict]:
        """trade_context.jsonl을 읽어 전략별 EV를 계산한다. (동기, I/O 가볍다)

        Returns:
            {strategy_name: {ev, suspended, n_trades, tp_hit_rate,
                              avg_tp_pct, avg_sl_pct, avg_fee_drag_pct, reason}}
        """
        if not self._jsonl_path.exists():
            return {}

        # ── 전략별 거래 분류 ──
        buckets: dict[str, list[dict]] = {}
        try:
            with open(self._jsonl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    strat = rec.get("strategy", "")
                    if not strat:
                        continue
                    buckets.setdefault(strat, []).append(rec)
        except Exception as e:
            logger.warning(f"[EVGuardian] jsonl read failed: {e}")
            return {}

        new_stats: dict[str, dict] = {}
        for strat, trades in buckets.items():
            closed = [t for t in trades if t.get("exit_reason", "")]
            n = len(closed)

            # 기본 stat 구조
            stat: dict = {
                "n_trades": n,
                "n_total": len(trades),
                "ev": 0.0,
                "tp_hit_rate": 0.0,
                "avg_tp_pct": 0.0,
                "avg_sl_pct": 0.0,
                "avg_fee_drag_pct": 0.0,
                "avg_net_pnl_pct": 0.0,
                "suspended": False,
                "reason": "insufficient_data" if n < self.MIN_SAMPLE else "ok",
            }

            if n < self.MIN_SAMPLE:
                # 데이터 부족 → 기존 suspended 상태 유지
                prev = self._ev_stats.get(strat, {})
                stat["suspended"] = prev.get("suspended", False)
                new_stats[strat] = stat
                continue

            # TP 히트 / 그 외 분류
            tp_hits = [t for t in closed if t.get("exit_reason") == "TP_HIT"]
            non_tp = [t for t in closed if t.get("exit_reason") != "TP_HIT"]

            tp_rate = len(tp_hits) / n
            avg_tp_pct = (
                sum(abs(t.get("pnl_pct", t.get("tp_distance_pct", 0.0) / 100)) for t in tp_hits) / len(tp_hits)
                if tp_hits else 0.0
            )
            avg_sl_pct = (
                sum(abs(t.get("pnl_pct", t.get("sl_distance_pct", 0.0) / 100)) for t in non_tp) / len(non_tp)
                if non_tp else 0.0
            )
            avg_fee = (
                sum(t.get("fee_drag_pct", 0.0) / 100 for t in closed) / n
            )
            avg_net_pnl = (
                sum(t.get("pnl_net_pct", t.get("pnl_pct", 0.0)) for t in closed) / n
            )

            # EV 공식 (수수료 이중 차감 방지: pnl_net_pct에 fee 이미 포함)
            # pnl_net_pct가 있으면 그대로 사용, 없으면 분해 공식
            # any(t.get(...))가 아닌 "key in t"로 체크: 0.0 값은 falsy라 any()가 오작동함
            if any("pnl_net_pct" in t for t in closed):
                ev = avg_net_pnl
                reason_ev = "net_pnl_avg"
            else:
                ev = tp_rate * avg_tp_pct - (1 - tp_rate) * avg_sl_pct - avg_fee
                reason_ev = "decomposed"

            # Suspended 판단
            was_suspended = self._ev_stats.get(strat, {}).get("suspended", False)
            if was_suspended:
                suspended = ev < self.EV_RESUME_THRESHOLD  # 회복 기준
                reason_str = "suspended_recovering" if not suspended else f"suspended_ev={ev:.4%}"
            else:
                suspended = ev < self.EV_THRESHOLD
                reason_str = f"ev={ev:.4%}_below_threshold" if suspended else "ok"

            stat.update(
                ev=round(ev, 6),
                tp_hit_rate=round(tp_rate, 4),
                avg_tp_pct=round(avg_tp_pct, 6),
                avg_sl_pct=round(avg_sl_pct, 6),
                avg_fee_drag_pct=round(avg_fee, 6),
                avg_net_pnl_pct=round(avg_net_pnl, 6),
                suspended=suspended,
                reason=reason_str,
                ev_method=reason_ev,
            )
            new_stats[strat] = stat

            # 상태 변화 로그
            prev_suspended = self._ev_stats.get(strat, {}).get("suspended", False)
            if suspended and not prev_suspended:
                logger.warning(
                    f"[EVGuardian] ⚠️ {strat} SUSPENDED — "
                    f"EV={ev:.4%} (threshold={self.EV_THRESHOLD:.4%}) "
                    f"n={n} tp_rate={tp_rate:.1%}"
                )
            elif not suspended and prev_suspended:
                logger.info(
                    f"[EVGuardian] ✅ {strat} RESUMED — EV={ev:.4%} recovered"
                )

        self._ev_stats = new_stats
        self._last_eval_ts = datetime.now(timezone.utc).timestamp()
        return new_stats

    def is_strategy_suspended(self, strategy: str) -> tuple[bool, str]:
        """F1: 전략이 현재 차단 상태인지 확인.

        Returns:
            (suspended: bool, reason: str)
        """
        stat = self._ev_stats.get(strategy)
        if stat is None:
            return False, "no_data"
        if stat["suspended"]:
            return True, stat.get("reason", "negative_ev")
        return False, "ok"

    # ── F5: 수수료 예산 ───────────────────────────────────

    def _reset_if_new_day(self) -> None:
        """UTC 자정이 지나면 일일 수수료 카운터 초기화."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._fee_day:
            if self._fee_day:
                logger.info(
                    f"[EVGuardian] Daily fee reset — yesterday total: "
                    f"${self._daily_fee_total:.4f} / budget: ${self._fee_budget_daily:.4f}"
                )
            self._fee_day = today
            self._daily_fees = {}
            self._daily_fee_total = 0.0

    def record_fee(self, strategy: str, fee_usdt: float) -> None:
        """F5: 거래 종료 시 실수수료를 누적한다. thread-safe (GIL 보호됨)."""
        self._reset_if_new_day()
        self._daily_fees[strategy] = self._daily_fees.get(strategy, 0.0) + fee_usdt
        self._daily_fee_total += fee_usdt
        logger.debug(
            f"[EVGuardian] Fee recorded: {strategy} +${fee_usdt:.4f} "
            f"| strategy_today=${self._daily_fees[strategy]:.4f} "
            f"| total_today=${self._daily_fee_total:.4f} "
            f"| budget=${self._fee_budget_daily:.4f}"
        )

    def is_fee_budget_exceeded(self) -> tuple[bool, str]:
        """F5: 일일 수수료 예산 초과 여부 확인.

        Returns:
            (exceeded: bool, detail: str)
        """
        self._reset_if_new_day()
        if self._daily_fee_total >= self._fee_budget_daily:
            return True, (
                f"daily_fee=${self._daily_fee_total:.4f} >= budget=${self._fee_budget_daily:.4f}"
            )
        return False, f"daily_fee=${self._daily_fee_total:.4f} / ${self._fee_budget_daily:.4f}"

    # ── 통합 진입 허가 체크 ───────────────────────────────

    def check_entry_allowed(self, strategy: str) -> tuple[bool, str]:
        """F1 + F5 통합 체크. tick() 호출 전에 반드시 확인한다.

        Returns:
            (allowed: bool, reason: str)
        """
        # F1: EV 차단 체크
        suspended, ev_reason = self.is_strategy_suspended(strategy)
        if suspended:
            return False, f"[F1_EV] {ev_reason}"

        # F5: 수수료 예산 체크
        exceeded, fee_reason = self.is_fee_budget_exceeded()
        if exceeded:
            return False, f"[F5_FEE] {fee_reason}"

        return True, "ok"

    # ── 리포트 ────────────────────────────────────────────

    def get_report(self) -> dict:
        """현재 상태 스냅샷 — Discord/로그/JSON 저장용."""
        self._reset_if_new_day()
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "fee_budget_daily": round(self._fee_budget_daily, 4),
            "daily_fee_total": round(self._daily_fee_total, 4),
            "daily_fee_pct_used": round(
                self._daily_fee_total / self._fee_budget_daily * 100, 2
            ) if self._fee_budget_daily > 0 else 0.0,
            "fee_budget_exceeded": self._daily_fee_total >= self._fee_budget_daily,
            "daily_fees_by_strategy": {k: round(v, 4) for k, v in self._daily_fees.items()},
            "strategies": {
                name: {
                    "ev": stat.get("ev", 0.0),
                    "suspended": stat.get("suspended", False),
                    "n_trades": stat.get("n_trades", 0),
                    "tp_hit_rate": stat.get("tp_hit_rate", 0.0),
                    "avg_tp_pct": stat.get("avg_tp_pct", 0.0),
                    "avg_sl_pct": stat.get("avg_sl_pct", 0.0),
                    "avg_fee_drag_pct": stat.get("avg_fee_drag_pct", 0.0),
                    "reason": stat.get("reason", ""),
                }
                for name, stat in self._ev_stats.items()
            },
        }

    def save_report(self) -> None:
        """리포트를 JSON으로 저장."""
        if not self._report_path:
            return
        try:
            self._report_path.parent.mkdir(parents=True, exist_ok=True)
            report = self.get_report()
            tmp = self._report_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            tmp.replace(self._report_path)
        except Exception as e:
            logger.error(f"[EVGuardian] save_report failed: {e}")

    def format_discord_summary(self) -> str:
        """Discord 브리핑용 요약 텍스트."""
        report = self.get_report()
        lines = []

        # F5 수수료 예산
        fee_bar = "🔴" if report["fee_budget_exceeded"] else (
            "🟡" if report["daily_fee_pct_used"] > 70 else "🟢"
        )
        lines.append(
            f"{fee_bar} **수수료 예산** `${report['daily_fee_total']:.4f}` / "
            f"`${report['fee_budget_daily']:.4f}` ({report['daily_fee_pct_used']:.1f}%)"
        )

        # F1 전략별 EV
        if report["strategies"]:
            lines.append("**전략별 EV 현황:**")
            for name, stat in report["strategies"].items():
                icon = "🚫" if stat["suspended"] else "✅"
                lines.append(
                    f"{icon} `{name}` EV=`{stat['ev']:.4%}` "
                    f"TP율=`{stat['tp_hit_rate']:.1%}` "
                    f"n=`{stat['n_trades']}`"
                    + (" — **SUSPENDED**" if stat["suspended"] else "")
                )
        else:
            lines.append("EV 데이터 없음 (거래 누적 중)")

        return "\n".join(lines)
