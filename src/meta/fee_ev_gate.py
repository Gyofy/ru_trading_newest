"""FeeEVGate — 수수료 포함 기대값(EV) 사전 검증.

진입 전에 expected_pnl = WR × avg_win - (1-WR) × avg_loss - fee 를 계산.
EV <= 0 이면 진입을 거부하여 구조적 음의 EV 거래를 원천 차단.

v8.5 교훈: 169건 Gross +$32 vs Fee $1,360 (4,225%) — 수수료가 모든 문제의 근원.
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

logger = logging.getLogger("fee_ev_gate")


class FeeEVGate:
    """Pre-trade EV check: reject trades with negative expected value."""

    MIN_SAMPLE = 15  # 최소 거래 수 (미달 시 통과 허용)

    def __init__(
        self,
        trade_context_path: Path,
        round_trip_fee_rate: float = 0.0020,  # v9.1: 양방향 taker 0.20%
    ):
        self._path = trade_context_path
        self._fee_rate = round_trip_fee_rate
        self._cache: dict[str, dict] = {}  # strategy → {wr, avg_win, avg_loss, n}
        self._last_refresh = 0.0

    def refresh(self) -> None:
        """trade_context.jsonl에서 전략별 WR/avg_win/avg_loss 재계산."""
        if not self._path.exists():
            return

        trades_by_strategy: dict[str, list[float]] = {}
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
                    if not rec.get("exit_reason"):
                        continue
                    strategy = rec.get("strategy", "")
                    pnl_pct = rec.get("pnl_net_pct", rec.get("pnl_pct", 0))
                    if strategy:
                        trades_by_strategy.setdefault(strategy, []).append(pnl_pct)
        except Exception as e:
            logger.warning(f"[FeeEV] Failed to load trades: {e}")
            return

        for strategy, pnls in trades_by_strategy.items():
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            n = len(pnls)
            wr = len(wins) / n if n > 0 else 0.5
            avg_win = statistics.mean(wins) if wins else 0.0
            avg_loss = abs(statistics.mean(losses)) if losses else 0.0

            self._cache[strategy] = {
                "wr": wr,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "n": n,
            }

    def check(
        self,
        strategy: str,
        sl_dist_pct: float,
        tp_dist_pct: float,
        notional: float,
    ) -> tuple[bool, float, str]:
        """진입 전 EV 검증.

        Args:
            strategy: 전략 이름
            sl_dist_pct: SL 거리 (decimal, e.g. 0.015 = 1.5%)
            tp_dist_pct: TP 거리 (decimal)
            notional: 주문 notional (USDT)

        Returns:
            (allowed, expected_ev, reason)
        """
        stats = self._cache.get(strategy)

        # 데이터 부족 → 통과 허용 (데이터 수집 우선)
        if stats is None or stats["n"] < self.MIN_SAMPLE:
            return True, 0.0, "insufficient_data"

        wr = stats["wr"]
        fee_pct = self._fee_rate  # round-trip fee as decimal

        # EV = WR × TP - (1-WR) × SL - fee
        ev = wr * tp_dist_pct - (1 - wr) * sl_dist_pct - fee_pct

        if ev <= 0:
            reason = (
                f"negative_ev: WR={wr:.1%} × TP={tp_dist_pct:.3%} "
                f"- {1-wr:.1%} × SL={sl_dist_pct:.3%} "
                f"- fee={fee_pct:.3%} = EV={ev:.4%}"
            )
            logger.info(f"[FeeEV] {strategy} REJECTED — {reason}")
            return False, ev, reason

        return True, ev, "positive_ev"

    def get_strategy_stats(self, strategy: str) -> dict | None:
        return self._cache.get(strategy)
