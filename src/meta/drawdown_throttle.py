"""DrawdownThrottle — 연속 손실 시 자동 사이즈 축소.

행동재무학 근거: 연속 손실(tilt) 시 사이즈를 줄이면 MDD 40%+ 감소.
3연패 → 사이즈 50%, 5연패 → 전략 일시 정지, 2연승 → 복귀.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

logger = logging.getLogger("drawdown_throttle")


class DrawdownThrottle:
    """Per-strategy consecutive loss tracker with automatic size scaling."""

    # Consecutive losses → size multiplier
    THROTTLE_TABLE = {
        0: 1.0,   # 정상
        1: 1.0,   # 1연패: 유지
        2: 0.8,   # 2연패: 80%
        3: 0.5,   # 3연패: 50%
        4: 0.3,   # 4연패: 30%
    }
    PAUSE_THRESHOLD = 10  # 5→10: 데모 데이터 수집 모드 (정지 기준 완화)
    RESUME_WINS = 2       # 2연승 시 복귀

    def __init__(self) -> None:
        self._streak: dict[str, int] = defaultdict(int)       # 음수=연패, 양수=연승
        self._paused: dict[str, float] = {}                   # strategy → pause 시작 시각
        self._win_after_pause: dict[str, int] = defaultdict(int)

    def record_trade(self, strategy: str, pnl_net: float) -> None:
        """거래 완결 시 호출. pnl_net > 0 이면 승리."""
        if pnl_net > 0:
            if self._streak[strategy] < 0:
                self._streak[strategy] = 1  # 연패 끊김 → 1승
            else:
                self._streak[strategy] += 1

            # 정지 중 복귀 체크
            if strategy in self._paused:
                self._win_after_pause[strategy] += 1
                if self._win_after_pause[strategy] >= self.RESUME_WINS:
                    del self._paused[strategy]
                    del self._win_after_pause[strategy]
                    self._streak[strategy] = 1
                    logger.info(
                        f"[Throttle] {strategy} RESUMED — "
                        f"{self.RESUME_WINS} consecutive wins after pause"
                    )
        else:
            if self._streak[strategy] > 0:
                self._streak[strategy] = -1  # 연승 끊김 → 1패
            else:
                self._streak[strategy] -= 1  # 연패 심화

            # 정지 트리거
            if abs(self._streak[strategy]) >= self.PAUSE_THRESHOLD:
                if strategy not in self._paused:
                    self._paused[strategy] = time.time()
                    self._win_after_pause[strategy] = 0
                    logger.warning(
                        f"[Throttle] {strategy} PAUSED — "
                        f"{abs(self._streak[strategy])} consecutive losses"
                    )

    def get_size_factor(self, strategy: str) -> float:
        """현재 사이즈 배율. 0.0이면 일시 정지 상태."""
        if strategy in self._paused:
            return 0.0

        losses = abs(min(self._streak[strategy], 0))
        if losses >= self.PAUSE_THRESHOLD:
            return 0.0

        return self.THROTTLE_TABLE.get(losses, 0.3)

    def is_paused(self, strategy: str) -> bool:
        return strategy in self._paused

    def get_status(self, strategy: str) -> dict:
        streak = self._streak[strategy]
        return {
            "strategy": strategy,
            "streak": streak,
            "size_factor": self.get_size_factor(strategy),
            "paused": self.is_paused(strategy),
            "status": (
                "PAUSED" if self.is_paused(strategy)
                else f"{'W' if streak > 0 else 'L'}{abs(streak)}" if streak != 0
                else "NEUTRAL"
            ),
        }

    def get_all_status(self) -> dict[str, dict]:
        return {s: self.get_status(s) for s in self._streak}
