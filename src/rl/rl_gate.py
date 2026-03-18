"""RL Gate — decision wrapper with warmup and v4.2 fallback.

Wraps LinUCB bandit with safety policies:
  - Warmup: always ACCEPT_1.00 until enough signals observed
  - Fallback: if accept rate drops below threshold, disable RL
  - Shadow mode: log RL decision but don't apply it
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np

from src.rl.bandit import LinUCB, N_ACTIONS, SIZING_MAP

logger = logging.getLogger("live_bot.rl.gate")

# Default action = ACCEPT_1.00 (index 2)
DEFAULT_ACTION = 2
# Jafar 2024 inspired: limit sizing change to ±25% vs recent week
CHANGE_CAP_RATIO = 0.25


class RLGate:
    """RL-based signal gate with safety fallback and change cap.

    Safety features (glucose RL literature inspired):
    - Warmup: always ACCEPT_1.00 until enough signals
    - Fallback: if accept rate < threshold, revert to v4.2 baseline
    - Change cap: sizing can't deviate >25% from recent week average (Jafar 2024)
    - Shadow mode: log RL decision without applying it
    """

    def __init__(
        self,
        bandit: LinUCB | None = None,
        warmup: int = 200,
        fallback_lookback: int = 100,
        fallback_min_rate: float = 0.25,
        shadow_mode: bool = True,
        model_path: Path | None = None,
    ):
        self.bandit = bandit
        self.warmup = warmup
        self.fallback_lookback = fallback_lookback
        self.fallback_min_rate = fallback_min_rate
        self.shadow_mode = shadow_mode
        self.total_signals = 0
        self._recent = deque(maxlen=fallback_lookback)
        self._weekly_actions: deque = deque(maxlen=12 * 7)  # ~1 week of cycles
        self._fallback_active = False

        # Try loading saved model
        if bandit is None and model_path and Path(model_path).exists():
            try:
                self.bandit = LinUCB.load(model_path)
            except Exception as e:
                logger.warning(f"[RLGate] Failed to load model: {e}")

    def decide(self, state: np.ndarray) -> tuple[int, float]:
        """Decide action for a signal.

        Returns
        -------
        (action, rl_score) : tuple[int, float]
            action: 0=REJECT, 1=ACCEPT_0.75, 2=ACCEPT_1.00, 3=ACCEPT_1.25
            rl_score: exploitation score (for ranking)
        """
        self.total_signals += 1

        # No bandit loaded
        if self.bandit is None:
            self._recent.append(DEFAULT_ACTION)
            return DEFAULT_ACTION, 0.0

        # Warmup phase
        if self.total_signals <= self.warmup:
            self._recent.append(DEFAULT_ACTION)
            return DEFAULT_ACTION, 0.0

        # Check fallback
        if self._should_fallback():
            self._fallback_active = True
            self._recent.append(DEFAULT_ACTION)
            return DEFAULT_ACTION, 0.0

        self._fallback_active = False

        # Normal RL decision
        action, score = self.bandit.score(state)
        action = self._apply_change_cap(action)
        self._recent.append(action)
        self._weekly_actions.append(action)
        return action, score

    def effective_action(self, rl_action: int) -> int:
        """In shadow mode, always return ACCEPT_1.00 regardless of RL decision."""
        if self.shadow_mode:
            return DEFAULT_ACTION
        return rl_action

    def _apply_change_cap(self, action: int) -> int:
        """Limit sizing deviation to ±25% vs recent week average (Jafar 2024)."""
        recent_accepts = [a for a in self._weekly_actions if a > 0]
        if len(recent_accepts) < 5:
            return action
        if action == 0:  # REJECT is always allowed
            return action
        avg_sizing = np.mean([SIZING_MAP[a] for a in recent_accepts])
        new_sizing = SIZING_MAP[action]
        if avg_sizing <= 0:
            return action
        if new_sizing > avg_sizing * (1 + CHANGE_CAP_RATIO):
            # Find closest action within cap
            cap = avg_sizing * (1 + CHANGE_CAP_RATIO)
            for a in [2, 3, 1]:  # prefer 1.0, then 1.25, then 0.75
                if SIZING_MAP[a] <= cap:
                    return a
        if new_sizing < avg_sizing * (1 - CHANGE_CAP_RATIO):
            floor = avg_sizing * (1 - CHANGE_CAP_RATIO)
            for a in [2, 1, 3]:
                if SIZING_MAP[a] >= floor:
                    return a
        return action

    def _should_fallback(self) -> bool:
        """Check if accept rate is too low."""
        if len(self._recent) < 50:
            return False
        accepts = sum(1 for a in self._recent if a > 0)
        rate = accepts / len(self._recent)
        if rate < self.fallback_min_rate:
            if not self._fallback_active:
                logger.warning(
                    f"[RLGate] Accept rate {rate:.0%} < {self.fallback_min_rate:.0%}, "
                    f"falling back to v4.2 baseline"
                )
            return True
        return False

    @property
    def is_active(self) -> bool:
        """Whether RL is making real decisions (not warmup/fallback/shadow)."""
        return (
            self.bandit is not None
            and self.total_signals > self.warmup
            and not self._fallback_active
            and not self.shadow_mode
        )

    @property
    def status(self) -> str:
        if self.bandit is None:
            return "no_model"
        if self.total_signals <= self.warmup:
            return f"warmup ({self.total_signals}/{self.warmup})"
        if self._fallback_active:
            return "fallback"
        if self.shadow_mode:
            return "shadow"
        return "active"
