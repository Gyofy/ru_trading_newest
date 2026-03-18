"""LinUCB Contextual Bandit — production-grade.

Features:
  - Sherman-Morrison rank-1 inverse update O(d^2)
  - Forgetting factor gamma for non-stationary environments
  - Intercept term included in state vector
  - 4-action space: [REJECT, ACCEPT_0.75, ACCEPT_1.00, ACCEPT_1.25]
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

from src.rl.state_builder import STATE_DIM, STATE_NAMES

logger = logging.getLogger("live_bot.rl.bandit")

N_ACTIONS = 4
ACTION_NAMES = ["REJECT", "ACCEPT_0.75", "ACCEPT_1.00", "ACCEPT_1.25"]
SIZING_MAP = {0: 0.0, 1: 0.75, 2: 1.0, 3: 1.25}


class LinUCB:
    """Linear Upper Confidence Bound contextual bandit.

    Per-action model: reward_a = x^T theta_a
    Selection: argmax_a [ x^T theta_a + alpha * sqrt(x^T A_a^{-1} x) ]

    Parameters
    ----------
    state_dim : int
        Dimension of state vector (default 31 = 30 features + intercept).
    n_actions : int
        Number of discrete actions (default 4).
    alpha : float
        Exploration parameter. Higher = more exploration.
    gamma : float
        Forgetting factor. 1.0 = no forgetting, 0.99 = ~100 sample half-life.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        n_actions: int = N_ACTIONS,
        alpha: float = 1.0,
        gamma: float = 0.995,
    ):
        self.d = state_dim
        self.K = n_actions
        self.alpha = alpha
        self.gamma = gamma

        self._A_inv = [np.eye(state_dim, dtype=np.float64) for _ in range(n_actions)]
        self.b = [np.zeros(state_dim, dtype=np.float64) for _ in range(n_actions)]
        self.n_updates = [0] * n_actions

    def select_action(self, state: np.ndarray) -> int:
        """UCB action selection (exploration + exploitation)."""
        best_a, best_ucb = 0, -np.inf
        for a in range(self.K):
            theta = self._A_inv[a] @ self.b[a]
            exploit = float(state @ theta)
            Ax = self._A_inv[a] @ state
            explore = self.alpha * np.sqrt(max(0.0, float(state @ Ax)))
            ucb = exploit + explore
            if ucb > best_ucb:
                best_ucb = ucb
                best_a = a
        return best_a

    def score(self, state: np.ndarray) -> tuple[int, float]:
        """Select action + return exploitation-only score (for ranking)."""
        action = self.select_action(state)
        theta = self._A_inv[action] @ self.b[action]
        return action, float(state @ theta)

    def update(self, state: np.ndarray, action: int, reward: float) -> None:
        """Sherman-Morrison rank-1 update with forgetting."""
        x = state.reshape(-1, 1)  # (d, 1)

        # Forgetting: decay old info
        if self.gamma < 1.0:
            self._A_inv[action] /= self.gamma
            self.b[action] *= self.gamma

        # Sherman-Morrison: (A + xx^T)^{-1} = A^{-1} - A^{-1}xx^T A^{-1} / (1 + x^T A^{-1} x)
        Ax = self._A_inv[action] @ x             # (d, 1)
        denom = 1.0 + float(x.T @ Ax)            # scalar
        if abs(denom) > 1e-12:
            self._A_inv[action] -= (Ax @ Ax.T) / denom

        self.b[action] += reward * state
        self.n_updates[action] += 1

        # Periodic re-regularization: prevent A_inv numerical explosion
        if self.n_updates[action] % 500 == 0:
            self._A_inv[action] += 1e-6 * np.eye(self.d, dtype=np.float64)
            logger.debug(f"[LinUCB] Re-regularized action {action} at update {self.n_updates[action]}")

    def get_theta(self, action: int) -> np.ndarray:
        """Return learned weight vector for action (interpretable)."""
        return self._A_inv[action] @ self.b[action]

    def feature_importance(self, action: int) -> dict[str, float]:
        """Named feature importance for given action."""
        theta = self.get_theta(action)
        names = STATE_NAMES if len(STATE_NAMES) == self.d else [f"f{i}" for i in range(self.d)]
        return {n: float(theta[i]) for i, n in enumerate(names)}

    def summary(self) -> dict:
        """Diagnostic summary."""
        return {
            "state_dim": self.d,
            "n_actions": self.K,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "updates_per_action": self.n_updates,
            "total_updates": sum(self.n_updates),
        }

    # ── Persistence ────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "A_inv": self._A_inv, "b": self.b,
            "n_updates": self.n_updates,
            "d": self.d, "K": self.K,
            "alpha": self.alpha, "gamma": self.gamma,
        }, path, compress=3)
        logger.info(f"[LinUCB] Saved to {path} ({sum(self.n_updates)} total updates)")

    @classmethod
    def load(cls, path: str | Path) -> "LinUCB":
        path = Path(path)
        d = joblib.load(path)
        obj = cls(state_dim=d["d"], n_actions=d["K"],
                  alpha=d["alpha"], gamma=d["gamma"])
        obj._A_inv = d["A_inv"]
        obj.b = d["b"]
        obj.n_updates = d["n_updates"]
        logger.info(f"[LinUCB] Loaded from {path} ({sum(obj.n_updates)} total updates)")
        return obj
