"""PortfolioRiskManager — cross-strategy risk overlay.

Enforces portfolio-level gates: exposure cap, direction concentration,
daily loss limit, per-strategy loss pause, and coin exclusivity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.strategies.multi_position_manager import MultiPositionManager

logger = logging.getLogger("portfolio_risk")


@dataclass
class PortfolioRiskConfig:
    total_exposure_pct: float = 0.80      # max 80% of equity in positions
    same_direction_max: int = 3           # max positions in same direction
    daily_loss_pct: float = 0.20          # -20% daily → kill all
    strategy_loss_pct: float = 0.40       # -40% of allocation → pause strategy
    min_notional_usdt: float = 5.0        # Binance minimum
    max_funding_rate: float = 0.001       # block if |funding| > 0.1%


class PortfolioRiskManager:
    """Portfolio-level risk checks applied before any entry."""

    def __init__(
        self,
        config: PortfolioRiskConfig,
        pos_manager: MultiPositionManager,
        initial_equity: float,
        strategy_allocations: dict[str, float] | None = None,
    ):
        self.config = config
        self.pos_manager = pos_manager
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.strategy_allocations = strategy_allocations or {}

        # Daily tracking
        self._day_start_equity = initial_equity
        self._day_start_date = datetime.now(timezone.utc).date()
        self._strategy_pnl: dict[str, float] = {}  # strategy -> cumulative daily PnL
        self._kill_switch = False
        self._kill_reason = ""
        self._paused_strategies: set[str] = set()

    # ── Main gate ─────────────────────────────────────────

    def approve_entry(
        self,
        strategy_name: str,
        coin: str,
        side: str,
        notional: float,
        funding_rate: float = 0.0,
    ) -> tuple[bool, str]:
        """Run all portfolio gates. Returns (approved, reason)."""

        # Gate 1: Kill switch
        if self._kill_switch:
            return False, f"kill_switch: {self._kill_reason}"

        # Gate 2: Daily loss
        daily_loss = self._daily_loss_pct()
        if daily_loss >= self.config.daily_loss_pct:
            self._kill_switch = True
            self._kill_reason = f"daily_loss {daily_loss:.1%}"
            return False, self._kill_reason

        # Gate 3: Strategy paused
        if strategy_name in self._paused_strategies:
            return False, f"strategy_paused: {strategy_name}"

        # Gate 4: Per-strategy loss
        strat_pnl = self._strategy_pnl.get(strategy_name, 0.0)
        alloc = self.strategy_allocations.get(strategy_name, 20.0)
        if alloc > 0 and strat_pnl < 0 and abs(strat_pnl) > alloc * self.config.strategy_loss_pct:
            self._paused_strategies.add(strategy_name)
            return False, f"strategy_loss: {strat_pnl:.2f} > {alloc * self.config.strategy_loss_pct:.2f}"

        # Gate 5: Total exposure cap
        current_notional = self.pos_manager.total_notional()
        max_notional = self.current_equity * self.config.total_exposure_pct
        if current_notional + notional > max_notional:
            return False, f"exposure_cap: {current_notional + notional:.2f} > {max_notional:.2f}"

        # Gate 6: Same direction max
        dir_counts = self.pos_manager.direction_counts()
        if dir_counts.get(side, 0) >= self.config.same_direction_max:
            return False, f"direction_cap: {side}={dir_counts[side]}"

        # Gate 7: Coin exclusivity
        if self.pos_manager.has_coin_any_strategy(coin):
            return False, f"coin_duplicate: {coin}"

        # Gate 8: Funding rate extreme — block if funding works AGAINST position
        # Note: 0.001 = 0.1% per 8h. During market overheating many coins exceed this.
        # Only blocks when funding is ADVERSE (we pay), not when we receive.
        if abs(funding_rate) > self.config.max_funding_rate:
            funding_against = (
                (side == "BUY" and funding_rate > self.config.max_funding_rate)
                or (side == "SELL" and funding_rate < -self.config.max_funding_rate)
            )
            if funding_against:
                logger.info(
                    f"[Gate8] {coin} {side} blocked: funding={funding_rate:.6f} "
                    f"(limit={self.config.max_funding_rate}, adverse direction)"
                )
                return False, f"funding_adverse: {funding_rate:.6f} vs limit {self.config.max_funding_rate}"

        # Gate 9: Minimum notional
        if notional < self.config.min_notional_usdt:
            return False, f"min_notional: {notional:.2f} < {self.config.min_notional_usdt}"

        return True, "approved"

    # ── PnL tracking ──────────────────────────────────────

    def record_trade_pnl(self, strategy_name: str, pnl_usdt: float) -> None:
        """Record a closed trade's PnL."""
        self._strategy_pnl[strategy_name] = (
            self._strategy_pnl.get(strategy_name, 0.0) + pnl_usdt
        )
        self.current_equity += pnl_usdt
        logger.info(
            f"[PortRisk] {strategy_name} PnL: {pnl_usdt:+.4f} | "
            f"equity: {self.current_equity:.2f} | "
            f"daily: {self._daily_loss_pct():.2%}"
        )

    def update_equity(self, equity: float) -> None:
        """Sync equity from exchange balance."""
        self.current_equity = equity

    def _daily_loss_pct(self) -> float:
        """Current daily drawdown as positive fraction."""
        if self._day_start_equity <= 0:
            return 0.0
        loss = (self._day_start_equity - self.current_equity) / self._day_start_equity
        return max(0.0, loss)

    # ── Daily reset ───────────────────────────────────────

    def maybe_reset_daily(self) -> None:
        """Reset daily counters on new UTC day."""
        today = datetime.now(timezone.utc).date()
        if today > self._day_start_date:
            logger.info(
                f"[PortRisk] Day reset: {self._day_start_date} → {today} | "
                f"equity: {self.current_equity:.2f}"
            )
            self._day_start_date = today
            self._day_start_equity = self.current_equity
            self._strategy_pnl.clear()
            self._paused_strategies.clear()
            if self._kill_switch and "daily_loss" in self._kill_reason:
                self._kill_switch = False
                self._kill_reason = ""

    # ── Status ────────────────────────────────────────────

    @property
    def is_killed(self) -> bool:
        return self._kill_switch

    def status(self) -> dict:
        return {
            "equity": round(self.current_equity, 2),
            "daily_loss_pct": round(self._daily_loss_pct(), 4),
            "positions": self.pos_manager.count(),
            "notional": round(self.pos_manager.total_notional(), 2),
            "kill_switch": self._kill_switch,
            "paused": list(self._paused_strategies),
            "strategy_pnl": {k: round(v, 4) for k, v in self._strategy_pnl.items()},
        }
