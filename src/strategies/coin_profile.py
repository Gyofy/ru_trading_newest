"""CoinProfileStore -- adaptive per-coin parameter optimization.

New coins start with global DEFAULT params.
As trades accumulate in trade_context.jsonl, per-coin adjustments are learned:
  - SL multiplier adjustment (too many SL hits -> widen SL)
  - CVD threshold adjustment (false signals -> tighten quantile)
  - Optimal trailing distance
  - Win rate / avg PnL tracking

Persisted to: data/reports/multi_strategy/coin_profiles.json
Updated: after every trade close (incremental)

Integration (callers, not modified here):
  - base.py _try_execute(): call store.get_params(coin, config.extra) before barriers
  - asymmetric_sniper.py: same pattern
  - run_multi_strategy.py _on_position_close(): call store.update_from_trade(trade)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("coin_profile")

# ── EMA smoothing constant (slow adaptation) ──
_EMA_ALPHA = 0.1

# ── Trades needed for full confidence ──
_FULL_CONFIDENCE_TRADES = 20

# ── Adjustment bounds (safety clamps) ──
_SL_MULT_MIN = 0.6
_SL_MULT_MAX = 2.0
_TRAIL_MULT_MIN = 0.5
_TRAIL_MULT_MAX = 2.0
_CVD_Q_ADJ_MIN = -0.05
_CVD_Q_ADJ_MAX = 0.10


@dataclass
class CoinProfile:
    """Per-coin learned parameter adjustments and performance stats."""

    coin: str = ""

    # ── Learned adjustments (multipliers on base strategy params) ──
    sl_mult_adj: float = 1.0       # SL multiplier adjustment (>1 = wider SL)
    cvd_quantile_adj: float = 0.0  # quantile offset (+0.01 = stricter)
    trail_mult_adj: float = 1.0    # trailing distance multiplier

    # ── Performance stats ──
    total_trades: int = 0
    wins: int = 0
    total_pnl_usdt: float = 0.0
    avg_pnl_pct: float = 0.0
    avg_mfe_pct: float = 0.0      # how far price goes in our favor
    avg_mae_pct: float = 0.0      # how far price goes against us
    avg_bars_held: float = 0.0
    sl_hit_rate: float = 0.0      # fraction of trades that hit SL

    # ── Confidence (how much to trust learned params vs defaults) ──
    confidence: float = 0.0       # 0.0 = use defaults, 1.0 = fully adapted
    last_updated: str = ""

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades


class CoinProfileStore:
    """Manages per-coin adaptive parameters with JSON persistence.

    Usage:
        store = CoinProfileStore(persist_path, trade_context_path)
        params = store.get_params("SOL", base_params)  # at entry
        store.update_from_trade(trade_record)           # at close
    """

    def __init__(
        self,
        persist_path: Path,
        trade_context_path: Optional[Path] = None,
    ):
        self._persist_path = Path(persist_path)
        self._trade_context_path = (
            Path(trade_context_path) if trade_context_path else None
        )
        self._profiles: dict[str, CoinProfile] = {}
        self._load()

    # ── Public API ─────────────────────────────────────────

    def get_params(self, coin: str, base_params: dict) -> dict:
        """Get adapted params for a coin, blending defaults with learned adjustments.

        If coin has no profile (new coin), returns base_params unchanged.
        As confidence grows (more trades), learned adjustments carry more weight.

        Blending:
            effective = base * (1 - confidence) + learned * confidence
            confidence = min(total_trades / 20, 1.0)

        Recognized base_params keys that get adapted:
            sl_atr_mult   -> multiplied by sl_mult_adj
            trail_mult    -> multiplied by trail_mult_adj
            cvd_quantile  -> offset by cvd_quantile_adj
        """
        profile = self._profiles.get(coin)
        if profile is None or profile.total_trades == 0:
            return dict(base_params)  # shallow copy, no changes

        c = profile.confidence
        adapted = dict(base_params)

        # SL multiplier blending
        if "sl_atr_mult" in adapted:
            base_val = adapted["sl_atr_mult"]
            learned_val = base_val * profile.sl_mult_adj
            adapted["sl_atr_mult"] = base_val * (1 - c) + learned_val * c

        # Trailing distance blending
        if "trail_mult" in adapted:
            base_val = adapted["trail_mult"]
            learned_val = base_val * profile.trail_mult_adj
            adapted["trail_mult"] = base_val * (1 - c) + learned_val * c

        # CVD quantile blending (additive offset, not multiplicative)
        if "cvd_quantile" in adapted:
            base_val = adapted["cvd_quantile"]
            learned_val = base_val + profile.cvd_quantile_adj
            adapted["cvd_quantile"] = base_val * (1 - c) + learned_val * c

        return adapted

    def update_from_trade(self, trade_record: dict) -> None:
        """Update coin profile from a completed trade.

        Expected trade_record keys (from TradeContext / trade_context.jsonl):
            coin, pnl_pct, pnl_usdt, mfe_pct, mae_pct, bars_held,
            exit_reason, sl_distance_pct
        """
        coin = trade_record.get("coin", "")
        if not coin:
            logger.warning("update_from_trade: missing coin field, skip")
            return

        profile = self._profiles.get(coin)
        if profile is None:
            profile = CoinProfile(coin=coin)
            self._profiles[coin] = profile

        alpha = _EMA_ALPHA
        n = profile.total_trades

        pnl_pct = trade_record.get("pnl_pct", 0.0)
        pnl_usdt = trade_record.get("pnl_usdt", 0.0)
        mfe_pct = trade_record.get("mfe_pct", 0.0)
        mae_pct = trade_record.get("mae_pct", 0.0)
        bars_held = trade_record.get("bars_held", 0)
        exit_reason = trade_record.get("exit_reason", "")
        is_win = pnl_pct > 0
        is_sl_hit = exit_reason.upper() in ("SL_HIT", "SL")

        # ── Update trade counts ──
        profile.total_trades += 1
        if is_win:
            profile.wins += 1
        profile.total_pnl_usdt += pnl_usdt

        # ── EMA stats update ──
        if n == 0:
            # First trade: initialize directly
            profile.avg_pnl_pct = pnl_pct
            profile.avg_mfe_pct = mfe_pct
            profile.avg_mae_pct = mae_pct
            profile.avg_bars_held = float(bars_held)
            profile.sl_hit_rate = 1.0 if is_sl_hit else 0.0
        else:
            profile.avg_pnl_pct = (1 - alpha) * profile.avg_pnl_pct + alpha * pnl_pct
            profile.avg_mfe_pct = (1 - alpha) * profile.avg_mfe_pct + alpha * mfe_pct
            profile.avg_mae_pct = (1 - alpha) * profile.avg_mae_pct + alpha * mae_pct
            profile.avg_bars_held = (
                (1 - alpha) * profile.avg_bars_held + alpha * bars_held
            )
            profile.sl_hit_rate = (
                (1 - alpha) * profile.sl_hit_rate + alpha * (1.0 if is_sl_hit else 0.0)
            )

        # ── Confidence update ──
        profile.confidence = min(profile.total_trades / _FULL_CONFIDENCE_TRADES, 1.0)

        # ── Adaptive adjustment rules ──
        old_sl = profile.sl_mult_adj
        old_trail = profile.trail_mult_adj
        old_cvd = profile.cvd_quantile_adj

        self._adapt_sl(profile)
        self._adapt_trail(profile)
        self._adapt_cvd(profile)

        # ── Timestamp ──
        profile.last_updated = datetime.now(timezone.utc).isoformat()

        # ── Log material changes (>10%) ──
        if old_sl != 0 and abs(profile.sl_mult_adj / old_sl - 1) > 0.10:
            logger.info(
                f"[{coin}] sl_mult_adj: {old_sl:.3f} -> {profile.sl_mult_adj:.3f}"
            )
        if old_trail != 0 and abs(profile.trail_mult_adj / old_trail - 1) > 0.10:
            logger.info(
                f"[{coin}] trail_mult_adj: {old_trail:.3f} -> {profile.trail_mult_adj:.3f}"
            )
        if abs(profile.cvd_quantile_adj - old_cvd) > 0.005:
            logger.info(
                f"[{coin}] cvd_quantile_adj: {old_cvd:.4f} -> {profile.cvd_quantile_adj:.4f}"
            )

        self._save()

    def get_profile(self, coin: str) -> Optional[CoinProfile]:
        """Return profile for a coin, or None if unseen."""
        return self._profiles.get(coin)

    def summary(self) -> dict:
        """Return summary of all coin profiles for logging."""
        result = {}
        for coin, p in sorted(self._profiles.items()):
            result[coin] = {
                "trades": p.total_trades,
                "wins": p.wins,
                "win_rate": round(p.win_rate, 3),
                "pnl_usdt": round(p.total_pnl_usdt, 2),
                "avg_pnl_pct": round(p.avg_pnl_pct, 4),
                "sl_hit_rate": round(p.sl_hit_rate, 3),
                "confidence": round(p.confidence, 2),
                "sl_mult_adj": round(p.sl_mult_adj, 3),
                "trail_mult_adj": round(p.trail_mult_adj, 3),
                "cvd_q_adj": round(p.cvd_quantile_adj, 4),
            }
        return result

    # ── Adaptation rules ──────────────────────────────────

    @staticmethod
    def _adapt_sl(profile: CoinProfile) -> None:
        """Adjust SL multiplier based on SL hit rate.

        - SL hit rate > 60% -> widen SL (increase multiplier)
        - SL hit rate < 30% -> tighten SL (decrease multiplier)
        - Between 30-60%   -> no change (healthy zone)

        Step size: 0.02 per trade (slow, ~50 trades to go from 1.0 to 2.0)
        """
        if profile.total_trades < 3:
            return  # need minimum data

        step = 0.02
        if profile.sl_hit_rate > 0.60:
            profile.sl_mult_adj += step
        elif profile.sl_hit_rate < 0.30:
            profile.sl_mult_adj -= step

        profile.sl_mult_adj = max(
            _SL_MULT_MIN, min(_SL_MULT_MAX, profile.sl_mult_adj)
        )

    @staticmethod
    def _adapt_trail(profile: CoinProfile) -> None:
        """Adjust trailing stop distance based on MFE vs realized PnL.

        If avg MFE >> avg actual PnL -> trail too tight, widen.
        If avg MFE ~ avg actual PnL -> trail is good, could tighten slightly.

        MFE utilization = avg_pnl_pct / avg_mfe_pct (how much of the move we capture)
        Target utilization: 40-60%
        """
        if profile.total_trades < 5 or profile.avg_mfe_pct <= 0:
            return

        utilization = profile.avg_pnl_pct / profile.avg_mfe_pct

        step = 0.02
        if utilization < 0.30:
            # Capturing < 30% of favorable move -> trail too tight
            profile.trail_mult_adj += step
        elif utilization > 0.60:
            # Capturing > 60% -> could tighten trail for faster lock-in
            profile.trail_mult_adj -= step

        profile.trail_mult_adj = max(
            _TRAIL_MULT_MIN, min(_TRAIL_MULT_MAX, profile.trail_mult_adj)
        )

    @staticmethod
    def _adapt_cvd(profile: CoinProfile) -> None:
        """Adjust CVD quantile threshold based on signal quality.

        If avg MAE is small but still losing (signal quality issue) -> tighten.
        If winning consistently with current threshold -> loosen slightly to allow
        more trades.

        Heuristic: losing trades with low MAE = signal triggered correctly but
        direction was wrong. Need stricter trigger.
        """
        if profile.total_trades < 5:
            return

        step = 0.003

        # Losing with small adverse excursion = bad signal quality
        if profile.avg_pnl_pct < 0 and profile.avg_mae_pct < 0.5:
            profile.cvd_quantile_adj += step  # stricter

        # Winning consistently -> relax slightly to allow more trades
        elif profile.win_rate > 0.55 and profile.avg_pnl_pct > 0:
            profile.cvd_quantile_adj -= step * 0.5  # slower relaxation

        profile.cvd_quantile_adj = max(
            _CVD_Q_ADJ_MIN, min(_CVD_Q_ADJ_MAX, profile.cvd_quantile_adj)
        )

    # ── Persistence ───────────────────────────────────────

    def _save(self) -> None:
        """Persist all profiles to JSON (crash-safe via temp write + rename)."""
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._persist_path.with_suffix(".tmp")

        data = {}
        for coin, profile in self._profiles.items():
            data[coin] = asdict(profile)

        try:
            tmp_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            tmp_path.replace(self._persist_path)
        except Exception as e:
            logger.error(f"Failed to save coin profiles: {e}")
            if tmp_path.exists():
                tmp_path.unlink()

    def _load(self) -> None:
        """Load profiles from JSON. If file missing, start empty."""
        if not self._persist_path.exists():
            logger.info(
                f"No existing coin profiles at {self._persist_path}, starting fresh"
            )
            return

        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for coin, fields in raw.items():
                profile = CoinProfile(**fields)
                self._profiles[coin] = profile
            logger.info(
                f"Loaded {len(self._profiles)} coin profiles from {self._persist_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load coin profiles: {e}, starting fresh")
            self._profiles = {}

    def rebuild_from_trade_context(self) -> int:
        """Rebuild all profiles from trade_context.jsonl (cold start / recalibration).

        Returns number of trades processed.
        """
        if self._trade_context_path is None or not self._trade_context_path.exists():
            logger.warning("No trade_context.jsonl found, nothing to rebuild")
            return 0

        self._profiles.clear()
        count = 0

        try:
            with open(self._trade_context_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        self.update_from_trade(record)
                        count += 1
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error rebuilding from trade_context: {e}")

        logger.info(f"Rebuilt coin profiles from {count} trades")
        return count
