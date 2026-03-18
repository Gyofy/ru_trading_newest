"""High-Resolution Barrier Simulator -- 1h granularity within 4h signals.

Problem: current 4h barrier check can't determine TP/SL order within a bar.
When both hit in same 4h bar, we default to SL (conservative but inaccurate).

Solution: use 1h OHLCV (already fetched) to check barrier hits with 4x resolution.
Signal still on 4h, but exit detection uses 1h bars.

Usage:
    from src.evaluation.hires_barrier import simulate_hires

    trades = simulate_hires(
        signal_df=oos_4h,        # 4h dataframe (signals)
        hires_df=oos_1h,         # 1h dataframe (barrier checking)
        s1_pred=s1_pred,
        s2_pred=s2_pred,
        k_upper=3.0, k_lower=0.6,
        ...
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.execution.cost_model import CostModel, ExitType


@dataclass
class HiResTrade:
    coin: str
    side: str
    entry_time: str
    entry_price: float
    tp_price: float
    sl_price: float
    atr: float
    exit_time: str = ""
    exit_price: float = 0.0
    exit_type: str = ""
    holding_bars_4h: int = 0
    holding_bars_1h: int = 0
    gross_pnl_eq: float = 0.0
    cost_eq: float = 0.0
    net_pnl_eq: float = 0.0
    regime: str = ""
    resolution: str = "1h"  # "1h" or "4h_fallback"


def simulate_hires(
    signal_df: pd.DataFrame,
    hires_df: pd.DataFrame,
    s1_pred: np.ndarray,
    s2_pred: np.ndarray,
    k_upper: float = 3.0,
    k_lower: float = 0.6,
    max_hold_4h: int = 18,
    risk_frac: float = 0.005,
    min_barrier_pct: float = 0.002,
    cost_model: CostModel = None,
    bar_minutes: int = 240,
    coin: str = "",
    blocked_regimes: list = None,
    regime_fn=None,
) -> list:
    """Simulate trades with 1h barrier resolution.

    Args:
        signal_df: 4h OHLCV (signals generated on this)
        hires_df: 1h OHLCV (barrier checking on this)
        s1_pred, s2_pred: predictions aligned to signal_df
        regime_fn: callable(df, i) -> str, optional
        blocked_regimes: list of regime strings to block
    """
    if cost_model is None:
        cost_model = CostModel()
    if blocked_regimes is None:
        blocked_regimes = []

    close_4h = signal_df["close"].values
    times_4h = signal_df.index
    n_4h = len(close_4h)

    # 1h data
    hi_close = hires_df["close"].values
    hi_high = hires_df["high"].values
    hi_low = hires_df["low"].values
    hi_times = hires_df.index
    n_1h = len(hi_close)

    # ATR from 4h (already computed)
    if "atr_14" in signal_df.columns:
        atr_4h = signal_df["atr_14"].values
    else:
        atr_4h = np.full(n_4h, 0.01)

    # Build 4h -> 1h time mapping
    # For each 4h bar, find corresponding 1h bar indices
    def find_1h_range(t_4h_start, n_bars_4h):
        """Find 1h bar indices covering n_bars_4h * 4h period after t_4h_start."""
        end_time = t_4h_start + pd.Timedelta(hours=4 * n_bars_4h)
        mask = (hi_times > t_4h_start) & (hi_times <= end_time)
        indices = np.where(mask)[0]
        return indices

    trades = []
    next_avail_4h = 0

    for i in range(n_4h - max_hold_4h):
        if i < next_avail_4h:
            continue
        if s1_pred[i] != 1:
            continue

        # Regime filter
        if regime_fn is not None:
            regime = regime_fn(signal_df, i)
            if regime in blocked_regimes:
                continue
        else:
            regime = "UNKNOWN"

        entry_price = close_4h[i]
        side = "BUY" if s2_pred[i] == 1 else "SELL"
        cur_atr = atr_4h[i] if not np.isnan(atr_4h[i]) else entry_price * 0.01

        u_dist = max(k_upper * cur_atr, min_barrier_pct * entry_price)
        l_dist = max(k_lower * cur_atr, min_barrier_pct * entry_price)

        if side == "BUY":
            tp_price = entry_price + u_dist
            sl_price = entry_price - l_dist
        else:
            tp_price = entry_price - u_dist
            sl_price = entry_price + l_dist

        # Find 1h bars for barrier checking
        entry_time = times_4h[i]
        hr_indices = find_1h_range(entry_time, max_hold_4h)

        exit_type = None
        exit_price = 0.0
        exit_time = ""
        exit_1h_idx = -1
        resolution = "1h"

        if len(hr_indices) > 0:
            # Check barriers on 1h bars
            for j in hr_indices:
                if side == "BUY":
                    hit_tp = hi_high[j] >= tp_price
                    hit_sl = hi_low[j] <= sl_price
                else:
                    hit_tp = hi_low[j] <= tp_price
                    hit_sl = hi_high[j] >= sl_price

                if hit_tp and hit_sl:
                    # Same 1h bar -- still conservative SL, but 4x better resolution
                    exit_type = "stop_loss"
                    exit_price = sl_price
                    exit_time = str(hi_times[j])
                    exit_1h_idx = j
                    break
                elif hit_tp:
                    exit_type = "take_profit"
                    exit_price = tp_price
                    exit_time = str(hi_times[j])
                    exit_1h_idx = j
                    break
                elif hit_sl:
                    exit_type = "stop_loss"
                    exit_price = sl_price
                    exit_time = str(hi_times[j])
                    exit_1h_idx = j
                    break

            if exit_type is None:
                # Time barrier -- use last 1h bar's close
                last_idx = hr_indices[-1] if len(hr_indices) > 0 else -1
                if last_idx >= 0:
                    exit_type = "time_stop"
                    exit_price = hi_close[last_idx]
                    exit_time = str(hi_times[last_idx])
                    exit_1h_idx = last_idx
        else:
            # Fallback to 4h resolution
            resolution = "4h_fallback"
            for j in range(i + 1, min(i + max_hold_4h + 1, n_4h)):
                h4_high = signal_df["high"].values[j]
                h4_low = signal_df["low"].values[j]
                if side == "BUY":
                    hit_tp = h4_high >= tp_price
                    hit_sl = h4_low <= sl_price
                else:
                    hit_tp = h4_low <= tp_price
                    hit_sl = h4_high >= sl_price

                if hit_tp and hit_sl:
                    exit_type, exit_price = "stop_loss", sl_price
                    exit_time = str(times_4h[j])
                    break
                elif hit_tp:
                    exit_type, exit_price = "take_profit", tp_price
                    exit_time = str(times_4h[j])
                    break
                elif hit_sl:
                    exit_type, exit_price = "stop_loss", sl_price
                    exit_time = str(times_4h[j])
                    break

            if exit_type is None:
                eb = min(i + max_hold_4h, n_4h - 1)
                exit_type = "time_stop"
                exit_price = close_4h[eb]
                exit_time = str(times_4h[eb])

        # Holding bars
        if exit_1h_idx >= 0:
            entry_1h = np.searchsorted(hi_times, entry_time)
            holding_1h = exit_1h_idx - entry_1h
        else:
            holding_1h = 0

        # Map to 4h bars for cost calculation
        holding_4h = max(holding_1h // 4, 1)

        # PnL
        if side == "BUY":
            gpnl_pct = (exit_price - entry_price) / entry_price
        else:
            gpnl_pct = (entry_price - exit_price) / entry_price

        sd = max(l_dist / entry_price, 0.003)
        nr = risk_frac / sd
        gpnl_eq = gpnl_pct * nr

        # Cost
        ee = {"take_profit": ExitType.TAKE_PROFIT,
              "stop_loss": ExitType.STOP_LOSS,
              "time_stop": ExitType.TIME_STOP}[exit_type]

        cost = cost_model.estimate_trade_cost(
            entry_price=entry_price, sl_price=sl_price, tp_price=tp_price,
            risk_frac=risk_frac, exit_type=ee,
            holding_bars=holding_4h, bar_minutes=bar_minutes,
            entry_is_maker=True)

        net = gpnl_eq - cost.total_eq

        trades.append(HiResTrade(
            coin=coin, side=side,
            entry_time=str(entry_time), entry_price=round(entry_price, 6),
            tp_price=round(tp_price, 6), sl_price=round(sl_price, 6),
            atr=round(cur_atr, 6),
            exit_time=exit_time, exit_price=round(exit_price, 6),
            exit_type=exit_type,
            holding_bars_4h=holding_4h, holding_bars_1h=holding_1h,
            gross_pnl_eq=round(gpnl_eq, 6), cost_eq=round(cost.total_eq, 6),
            net_pnl_eq=round(net, 6), regime=regime, resolution=resolution,
        ))

        # Non-overlapping: skip based on 4h bars
        next_avail_4h = i + holding_4h + 1

    return trades
