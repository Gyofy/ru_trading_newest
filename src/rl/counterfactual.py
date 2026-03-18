"""Counterfactual PnL estimation for rejected signals.

When RL rejects a signal, we track what would have happened
if we had entered, using actual price data + Triple Barrier logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_counterfactual(
    df_future: pd.DataFrame,
    entry_price: float,
    side: str,
    sl_price: float,
    tp_price: float,
    max_horizon: int = 18,
    cost_pct: float = 0.002,
) -> float | None:
    """Estimate PnL if the signal had been accepted.

    Parameters
    ----------
    df_future : pd.DataFrame
        Bars after the signal (must have 'close' column, len >= 1).
    entry_price : float
        Hypothetical entry price.
    side : str
        "BUY" or "SELL".
    sl_price, tp_price : float
        Barrier levels.
    max_horizon : int
        Maximum bars to hold.
    cost_pct : float
        Estimated round-trip cost (default 0.2%).

    Returns
    -------
    float or None
        Net PnL percentage, or None if insufficient data.
    """
    required = {"close", "high", "low"}
    if not required.issubset(df_future.columns) or len(df_future) == 0:
        return None

    n = min(len(df_future), max_horizon)

    for i in range(n):
        bar_high = df_future["high"].iloc[i]
        bar_low = df_future["low"].iloc[i]
        if side == "BUY":
            if bar_high >= tp_price:
                return (tp_price - entry_price) / entry_price - cost_pct
            if bar_low <= sl_price:
                return (sl_price - entry_price) / entry_price - cost_pct
        else:  # SELL
            if bar_low <= tp_price:
                return (entry_price - tp_price) / entry_price - cost_pct
            if bar_high >= sl_price:
                return (entry_price - sl_price) / entry_price - cost_pct

    # TTL expiry — use final close
    final = df_future["close"].iloc[n - 1]
    if side == "BUY":
        return (final - entry_price) / entry_price - cost_pct
    return (entry_price - final) / entry_price - cost_pct
