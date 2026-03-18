"""Build 30-dim state vector for LinUCB from pipeline data.

State = [signal(6) + market(4) + micro(3) + cost(2) + portfolio(4)
         + coin_history(4) + cross(2) + coin_id(5) + intercept(1)] = 31 dim
(30 features + 1 intercept, but we call it "30-dim" by convention.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COIN_INDEX = {"DOT": 0, "ADA": 1, "XRP": 2, "SOL": 3, "LINK": 4}


def _clip(val: float, lo: float, hi: float) -> float:
    if np.isnan(val):
        return 0.0
    return max(lo, min(hi, val))


def _safe_col(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    if col in df.columns:
        v = df[col].iloc[-1]
        return default if (np.isnan(v) or np.isinf(v)) else float(v)
    return default


def build_rl_state(
    df: pd.DataFrame,
    pred_side: str,
    p_trade: float,
    p_direction: float,
    s1_threshold: float,
    coin: str,
    equity: float,
    daily_pnl: float,
    weekly_pnl: float,
    dd_ratio: float,
    open_count: int,
    coin_win_rate_5: float,
    coin_avg_pnl_5: float,
    coin_streak: int,
    bars_since_last: int,
    max_horizon: int = 18,
    last_funding: float = 0.0,
) -> np.ndarray:
    """Build 30-dim + intercept state vector.

    All features are clipped/normalized to stable ranges for LinUCB.
    """
    confidence = p_trade * p_direction

    # Signal quality (6)
    trade_margin = _clip(p_trade - s1_threshold, -0.5, 0.5)
    dir_margin = _clip(abs(p_direction - 0.5) * 2, 0.0, 1.0)
    side_sign = 1.0 if pred_side == "BUY" else -1.0

    # Market regime (4)
    adx = _safe_col(df, "adx_14", 20.0)
    di_diff = _safe_col(df, "di_diff", 0.0)
    if di_diff == 0.0 and "plus_di_14" in df.columns and "minus_di_14" in df.columns:
        di_diff = _safe_col(df, "plus_di_14") - _safe_col(df, "minus_di_14")
    regime_trend = 1.0 if adx > 25 else 0.0
    regime_up = 1.0 if (adx > 25 and di_diff > 0) else 0.0
    close_price = _safe_col(df, "close", 1.0)
    atr_pct = _clip(_safe_col(df, "atr_14", close_price * 0.01) / (close_price + 1e-10), 0.0, 0.1)
    hurst = _clip(_safe_col(df, "hurst", 0.5), 0.0, 1.0)

    # Microstructure (3)
    cvd_ratio = _clip(_safe_col(df, "cvd_ratio_6", 0.0), -3.0, 3.0)
    vol_sma = _safe_col(df, "volume_sma_20", 1.0)
    ofi_norm = _clip(_safe_col(df, "ofi_sum_3", 0.0) / (vol_sma + 1e-10), -3.0, 3.0)
    ms_composite = _clip(_safe_col(df, "ms_composite", 0.0), -1.0, 1.0)

    # Cost proxy (2)
    high = _safe_col(df, "high", close_price)
    low = _safe_col(df, "low", close_price)
    spread_proxy = _clip((high - low) / (close_price + 1e-10), 0.0, 0.05)
    funding_clipped = _clip(last_funding, -0.003, 0.003)

    # Portfolio state (4)
    open_norm = open_count / 5.0
    daily_pnl_pct = _clip(daily_pnl / (equity + 1e-10), -0.05, 0.05)
    weekly_pnl_pct = _clip(weekly_pnl / (equity + 1e-10), -0.10, 0.10)
    dd_clipped = _clip(dd_ratio, 0.0, 1.0)

    # Coin history (4)
    wr5 = _clip(coin_win_rate_5, 0.0, 1.0)
    avg5 = _clip(coin_avg_pnl_5, -0.05, 0.05)
    streak_norm = _clip(coin_streak / 5.0, -1.0, 1.0)
    bars_norm = _clip(bars_since_last / max_horizon, 0.0, 1.0) if max_horizon > 0 else 0.0

    # Cross-market (2)
    btc_ret = 0.0
    if "close" in df.columns and len(df) >= 6:
        btc_ret = _clip(float(df["close"].pct_change(6).iloc[-1]), -0.10, 0.10)
    corr_btc = _clip(_safe_col(df, "corr_btc", 0.0), -1.0, 1.0)

    # Coin identity (5)
    coin_oh = [0.0] * 5
    if coin in COIN_INDEX:
        coin_oh[COIN_INDEX[coin]] = 1.0

    # Intercept (1)
    intercept = 1.0

    return np.array([
        p_trade, p_direction, confidence, trade_margin, dir_margin, side_sign,
        regime_trend, regime_up, atr_pct, hurst,
        cvd_ratio, ofi_norm, ms_composite,
        spread_proxy, funding_clipped,
        open_norm, daily_pnl_pct, weekly_pnl_pct, dd_clipped,
        wr5, avg5, streak_norm, bars_norm,
        btc_ret, corr_btc,
        *coin_oh,
        intercept,
    ], dtype=np.float64)  # shape: (31,)


STATE_DIM = 31  # 30 features + 1 intercept
STATE_NAMES = [
    "p_trade", "p_direction", "confidence", "trade_margin", "dir_margin", "side_sign",
    "regime_trend", "regime_up", "atr_pct", "hurst",
    "cvd_ratio", "ofi_norm", "ms_composite",
    "spread_proxy", "last_funding",
    "open_positions", "daily_pnl_pct", "weekly_pnl_pct", "dd_ratio",
    "coin_win_rate_5", "coin_avg_pnl_5", "coin_streak", "bars_since_last",
    "btc_return_24h", "corr_btc",
    "is_dot", "is_ada", "is_xrp", "is_sol", "is_link",
    "intercept",
]
