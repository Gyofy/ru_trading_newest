"""Build 31-dim state vector for LinUCB from pipeline data.

v5.0: TSMOM rule-based signals (no ML probabilities).
State = [signal(6) + market(4) + micro(3) + cost(2) + portfolio(4)
         + coin_history(4) + cross(2) + coin_id(5) + intercept(1)] = 31 dim

v4.3 → v5.0 signal quality 변경:
  p_trade      → tsmom_strength  (|28d return|, 모멘텀 크기)
  p_direction  → rsi_normalized  (RSI/100)
  confidence   → cvd_extremeness (CVD가 얼마나 극단인지)
  trade_margin → oi_zscore       (OI 과열 정도)
  dir_margin   → tsmom_rsi_agree (TSMOM-RSI 방향 일치)
  side_sign    → side_sign       (동일)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COIN_INDEX = {"BTC": 0, "ETH": 1, "SOL": 2, "XRP": 3, "ADA": 4, "DOT": 5, "LINK": 6}


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
    # v5.0 TSMOM signal inputs
    tsmom_strength: float = 0.0,
    rsi_value: float = 50.0,
    cvd_extremeness: float = 0.0,
    oi_zscore: float = 0.0,
    tsmom_rsi_agree: bool = True,
    # legacy compat (ignored in v5.0, kept for API stability)
    p_trade: float = 0.0,
    p_direction: float = 0.0,
    s1_threshold: float = 0.0,
    max_horizon: int = 24,
    last_funding: float = 0.0,
    btc_df: pd.DataFrame = None,
) -> np.ndarray:
    """Build 30-dim + intercept state vector (v5.0 TSMOM).

    All features are clipped/normalized to stable ranges for LinUCB.
    """
    # Signal quality (6) — v5.0 TSMOM-based
    tsmom_str = _clip(tsmom_strength, 0.0, 0.3)
    rsi_norm = _clip(rsi_value / 100.0, 0.0, 1.0)
    cvd_ext = _clip(cvd_extremeness, 0.0, 1.0)
    oi_z = _clip(oi_zscore, -3.0, 3.0)
    agree = 1.0 if tsmom_rsi_agree else 0.0
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
    src = btc_df if btc_df is not None else df
    if "close" in src.columns and len(src) >= 6:
        btc_ret = _clip(float(src["close"].pct_change(6).iloc[-1]), -0.10, 0.10)
    corr_btc = _clip(_safe_col(df, "corr_btc", 0.0), -1.0, 1.0)

    # Coin identity (7 coins in v5.0)
    coin_oh = [0.0] * 7
    if coin in COIN_INDEX:
        coin_oh[COIN_INDEX[coin]] = 1.0

    # Intercept (1)
    intercept = 1.0

    return np.array([
        tsmom_str, rsi_norm, cvd_ext, oi_z, agree, side_sign,
        regime_trend, regime_up, atr_pct, hurst,
        cvd_ratio, ofi_norm, ms_composite,
        spread_proxy, funding_clipped,
        open_norm, daily_pnl_pct, weekly_pnl_pct, dd_clipped,
        wr5, avg5, streak_norm, bars_norm,
        btc_ret, corr_btc,
        *coin_oh,
        intercept,
    ], dtype=np.float64)  # shape: (33,)


STATE_DIM = 33  # 32 features + 1 intercept (7 coins instead of 5)
STATE_NAMES = [
    "tsmom_strength", "rsi_normalized", "cvd_extremeness",
    "oi_zscore", "tsmom_rsi_agree", "side_sign",
    "regime_trend", "regime_up", "atr_pct", "hurst",
    "cvd_ratio", "ofi_norm", "ms_composite",
    "spread_proxy", "last_funding",
    "open_positions", "daily_pnl_pct", "weekly_pnl_pct", "dd_ratio",
    "coin_win_rate_5", "coin_avg_pnl_5", "coin_streak", "bars_since_last",
    "btc_return_24h", "corr_btc",
    "is_btc", "is_eth", "is_sol", "is_xrp", "is_ada", "is_dot", "is_link",
    "intercept",
]
