"""TSMOM v5.1 Core — shared signal generation, backtest, metrics.

All experiment scripts and paper bot import from here.
Single source of truth for strategy logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

COST_ROUNDTRIP = 0.0020

COINS_10 = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
    "XRP": "XRPUSDT", "ADA": "ADAUSDT", "DOT": "DOTUSDT", "LINK": "LINKUSDT",
    "DOGE": "DOGEUSDT", "AVAX": "AVAXUSDT", "BNB": "BNBUSDT",
}

YAHOO_MAP = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD",
    "XRP": "XRP-USD", "ADA": "ADA-USD", "DOT": "DOT-USD", "LINK": "LINK-USD",
    "DOGE": "DOGE-USD", "AVAX": "AVAX-USD", "BNB": "BNB-USD",
}


# ── CVD ──────────────────────────────────────────────────

def compute_cvd_ratio(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """BVC-based CVD ratio (causal, no lookahead)."""
    hr = (df["high"] - df["low"]).replace(0, np.nan)
    buy_frac = ((df["close"] - df["low"]) / hr).fillna(0.5).clip(0, 1)
    vd = (2 * buy_frac - 1) * df["volume"]
    cvd = vd.cumsum()
    cvd_ma = cvd.rolling(window, min_periods=max(6, window // 4)).mean()
    return ((cvd - cvd_ma) / cvd_ma.abs().replace(0, np.nan)).fillna(0)


# ── Signal Generation ────────────────────────────────────

def generate_dual_signals(
    df: pd.DataFrame,
    lb_short: int = 7,
    lb_long: int = 28,
    cvd_quantile: float = 0.75,
    cvd_window: int = 120,
    use_oi: bool = True,
    oi_max: float = 2.0,
) -> pd.Series:
    """Dual TSMOM + RSI + CVD + OI signal generation.

    Returns: Series of +1 (LONG), -1 (SHORT), 0 (FLAT).
    """
    bs, bl = lb_short * 6, lb_long * 6

    tsmom_s = np.sign(df["close"].pct_change(bs))
    tsmom_l = np.sign(df["close"].pct_change(bl))

    rsi = df.get("rsi_14", pd.Series(50.0, index=df.index))
    rsi_ok = ((tsmom_s == 1) & (rsi > 50)) | ((tsmom_s == -1) & (rsi < 50))

    cvd = compute_cvd_ratio(df, 24)
    q_hi = cvd.rolling(cvd_window, min_periods=30).quantile(cvd_quantile)
    q_lo = cvd.rolling(cvd_window, min_periods=30).quantile(1 - cvd_quantile)
    cvd_ok = ((tsmom_s == -1) & (cvd > q_hi)) | ((tsmom_s == 1) & (cvd < q_lo))

    dual_ok = tsmom_s == tsmom_l
    mask = rsi_ok & cvd_ok & dual_ok

    if use_oi and "oi_zscore" in df.columns:
        mask = mask & (df["oi_zscore"].abs().fillna(0) < oi_max)

    sig = tsmom_s.copy()
    sig[~mask] = 0
    return sig.fillna(0).astype(int)


# ── Backtest ─────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    k_upper: float = 5.0,
    k_lower: float = 1.0,
    max_hold: int = 24,
    cost: float = COST_ROUNDTRIP,
) -> np.ndarray:
    """Fast barrier backtest. Returns array of net PnLs per trade."""
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values

    if "atr_14" in df.columns:
        atr = df["atr_14"].values
    else:
        tr = np.maximum(h - l, np.maximum(
            np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

    sig = signals.values if hasattr(signals, "values") else signals
    trades = []
    nxt = 0

    for i in range(len(df) - max_hold):
        if i < nxt or sig[i] == 0 or np.isnan(atr[i]) or atr[i] <= 0:
            continue

        side = int(sig[i])
        entry = c[i]
        a = atr[i]
        tp_d = max(k_upper * a, entry * 0.002)
        sl_d = max(k_lower * a, entry * 0.002)
        tp = entry + tp_d * side
        sl = entry - sl_d * side

        ep = c[min(i + max_hold, len(df) - 1)]
        eb = min(i + max_hold, len(df) - 1)

        for j in range(i + 1, min(i + max_hold + 1, len(df))):
            if side == 1:
                if l[j] <= sl:
                    ep, eb = sl, j
                    break
                if h[j] >= tp:
                    ep, eb = tp, j
                    break
            else:
                if h[j] >= sl:
                    ep, eb = sl, j
                    break
                if l[j] <= tp:
                    ep, eb = tp, j
                    break

        pnl = ((ep - entry) / entry) * side - cost
        trades.append(pnl)
        nxt = eb + 1

    return np.array(trades)


# ── Metrics ──────────────────────────────────────────────

@dataclass
class StrategyResult:
    n: int = 0
    wr: float = 0.0
    avg: float = 0.0
    total: float = 0.0
    sharpe: float = 0.0
    mdd: float = 0.0
    pf: float = 0.0
    max_consec_loss: int = 0

    def summary(self) -> str:
        return (f"n={self.n:3d} WR={self.wr:.1%} avg={self.avg:+.4%} "
                f"Sharpe={self.sharpe:.2f} MDD={self.mdd:.2%} PF={self.pf:.2f}")


def calc_metrics(pnls) -> StrategyResult:
    """Calculate strategy metrics from PnL array."""
    if len(pnls) == 0:
        return StrategyResult()

    a = np.array(pnls)
    eq = np.cumsum(a)
    dd = eq - np.maximum.accumulate(eq)
    w = sum(p for p in a if p > 0)
    lo = abs(sum(p for p in a if p < 0))

    streak, max_streak = 0, 0
    for p in a:
        if p < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return StrategyResult(
        n=len(a),
        wr=float(np.mean(a > 0)),
        avg=float(np.mean(a)),
        total=float(np.sum(a)),
        sharpe=float(np.mean(a) / np.std(a) * np.sqrt(len(a))) if np.std(a) > 0 else 0.0,
        mdd=float(np.min(dd)) if len(dd) > 0 else 0.0,
        pf=float(w / lo) if lo > 0 else float("inf"),
        max_consec_loss=max_streak,
    )


# ── Data Loading ─────────────────────────────────────────

def load_ohlcv_10(period: str = "365d") -> dict[str, pd.DataFrame]:
    """Load 10 coins with OHLCV + indicators + microstructure + Binance metrics."""
    import os
    from src.data.crawlers.crypto_ohlcv import fetch_ohlcv, resample_to_4h, add_technical_indicators
    from src.data.crawlers.microstructure_rollup import add_microstructure_rollup

    data = {}
    for coin, yahoo_sym in YAHOO_MAP.items():
        df = fetch_ohlcv(coin, yahoo_sym, period=period, interval="1h")
        if df.empty:
            continue
        df = resample_to_4h(df)
        df = add_technical_indicators(df)
        df = add_microstructure_rollup(df)

        # Binance metrics
        metrics_dir = f"data/raw/binance_public/metrics/{COINS_10[coin]}"
        if os.path.exists(metrics_dir):
            csvs = sorted([f for f in os.listdir(metrics_dir) if f.endswith(".csv")])
            if csvs:
                dfs_m = []
                for f in csvs:
                    try:
                        dfs_m.append(pd.read_csv(os.path.join(metrics_dir, f)))
                    except Exception:
                        continue
                if dfs_m:
                    merged = pd.concat(dfs_m, ignore_index=True)
                    merged["create_time"] = pd.to_datetime(merged["create_time"])
                    merged = merged.sort_values("create_time").set_index("create_time")
                    merged = merged.drop(columns=["symbol"], errors="ignore")
                    m4h = merged.resample("4h").last().dropna(how="all")
                    if df.index.tz is not None and m4h.index.tz is None:
                        m4h.index = m4h.index.tz_localize(df.index.tz)
                    elif df.index.tz is None and m4h.index.tz is not None:
                        m4h.index = m4h.index.tz_localize(None)
                    ma = m4h.reindex(df.index, method="ffill")
                    if "sum_open_interest_value" in ma.columns:
                        oi = ma["sum_open_interest_value"].astype(float)
                        df["oi_zscore"] = (oi - oi.rolling(48, min_periods=12).mean()) / \
                            oi.rolling(48, min_periods=12).std()

        data[coin] = df
    return data


def split_is_oos(data: dict, ratio: float = 0.70):
    """Split data into IS (first ratio%) and OOS (rest)."""
    is_d, oos_d = {}, {}
    for coin, df in data.items():
        cut = int(len(df) * ratio)
        is_d[coin] = df.iloc[:cut].copy()
        oos_d[coin] = df.iloc[cut:].copy()
    return is_d, oos_d
