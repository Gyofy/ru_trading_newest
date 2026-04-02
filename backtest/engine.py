"""BacktestEngine — bar-by-bar replay of 6 strategies on historical data.

Strict point-in-time: no lookahead bias.
Fee-aware PnL with slippage.
Compatible with StrategySolver trade_context.jsonl format.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from backtest.data_loader import BacktestDataHub

logger = logging.getLogger("backtest.engine")

# ── Cost Model ──────────────────────────────────────────
ENTRY_FEE_RATE = 0.0005   # taker 0.05%
EXIT_FEE_RATE = 0.0005    # taker 0.05%
ENTRY_SLIP = 0.0005       # 0.05% slippage
EXIT_SLIP = 0.0005        # 0.05% slippage
ROUNDTRIP_COST = ENTRY_FEE_RATE + EXIT_FEE_RATE + ENTRY_SLIP + EXIT_SLIP  # 0.20%


@dataclass
class SimPosition:
    """One open simulated position."""
    trade_id: str
    coin: str
    strategy: str
    side: str          # "BUY" or "SELL"
    entry_price: float
    entry_ts: pd.Timestamp
    qty: float
    notional: float
    leverage: int
    sl_price: float
    tp_price: float
    ttl_bars: int
    bars_held: int = 0
    mfe_price: float = 0.0
    mae_price: float = 0.0
    signal_extra: dict = field(default_factory=dict)
    entry_atr: float = 0.0


# ── Strategy Evaluation Functions ────────────────────────

def eval_cvd_extreme(
    hub: BacktestDataHub, coin: str, ts: pd.Timestamp, cfg: dict
) -> dict | None:
    """CVD Extreme: quantile + z-score dual condition → counter-trend."""
    extra = cfg.get("extra", {})
    roll_window = extra.get("cvd_roll_window", 240)
    quantile = extra.get("cvd_quantile", 0.90)
    sigma_mult = extra.get("cvd_sigma_mult", 1.2)
    volume_spike_mult = extra.get("volume_spike_mult", 1.0)

    df = hub.get_ohlcv(coin, "1m", limit=roll_window + 50, current_ts=ts)
    if df is None or len(df) < roll_window + 10:
        return None

    cvd = hub.compute_cvd(df)
    cvd_delta = cvd.diff()

    if len(cvd_delta) < roll_window:
        return None

    window_data = cvd_delta.iloc[-roll_window:]
    current = float(cvd_delta.iloc[-1])
    mean = float(window_data.mean())
    std = float(window_data.std())
    if std < 1e-10:
        return None

    z_score = (current - mean) / std

    q_high = float(window_data.quantile(quantile))
    q_low = float(window_data.quantile(1 - quantile))

    # Volume confirmation
    vol = df["volume"].iloc[-1]
    vol_mean = df["volume"].iloc[-roll_window:-1].mean()
    if vol_mean > 0 and vol < vol_mean * volume_spike_mult:
        return None

    # Dual condition
    side = None
    if current > q_high and z_score > sigma_mult:
        side = "SELL"  # extreme buying → sell reversal
    elif current < q_low and z_score < -sigma_mult:
        side = "BUY"   # extreme selling → buy reversal

    if side is None:
        return None

    strength = abs(z_score)
    return {
        "side": side,
        "confidence": min(strength / 5.0, 1.0),
        "extra": {
            "trigger": "cvd_extreme",
            "cvd_z_score": z_score,
            "cvd_value": current,
            "strength": strength,
            "signal_strength": strength,
        },
    }


def eval_liquidation_fade(
    hub: BacktestDataHub, coin: str, ts: pd.Timestamp, cfg: dict
) -> dict | None:
    """Liquidation Fade: OI drop + volume spike → counter-cascade."""
    extra = cfg.get("extra", {})
    oi_sigma = extra.get("oi_sigma_threshold", 0.5)
    taker_mult = extra.get("taker_spike_mult", 0.8)

    df_5m = hub.get_ohlcv(coin, "5m", limit=110, current_ts=ts)
    if df_5m is None or len(df_5m) < 55:
        return None

    # Volume spike check
    vol = df_5m["volume"].values
    vol_avg = float(np.mean(vol[-51:-1]))
    vol_current = float(vol[-1])
    if vol_avg <= 0:
        return None
    taker_ratio = vol_current / vol_avg
    if taker_ratio < taker_mult:
        return None

    # Recent price move (last 6 bars of 5m = 30min)
    close = df_5m["close"].values
    recent_return = close[-1] / close[-6] - 1 if close[-6] > 0 else 0
    min_move = extra.get("min_move_pct", 0.005)
    if abs(recent_return) < min_move:
        return None

    # OI check (real data or proxy)
    oi_now = hub.get_open_interest(coin, ts)
    oi_drop_est = 0.0
    if oi_now is not None:
        # Try to get OI from ~30min ago
        oi_prev_ts = ts - pd.Timedelta(minutes=30)
        oi_prev = hub.get_open_interest(coin, oi_prev_ts)
        if oi_prev and oi_prev > 0:
            oi_change_pct = (oi_now - oi_prev) / oi_prev
            if oi_change_pct < -0.001:
                oi_drop_est = abs(oi_change_pct) * 100 * taker_ratio
            else:
                oi_drop_est = taker_ratio * abs(recent_return) * 100
        else:
            oi_drop_est = taker_ratio * abs(recent_return) * 100
    else:
        # Proxy: volume × price move
        oi_drop_est = taker_ratio * abs(recent_return) * 100

    if oi_drop_est < oi_sigma:
        return None

    # Fade the cascade
    side = "BUY" if recent_return < 0 else "SELL"

    return {
        "side": side,
        "confidence": min(taker_ratio / 4.0, 1.0),
        "extra": {
            "trigger": "liq_cascade",
            "strength": oi_drop_est,
            "signal_strength": oi_drop_est,
            "ofi_value": taker_ratio,
            "recent_return_pct": recent_return * 100,
        },
    }


def eval_vwap_reversion(
    hub: BacktestDataHub, coin: str, ts: pd.Timestamp, cfg: dict
) -> dict | None:
    """VWAP Exhaustion: z-score deviation from VWAP + volume decay."""
    extra = cfg.get("extra", {})
    vwap_window = extra.get("vwap_window", 60)
    sigma_mult = extra.get("sigma_mult", 1.2)
    vol_decay_threshold = extra.get("vol_decay_threshold", 0.85)

    df = hub.get_ohlcv(coin, "1m", limit=vwap_window + 20, current_ts=ts)
    if df is None or len(df) < vwap_window + 5:
        return None

    vwap = hub.compute_vwap(df, window=vwap_window)
    if vwap.isna().all():
        return None

    close_now = float(df["close"].iloc[-1])
    vwap_now = float(vwap.iloc[-1])
    if vwap_now <= 0 or np.isnan(vwap_now):
        return None

    # Z-score of VWAP deviation
    dev_series = (df["close"] - vwap) / vwap
    dev_series = dev_series.dropna()
    if len(dev_series) < 10:
        return None

    dev_std = float(dev_series.iloc[-vwap_window:].std()) if len(dev_series) >= vwap_window else float(dev_series.std())
    if dev_std < 1e-10:
        return None

    current_dev = float(dev_series.iloc[-1])
    z_score = current_dev / dev_std

    if abs(z_score) < sigma_mult:
        return None

    # Volume exhaustion filter
    vol_recent = df["volume"].iloc[-20:]
    if len(vol_recent) < 10:
        return None
    vol_peak = float(vol_recent.iloc[-10:-1].max())
    vol_current = float(vol_recent.iloc[-1])
    if vol_peak > 0:
        vol_ratio = vol_current / vol_peak
        if vol_ratio > vol_decay_threshold:
            return None
    else:
        return None

    # Counter-trend
    side = "SELL" if z_score > 0 else "BUY"

    return {
        "side": side,
        "confidence": min(abs(z_score) / 4.0, 1.0),
        "extra": {
            "trigger": "vwap_vol_exhaustion",
            "z_score": z_score,
            "vwap_dev_pct": current_dev * 100,
            "vol_decay_ratio": vol_ratio,
            "signal_strength": abs(z_score),
        },
    }


def eval_funding_arb(
    hub: BacktestDataHub, coin: str, ts: pd.Timestamp, cfg: dict
) -> dict | None:
    """MTF Momentum: 1m+15m+1h EMA alignment."""
    extra = cfg.get("extra", {})
    fast_ema = extra.get("fast_ema", 8)
    slow_ema = extra.get("slow_ema", 21)
    min_body_pct = extra.get("min_body_pct", 0.0005)

    df_1m = hub.get_ohlcv(coin, "1m", limit=50, current_ts=ts)
    df_15m = hub.get_ohlcv(coin, "15m", limit=30, current_ts=ts)
    df_1h = hub.get_ohlcv(coin, "1h", limit=20, current_ts=ts)

    if df_1m is None or len(df_1m) < 30:
        return None
    if df_15m is None or len(df_15m) < 22:
        return None
    if df_1h is None or len(df_1h) < 10:
        return None

    def ema_direction(df: pd.DataFrame, fast: int, slow: int) -> int:
        close = df["close"]
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        if len(ema_fast) < 2:
            return 0
        if ema_fast.iloc[-1] > ema_slow.iloc[-1] and ema_fast.iloc[-2] > ema_slow.iloc[-2]:
            return 1
        if ema_fast.iloc[-1] < ema_slow.iloc[-1] and ema_fast.iloc[-2] < ema_slow.iloc[-2]:
            return -1
        return 0

    dir_1m = ema_direction(df_1m, fast_ema, slow_ema)
    dir_15m = ema_direction(df_15m, fast_ema, slow_ema)
    dir_1h = ema_direction(df_1h, fast_ema, slow_ema)

    # ALL three must align
    if dir_1m == 0 or dir_15m == 0 or dir_1h == 0:
        return None
    if dir_1m != dir_15m or dir_15m != dir_1h:
        return None

    # Body confirmation
    close_now = float(df_1m["close"].iloc[-1])
    open_now = float(df_1m["open"].iloc[-1])
    body_pct = (close_now - open_now) / open_now if open_now > 0 else 0

    if dir_1m == 1 and body_pct < min_body_pct:
        return None
    if dir_1m == -1 and body_pct > -min_body_pct:
        return None

    side = "BUY" if dir_1m == 1 else "SELL"
    strength = abs(body_pct) * 100

    return {
        "side": side,
        "confidence": min(strength / 0.5, 1.0),
        "extra": {
            "trigger": "mtf_alignment",
            "dir_1m": dir_1m,
            "dir_15m": dir_15m,
            "dir_1h": dir_1h,
            "body_pct": body_pct,
            "signal_strength": strength,
        },
    }


def eval_volume_impulse(
    hub: BacktestDataHub, coin: str, ts: pd.Timestamp, cfg: dict
) -> dict | None:
    """Volume Impulse: extreme volume spike + directional body."""
    extra = cfg.get("extra", {})
    volume_mult = extra.get("volume_mult", 2.5)
    min_body_pct = extra.get("min_body_pct", 0.001)
    lookback = extra.get("lookback", 50)

    df = hub.get_ohlcv(coin, "1m", limit=lookback + 10, current_ts=ts)
    if df is None or len(df) < lookback + 1:
        return None

    vol = df["volume"].values
    vol_avg = float(np.mean(vol[-lookback - 1:-1]))
    vol_current = float(vol[-1])
    if vol_avg <= 0:
        return None

    vol_ratio = vol_current / vol_avg
    if vol_ratio < volume_mult:
        return None

    # Body check
    close_now = float(df["close"].iloc[-1])
    open_now = float(df["open"].iloc[-1])
    body_pct = (close_now - open_now) / open_now if open_now > 0 else 0
    if abs(body_pct) < min_body_pct:
        return None

    if body_pct > 0:
        side = "BUY"
        sl_anchor = float(df["low"].iloc[-1])
    else:
        side = "SELL"
        sl_anchor = float(df["high"].iloc[-1])

    # CVD alignment (logged, not filtered)
    cvd = hub.compute_cvd(df)
    cvd_delta = cvd.diff()
    cvd_aligned = False
    if len(cvd_delta) >= 2:
        cvd_aligned = (body_pct > 0 and cvd_delta.iloc[-1] > 0) or (body_pct < 0 and cvd_delta.iloc[-1] < 0)

    return {
        "side": side,
        "confidence": min(vol_ratio / 10.0, 1.0),
        "extra": {
            "trigger": "volume_impulse",
            "vol_ratio": vol_ratio,
            "body_pct": body_pct,
            "sl_anchor": sl_anchor,
            "cvd_aligned": cvd_aligned,
            "signal_strength": vol_ratio,
        },
    }


def eval_oi_divergence(
    hub: BacktestDataHub,
    coin: str,
    ts: pd.Timestamp,
    cfg: dict,
    oi_history: dict[str, list],
) -> dict | None:
    """OI Divergence: 4 price-OI patterns."""
    extra = cfg.get("extra", {})
    price_lookback = extra.get("price_lookback", 12)
    min_price_move = extra.get("min_price_move_pct", 0.001)
    oi_threshold = extra.get("oi_change_threshold", 0.002)

    # OI reading
    oi_now = hub.get_open_interest(coin, ts)
    if oi_now is None:
        return None

    key = coin
    if key not in oi_history:
        oi_history[key] = []
    oi_history[key].append(oi_now)
    if len(oi_history[key]) > 20:
        oi_history[key] = oi_history[key][-20:]

    if len(oi_history[key]) < 3:
        return None

    oi_recent = oi_history[key][-3:]
    oi_change_pct = (oi_recent[-1] - oi_recent[0]) / oi_recent[0] if oi_recent[0] > 0 else 0

    # Price trend from 5m data
    df_5m = hub.get_ohlcv(coin, "5m", limit=price_lookback + 5, current_ts=ts)
    if df_5m is None or len(df_5m) < price_lookback:
        return None

    close = df_5m["close"].values
    price_change_pct = (close[-1] - close[-price_lookback]) / close[-price_lookback] if close[-price_lookback] > 0 else 0

    if abs(price_change_pct) < min_price_move or abs(oi_change_pct) < oi_threshold:
        return None

    # Divergence matrix
    side = None
    trigger = ""
    if price_change_pct > 0 and oi_change_pct < -oi_threshold:
        side = "SELL"
        trigger = "price_up_oi_down"
    elif price_change_pct < 0 and oi_change_pct < -oi_threshold:
        side = "BUY"
        trigger = "price_down_oi_down"
    elif price_change_pct > 0 and oi_change_pct > oi_threshold:
        side = "BUY"
        trigger = "price_up_oi_up"
    elif price_change_pct < 0 and oi_change_pct > oi_threshold:
        side = "SELL"
        trigger = "price_down_oi_up"

    if side is None:
        return None

    return {
        "side": side,
        "confidence": min(abs(oi_change_pct) / 0.02, 1.0),
        "extra": {
            "trigger": trigger,
            "price_change_pct": price_change_pct * 100,
            "oi_change_pct": oi_change_pct * 100,
            "oi_value": oi_now,
            "signal_strength": abs(oi_change_pct) * 100,
        },
    }


# ── Barrier Computation ──────────────────────────────────

def compute_barriers(
    strategy: str, side: str, price: float, atr: float, cfg: dict
) -> tuple[float, float]:
    """Compute SL/TP prices per strategy barrier logic.

    Returns (sl_price, tp_price).
    """
    extra = cfg.get("extra", {})
    sl_mult = cfg.get("sl_atr_mult", 4.0)

    if strategy in ("cvd_extreme", "vwap_reversion", "funding_arb"):
        # Scale 1m ATR → 15m equivalent
        atr_scaled = atr * math.sqrt(15)
        sl_floor_pct = 0.004  # 0.40%
        sl_dist = max(atr_scaled * sl_mult, price * sl_floor_pct)
        tp_rr = extra.get("tp_rr", 3.0)
        tp_dist = sl_dist * tp_rr

    elif strategy == "liquidation_fade":
        # Scale 1m ATR → 5m equivalent
        atr_scaled = atr * math.sqrt(5)
        sl_floor_pct = 0.004
        sl_dist = max(atr_scaled * sl_mult, price * sl_floor_pct)
        tp_mult = extra.get("tp_atr_mult", 7.0)
        tp_dist = max(atr_scaled * tp_mult, sl_dist * 1.5)

    elif strategy == "volume_impulse":
        # SL from impulse bar extremity (passed via signal.extra.sl_anchor)
        # Fallback to ATR-based
        atr_scaled = atr * math.sqrt(15)
        sl_dist = max(atr_scaled * 2.0, price * 0.004)
        tp_rr = extra.get("tp_rr", 4.0)
        tp_dist = sl_dist * tp_rr

    elif strategy == "oi_divergence":
        # Scale 1m ATR → 5m equivalent
        atr_scaled = atr * math.sqrt(5)
        sl_floor_pct = 0.005  # 0.50% floor for 5m strategy
        sl_dist = max(atr_scaled * sl_mult, price * sl_floor_pct)
        tp_rr = extra.get("tp_rr", 3.0)
        tp_dist = sl_dist * tp_rr

    else:
        atr_scaled = atr * math.sqrt(15)
        sl_dist = max(atr_scaled * sl_mult, price * 0.004)
        tp_rr = extra.get("tp_rr", 2.5)
        tp_dist = sl_dist * tp_rr

    # SL floor: max(fee × 2.5, 0.45%) — prevent SL < roundtrip fee
    min_sl_dist = price * 0.0045  # 0.45%
    sl_dist = max(sl_dist, min_sl_dist)

    if side == "BUY":
        return (price - sl_dist, price + tp_dist)
    else:
        return (price + sl_dist, price - tp_dist)


# ── Volume Impulse special barrier ─────────────────────

def compute_barriers_volume_impulse(
    side: str, price: float, atr: float, cfg: dict, sl_anchor: float
) -> tuple[float, float]:
    """Volume Impulse uses impulse bar extremity as SL anchor."""
    extra = cfg.get("extra", {})
    tp_rr = extra.get("tp_rr", 4.0)

    sl_dist = abs(price - sl_anchor) if sl_anchor > 0 else atr * math.sqrt(15) * 2.0
    sl_dist = max(sl_dist, price * 0.004)  # 0.40% floor
    sl_dist = max(sl_dist, price * 0.0045)  # SL floor
    tp_dist = sl_dist * tp_rr

    if side == "BUY":
        return (price - sl_dist, price + tp_dist)
    else:
        return (price + sl_dist, price - tp_dist)


# ── Strategy Registry ────────────────────────────────────

STRATEGY_EVAL_FNS: dict[str, Callable] = {
    "cvd_extreme": eval_cvd_extreme,
    "liquidation_fade": eval_liquidation_fade,
    "vwap_reversion": eval_vwap_reversion,
    "funding_arb": eval_funding_arb,
    "volume_impulse": eval_volume_impulse,
    # oi_divergence handled separately (needs oi_history state)
}

# Minimum bars warmup per strategy
WARMUP_BARS = {
    "cvd_extreme": 300,      # 240 + 50 + buffer
    "liquidation_fade": 600,  # needs 5m data = 100×5=500 1m bars + buffer
    "vwap_reversion": 100,
    "funding_arb": 100,       # needs 1h data built up
    "volume_impulse": 70,
    "oi_divergence": 600,     # needs 5m data + OI history
}


class BacktestEngine:
    """Bar-by-bar replay engine for 6 strategies."""

    def __init__(
        self,
        data_hub: BacktestDataHub,
        config: dict,
        trade_coins: list[str],
    ):
        self._hub = data_hub
        self._config = config
        self._trade_coins = trade_coins
        self._open_positions: list[SimPosition] = []
        self._closed_trades: list[dict] = []
        self._total_trade_count = 0

        # Per-strategy state
        self._oi_history: dict[str, list] = {}  # for oi_divergence
        self._last_entry_ts: dict[str, pd.Timestamp] = {}  # cooldown tracking
        self._daily_trade_count: dict[str, int] = {}
        self._current_date: str = ""

        # Active strategy configs
        self._strategy_configs: dict[str, dict] = {}
        for name, scfg in config.get("strategies", {}).items():
            if scfg.get("enabled", False):
                self._strategy_configs[name] = dict(scfg)

        logger.info(
            f"Engine initialized: {len(self._strategy_configs)} strategies, "
            f"{len(trade_coins)} coins"
        )

    def run(self, progress_interval: int = 5000) -> list[dict]:
        """Main backtest loop. Returns list of closed trade dicts."""
        all_ts = self._hub.get_all_1m_timestamps(self._trade_coins)
        warmup = max(WARMUP_BARS.values())

        logger.info(f"Replaying {len(all_ts)} bars (warmup={warmup})")

        for i, ts in enumerate(all_ts):
            if i < warmup:
                continue

            # Daily reset
            date_str = str(ts.date())
            if date_str != self._current_date:
                self._current_date = date_str
                self._daily_trade_count.clear()

            # Phase 1: Check exits for all open positions
            self._check_all_exits(ts)

            # Phase 2: Evaluate strategies for new signals
            self._evaluate_strategies(ts)

            # Progress logging
            if (i - warmup) % progress_interval == 0 and i > warmup:
                logger.info(
                    f"  Bar {i}/{len(all_ts)} | "
                    f"Open: {len(self._open_positions)} | "
                    f"Closed: {len(self._closed_trades)} | "
                    f"Date: {date_str}"
                )

        # Close remaining positions at last bar
        if self._open_positions:
            last_ts = all_ts[-1]
            for pos in list(self._open_positions):
                self._close_position(pos, "BACKTEST_END", last_ts)

        logger.info(
            f"Backtest complete: {len(self._closed_trades)} trades closed"
        )
        return self._closed_trades

    def _check_all_exits(self, ts: pd.Timestamp):
        """Check SL/TP/TTL for all open positions."""
        for pos in list(self._open_positions):
            bar = self._hub.get_bar(pos.coin, "1m", ts)
            if bar is None:
                continue

            pos.bars_held += 1
            h, l = bar["high"], bar["low"]

            # Update MFE/MAE
            if pos.side == "BUY":
                if pos.mfe_price == 0:
                    pos.mfe_price = h
                    pos.mae_price = l
                else:
                    pos.mfe_price = max(pos.mfe_price, h)
                    pos.mae_price = min(pos.mae_price, l)
            else:
                if pos.mfe_price == 0:
                    pos.mfe_price = l
                    pos.mae_price = h
                else:
                    pos.mfe_price = min(pos.mfe_price, l)
                    pos.mae_price = max(pos.mae_price, h)

            # SL/TP check — conservative: if both hit in same bar, SL wins
            exit_reason = None
            if pos.side == "BUY":
                if l <= pos.sl_price:
                    exit_reason = "SL_HIT"
                elif h >= pos.tp_price:
                    exit_reason = "TP_HIT"
            else:
                if h >= pos.sl_price:
                    exit_reason = "SL_HIT"
                elif l <= pos.tp_price:
                    exit_reason = "TP_HIT"

            # TTL
            if exit_reason is None and pos.ttl_bars > 0 and pos.bars_held >= pos.ttl_bars:
                exit_reason = "TIME_STOP"

            if exit_reason:
                self._close_position(pos, exit_reason, ts)

    def _evaluate_strategies(self, ts: pd.Timestamp):
        """Evaluate all strategies for new entry signals."""
        for strat_name, scfg in self._strategy_configs.items():
            max_pos = scfg.get("max_positions", 15)
            open_count = sum(1 for p in self._open_positions if p.strategy == strat_name)
            if open_count >= max_pos:
                continue

            # Daily trade cap
            daily_max = scfg.get("extra", {}).get("max_daily_trades", 100)
            daily_count = self._daily_trade_count.get(strat_name, 0)
            if daily_count >= daily_max:
                continue

            # Cooldown
            cooldown = scfg.get("extra", {}).get("cooldown_sec", 60)
            cooldown_key = strat_name
            if cooldown_key in self._last_entry_ts:
                elapsed = (ts - self._last_entry_ts[cooldown_key]).total_seconds()
                if elapsed < cooldown:
                    continue

            for coin in self._trade_coins:
                # One position per coin across all strategies
                if any(p.coin == coin for p in self._open_positions):
                    continue

                signal = self._eval_strategy(strat_name, coin, ts, scfg)
                if signal is not None:
                    self._open_position(strat_name, coin, signal, ts, scfg)
                    self._last_entry_ts[cooldown_key] = ts
                    self._daily_trade_count[strat_name] = daily_count + 1
                    break  # one entry per strategy per bar

    def _eval_strategy(
        self, name: str, coin: str, ts: pd.Timestamp, cfg: dict
    ) -> dict | None:
        """Dispatch to strategy-specific evaluation."""
        try:
            if name == "oi_divergence":
                return eval_oi_divergence(self._hub, coin, ts, cfg, self._oi_history)
            elif name in STRATEGY_EVAL_FNS:
                return STRATEGY_EVAL_FNS[name](self._hub, coin, ts, cfg)
            return None
        except Exception as e:
            logger.debug(f"Strategy {name} eval error for {coin}: {e}")
            return None

    def _open_position(
        self, strategy: str, coin: str, signal: dict, ts: pd.Timestamp, cfg: dict
    ):
        """Simulate entry fill."""
        df = self._hub.get_ohlcv(coin, "1m", limit=100, current_ts=ts)
        if df is None or len(df) < 15:
            return

        close_price = float(df["close"].iloc[-1])
        atr = self._hub.compute_atr(df)
        if atr <= 0:
            return

        side = signal["side"]

        # Apply entry slippage
        if side == "BUY":
            fill_price = close_price * (1 + ENTRY_SLIP)
        else:
            fill_price = close_price * (1 - ENTRY_SLIP)

        # Compute barriers
        sl_anchor = signal.get("extra", {}).get("sl_anchor", 0)
        if strategy == "volume_impulse" and sl_anchor > 0:
            sl_price, tp_price = compute_barriers_volume_impulse(
                side, fill_price, atr, cfg, sl_anchor
            )
        else:
            sl_price, tp_price = compute_barriers(
                strategy, side, fill_price, atr, cfg
            )

        # Position sizing: simplified allocation / max_positions
        allocation = cfg.get("allocation_usdt", 2000)
        leverage = cfg.get("leverage", 3)
        max_pos = cfg.get("max_positions", 15)
        notional = allocation * leverage / max_pos
        qty = notional / fill_price

        # TTL
        ttl_bars = 120  # 2 hours for 1m bars

        pos = SimPosition(
            trade_id=str(uuid.uuid4())[:8],
            coin=coin,
            strategy=strategy,
            side=side,
            entry_price=fill_price,
            entry_ts=ts,
            qty=qty,
            notional=notional,
            leverage=leverage,
            sl_price=sl_price,
            tp_price=tp_price,
            ttl_bars=ttl_bars,
            signal_extra=signal.get("extra", {}),
            entry_atr=atr,
        )
        self._open_positions.append(pos)
        self._total_trade_count += 1

    def _close_position(self, pos: SimPosition, exit_reason: str, ts: pd.Timestamp):
        """Close position and record trade."""
        if pos not in self._open_positions:
            return
        self._open_positions.remove(pos)

        # Exit price
        if exit_reason == "SL_HIT":
            exit_price = pos.sl_price
        elif exit_reason == "TP_HIT":
            exit_price = pos.tp_price
        else:
            bar = self._hub.get_bar(pos.coin, "1m", ts)
            exit_price = bar["close"] if bar else pos.entry_price

        # Apply exit slippage
        if pos.side == "BUY":
            exit_price *= (1 - EXIT_SLIP)
        else:
            exit_price *= (1 + EXIT_SLIP)

        # PnL calculation
        if pos.side == "BUY":
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
            mfe_pct = (pos.mfe_price - pos.entry_price) / pos.entry_price if pos.mfe_price > 0 else 0
            mae_pct = (pos.mae_price - pos.entry_price) / pos.entry_price if pos.mae_price > 0 else 0
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price
            mfe_pct = (pos.entry_price - pos.mfe_price) / pos.entry_price if pos.mfe_price > 0 else 0
            mae_pct = (pos.entry_price - pos.mae_price) / pos.entry_price if pos.mae_price > 0 else 0

        pnl_usdt = pnl_pct * pos.notional
        fee_usdt = pos.notional * ROUNDTRIP_COST
        pnl_net_usdt = pnl_usdt - fee_usdt
        pnl_net_pct = pnl_net_usdt / (pos.notional / pos.leverage) if pos.leverage > 0 else 0

        sl_dist_pct = abs(pos.sl_price - pos.entry_price) / pos.entry_price * 100
        tp_dist_pct = abs(pos.tp_price - pos.entry_price) / pos.entry_price * 100

        trade = {
            "trade_id": pos.trade_id,
            "ts_entry": pos.entry_ts.isoformat(),
            "ts_exit": ts.isoformat(),
            "coin": pos.coin,
            "strategy": pos.strategy,
            "side": pos.side,
            "entry_price": round(pos.entry_price, 6),
            "exit_price": round(exit_price, 6),
            "sl_price": round(pos.sl_price, 6),
            "tp_price": round(pos.tp_price, 6),
            "sl_distance_pct": round(sl_dist_pct, 4),
            "tp_distance_pct": round(tp_dist_pct, 4),
            "exit_reason": exit_reason,
            "pnl_usdt": round(pnl_usdt, 4),
            "pnl_pct": round(pnl_pct, 6),
            "pnl_net_usdt": round(pnl_net_usdt, 4),
            "pnl_net_pct": round(pnl_net_pct, 6),
            "fee_usdt": round(fee_usdt, 4),
            "bars_held": pos.bars_held,
            "mfe_pct": round(mfe_pct, 6),
            "mae_pct": round(mae_pct, 6),
            "notional_usdt": round(pos.notional, 2),
            "leverage": pos.leverage,
            "entry_atr": round(pos.entry_atr, 6),
            **{k: v for k, v in pos.signal_extra.items() if isinstance(v, (int, float, str, bool))},
        }
        self._closed_trades.append(trade)
