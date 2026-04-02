"""HTF (Higher Timeframe) Backtest Engine — 15m + 1h dual-timeframe strategies.

Key difference from 1m engine:
- SL/TP checked on 1m bars (intrabar resolution) even for 15m/1h signals
- ATR is native (no sqrt scaling needed)
- Holding periods: 15m→12h, 1h→48h
- Fee impact: 39% (15m) and 20% (1h) of ATR — tradeable territory
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.data_loader import BacktestDataHub

logger = logging.getLogger("backtest.engine_htf")

# Cost model (same as live)
FEE_RATE = 0.20 / 100  # 0.20% roundtrip
ENTRY_SLIP = 0.0003     # 0.03% (less slip at HTF — not chasing 1m moves)
EXIT_SLIP = 0.0003


@dataclass
class Position:
    trade_id: str
    coin: str
    strategy: str
    timeframe: str
    side: str
    entry_price: float
    entry_ts: pd.Timestamp
    notional: float
    leverage: int
    sl_price: float
    tp_price: float
    ttl_bars: int       # in 1m bars
    bars_held: int = 0
    mfe_price: float = 0.0
    mae_price: float = 0.0
    entry_atr: float = 0.0
    signal_extra: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════
#  15-MINUTE STRATEGIES
# ══════════════════════════════════════════════════════════════

def eval_tsmom_15m(hub: BacktestDataHub, coin: str, ts: pd.Timestamp, cfg: dict) -> dict | None:
    """Time-Series Momentum on 15m: 4h lookback direction + RSI filter."""
    df_15m = hub.get_ohlcv(coin, "15m", limit=50, current_ts=ts)
    if df_15m is None or len(df_15m) < 30:
        return None

    close = df_15m["close"].values

    # 4h momentum (16 bars of 15m = 4 hours)
    if len(close) < 17:
        return None
    mom_4h = (close[-1] - close[-16]) / close[-16]

    # 1h momentum (4 bars of 15m)
    mom_1h = (close[-1] - close[-4]) / close[-4]

    # Both must align
    if mom_4h > 0 and mom_1h > 0:
        side = "BUY"
    elif mom_4h < 0 and mom_1h < 0:
        side = "SELL"
    else:
        return None

    # RSI filter (14-bar)
    rsi = _compute_rsi(close, 14)
    if rsi is None:
        return None
    # Don't buy overbought, don't sell oversold
    if side == "BUY" and rsi > 70:
        return None
    if side == "SELL" and rsi < 30:
        return None

    # Minimum move filter
    min_move = cfg.get("min_move_pct", 0.003)
    if abs(mom_4h) < min_move:
        return None

    return {
        "side": side,
        "confidence": min(abs(mom_4h) / 0.02, 1.0),
        "extra": {
            "trigger": "tsmom_15m",
            "mom_4h": mom_4h,
            "mom_1h": mom_1h,
            "rsi": rsi,
            "signal_strength": abs(mom_4h) * 100,
        },
    }


def eval_vwap_mean_reversion_15m(hub: BacktestDataHub, coin: str, ts: pd.Timestamp, cfg: dict) -> dict | None:
    """VWAP Mean Reversion on 15m: extreme VWAP deviation + volume exhaustion."""
    df_15m = hub.get_ohlcv(coin, "15m", limit=40, current_ts=ts)
    if df_15m is None or len(df_15m) < 25:
        return None

    vwap = hub.compute_vwap(df_15m, window=20)
    if vwap.isna().all():
        return None

    close_now = float(df_15m["close"].iloc[-1])
    vwap_now = float(vwap.iloc[-1])
    if vwap_now <= 0 or np.isnan(vwap_now):
        return None

    dev_pct = (close_now - vwap_now) / vwap_now
    sigma = cfg.get("sigma_mult", 1.5)

    # Z-score of deviation
    dev_series = (df_15m["close"] - vwap) / vwap
    dev_series = dev_series.dropna()
    if len(dev_series) < 10:
        return None
    dev_std = float(dev_series.std())
    if dev_std < 1e-10:
        return None
    z_score = dev_pct / dev_std

    if abs(z_score) < sigma:
        return None

    # Volume declining (exhaustion)
    vol = df_15m["volume"].values
    vol_ma = np.mean(vol[-10:-1])
    vol_now = vol[-1]
    if vol_ma > 0 and vol_now > vol_ma * 0.8:
        return None  # Volume still strong → trend may continue

    side = "SELL" if z_score > 0 else "BUY"

    return {
        "side": side,
        "confidence": min(abs(z_score) / 4.0, 1.0),
        "extra": {
            "trigger": "vwap_mr_15m",
            "z_score": z_score,
            "vwap_dev_pct": dev_pct * 100,
            "signal_strength": abs(z_score),
        },
    }


def eval_cvd_divergence_15m(hub: BacktestDataHub, coin: str, ts: pd.Timestamp, cfg: dict) -> dict | None:
    """CVD Divergence on 15m: price-CVD divergence over 2h window."""
    df_15m = hub.get_ohlcv(coin, "15m", limit=30, current_ts=ts)
    if df_15m is None or len(df_15m) < 12:
        return None

    close = df_15m["close"].values
    cvd = hub.compute_cvd(df_15m)
    if len(cvd) < 12:
        return None

    # 2h lookback (8 bars of 15m)
    price_change = (close[-1] - close[-8]) / close[-8]
    cvd_change = cvd.iloc[-1] - cvd.iloc[-8]
    cvd_norm = cvd_change / (df_15m["volume"].iloc[-8:].sum() + 1e-10)

    min_price = cfg.get("min_price_move", 0.003)
    if abs(price_change) < min_price:
        return None

    # Divergence: price up but CVD down (hidden selling) or vice versa
    if price_change > min_price and cvd_norm < -0.05:
        side = "SELL"  # bearish divergence
        trigger = "price_up_cvd_down"
    elif price_change < -min_price and cvd_norm > 0.05:
        side = "BUY"   # bullish divergence
        trigger = "price_down_cvd_up"
    else:
        return None

    return {
        "side": side,
        "confidence": min(abs(cvd_norm) / 0.2, 1.0),
        "extra": {
            "trigger": trigger,
            "price_change_pct": price_change * 100,
            "cvd_norm": cvd_norm,
            "signal_strength": abs(cvd_norm) * 10,
        },
    }


# ══════════════════════════════════════════════════════════════
#  1-HOUR STRATEGIES
# ══════════════════════════════════════════════════════════════

def eval_tsmom_1h(hub: BacktestDataHub, coin: str, ts: pd.Timestamp, cfg: dict) -> dict | None:
    """TSMOM on 1h: 24h momentum direction + multi-filter."""
    df_1h = hub.get_ohlcv(coin, "1h", limit=50, current_ts=ts)
    if df_1h is None or len(df_1h) < 30:
        return None

    close = df_1h["close"].values

    # 24h momentum (24 bars)
    mom_24h = (close[-1] - close[-24]) / close[-24] if len(close) >= 25 else None
    if mom_24h is None:
        return None

    # 6h momentum (confirmation)
    mom_6h = (close[-1] - close[-6]) / close[-6]

    # Must align
    if mom_24h > 0 and mom_6h > 0:
        side = "BUY"
    elif mom_24h < 0 and mom_6h < 0:
        side = "SELL"
    else:
        return None

    # RSI filter
    rsi = _compute_rsi(close, 14)
    if rsi is None:
        return None
    if side == "BUY" and rsi > 72:
        return None
    if side == "SELL" and rsi < 28:
        return None

    # Minimum move
    min_move = cfg.get("min_move_pct", 0.005)
    if abs(mom_24h) < min_move:
        return None

    # Volume trend (increasing volume = conviction)
    vol = df_1h["volume"].values
    vol_recent = np.mean(vol[-6:])
    vol_prior = np.mean(vol[-12:-6])
    vol_trend = vol_recent / (vol_prior + 1e-10)

    return {
        "side": side,
        "confidence": min(abs(mom_24h) / 0.03, 1.0),
        "extra": {
            "trigger": "tsmom_1h",
            "mom_24h": mom_24h,
            "mom_6h": mom_6h,
            "rsi": rsi,
            "vol_trend": vol_trend,
            "signal_strength": abs(mom_24h) * 100,
        },
    }


def eval_relative_strength_1h(hub: BacktestDataHub, coin: str, ts: pd.Timestamp, cfg: dict,
                               btc_data: BacktestDataHub = None) -> dict | None:
    """Relative Strength: coin outperforming/underperforming BTC → trend-follow."""
    df_coin = hub.get_ohlcv(coin, "1h", limit=30, current_ts=ts)
    df_btc = hub.get_ohlcv("BTC", "1h", limit=30, current_ts=ts)

    if df_coin is None or df_btc is None or len(df_coin) < 25 or len(df_btc) < 25:
        return None

    close_coin = df_coin["close"].values
    close_btc = df_btc["close"].values

    # Use minimum length
    n = min(len(close_coin), len(close_btc))
    close_coin = close_coin[-n:]
    close_btc = close_btc[-n:]

    # 12h relative return (coin - BTC)
    lookback = min(12, n - 1)
    coin_ret = (close_coin[-1] - close_coin[-lookback]) / close_coin[-lookback]
    btc_ret = (close_btc[-1] - close_btc[-lookback]) / close_btc[-lookback]
    rel_strength = coin_ret - btc_ret

    # Need meaningful divergence
    min_rs = cfg.get("min_rs_pct", 0.01)
    if abs(rel_strength) < min_rs:
        return None

    # BTC direction as base bias
    if btc_ret > 0.002:
        btc_bias = "BUY"
    elif btc_ret < -0.002:
        btc_bias = "SELL"
    else:
        return None  # No clear BTC trend

    # Coin stronger than BTC + BTC bullish → LONG coin
    # Coin weaker than BTC + BTC bearish → SHORT coin
    if rel_strength > min_rs and btc_bias == "BUY":
        side = "BUY"
    elif rel_strength < -min_rs and btc_bias == "SELL":
        side = "SELL"
    else:
        return None

    return {
        "side": side,
        "confidence": min(abs(rel_strength) / 0.03, 1.0),
        "extra": {
            "trigger": "rel_strength_1h",
            "coin_ret": coin_ret * 100,
            "btc_ret": btc_ret * 100,
            "rel_strength": rel_strength * 100,
            "signal_strength": abs(rel_strength) * 100,
        },
    }


def eval_breakout_1h(hub: BacktestDataHub, coin: str, ts: pd.Timestamp, cfg: dict) -> dict | None:
    """Range Breakout on 1h: Donchian channel breakout with volume confirmation."""
    df_1h = hub.get_ohlcv(coin, "1h", limit=30, current_ts=ts)
    if df_1h is None or len(df_1h) < 22:
        return None

    close = df_1h["close"].values
    high = df_1h["high"].values
    low = df_1h["low"].values
    vol = df_1h["volume"].values

    # 20-bar Donchian channel
    dc_high = np.max(high[-21:-1])  # exclude current bar
    dc_low = np.min(low[-21:-1])
    dc_mid = (dc_high + dc_low) / 2

    current_close = close[-1]

    # Breakout detection
    if current_close > dc_high:
        side = "BUY"
    elif current_close < dc_low:
        side = "SELL"
    else:
        return None

    # Volume confirmation: current > 1.5x average
    vol_avg = np.mean(vol[-20:-1])
    vol_now = vol[-1]
    if vol_avg > 0 and vol_now < vol_avg * 1.3:
        return None

    # Channel width as volatility measure
    channel_width = (dc_high - dc_low) / dc_mid

    return {
        "side": side,
        "confidence": min(abs(current_close - dc_mid) / (dc_high - dc_low + 1e-10), 1.0),
        "extra": {
            "trigger": "breakout_1h",
            "dc_high": dc_high,
            "dc_low": dc_low,
            "channel_width": channel_width * 100,
            "vol_ratio": vol_now / (vol_avg + 1e-10),
            "signal_strength": channel_width * 100,
        },
    }


# ══════════════════════════════════════════════════════════════
#  BARRIER COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_htf_barriers(
    strategy: str, side: str, price: float, atr: float, cfg: dict
) -> tuple[float, float]:
    """Compute SL/TP for HTF strategies. ATR is already native timeframe."""
    sl_mult = cfg.get("sl_mult", 1.0)
    tp_mult = cfg.get("tp_mult", 2.0)

    sl_dist = atr * sl_mult
    tp_dist = atr * tp_mult

    # SL floor: max(0.45%, sl_dist)
    sl_dist = max(sl_dist, price * 0.0045)

    if side == "BUY":
        return (price - sl_dist, price + tp_dist)
    else:
        return (price + sl_dist, price - tp_dist)


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _compute_rsi(close: np.ndarray, period: int = 14) -> float | None:
    if len(close) < period + 1:
        return None
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ══════════════════════════════════════════════════════════════
#  ENGINE
# ══════════════════════════════════════════════════════════════

# Strategy registry: {name: (eval_fn, timeframe, ttl_minutes, default_cfg)}
STRATEGIES = {
    # 15m strategies
    "tsmom_15m": (eval_tsmom_15m, "15m", 720, {"sl_mult": 1.0, "tp_mult": 2.0, "min_move_pct": 0.003}),
    "vwap_mr_15m": (eval_vwap_mean_reversion_15m, "15m", 720, {"sl_mult": 1.0, "tp_mult": 2.0, "sigma_mult": 1.5}),
    "cvd_div_15m": (eval_cvd_divergence_15m, "15m", 720, {"sl_mult": 1.0, "tp_mult": 2.0, "min_price_move": 0.003}),
    # 1h strategies
    "tsmom_1h": (eval_tsmom_1h, "1h", 2880, {"sl_mult": 1.0, "tp_mult": 2.0, "min_move_pct": 0.005}),
    "rel_strength_1h": (eval_relative_strength_1h, "1h", 2880, {"sl_mult": 1.0, "tp_mult": 2.0, "min_rs_pct": 0.01}),
    "breakout_1h": (eval_breakout_1h, "1h", 2880, {"sl_mult": 1.0, "tp_mult": 2.0}),
}


class HTFBacktestEngine:
    """Dual-timeframe backtest engine (15m + 1h)."""

    def __init__(
        self,
        hub: BacktestDataHub,
        coins: list[str],
        strategies: dict[str, dict] | None = None,
        capital: float = 5000,
        leverage: int = 3,
        max_positions: int = 4,
    ):
        self._hub = hub
        self._coins = coins
        self._capital = capital
        self._leverage = leverage
        self._max_pos = max_positions
        self._open: list[Position] = []
        self._closed: list[dict] = []

        # Strategy configs (override defaults)
        self._strat_cfgs = {}
        for name, (fn, tf, ttl, defaults) in STRATEGIES.items():
            cfg = dict(defaults)
            if strategies and name in strategies:
                cfg.update(strategies[name])
            self._strat_cfgs[name] = cfg

        # Cooldowns: {strategy:coin -> last_entry_ts}
        self._cooldowns: dict[str, pd.Timestamp] = {}

    def run(self, enabled_strategies: list[str] | None = None) -> list[dict]:
        """Main loop: iterate 1m bars, check exits, evaluate on 15m/1h boundaries."""
        all_ts = self._hub.get_all_1m_timestamps(self._coins)
        warmup = 1500  # 25 hours

        active_strats = enabled_strategies or list(STRATEGIES.keys())
        logger.info(f"HTF Backtest: {len(all_ts)} bars, {len(active_strats)} strategies, {len(self._coins)} coins")

        for i, ts in enumerate(all_ts):
            if i < warmup:
                continue

            # Phase 1: Check exits on 1m resolution
            self._check_exits(ts)

            # Phase 2: Evaluate strategies at their native bar boundaries
            minute = ts.minute
            hour = ts.hour

            # 15m strategies: evaluate at :00, :15, :30, :45
            if minute % 15 == 0:
                for name in active_strats:
                    if name not in STRATEGIES:
                        continue
                    _, tf, _, _ = STRATEGIES[name]
                    if tf == "15m":
                        self._evaluate(name, ts)

            # 1h strategies: evaluate at :00
            if minute == 0:
                for name in active_strats:
                    if name not in STRATEGIES:
                        continue
                    _, tf, _, _ = STRATEGIES[name]
                    if tf == "1h":
                        self._evaluate(name, ts)

            # Progress
            if i % 5000 == 0 and i > warmup:
                logger.info(f"  Bar {i}/{len(all_ts)} | Open: {len(self._open)} | Closed: {len(self._closed)}")

        # Close remaining
        if self._open:
            for pos in list(self._open):
                self._close(pos, "BACKTEST_END", all_ts[-1])

        logger.info(f"HTF Backtest complete: {len(self._closed)} trades")
        return self._closed

    def _check_exits(self, ts: pd.Timestamp):
        for pos in list(self._open):
            bar = self._hub.get_bar(pos.coin, "1m", ts)
            if bar is None:
                continue

            pos.bars_held += 1
            h, l = bar["high"], bar["low"]

            # MFE/MAE
            if pos.side == "BUY":
                pos.mfe_price = max(pos.mfe_price, h) if pos.mfe_price > 0 else h
                pos.mae_price = min(pos.mae_price, l) if pos.mae_price > 0 else l
                if l <= pos.sl_price:
                    self._close(pos, "SL_HIT", ts)
                elif h >= pos.tp_price:
                    self._close(pos, "TP_HIT", ts)
                elif pos.bars_held >= pos.ttl_bars:
                    self._close(pos, "TIME_STOP", ts)
            else:
                pos.mfe_price = min(pos.mfe_price, l) if pos.mfe_price > 0 else l
                pos.mae_price = max(pos.mae_price, h) if pos.mae_price > 0 else h
                if h >= pos.sl_price:
                    self._close(pos, "SL_HIT", ts)
                elif l <= pos.tp_price:
                    self._close(pos, "TP_HIT", ts)
                elif pos.bars_held >= pos.ttl_bars:
                    self._close(pos, "TIME_STOP", ts)

    def _evaluate(self, strat_name: str, ts: pd.Timestamp):
        if len(self._open) >= self._max_pos:
            return

        fn, tf, ttl, _ = STRATEGIES[strat_name]
        cfg = self._strat_cfgs[strat_name]

        for coin in self._coins:
            # One position per coin
            if any(p.coin == coin for p in self._open):
                continue

            # Cooldown (1 bar of native timeframe)
            cd_key = f"{strat_name}:{coin}"
            cd_minutes = 15 if tf == "15m" else 60
            if cd_key in self._cooldowns:
                if (ts - self._cooldowns[cd_key]).total_seconds() < cd_minutes * 60:
                    continue

            try:
                signal = fn(self._hub, coin, ts, cfg)
            except Exception:
                continue

            if signal is None:
                continue

            # Open position
            df = self._hub.get_ohlcv(coin, tf, limit=20, current_ts=ts)
            if df is None or len(df) < 15:
                continue

            atr = self._hub.compute_atr(df)
            if atr <= 0:
                continue

            price = float(df["close"].iloc[-1])
            side = signal["side"]

            # Slippage
            fill = price * (1 + ENTRY_SLIP) if side == "BUY" else price * (1 - ENTRY_SLIP)

            sl, tp = compute_htf_barriers(strat_name, side, fill, atr, cfg)

            notional = self._capital * 0.10 * self._leverage
            pos = Position(
                trade_id=str(uuid.uuid4())[:8],
                coin=coin,
                strategy=strat_name,
                timeframe=tf,
                side=side,
                entry_price=fill,
                entry_ts=ts,
                notional=notional,
                leverage=self._leverage,
                sl_price=sl,
                tp_price=tp,
                ttl_bars=ttl,
                entry_atr=atr,
                signal_extra=signal.get("extra", {}),
            )
            self._open.append(pos)
            self._cooldowns[cd_key] = ts
            break  # one entry per strategy per evaluation

    def _close(self, pos: Position, reason: str, ts: pd.Timestamp):
        if pos not in self._open:
            return
        self._open.remove(pos)

        if reason == "SL_HIT":
            exit_price = pos.sl_price
        elif reason == "TP_HIT":
            exit_price = pos.tp_price
        else:
            bar = self._hub.get_bar(pos.coin, "1m", ts)
            exit_price = bar["close"] if bar else pos.entry_price

        # Exit slippage
        if pos.side == "BUY":
            exit_price *= (1 - EXIT_SLIP)
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
            mfe_pct = (pos.mfe_price - pos.entry_price) / pos.entry_price if pos.mfe_price > 0 else 0
            mae_pct = (pos.mae_price - pos.entry_price) / pos.entry_price if pos.mae_price > 0 else 0
        else:
            exit_price *= (1 + EXIT_SLIP)
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price
            mfe_pct = (pos.entry_price - pos.mfe_price) / pos.entry_price if pos.mfe_price > 0 else 0
            mae_pct = (pos.entry_price - pos.mae_price) / pos.entry_price if pos.mae_price > 0 else 0

        fee = pos.notional * FEE_RATE
        pnl_gross = pnl_pct * pos.notional
        pnl_net = pnl_gross - fee

        self._closed.append({
            "trade_id": pos.trade_id,
            "ts_entry": pos.entry_ts.isoformat(),
            "ts_exit": ts.isoformat(),
            "coin": pos.coin,
            "strategy": pos.strategy,
            "timeframe": pos.timeframe,
            "side": pos.side,
            "entry_price": round(pos.entry_price, 6),
            "exit_price": round(exit_price, 6),
            "sl_price": round(pos.sl_price, 6),
            "tp_price": round(pos.tp_price, 6),
            "sl_distance_pct": round(abs(pos.sl_price - pos.entry_price) / pos.entry_price * 100, 4),
            "tp_distance_pct": round(abs(pos.tp_price - pos.entry_price) / pos.entry_price * 100, 4),
            "exit_reason": reason,
            "pnl_gross": round(pnl_gross, 4),
            "pnl_net": round(pnl_net, 4),
            "pnl_pct": round(pnl_pct * 100, 4),
            "fee": round(fee, 4),
            "bars_held": pos.bars_held,
            "mfe_pct": round(mfe_pct * 100, 4),
            "mae_pct": round(mae_pct * 100, 4),
            "entry_atr": round(pos.entry_atr, 6),
            **{k: v for k, v in pos.signal_extra.items() if isinstance(v, (int, float, str, bool))},
        })
