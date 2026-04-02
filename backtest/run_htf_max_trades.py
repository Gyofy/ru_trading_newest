"""Maximize 1h trades: more strategies + more coins + relaxed filters.

Usage: python -m backtest.run_htf_max_trades
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.data_loader import BacktestDataHub
from backtest.engine_htf import (
    _compute_rsi, compute_htf_barriers, Position,
    FEE_RATE, ENTRY_SLIP, EXIT_SLIP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest.max_trades")


# ══════════════════════════════════════════════════════════════
#  EXPANDED 1H STRATEGIES (8 total)
# ══════════════════════════════════════════════════════════════

def eval_tsmom_1h(hub, coin, ts, cfg):
    """TSMOM: 24h + 6h momentum alignment."""
    df = hub.get_ohlcv(coin, "1h", limit=50, current_ts=ts)
    if df is None or len(df) < 26:
        return None
    close = df["close"].values
    mom_24h = (close[-1] - close[-24]) / close[-24]
    mom_6h = (close[-1] - close[-6]) / close[-6]
    if not ((mom_24h > 0 and mom_6h > 0) or (mom_24h < 0 and mom_6h < 0)):
        return None
    side = "BUY" if mom_24h > 0 else "SELL"
    rsi = _compute_rsi(close, 14)
    if rsi and ((side == "BUY" and rsi > 75) or (side == "SELL" and rsi < 25)):
        return None
    min_move = cfg.get("min_move_pct", 0.004)
    if abs(mom_24h) < min_move:
        return None
    return {"side": side, "extra": {"trigger": "tsmom_1h", "mom_24h": mom_24h, "mom_6h": mom_6h,
                                     "rsi": rsi, "signal_strength": abs(mom_24h) * 100}}


def eval_rel_strength_1h(hub, coin, ts, cfg):
    """Relative Strength vs BTC."""
    df_coin = hub.get_ohlcv(coin, "1h", limit=30, current_ts=ts)
    df_btc = hub.get_ohlcv("BTC", "1h", limit=30, current_ts=ts)
    if df_coin is None or df_btc is None or len(df_coin) < 14 or len(df_btc) < 14:
        return None
    n = min(len(df_coin), len(df_btc))
    cc, cb = df_coin["close"].values[-n:], df_btc["close"].values[-n:]
    lookback = min(12, n - 1)
    coin_ret = (cc[-1] - cc[-lookback]) / cc[-lookback]
    btc_ret = (cb[-1] - cb[-lookback]) / cb[-lookback]
    rs = coin_ret - btc_ret
    min_rs = cfg.get("min_rs_pct", 0.008)
    if abs(rs) < min_rs:
        return None
    if btc_ret > 0.001 and rs > min_rs:
        return {"side": "BUY", "extra": {"trigger": "rel_strength_1h", "rel_strength": rs * 100,
                                          "coin_ret": coin_ret * 100, "btc_ret": btc_ret * 100,
                                          "signal_strength": abs(rs) * 100}}
    elif btc_ret < -0.001 and rs < -min_rs:
        return {"side": "SELL", "extra": {"trigger": "rel_strength_1h", "rel_strength": rs * 100,
                                           "coin_ret": coin_ret * 100, "btc_ret": btc_ret * 100,
                                           "signal_strength": abs(rs) * 100}}
    return None


def eval_tsmom_12h(hub, coin, ts, cfg):
    """TSMOM: 12h + 4h shorter momentum."""
    df = hub.get_ohlcv(coin, "1h", limit=30, current_ts=ts)
    if df is None or len(df) < 14:
        return None
    close = df["close"].values
    mom_12h = (close[-1] - close[-12]) / close[-12]
    mom_4h = (close[-1] - close[-4]) / close[-4]
    if not ((mom_12h > 0 and mom_4h > 0) or (mom_12h < 0 and mom_4h < 0)):
        return None
    side = "BUY" if mom_12h > 0 else "SELL"
    min_move = cfg.get("min_move_pct", 0.003)
    if abs(mom_12h) < min_move:
        return None
    rsi = _compute_rsi(close, 14)
    if rsi and ((side == "BUY" and rsi > 75) or (side == "SELL" and rsi < 25)):
        return None
    return {"side": side, "extra": {"trigger": "tsmom_12h", "mom_12h": mom_12h, "mom_4h": mom_4h,
                                     "signal_strength": abs(mom_12h) * 100}}


def eval_ema_cross_1h(hub, coin, ts, cfg):
    """EMA Crossover: 8/21 EMA cross with volume confirmation."""
    df = hub.get_ohlcv(coin, "1h", limit=40, current_ts=ts)
    if df is None or len(df) < 25:
        return None
    close = df["close"]
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    # Fresh cross: current bar crossed, previous bar hadn't
    if ema8.iloc[-1] > ema21.iloc[-1] and ema8.iloc[-2] <= ema21.iloc[-2]:
        side = "BUY"
    elif ema8.iloc[-1] < ema21.iloc[-1] and ema8.iloc[-2] >= ema21.iloc[-2]:
        side = "SELL"
    else:
        return None
    # Volume confirmation
    vol = df["volume"].values
    vol_avg = np.mean(vol[-20:-1])
    if vol_avg > 0 and vol[-1] < vol_avg * 1.0:
        return None
    return {"side": side, "extra": {"trigger": "ema_cross_1h",
                                     "signal_strength": abs(float(ema8.iloc[-1] - ema21.iloc[-1])) / float(close.iloc[-1]) * 100}}


def eval_rsi_reversal_1h(hub, coin, ts, cfg):
    """RSI Reversal: RSI extreme + reversal bar."""
    df = hub.get_ohlcv(coin, "1h", limit=30, current_ts=ts)
    if df is None or len(df) < 16:
        return None
    close = df["close"].values
    rsi = _compute_rsi(close, 14)
    if rsi is None:
        return None
    # RSI extreme + turning
    rsi_prev = _compute_rsi(close[:-1], 14)
    if rsi_prev is None:
        return None
    oversold = cfg.get("oversold", 28)
    overbought = cfg.get("overbought", 72)
    if rsi_prev < oversold and rsi > rsi_prev + 2:
        side = "BUY"
    elif rsi_prev > overbought and rsi < rsi_prev - 2:
        side = "SELL"
    else:
        return None
    # Confirmation: bar body in direction
    body = (close[-1] - df["open"].values[-1]) / df["open"].values[-1]
    if side == "BUY" and body < 0:
        return None
    if side == "SELL" and body > 0:
        return None
    return {"side": side, "extra": {"trigger": "rsi_reversal_1h", "rsi": rsi, "rsi_prev": rsi_prev,
                                     "signal_strength": abs(rsi - 50)}}


def eval_volume_breakout_1h(hub, coin, ts, cfg):
    """Volume Breakout: 2x volume + directional move."""
    df = hub.get_ohlcv(coin, "1h", limit=30, current_ts=ts)
    if df is None or len(df) < 22:
        return None
    vol = df["volume"].values
    close = df["close"].values
    vol_avg = np.mean(vol[-21:-1])
    vol_mult = cfg.get("vol_mult", 2.0)
    if vol_avg <= 0 or vol[-1] < vol_avg * vol_mult:
        return None
    # Price move
    bar_ret = (close[-1] - df["open"].values[-1]) / df["open"].values[-1]
    if abs(bar_ret) < 0.003:
        return None
    side = "BUY" if bar_ret > 0 else "SELL"
    return {"side": side, "extra": {"trigger": "vol_breakout_1h", "vol_ratio": vol[-1] / vol_avg,
                                     "bar_ret": bar_ret * 100, "signal_strength": vol[-1] / vol_avg}}


def eval_donchian_1h(hub, coin, ts, cfg):
    """Donchian Channel Breakout: 20-bar high/low break."""
    df = hub.get_ohlcv(coin, "1h", limit=30, current_ts=ts)
    if df is None or len(df) < 22:
        return None
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    dc_high = np.max(high[-21:-1])
    dc_low = np.min(low[-21:-1])
    if close[-1] > dc_high:
        side = "BUY"
    elif close[-1] < dc_low:
        side = "SELL"
    else:
        return None
    vol = df["volume"].values
    vol_avg = np.mean(vol[-20:-1])
    if vol_avg > 0 and vol[-1] < vol_avg * 1.2:
        return None
    return {"side": side, "extra": {"trigger": "donchian_1h", "channel_width": (dc_high - dc_low) / close[-1] * 100,
                                     "signal_strength": abs(close[-1] - (dc_high + dc_low) / 2) / close[-1] * 100}}


def eval_macd_cross_1h(hub, coin, ts, cfg):
    """MACD Crossover: MACD line crosses signal line."""
    df = hub.get_ohlcv(coin, "1h", limit=40, current_ts=ts)
    if df is None or len(df) < 30:
        return None
    close = df["close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    # Fresh cross
    if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
        side = "BUY"
    elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
        side = "SELL"
    else:
        return None
    # Trend filter: only trade in direction of 24h trend
    mom_24h = (float(close.iloc[-1]) - float(close.iloc[-24])) / float(close.iloc[-24]) if len(close) >= 25 else 0
    if side == "BUY" and mom_24h < -0.01:
        return None
    if side == "SELL" and mom_24h > 0.01:
        return None
    return {"side": side, "extra": {"trigger": "macd_cross_1h", "macd": float(macd.iloc[-1]),
                                     "signal_strength": abs(float(macd.iloc[-1] - signal.iloc[-1])) / float(close.iloc[-1]) * 100}}


# ══════════════════════════════════════════════════════════════
#  ENGINE
# ══════════════════════════════════════════════════════════════

STRATEGIES_1H = {
    "tsmom_1h":        (eval_tsmom_1h,        {"sl_mult": 1.5, "tp_mult": 5.0, "min_move_pct": 0.004}),
    "rel_strength_1h": (eval_rel_strength_1h,  {"sl_mult": 2.0, "tp_mult": 5.0, "min_rs_pct": 0.008}),
    "tsmom_12h":       (eval_tsmom_12h,        {"sl_mult": 1.5, "tp_mult": 5.0, "min_move_pct": 0.003}),
    "ema_cross_1h":    (eval_ema_cross_1h,     {"sl_mult": 1.5, "tp_mult": 4.0}),
    "rsi_reversal_1h": (eval_rsi_reversal_1h,  {"sl_mult": 1.5, "tp_mult": 3.0}),
    "vol_breakout_1h": (eval_volume_breakout_1h, {"sl_mult": 1.5, "tp_mult": 4.0, "vol_mult": 2.0}),
    "donchian_1h":     (eval_donchian_1h,      {"sl_mult": 1.5, "tp_mult": 4.0}),
    "macd_cross_1h":   (eval_macd_cross_1h,    {"sl_mult": 1.5, "tp_mult": 4.0}),
}

TTL_BARS = 2880  # 48h in 1m bars


class MaxTradesEngine:
    def __init__(self, hub, coins, capital=5000, leverage=3, max_pos=8):
        self._hub = hub
        self._coins = coins
        self._capital = capital
        self._leverage = leverage
        self._max_pos = max_pos
        self._open: list[Position] = []
        self._closed: list[dict] = []
        self._cooldowns: dict[str, pd.Timestamp] = {}

    def run(self, strategies: dict | None = None) -> list[dict]:
        strats = strategies or STRATEGIES_1H
        all_ts = self._hub.get_all_1m_timestamps(self._coins[:4])  # use first 4 for timestamp index
        warmup = 1500

        for i, ts in enumerate(all_ts):
            if i < warmup:
                continue
            self._check_exits(ts)
            if ts.minute == 0:  # 1h boundary
                self._evaluate_all(ts, strats)
            if i % 5000 == 0 and i > warmup:
                logger.info(f"  Bar {i}/{len(all_ts)} | Open:{len(self._open)} Closed:{len(self._closed)}")

        for pos in list(self._open):
            self._close(pos, "BACKTEST_END", all_ts[-1])

        return self._closed

    def _check_exits(self, ts):
        for pos in list(self._open):
            bar = self._hub.get_bar(pos.coin, "1m", ts)
            if bar is None:
                continue
            pos.bars_held += 1
            h, l = bar["high"], bar["low"]
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

    def _evaluate_all(self, ts, strats):
        if len(self._open) >= self._max_pos:
            return
        for name, (fn, cfg) in strats.items():
            if len(self._open) >= self._max_pos:
                break
            for coin in self._coins:
                if any(p.coin == coin and p.strategy == name for p in self._open):
                    continue
                cd_key = f"{name}:{coin}"
                if cd_key in self._cooldowns and (ts - self._cooldowns[cd_key]).total_seconds() < 3600:
                    continue
                try:
                    sig = fn(self._hub, coin, ts, cfg)
                except Exception:
                    continue
                if sig is None:
                    continue
                self._open_pos(name, coin, sig, ts, cfg)
                self._cooldowns[cd_key] = ts

    def _open_pos(self, strat, coin, sig, ts, cfg):
        df = self._hub.get_ohlcv(coin, "1h", limit=20, current_ts=ts)
        if df is None or len(df) < 15:
            return
        atr = self._hub.compute_atr(df)
        if atr <= 0:
            return
        price = float(df["close"].iloc[-1])
        side = sig["side"]
        fill = price * (1 + ENTRY_SLIP) if side == "BUY" else price * (1 - ENTRY_SLIP)
        sl, tp = compute_htf_barriers(strat, side, fill, atr, cfg)
        notional = self._capital * 0.08 * self._leverage  # 8% per trade (more positions)
        self._open.append(Position(
            trade_id=str(uuid.uuid4())[:8], coin=coin, strategy=strat, timeframe="1h",
            side=side, entry_price=fill, entry_ts=ts, notional=notional, leverage=self._leverage,
            sl_price=sl, tp_price=tp, ttl_bars=TTL_BARS, entry_atr=atr,
            signal_extra=sig.get("extra", {}),
        ))

    def _close(self, pos, reason, ts):
        if pos not in self._open:
            return
        self._open.remove(pos)
        if reason == "SL_HIT":
            ep = pos.sl_price
        elif reason == "TP_HIT":
            ep = pos.tp_price
        else:
            bar = self._hub.get_bar(pos.coin, "1m", ts)
            ep = bar["close"] if bar else pos.entry_price
        if pos.side == "BUY":
            ep *= (1 - EXIT_SLIP)
            pnl_pct = (ep - pos.entry_price) / pos.entry_price
            mfe = (pos.mfe_price - pos.entry_price) / pos.entry_price if pos.mfe_price > 0 else 0
            mae = (pos.mae_price - pos.entry_price) / pos.entry_price if pos.mae_price > 0 else 0
        else:
            ep *= (1 + EXIT_SLIP)
            pnl_pct = (pos.entry_price - ep) / pos.entry_price
            mfe = (pos.entry_price - pos.mfe_price) / pos.entry_price if pos.mfe_price > 0 else 0
            mae = (pos.entry_price - pos.mae_price) / pos.entry_price if pos.mae_price > 0 else 0
        fee = pos.notional * FEE_RATE
        gross = pnl_pct * pos.notional
        net = gross - fee
        self._closed.append({
            "trade_id": pos.trade_id, "ts_entry": pos.entry_ts.isoformat(),
            "ts_exit": ts.isoformat(), "coin": pos.coin, "strategy": pos.strategy,
            "side": pos.side, "entry_price": round(pos.entry_price, 6),
            "exit_price": round(ep, 6), "exit_reason": reason,
            "pnl_gross": round(gross, 4), "pnl_net": round(net, 4),
            "pnl_pct": round(pnl_pct * 100, 4), "fee": round(fee, 4),
            "bars_held": pos.bars_held, "mfe_pct": round(mfe * 100, 4),
            "mae_pct": round(mae * 100, 4),
            "sl_dist_pct": round(abs(pos.sl_price - pos.entry_price) / pos.entry_price * 100, 4),
            "tp_dist_pct": round(abs(pos.tp_price - pos.entry_price) / pos.entry_price * 100, 4),
            **{k: v for k, v in pos.signal_extra.items() if isinstance(v, (int, float, str, bool))},
        })


def analyze(trades, label=""):
    if not trades:
        return {"label": label, "n": 0}
    df = pd.DataFrame(trades)
    n = len(df)
    wins = df[df["pnl_net"] > 0]
    wr = len(wins) / n * 100
    gross = df["pnl_gross"].sum()
    fees = df["fee"].sum()
    net = df["pnl_net"].sum()
    w_sum = wins["pnl_net"].sum() if len(wins) > 0 else 0
    l_sum = abs(df[df["pnl_net"] <= 0]["pnl_net"].sum()) + 0.01
    pf = w_sum / l_sum
    exits = df["exit_reason"].value_counts().to_dict()
    tp_r = exits.get("TP_HIT", 0) / n * 100
    sl_r = exits.get("SL_HIT", 0) / n * 100
    return {"label": label, "n": n, "wr": round(wr, 1), "gross": round(gross, 2),
            "fees": round(fees, 2), "net": round(net, 2), "pf": round(pf, 2),
            "tp_rate": round(tp_r, 1), "sl_rate": round(sl_r, 1),
            "avg_bars": round(df["bars_held"].mean(), 0), "exits": exits}


def print_table(results):
    hdr = f"{'Strategy':<22} {'N':>4} {'WR%':>6} {'Gross$':>9} {'Fee$':>7} {'Net$':>9} {'PF':>5} {'TP%':>5} {'SL%':>5} {'Bars':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if r["n"] == 0:
            print(f"{r['label']:<22}   --")
            continue
        print(f"{r['label']:<22} {r['n']:>4} {r['wr']:>5.1f}% {r['gross']:>+9.2f} {r['fees']:>7.2f} "
              f"{r['net']:>+9.2f} {r['pf']:>5.2f} {r['tp_rate']:>4.1f}% {r['sl_rate']:>4.1f}% {r['avg_bars']:>6.0f}")


def main():
    start = time.time()

    data_dir = ROOT / "data" / "raw" / "binance"
    # ALL available coins (excluding BTC/ETH as reference only)
    all_available = ["SOL", "XRP", "ADA", "DOT", "DOGE", "AVAX", "BNB", "LINK", "OP", "ARB", "SUI", "APT", "TAO"]
    ref_coins = ["BTC", "ETH"]

    logger.info(f"Loading {len(all_available) + len(ref_coins)} coins...")
    hub = BacktestDataHub(data_dir, all_available + ref_coins)

    # ══════════════════════════════════════════════════════════
    # Test 1: Original 4 coins, 8 strategies
    # ══════════════════════════════════════════════════════════
    coins_4 = ["SOL", "XRP", "ADA", "DOT"]

    print("\n" + "=" * 95)
    print("  TEST 1: 4 COINS x 8 STRATEGIES (1h)")
    print("=" * 95)

    eng = MaxTradesEngine(hub, coins_4, max_pos=8)
    trades = eng.run()
    df = pd.DataFrame(trades) if trades else pd.DataFrame()
    results = []
    if not df.empty:
        for s in sorted(df["strategy"].unique()):
            results.append(analyze(df[df["strategy"] == s].to_dict("records"), s))
        results.append(analyze(trades, "TOTAL"))
    print()
    print_table(results)

    # ══════════════════════════════════════════════════════════
    # Test 2: 13 coins, 8 strategies
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 95)
    print(f"  TEST 2: {len(all_available)} COINS x 8 STRATEGIES (1h)")
    print("=" * 95)

    eng2 = MaxTradesEngine(hub, all_available, max_pos=12)
    trades2 = eng2.run()
    df2 = pd.DataFrame(trades2) if trades2 else pd.DataFrame()
    results2 = []
    if not df2.empty:
        for s in sorted(df2["strategy"].unique()):
            results2.append(analyze(df2[df2["strategy"] == s].to_dict("records"), s))
        results2.append(analyze(trades2, "TOTAL"))
    print()
    print_table(results2)

    # Per-coin breakdown
    if not df2.empty:
        print(f"\n  Per-coin breakdown (13 coins):")
        coin_stats = df2.groupby("coin").agg(
            n=("pnl_net", "count"),
            net=("pnl_net", "sum"),
            gross=("pnl_gross", "sum"),
        ).sort_values("net", ascending=False)
        for coin, row in coin_stats.iterrows():
            wr = len(df2[(df2["coin"] == coin) & (df2["pnl_net"] > 0)]) / row["n"] * 100
            print(f"    {coin:<6} {int(row['n']):>4} trades  WR={wr:>5.1f}%  Gross=${row['gross']:>+8.2f}  Net=${row['net']:>+8.2f}")

    # ══════════════════════════════════════════════════════════
    # Test 3: Only profitable strategies, 13 coins
    # ══════════════════════════════════════════════════════════
    if not df2.empty:
        profitable_strats = {}
        for s in df2["strategy"].unique():
            sdf = df2[df2["strategy"] == s]
            if sdf["pnl_net"].sum() > 0 and len(sdf) >= 5:
                profitable_strats[s] = STRATEGIES_1H[s]

        if profitable_strats:
            print("\n" + "=" * 95)
            print(f"  TEST 3: {len(all_available)} COINS x {len(profitable_strats)} PROFITABLE STRATEGIES ONLY")
            print("=" * 95)

            eng3 = MaxTradesEngine(hub, all_available, max_pos=12)
            trades3 = eng3.run(strategies=profitable_strats)
            df3 = pd.DataFrame(trades3) if trades3 else pd.DataFrame()
            results3 = []
            if not df3.empty:
                for s in sorted(df3["strategy"].unique()):
                    results3.append(analyze(df3[df3["strategy"] == s].to_dict("records"), s))
                results3.append(analyze(trades3, "TOTAL"))
            print()
            print_table(results3)

            # Final per-coin
            if not df3.empty:
                print(f"\n  Per-coin (profitable strategies only):")
                coin_stats3 = df3.groupby("coin").agg(
                    n=("pnl_net", "count"), net=("pnl_net", "sum"), gross=("pnl_gross", "sum"),
                ).sort_values("net", ascending=False)
                for coin, row in coin_stats3.iterrows():
                    wr = len(df3[(df3["coin"] == coin) & (df3["pnl_net"] > 0)]) / row["n"] * 100
                    print(f"    {coin:<6} {int(row['n']):>4} trades  WR={wr:>5.1f}%  Gross=${row['gross']:>+8.2f}  Net=${row['net']:>+8.2f}")

    # Save
    output_dir = ROOT / "backtest" / "output"
    if trades2:
        with open(output_dir / "htf_max_trades.json", "w") as f:
            json.dump(trades2, f, indent=2, default=str)

    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
