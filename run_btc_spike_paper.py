#!/usr/bin/env python3
"""BTC Spike -> Alt Follow Paper Trading Bot.

Strategy:
  - Monitor BTC 1h candle close-to-close return
  - When |ret| > threshold (1.2%), enter ALT coins in same direction
  - TP = 1.5 * ATR(14), SL = 1.0 * ATR(14)
  - Max hold = 6h, leverage = 3x (paper)
  - Enter at next candle open (realistic timing)

Evidence:
  - 180d backtest (BTC -38.5% period): n=276, WR=48.6%, avg=+0.11%
  - Works in both UP and DOWN spikes (not long-only bias)
  - Edge source: volatility clustering + momentum continuation

Usage:
    python run_btc_spike_paper.py
    python run_btc_spike_paper.py --threshold 0.015 --leverage 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import ccxt.async_support as ccxt_async
except ImportError:
    ccxt_async = None

# ── Config ──────────────────────────────────────────────
DEFAULT_CONFIG = {
    "btc_threshold": 0.012,       # BTC 1h |ret| > 1.2%
    "alt_coins": ["SOL", "ETH", "XRP", "ADA"],
    "tp_atr_mult": 1.5,
    "sl_atr_mult": 1.0,
    "max_hold_bars": 6,           # 6h on 1h bars
    "leverage": 3,
    "risk_frac": 0.02,            # 2% of equity per trade
    "poll_interval_sec": 60,      # check every 60s
    "atr_period": 14,
    "equity": 500.0,              # paper equity
}

SYMBOL_MAP = {
    "BTC": "BTC/USDT:USDT", "ETH": "ETH/USDT:USDT",
    "SOL": "SOL/USDT:USDT", "XRP": "XRP/USDT:USDT",
    "ADA": "ADA/USDT:USDT", "LINK": "LINK/USDT:USDT",
}

LOG_DIR = PROJECT_ROOT / "data" / "reports" / "btc_spike_paper"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("btc_spike")


# ── Data Classes ────────────────────────────────────────

class PaperPosition:
    def __init__(self, coin, side, entry_price, qty, sl, tp, entry_time, max_hold_bars):
        self.coin = coin
        self.side = side
        self.entry_price = entry_price
        self.qty = qty
        self.sl = sl
        self.tp = tp
        self.entry_time = entry_time
        self.max_hold_bars = max_hold_bars
        self.bars_held = 0
        self.pnl = 0.0
        self.exit_reason = ""
        self.closed = False

    def check_exit(self, high, low, close):
        """Check SL/TP/TTL on new bar."""
        self.bars_held += 1
        if self.side == "BUY":
            if low <= self.sl:
                self.pnl = (self.sl - self.entry_price) / self.entry_price
                self.exit_reason = "SL"
                self.closed = True
            elif high >= self.tp:
                self.pnl = (self.tp - self.entry_price) / self.entry_price
                self.exit_reason = "TP"
                self.closed = True
        else:
            if high >= self.sl:
                self.pnl = (self.entry_price - self.sl) / self.entry_price
                self.exit_reason = "SL"
                self.closed = True
            elif low <= self.tp:
                self.pnl = (self.entry_price - self.tp) / self.entry_price
                self.exit_reason = "TP"
                self.closed = True

        if not self.closed and self.bars_held >= self.max_hold_bars:
            if self.side == "BUY":
                self.pnl = (close - self.entry_price) / self.entry_price
            else:
                self.pnl = (self.entry_price - close) / self.entry_price
            self.exit_reason = "TTL"
            self.closed = True

        return self.closed


class BtcSpikeBot:
    def __init__(self, config: dict):
        self.cfg = config
        self.equity = config["equity"]
        self.positions: list[PaperPosition] = []
        self.trade_log: list[dict] = []
        self.btc_candles: list[dict] = []  # recent 1h candles
        self.alt_candles: dict[str, list[dict]] = {c: [] for c in config["alt_coins"]}
        self.exchange = None
        self.running = True
        self._last_btc_close = None
        self._last_spike_hour = None

    async def initialize(self):
        """Initialize exchange connection (ccxt or yfinance fallback)."""
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")

        if ccxt_async and api_key:
            self.exchange = ccxt_async.binanceusdm({
                "apiKey": api_key,
                "secret": api_secret,
                "options": {"defaultType": "future"},
            })
            try:
                await self.exchange.load_markets()
                self._use_yfinance = False
                logger.info("Exchange connected (Binance live)")
                return True
            except Exception as e:
                logger.warning("Binance failed (%s), falling back to yfinance" % e)

        # yfinance fallback (paper only, no API key needed)
        try:
            import yfinance as yf
            self._yf = yf
            self._use_yfinance = True
            logger.info("Using yfinance fallback (paper mode, no API key)")
            return True
        except ImportError:
            logger.error("Neither ccxt+API nor yfinance available")
            return False

    def _yf_fetch(self, symbol_yf, period="5d", interval="1h"):
        """Fetch OHLCV via yfinance (sync)."""
        df = self._yf.Ticker(symbol_yf).history(period=period, interval=interval)
        df.columns = [c.lower() for c in df.columns]
        candles = []
        for ts, row in df.iterrows():
            candles.append({
                "ts": int(ts.timestamp() * 1000),
                "open": float(row["open"]), "high": float(row["high"]),
                "low": float(row["low"]), "close": float(row["close"]),
                "volume": float(row["volume"]),
            })
        return candles

    async def fetch_btc_1h(self):
        """Fetch latest BTC 1h candles."""
        try:
            if self._use_yfinance:
                self.btc_candles = self._yf_fetch("BTC-USD", "5d", "1h")
            else:
                ohlcv = await self.exchange.fetch_ohlcv("BTC/USDT:USDT", "1h", limit=20)
                self.btc_candles = [
                    {"ts": o[0], "open": o[1], "high": o[2], "low": o[3], "close": o[4], "volume": o[5]}
                    for o in ohlcv
                ]
            return len(self.btc_candles) >= 3
        except Exception as e:
            logger.error("BTC OHLCV fetch failed: %s" % e)
            return False

    async def fetch_alt_data(self, coin):
        """Fetch alt coin 1h candles for ATR."""
        try:
            if self._use_yfinance:
                yf_sym = coin + "-USD"
                self.alt_candles[coin] = self._yf_fetch(yf_sym, "5d", "1h")
            else:
                sym = SYMBOL_MAP.get(coin, f"{coin}/USDT:USDT")
                ohlcv = await self.exchange.fetch_ohlcv(sym, "1h", limit=20)
                self.alt_candles[coin] = [
                    {"ts": o[0], "open": o[1], "high": o[2], "low": o[3], "close": o[4], "volume": o[5]}
                    for o in ohlcv
                ]
            return len(self.alt_candles[coin]) >= 3
        except Exception as e:
            logger.error("%s OHLCV fetch failed: %s" % (coin, e))
            return False

    def compute_atr(self, candles, period=14):
        """Compute ATR from candle list."""
        if len(candles) < period + 1:
            return None
        trs = []
        for i in range(1, len(candles)):
            h = candles[i]["high"]; l = candles[i]["low"]
            pc = candles[i-1]["close"]
            tr = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(tr)
        if len(trs) < period:
            return np.mean(trs) if trs else None
        return np.mean(trs[-period:])

    def detect_btc_spike(self):
        """Check if last completed BTC 1h bar had a spike."""
        if len(self.btc_candles) < 3:
            return None, 0.0

        # Last COMPLETED bar (not current)
        completed = self.btc_candles[-2]
        prev = self.btc_candles[-3]

        ret = (completed["close"] - prev["close"]) / prev["close"]
        hour_key = completed["ts"]

        # Don't trigger twice on same bar
        if hour_key == self._last_spike_hour:
            return None, ret

        if abs(ret) >= self.cfg["btc_threshold"]:
            self._last_spike_hour = hour_key
            direction = "BUY" if ret > 0 else "SELL"
            logger.info("BTC SPIKE: %+.2f%% -> %s signal" % (ret * 100, direction))
            return direction, ret

        return None, ret

    async def enter_alts(self, direction):
        """Enter all alt coins in the given direction."""
        for coin in self.cfg["alt_coins"]:
            # Skip if already in position
            if any(p.coin == coin and not p.closed for p in self.positions):
                logger.info("  %s: already in position, skip" % coin)
                continue

            await self.fetch_alt_data(coin)
            candles = self.alt_candles.get(coin, [])
            if len(candles) < 3:
                continue

            atr = self.compute_atr(candles, self.cfg["atr_period"])
            if atr is None or atr < 1e-10:
                continue

            # Entry at current price (next bar open proxy)
            entry_price = candles[-1]["open"]  # current bar's open
            tp_dist = atr * self.cfg["tp_atr_mult"]
            sl_dist = atr * self.cfg["sl_atr_mult"]

            if direction == "BUY":
                tp = entry_price + tp_dist
                sl = entry_price - sl_dist
            else:
                tp = entry_price - tp_dist
                sl = entry_price + sl_dist

            # Position sizing
            risk_usdt = self.equity * self.cfg["risk_frac"]
            stop_dist_pct = sl_dist / entry_price
            notional = risk_usdt / stop_dist_pct
            qty = notional / entry_price

            pos = PaperPosition(
                coin=coin, side=direction, entry_price=entry_price,
                qty=qty, sl=sl, tp=tp,
                entry_time=datetime.now(timezone.utc).isoformat(),
                max_hold_bars=self.cfg["max_hold_bars"],
            )
            self.positions.append(pos)

            logger.info("  ENTER %s %s @ %.4f | SL=%.4f TP=%.4f | qty=%.4f not=$%.1f" % (
                coin, direction, entry_price, sl, tp, qty, notional))

    async def check_positions(self):
        """Check all open positions for exits."""
        for pos in self.positions:
            if pos.closed:
                continue

            candles = self.alt_candles.get(pos.coin, [])
            if len(candles) < 2:
                continue

            # Use latest completed bar
            bar = candles[-2]
            exited = pos.check_exit(bar["high"], bar["low"], bar["close"])

            if exited:
                cost = 0.0018  # round trip
                net_pnl = pos.pnl - cost
                pnl_leveraged = net_pnl * self.cfg["leverage"]
                pnl_dollar = self.equity * pnl_leveraged * (self.cfg["risk_frac"] / (pos.sl / pos.entry_price if pos.side == "BUY" else pos.sl / pos.entry_price))

                # Simple: use risk_frac * leverage for equity impact
                equity_impact = self.equity * net_pnl * self.cfg["leverage"]
                self.equity += equity_impact

                trade = {
                    "coin": pos.coin, "side": pos.side,
                    "entry": pos.entry_price, "exit_reason": pos.exit_reason,
                    "pnl_pct": round(net_pnl * 100, 4),
                    "pnl_lev": round(pnl_leveraged * 100, 4),
                    "equity_after": round(self.equity, 2),
                    "bars_held": pos.bars_held,
                    "entry_time": pos.entry_time,
                    "exit_time": datetime.now(timezone.utc).isoformat(),
                }
                self.trade_log.append(trade)
                self._save_trade(trade)

                logger.info("  EXIT %s %s: %s pnl=%+.2f%% (lev %+.2f%%) equity=$%.2f bars=%d" % (
                    pos.coin, pos.side, pos.exit_reason,
                    net_pnl * 100, pnl_leveraged * 100, self.equity, pos.bars_held))

    def _save_trade(self, trade):
        """Append trade to JSONL log."""
        with open(LOG_DIR / "trades.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(trade, ensure_ascii=False) + "\n")

    def _save_state(self):
        """Save current state."""
        state = {
            "equity": self.equity,
            "open_positions": len([p for p in self.positions if not p.closed]),
            "total_trades": len(self.trade_log),
            "last_update": datetime.now(timezone.utc).isoformat(),
        }
        with open(LOG_DIR / "state.json", "w") as f:
            json.dump(state, f, indent=2)

    async def run(self):
        """Main loop."""
        logger.info("=" * 60)
        logger.info("BTC Spike Paper Bot Started")
        logger.info("  Equity: $%.2f" % self.equity)
        logger.info("  Threshold: %.1f%%" % (self.cfg["btc_threshold"] * 100))
        logger.info("  Leverage: %dx" % self.cfg["leverage"])
        logger.info("  Coins: %s" % self.cfg["alt_coins"])
        logger.info("  TP: %.1f ATR, SL: %.1f ATR, Max hold: %dh" % (
            self.cfg["tp_atr_mult"], self.cfg["sl_atr_mult"], self.cfg["max_hold_bars"]))
        logger.info("=" * 60)

        if not await self.initialize():
            return

        try:
            while self.running:
                try:
                    # Fetch BTC data
                    if not await self.fetch_btc_1h():
                        await asyncio.sleep(30)
                        continue

                    # Fetch alt data for open positions
                    for coin in self.cfg["alt_coins"]:
                        await self.fetch_alt_data(coin)

                    # Check existing positions
                    await self.check_positions()

                    # Detect BTC spike
                    direction, ret = self.detect_btc_spike()
                    if direction:
                        await self.enter_alts(direction)

                    # Save state
                    self._save_state()

                    # Summary
                    open_pos = [p for p in self.positions if not p.closed]
                    if self.trade_log:
                        recent = self.trade_log[-1]
                        logger.info(
                            "Equity=$%.2f | Open=%d | Trades=%d | BTC_ret=%+.2f%% | Last: %s %s %s %+.2f%%" % (
                                self.equity, len(open_pos), len(self.trade_log),
                                ret * 100, recent["coin"], recent["side"],
                                recent["exit_reason"], recent["pnl_lev"]))
                    else:
                        logger.info("Equity=$%.2f | Open=%d | Waiting for BTC spike (>%.1f%%)..." % (
                            self.equity, len(open_pos), self.cfg["btc_threshold"] * 100))

                    await asyncio.sleep(self.cfg["poll_interval_sec"])

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Loop error: %s" % e)
                    await asyncio.sleep(30)

        finally:
            if self.exchange and not self._use_yfinance:
                await self.exchange.close()
            self._save_state()
            logger.info("Bot stopped. Final equity: $%.2f, Trades: %d" % (
                self.equity, len(self.trade_log)))


def main():
    parser = argparse.ArgumentParser(description="BTC Spike Paper Trading Bot")
    parser.add_argument("--threshold", type=float, default=0.012, help="BTC spike threshold (default 0.012)")
    parser.add_argument("--leverage", type=int, default=3, help="Paper leverage (default 3)")
    parser.add_argument("--equity", type=float, default=500.0, help="Starting equity (default 500)")
    parser.add_argument("--coins", nargs="+", default=["SOL", "ETH", "XRP", "ADA"])
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["btc_threshold"] = args.threshold
    config["leverage"] = args.leverage
    config["equity"] = args.equity
    config["alt_coins"] = args.coins

    bot = BtcSpikeBot(config)

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    main()
