"""TSMOM + RSI + CVD + OI Paper Trading Bot.

Strategy: v5.0 TSMOM Enhanced
  Layer 1: TSMOM 28d volume-weighted direction
  Layer 2: RSI > 50 (LONG) / RSI < 50 (SHORT) trend filter
  Layer 3: CVD Q85 extreme timing (counter-direction overextension)
  Layer 4: OI z-score < 2.0 (avoid crowded positioning)
  Barrier: TP=5xATR, SL=1.0xATR, TTL=24 bars (96h)

Validated: IS Sharpe 2.67, OOS Sharpe 2.80, p-value 0.006
"""

import sys, os, json, time, logging, asyncio, multiprocessing
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# Data source: Binance ccxt (live) or yfinance (fallback)
DATA_SOURCE = os.environ.get("DATA_SOURCE", "yfinance")  # "binance" or "yfinance"

try:
    import yfinance as yf
except ImportError:
    yf = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/reports/tsmom_paper/bot.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("tsmom_paper")

from src.rl.state_builder import build_rl_state
from src.rl.signal_logger import SignalLogger
from src.rl.bandit import SIZING_MAP

# ══════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════

COINS_YAHOO = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD",
    "XRP": "XRP-USD", "ADA": "ADA-USD", "DOT": "DOT-USD", "LINK": "LINK-USD",
    "DOGE": "DOGE-USD", "AVAX": "AVAX-USD", "BNB": "BNB-USD",
}
COINS_BINANCE = {
    "BTC": "BTC/USDT:USDT", "ETH": "ETH/USDT:USDT", "SOL": "SOL/USDT:USDT",
    "XRP": "XRP/USDT:USDT", "ADA": "ADA/USDT:USDT", "DOT": "DOT/USDT:USDT",
    "LINK": "LINK/USDT:USDT", "DOGE": "DOGE/USDT:USDT", "AVAX": "AVAX/USDT:USDT",
    "BNB": "BNB/USDT:USDT",
}
COINS = COINS_YAHOO  # default, overridden by DATA_SOURCE

# Strategy params (best OOS config)
CFG = {
    "lookback_days": 7,          # short-term TSMOM (direction)
    "lookback_long_days": 28,    # long-term TSMOM (regime gate)
    "dual_lookback": False,      # not used as filter — used for regime classification
    "volume_weighted": False,
    "cvd_quantile": 0.75,
    "cvd_roll_window": 120,
    "k_upper": 5.0,
    "k_lower": 2.0,
    "max_hold_bars": 24,
    "use_oi": True,
    "oi_zscore_max": 2.0,
    "cost_roundtrip": 0.0020,
    "leverage_a": 4,             # v5.3: sniper (7d+28d agree + CVD Q90) → 4x
    "leverage_b": 2,             # v5.3: steady (7d only + CVD Q75) → 2x
    "cvd_quantile_a": 0.90,     # sniper: extreme CVD only (top/bottom 10%)
    "leverage": 2,               # fallback default
    "equity_risk_pct": 0.02,     # 2% risk per trade
    "max_positions": 3,
}

STATE_DIR = Path("data/reports/tsmom_paper")
STATE_FILE = STATE_DIR / "state.json"
TRADES_FILE = STATE_DIR / "trades.jsonl"
LOCK_FILE = STATE_DIR / "bot.lock"

INITIAL_EQUITY = 1000.0  # paper equity
BAR_SECONDS = 4 * 3600   # 4h
POLL_SECONDS = 30         # SL/TP check interval
N_WORKERS = max(1, int(multiprocessing.cpu_count() * 0.7))  # 70% of cores


# ══════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════

@dataclass
class Position:
    coin: str
    side: int          # +1 LONG, -1 SHORT
    entry_price: float
    entry_time: str
    entry_bar: int
    tp_price: float
    sl_price: float
    atr_at_entry: float
    size_usd: float
    bars_held: int = 0
    status: str = "OPEN"


class PaperBot:
    def __init__(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)

        # Single instance lock — prevent duplicate processes
        if LOCK_FILE.exists():
            try:
                lock_data = json.loads(LOCK_FILE.read_text())
                old_pid = lock_data.get("pid", 0)
                # Check if old process is still alive
                import psutil
                if psutil.pid_exists(old_pid):
                    log.error(f"Another bot instance running (PID {old_pid}). Exiting.")
                    sys.exit(1)
                else:
                    log.warning(f"Stale lock from PID {old_pid}, taking over.")
            except (ImportError, Exception):
                # psutil not available, check by timestamp
                lock_age = time.time() - LOCK_FILE.stat().st_mtime
                if lock_age < 300:  # less than 5 min old
                    log.error("Another bot instance may be running (lock < 5min old). Exiting.")
                    sys.exit(1)

        # Write our lock
        LOCK_FILE.write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))

        self.equity = INITIAL_EQUITY
        self.positions: dict[str, Position] = {}
        self.trade_count = 0
        self.signal_logger = SignalLogger(STATE_DIR / "signal_log.jsonl")
        self.coin_history: dict[str, list[float]] = {}
        self.load_state()

    def load_state(self):
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                s = json.load(f)
            self.equity = s.get("equity", INITIAL_EQUITY)
            self.trade_count = s.get("trade_count", 0)
            for coin, p in s.get("positions", {}).items():
                self.positions[coin] = Position(**p)
            log.info(f"State loaded: equity=${self.equity:.2f}, "
                     f"positions={len(self.positions)}, trades={self.trade_count}")

    def save_state(self):
        s = {
            "equity": self.equity,
            "trade_count": self.trade_count,
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(STATE_FILE, "w") as f:
            json.dump(s, f, indent=2)

    def log_trade(self, trade: dict):
        with open(TRADES_FILE, "a") as f:
            f.write(json.dumps(trade, default=str) + "\n")

    # ── Data Fetching ──

    def fetch_4h(self, coin: str) -> pd.DataFrame:
        """Fetch OHLCV (yfinance or Binance ccxt), resample to 4h, add indicators."""
        if DATA_SOURCE == "binance":
            return self._fetch_4h_binance(coin)
        return self._fetch_4h_yfinance(coin)

    def _fetch_4h_yfinance(self, coin: str) -> pd.DataFrame:
        """yfinance fallback (no API key needed)."""
        if yf is None:
            log.warning(f"[{coin}] yfinance not installed")
            return pd.DataFrame()

        sym = COINS_YAHOO.get(coin, f"{coin}-USD")
        df = yf.download(sym, period="60d", interval="1h", progress=False)
        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={"Open":"open","High":"high","Low":"low",
                                 "Close":"close","Volume":"volume"})

        df = df.resample("4h").agg({
            "open": lambda x: x.iloc[0] if len(x) > 0 else np.nan,
            "high": "max", "low": "min",
            "close": lambda x: x.iloc[-1] if len(x) > 0 else np.nan,
            "volume": "sum",
        }).dropna()
        return self._add_indicators(df)

    def _fetch_4h_binance(self, coin: str) -> pd.DataFrame:
        """Binance ccxt (live API, real-time data)."""
        try:
            import ccxt
            exchange = ccxt.binanceusdm({
                "apiKey": os.environ.get("BINANCE_API_KEY", ""),
                "secret": os.environ.get("BINANCE_API_SECRET", ""),
                "options": {"defaultType": "future"},
            })

            ccxt_sym = COINS_BINANCE.get(coin, f"{coin}/USDT:USDT")
            ohlcv = exchange.fetch_ohlcv(ccxt_sym, "4h", limit=500)

            if not ohlcv:
                return pd.DataFrame()

            df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            return self._add_indicators(df)

        except Exception as e:
            log.warning(f"[{coin}] Binance fetch failed: {e}, falling back to yfinance")
            return self._fetch_4h_yfinance(coin)

    @staticmethod
    def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add ATR, RSI, CVD to raw OHLCV."""
        if len(df) < 20:
            return df

        # ATR
        tr = np.maximum(df["high"]-df["low"],
                        np.maximum(np.abs(df["high"]-df["close"].shift(1)),
                                   np.abs(df["low"]-df["close"].shift(1))))
        df["atr_14"] = tr.rolling(14, min_periods=1).mean()

        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # CVD
        hr = (df["high"] - df["low"]).replace(0, np.nan)
        buy_frac = ((df["close"] - df["low"]) / hr).fillna(0.5).clip(0, 1)
        vd = (2 * buy_frac - 1) * df["volume"]
        cvd = vd.cumsum()
        cvd_ma = cvd.rolling(24, min_periods=6).mean()
        df["cvd_ratio_24"] = ((cvd - cvd_ma) / cvd_ma.abs().replace(0, np.nan)).fillna(0)

        return df

    def fetch_binance_oi(self, coin: str) -> float:
        """Get latest OI z-score from saved Binance data."""
        sym = coin + "USDT"
        oi_dir = Path("data/raw/binance_public/metrics") / sym
        csvs = sorted(oi_dir.glob("*.csv"))
        if not csvs:
            return 0.0

        # Read last 30 days
        dfs = []
        for f in csvs[-30:]:
            try:
                dfs.append(pd.read_csv(f))
            except Exception:
                continue
        if not dfs:
            return 0.0

        merged = pd.concat(dfs, ignore_index=True)
        oi = merged["sum_open_interest_value"].astype(float)
        if len(oi) < 10:
            return 0.0

        mean = oi.rolling(len(oi), min_periods=10).mean().iloc[-1]
        std = oi.rolling(len(oi), min_periods=10).std().iloc[-1]
        if std > 0:
            return (oi.iloc[-1] - mean) / std
        return 0.0

    # ── Signal Generation ──

    def generate_signal(self, df: pd.DataFrame, coin: str) -> tuple[int, dict]:
        """Generate TSMOM + RSI + CVD + OI signal. Returns (signal, debug_info)."""
        if len(df) < CFG["lookback_days"] * 6 + 10:
            return 0, {"reason": "insufficient_bars"}

        lb_bars = CFG["lookback_days"] * 6

        # TSMOM direction (short-term)
        past_ret_short = df["close"].pct_change(lb_bars).iloc[-1]
        tsmom_short = np.sign(past_ret_short)

        if tsmom_short == 0 or np.isnan(tsmom_short):
            return 0, {"reason": "no_momentum"}

        # Dual lookback: long-term must agree
        if CFG.get("dual_lookback", False):
            lb_long = CFG["lookback_long_days"] * 6
            past_ret_long = df["close"].pct_change(lb_long).iloc[-1]
            tsmom_long = np.sign(past_ret_long)
            if tsmom_short != tsmom_long:
                return 0, {"reason": "dual_disagree",
                           "short": int(tsmom_short), "long": int(tsmom_long)}

        tsmom = tsmom_short

        direction = int(tsmom)

        # RSI filter
        rsi = df["rsi_14"].iloc[-1]
        if direction == 1 and rsi <= 50:
            return 0, {"reason": "rsi_filter", "rsi": rsi, "dir": "LONG"}
        if direction == -1 and rsi >= 50:
            return 0, {"reason": "rsi_filter", "rsi": rsi, "dir": "SHORT"}

        # CVD timing
        cvd = df["cvd_ratio_24"]
        cq = CFG["cvd_quantile"]
        cw = CFG["cvd_roll_window"]
        q_hi = cvd.rolling(cw, min_periods=30).quantile(cq).iloc[-1]
        q_lo = cvd.rolling(cw, min_periods=30).quantile(1 - cq).iloc[-1]
        cvd_now = cvd.iloc[-1]

        cvd_ok = False
        if direction == -1 and cvd_now > q_hi:
            cvd_ok = True
        elif direction == 1 and cvd_now < q_lo:
            cvd_ok = True

        if not cvd_ok:
            return 0, {"reason": "cvd_timing", "cvd": cvd_now,
                       "q_hi": q_hi, "q_lo": q_lo, "dir": "LONG" if direction == 1 else "SHORT"}

        # OI filter
        if CFG["use_oi"]:
            oi_z = self.fetch_binance_oi(coin)
            if abs(oi_z) > CFG["oi_zscore_max"]:
                return 0, {"reason": "oi_crowded", "oi_zscore": oi_z}

        # Compute CVD extremeness for RL state
        cvd_range = q_hi - q_lo if q_hi != q_lo else 1.0
        cvd_ext = abs(cvd_now - (q_hi + q_lo) / 2) / (cvd_range / 2 + 1e-10)
        cvd_ext = min(cvd_ext, 1.0)

        # TSMOM strength
        past_ret = df["close"].pct_change(CFG["lookback_days"] * 6).iloc[-1]
        tsmom_str = abs(past_ret) if not np.isnan(past_ret) else 0.0

        # OI z-score (already fetched or 0)
        oi_z = self.fetch_binance_oi(coin) if CFG["use_oi"] else 0.0

        # Regime classification: 28d agree + CVD Q90 = sniper (A), else steady (B)
        lb_long = CFG["lookback_long_days"] * 6
        past_ret_long = df["close"].pct_change(lb_long).iloc[-1]
        tsmom_long = np.sign(past_ret_long) if not np.isnan(past_ret_long) else 0
        regime_agree = (direction == int(tsmom_long))

        # A sniper requires stricter CVD (Q90)
        if regime_agree:
            cq_a = CFG.get("cvd_quantile_a", 0.90)
            q_hi_a = cvd.rolling(cw, min_periods=30).quantile(cq_a).iloc[-1]
            q_lo_a = cvd.rolling(cw, min_periods=30).quantile(1 - cq_a).iloc[-1]
            cvd_extreme_a = (direction == -1 and cvd_now > q_hi_a) or \
                            (direction == 1 and cvd_now < q_lo_a)
            strategy_type = "A_sniper" if cvd_extreme_a else "B_steady"
        else:
            strategy_type = "B_steady"

        return direction, {
            "tsmom": direction, "rsi": rsi, "cvd": cvd_now,
            "q_hi": q_hi, "q_lo": q_lo,
            "side": "LONG" if direction == 1 else "SHORT",
            "tsmom_strength": tsmom_str,
            "cvd_extremeness": cvd_ext,
            "oi_zscore": oi_z,
            "tsmom_rsi_agree": True,
            "strategy_type": strategy_type,
            "regime_agree": regime_agree,
        }

    # ── Position Management ──

    def open_position(self, coin: str, side: int, df: pd.DataFrame, signal_info: dict = None):
        price = df["close"].iloc[-1]
        atr = df["atr_14"].iloc[-1]

        tp_dist = max(CFG["k_upper"] * atr, price * 0.002)
        sl_dist = max(CFG["k_lower"] * atr, price * 0.002)

        tp = price + tp_dist * side
        sl = price - sl_dist * side

        # v5.3 Dual leverage: A(sniper 3x) vs B(steady 2x)
        strategy_type = signal_info.get("strategy_type", "B_steady") if signal_info else "B_steady"
        if strategy_type == "A_sniper":
            leverage = CFG["leverage_a"]
        else:
            leverage = CFG["leverage_b"]

        # Position sizing: risk_pct * equity / sl_distance
        risk_usd = self.equity * CFG["equity_risk_pct"]
        sl_pct = sl_dist / price
        size_usd = min(risk_usd / sl_pct, self.equity * 0.2) * leverage

        pos = Position(
            coin=coin, side=side, entry_price=price,
            entry_time=datetime.now(timezone.utc).isoformat(),
            entry_bar=0, tp_price=tp, sl_price=sl,
            atr_at_entry=atr, size_usd=size_usd,
        )
        self.positions[coin] = pos

        # RL state logging
        if signal_info:
            hist = self.coin_history.get(coin, [])
            wr5 = sum(1 for p in hist[-5:] if p > 0) / max(len(hist[-5:]), 1)
            avg5 = np.mean(hist[-5:]) if hist else 0.0
            streak = 0
            for p in reversed(hist):
                if p > 0: streak += 1
                else: break

            state = build_rl_state(
                df=df, pred_side="BUY" if side == 1 else "SELL", coin=coin,
                equity=self.equity, daily_pnl=0.0, weekly_pnl=0.0,
                dd_ratio=0.0, open_count=len(self.positions),
                coin_win_rate_5=wr5, coin_avg_pnl_5=avg5,
                coin_streak=streak, bars_since_last=0,
                tsmom_strength=signal_info.get("tsmom_strength", 0.0),
                rsi_value=signal_info.get("rsi", 50.0),
                cvd_extremeness=signal_info.get("cvd_extremeness", 0.0),
                oi_zscore=signal_info.get("oi_zscore", 0.0),
                tsmom_rsi_agree=signal_info.get("tsmom_rsi_agree", True),
            )

            self.signal_logger.log(
                coin=coin, side="BUY" if side == 1 else "SELL",
                regime="UNKNOWN", state=state.tolist(),
                p_trade=signal_info.get("tsmom_strength", 0.0),
                p_direction=signal_info.get("rsi", 50.0) / 100.0,
                action=3, rl_score=0.0,
                entry_price=price, sl_price=sl, tp_price=tp,
                risk_gate_passed=True, executed=True,
            )

        side_str = "LONG" if side == 1 else "SHORT"
        log.info(f"OPEN {side_str} {coin} @ ${price:.4f} | TP=${tp:.4f} SL=${sl:.4f} | "
                 f"size=${size_usd:.2f} ({leverage}x {strategy_type})")

    def check_exit(self, coin: str, df: pd.DataFrame) -> bool:
        """Check if position should be closed."""
        if coin not in self.positions:
            return False

        pos = self.positions[coin]
        current_high = df["high"].iloc[-1]
        current_low = df["low"].iloc[-1]
        current_close = df["close"].iloc[-1]
        pos.bars_held += 1

        exit_price = None
        exit_type = None

        if pos.side == 1:  # LONG
            if current_low <= pos.sl_price:
                exit_price, exit_type = pos.sl_price, "SL"
            elif current_high >= pos.tp_price:
                exit_price, exit_type = pos.tp_price, "TP"
        else:  # SHORT
            if current_high >= pos.sl_price:
                exit_price, exit_type = pos.sl_price, "SL"
            elif current_low <= pos.tp_price:
                exit_price, exit_type = pos.tp_price, "TP"

        # TTL
        if exit_price is None and pos.bars_held >= CFG["max_hold_bars"]:
            exit_price, exit_type = current_close, "TTL"

        if exit_price is not None:
            self.close_position(coin, exit_price, exit_type)
            return True

        return False

    def close_position(self, coin: str, exit_price: float, exit_type: str):
        pos = self.positions[coin]

        if pos.side == 1:
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        pnl_pct_net = pnl_pct - CFG["cost_roundtrip"]
        pnl_usd = pnl_pct_net * pos.size_usd
        self.equity += pnl_usd
        self.trade_count += 1

        side_str = "LONG" if pos.side == 1 else "SHORT"
        log.info(f"CLOSE {side_str} {coin} @ ${exit_price:.4f} | {exit_type} | "
                 f"PnL={pnl_pct_net:+.2%} (${pnl_usd:+.2f}) | "
                 f"bars={pos.bars_held} | equity=${self.equity:.2f}")

        trade = {
            "coin": coin, "side": side_str, "entry_price": pos.entry_price,
            "exit_price": exit_price, "exit_type": exit_type,
            "pnl_pct": pnl_pct, "pnl_net": pnl_pct_net, "pnl_usd": pnl_usd,
            "bars_held": pos.bars_held, "entry_time": pos.entry_time,
            "exit_time": datetime.now(timezone.utc).isoformat(),
            "equity_after": self.equity, "trade_num": self.trade_count,
            "leverage": CFG["leverage"],
        }
        self.log_trade(trade)

        # RL signal result backfill
        self.signal_logger.update_result(
            coin=coin, pnl_pct=pnl_pct_net,
            exit_reason=exit_type, bars_held=pos.bars_held,
        )

        # Update coin history for RL state
        if coin not in self.coin_history:
            self.coin_history[coin] = []
        self.coin_history[coin].append(pnl_pct_net)

        del self.positions[coin]

    # ── Main Loop ──

    async def run(self):
        log.info(f"TSMOM Paper Bot started | equity=${self.equity:.2f} | "
                 f"config: lb={CFG['lookback_days']} cq={CFG['cvd_quantile']} "
                 f"ku={CFG['k_upper']} kl={CFG['k_lower']} lev={CFG['leverage']}x")
        log.info(f"Coins: {', '.join(COINS.keys())}")

        while True:
            try:
                now = datetime.now(timezone.utc)
                log.info(f"=== Cycle @ {now.strftime('%H:%M')} UTC | "
                         f"equity=${self.equity:.2f} | pos={len(self.positions)} ===")

                # Parallel data fetch (all coins at once)
                from concurrent.futures import ThreadPoolExecutor
                coin_data = {}
                with ThreadPoolExecutor(max_workers=min(len(COINS), N_WORKERS)) as pool:
                    futures = {pool.submit(self.fetch_4h, c): c for c in COINS}
                    for future in futures:
                        coin = futures[future]
                        try:
                            coin_data[coin] = future.result()
                        except Exception as e:
                            log.warning(f"[{coin}] Fetch error: {e}")

                # Process signals sequentially (order matters for position limits)
                for coin, df in coin_data.items():
                    try:
                        if df.empty:
                            continue

                        if coin in self.positions:
                            self.check_exit(coin, df)
                            continue

                        if len(self.positions) >= CFG["max_positions"]:
                            continue

                        signal, info = self.generate_signal(df, coin)

                        if signal != 0:
                            self.open_position(coin, signal, df, signal_info=info)
                    except Exception as coin_err:
                        log.warning(f"[{coin}] Error: {coin_err}")
                        continue

                self.save_state()

                # Sleep until next 4h bar
                next_bar = now.replace(minute=0, second=0, microsecond=0)
                next_bar += timedelta(hours=(4 - next_bar.hour % 4) % 4 or 4)
                sleep_sec = (next_bar - now).total_seconds()
                sleep_sec = max(60, min(sleep_sec, BAR_SECONDS))

                log.info(f"Next cycle in {sleep_sec/60:.0f}min")
                await asyncio.sleep(sleep_sec)

            except KeyboardInterrupt:
                log.info("Shutting down...")
                self.save_state()
                self._release_lock()
                break
            except Exception as e:
                log.error(f"Error: {e}", exc_info=True)
                await asyncio.sleep(60)

    def _release_lock(self):
        try:
            LOCK_FILE.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    bot = PaperBot()
    asyncio.run(bot.run())
