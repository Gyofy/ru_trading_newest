"""TSMOM v5.3 Dual Regime Paper Trading Bot.

Strategy:
  A (Sniper): 7d+28d TSMOM agree + RSI + CVD Q90 + OI → 4x leverage
  B (Steady): 7d TSMOM only + RSI + CVD Q75 + OI      → 2x leverage
  Barrier:    TP=5×ATR, SL=2×ATR, TTL=24 bars (96h)
  Universe:   10 coins (BTC ETH SOL XRP ADA DOT LINK DOGE AVAX BNB)

Data source: yfinance (default) or Binance ccxt (DATA_SOURCE=binance)
"""

import sys, os, json, time, logging, asyncio, multiprocessing
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# ── Data Source ──
DATA_SOURCE = os.environ.get("DATA_SOURCE", "yfinance")

try:
    import yfinance as yf
except ImportError:
    yf = None

# ── Logging ──
STATE_DIR = Path("data/reports/tsmom_paper")
STATE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(STATE_DIR / "bot.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("tsmom_paper")

from src.rl.state_builder import build_rl_state
from src.rl.signal_logger import SignalLogger

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
COIN_LIST = list(COINS_YAHOO.keys())

CFG = {
    "lookback_days": 7,
    "lookback_long_days": 28,
    "cvd_quantile": 0.75,
    "cvd_quantile_a": 0.90,
    "cvd_roll_window": 120,
    "k_upper": 5.0,
    "k_lower": 2.0,
    "max_hold_bars": 24,
    "use_oi": True,
    "oi_zscore_max": 2.0,
    "cost_roundtrip": 0.0020,
    "leverage_a": 4,
    "leverage_b": 2,
    "equity_risk_pct": 0.02,
    "max_positions": 3,
}

STATE_FILE = STATE_DIR / "state.json"
TRADES_FILE = STATE_DIR / "trades.jsonl"
LOCK_FILE = STATE_DIR / "bot.lock"

INITIAL_EQUITY = 1000.0
BAR_SECONDS = 4 * 3600
N_WORKERS = max(1, int(multiprocessing.cpu_count() * 0.7))


# ══════════════════════════════════════════════════════════
# Position
# ══════════════════════════════════════════════════════════

@dataclass
class Position:
    coin: str
    side: int
    entry_price: float
    entry_time: str
    entry_bar: int
    tp_price: float
    sl_price: float
    atr_at_entry: float
    size_usd: float
    leverage: int = 2
    strategy_type: str = "B_steady"
    bars_held: int = 0
    status: str = "OPEN"


# ══════════════════════════════════════════════════════════
# Bot
# ══════════════════════════════════════════════════════════

class PaperBot:

    def __init__(self):
        self._acquire_lock()
        self.equity = INITIAL_EQUITY
        self.positions: dict[str, Position] = {}
        self.trade_count = 0
        self.signal_logger = SignalLogger(STATE_DIR / "signal_log.jsonl")
        self.coin_history: dict[str, list[float]] = {}
        self._load_state()

    # ── Lock ──

    def _acquire_lock(self):
        if LOCK_FILE.exists():
            try:
                import psutil
                old_pid = json.loads(LOCK_FILE.read_text()).get("pid", 0)
                if psutil.pid_exists(old_pid):
                    log.error(f"Another instance running (PID {old_pid}). Exiting.")
                    sys.exit(1)
                log.warning(f"Stale lock from PID {old_pid}, taking over.")
            except ImportError:
                if time.time() - LOCK_FILE.stat().st_mtime < 300:
                    log.error("Lock < 5min old. Exiting.")
                    sys.exit(1)
        LOCK_FILE.write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))

    def _release_lock(self):
        LOCK_FILE.unlink(missing_ok=True)

    # ── State ──

    def _load_state(self):
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                s = json.load(f)
            self.equity = s.get("equity", INITIAL_EQUITY)
            self.trade_count = s.get("trade_count", 0)
            for coin, p in s.get("positions", {}).items():
                self.positions[coin] = Position(**p)
            log.info(f"State loaded: equity=${self.equity:.2f}, "
                     f"positions={len(self.positions)}, trades={self.trade_count}")

    def _save_state(self):
        with open(STATE_FILE, "w") as f:
            json.dump({
                "equity": self.equity,
                "trade_count": self.trade_count,
                "positions": {k: asdict(v) for k, v in self.positions.items()},
                "updated": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)

    # ── Data ──

    def fetch_4h(self, coin: str) -> pd.DataFrame:
        if DATA_SOURCE == "binance":
            return self._fetch_binance(coin)
        return self._fetch_yfinance(coin)

    def _fetch_yfinance(self, coin: str) -> pd.DataFrame:
        if yf is None:
            return pd.DataFrame()
        sym = COINS_YAHOO.get(coin, f"{coin}-USD")
        df = yf.download(sym, period="60d", interval="1h", progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                 "Close": "close", "Volume": "volume"})
        df = df.resample("4h").agg({
            "open": lambda x: x.iloc[0] if len(x) > 0 else np.nan,
            "high": "max", "low": "min",
            "close": lambda x: x.iloc[-1] if len(x) > 0 else np.nan,
            "volume": "sum",
        }).dropna()
        return self._add_indicators(df)

    def _fetch_binance(self, coin: str) -> pd.DataFrame:
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
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            return self._add_indicators(df)
        except Exception as e:
            log.warning(f"[{coin}] Binance failed: {e}, fallback yfinance")
            return self._fetch_yfinance(coin)

    @staticmethod
    def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 20:
            return df
        tr = np.maximum(df["high"] - df["low"],
                        np.maximum(np.abs(df["high"] - df["close"].shift(1)),
                                   np.abs(df["low"] - df["close"].shift(1))))
        df["atr_14"] = tr.rolling(14, min_periods=1).mean()
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["rsi_14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        hr = (df["high"] - df["low"]).replace(0, np.nan)
        bf = ((df["close"] - df["low"]) / hr).fillna(0.5).clip(0, 1)
        vd = (2 * bf - 1) * df["volume"]
        cvd = vd.cumsum()
        cvd_ma = cvd.rolling(24, min_periods=6).mean()
        df["cvd_ratio_24"] = ((cvd - cvd_ma) / cvd_ma.abs().replace(0, np.nan)).fillna(0)
        return df

    def _fetch_oi_zscore(self, coin: str) -> float:
        oi_dir = Path("data/raw/binance_public/metrics") / (coin + "USDT")
        csvs = sorted(oi_dir.glob("*.csv")) if oi_dir.exists() else []
        if not csvs:
            return 0.0
        dfs = []
        for f in csvs[-30:]:
            try:
                dfs.append(pd.read_csv(f))
            except Exception:
                continue
        if not dfs:
            return 0.0
        oi = pd.concat(dfs, ignore_index=True)["sum_open_interest_value"].astype(float)
        if len(oi) < 10:
            return 0.0
        mean = oi.rolling(len(oi), min_periods=10).mean().iloc[-1]
        std = oi.rolling(len(oi), min_periods=10).std().iloc[-1]
        return (oi.iloc[-1] - mean) / std if std > 0 else 0.0

    # ── Signal ──

    def generate_signal(self, df: pd.DataFrame, coin: str) -> tuple[int, dict]:
        lb_bars = CFG["lookback_days"] * 6
        if len(df) < lb_bars + 10:
            return 0, {"reason": "insufficient_bars"}

        # Direction
        past_ret = df["close"].pct_change(lb_bars).iloc[-1]
        direction = int(np.sign(past_ret))
        if direction == 0 or np.isnan(past_ret):
            return 0, {"reason": "no_momentum"}

        # RSI
        rsi = df["rsi_14"].iloc[-1]
        if (direction == 1 and rsi <= 50) or (direction == -1 and rsi >= 50):
            return 0, {"reason": "rsi_filter", "rsi": rsi}

        # CVD
        cvd = df["cvd_ratio_24"]
        cw = CFG["cvd_roll_window"]
        cq = CFG["cvd_quantile"]
        q_hi = cvd.rolling(cw, min_periods=30).quantile(cq).iloc[-1]
        q_lo = cvd.rolling(cw, min_periods=30).quantile(1 - cq).iloc[-1]
        cvd_now = cvd.iloc[-1]

        if not ((direction == -1 and cvd_now > q_hi) or (direction == 1 and cvd_now < q_lo)):
            return 0, {"reason": "cvd_timing", "cvd": cvd_now}

        # OI
        oi_z = self._fetch_oi_zscore(coin) if CFG["use_oi"] else 0.0
        if abs(oi_z) > CFG["oi_zscore_max"]:
            return 0, {"reason": "oi_crowded", "oi_zscore": oi_z}

        # Regime: A sniper (7d+28d agree + CVD Q90) vs B steady
        lb_long = CFG["lookback_long_days"] * 6
        ret_long = df["close"].pct_change(lb_long).iloc[-1]
        tsmom_long = int(np.sign(ret_long)) if not np.isnan(ret_long) else 0
        regime_agree = (direction == tsmom_long)

        strategy_type = "B_steady"
        if regime_agree:
            cq_a = CFG["cvd_quantile_a"]
            q_hi_a = cvd.rolling(cw, min_periods=30).quantile(cq_a).iloc[-1]
            q_lo_a = cvd.rolling(cw, min_periods=30).quantile(1 - cq_a).iloc[-1]
            if (direction == -1 and cvd_now > q_hi_a) or (direction == 1 and cvd_now < q_lo_a):
                strategy_type = "A_sniper"

        tsmom_str = abs(past_ret)
        cvd_range = max(q_hi - q_lo, 1e-10)
        cvd_ext = min(abs(cvd_now - (q_hi + q_lo) / 2) / (cvd_range / 2), 1.0)

        return direction, {
            "side": "LONG" if direction == 1 else "SHORT",
            "rsi": rsi, "cvd": cvd_now, "oi_zscore": oi_z,
            "tsmom_strength": tsmom_str, "cvd_extremeness": cvd_ext,
            "strategy_type": strategy_type, "regime_agree": regime_agree,
            "tsmom_rsi_agree": True,
        }

    # ── Position Management ──

    def _open_position(self, coin: str, side: int, df: pd.DataFrame, info: dict):
        price = df["close"].iloc[-1]
        atr = df["atr_14"].iloc[-1]
        tp_d = max(CFG["k_upper"] * atr, price * 0.002)
        sl_d = max(CFG["k_lower"] * atr, price * 0.002)
        tp = price + tp_d * side
        sl = price - sl_d * side

        strategy_type = info.get("strategy_type", "B_steady")
        leverage = CFG["leverage_a"] if strategy_type == "A_sniper" else CFG["leverage_b"]

        risk_usd = self.equity * CFG["equity_risk_pct"]
        sl_pct = sl_d / price
        size_usd = min(risk_usd / sl_pct, self.equity * 0.2) * leverage

        self.positions[coin] = Position(
            coin=coin, side=side, entry_price=price,
            entry_time=datetime.now(timezone.utc).isoformat(),
            entry_bar=0, tp_price=tp, sl_price=sl,
            atr_at_entry=atr, size_usd=size_usd,
            leverage=leverage, strategy_type=strategy_type,
        )

        # RL logging
        hist = self.coin_history.get(coin, [])
        wr5 = sum(1 for p in hist[-5:] if p > 0) / max(len(hist[-5:]), 1)
        avg5 = float(np.mean(hist[-5:])) if hist else 0.0
        streak = 0
        for p in reversed(hist):
            if p > 0:
                streak += 1
            else:
                break

        state = build_rl_state(
            df=df, pred_side="BUY" if side == 1 else "SELL", coin=coin,
            equity=self.equity, daily_pnl=0.0, weekly_pnl=0.0,
            dd_ratio=0.0, open_count=len(self.positions),
            coin_win_rate_5=wr5, coin_avg_pnl_5=avg5,
            coin_streak=streak, bars_since_last=0,
            tsmom_strength=info.get("tsmom_strength", 0.0),
            rsi_value=info.get("rsi", 50.0),
            cvd_extremeness=info.get("cvd_extremeness", 0.0),
            oi_zscore=info.get("oi_zscore", 0.0),
            tsmom_rsi_agree=True,
        )
        self.signal_logger.log(
            coin=coin, side="BUY" if side == 1 else "SELL",
            regime="UNKNOWN", state=state.tolist(),
            p_trade=info.get("tsmom_strength", 0.0),
            p_direction=info.get("rsi", 50.0) / 100.0,
            action=3, rl_score=0.0,
            entry_price=price, sl_price=sl, tp_price=tp,
            risk_gate_passed=True, executed=True,
        )

        side_str = "LONG" if side == 1 else "SHORT"
        log.info(f"OPEN {side_str} {coin} @ ${price:.4f} | TP=${tp:.4f} SL=${sl:.4f} | "
                 f"size=${size_usd:.2f} ({leverage}x {strategy_type})")

    def _check_exit(self, coin: str, df: pd.DataFrame) -> bool:
        if coin not in self.positions:
            return False
        pos = self.positions[coin]
        hi, lo, cl = df["high"].iloc[-1], df["low"].iloc[-1], df["close"].iloc[-1]
        pos.bars_held += 1

        exit_price, exit_type = None, None
        if pos.side == 1:
            if lo <= pos.sl_price:
                exit_price, exit_type = pos.sl_price, "SL"
            elif hi >= pos.tp_price:
                exit_price, exit_type = pos.tp_price, "TP"
        else:
            if hi >= pos.sl_price:
                exit_price, exit_type = pos.sl_price, "SL"
            elif lo <= pos.tp_price:
                exit_price, exit_type = pos.tp_price, "TP"

        if exit_price is None and pos.bars_held >= CFG["max_hold_bars"]:
            exit_price, exit_type = cl, "TTL"

        if exit_price is not None:
            self._close_position(coin, exit_price, exit_type)
            return True
        return False

    def _close_position(self, coin: str, exit_price: float, exit_type: str):
        pos = self.positions[coin]
        pnl_pct = ((exit_price - pos.entry_price) / pos.entry_price) * pos.side
        pnl_net = pnl_pct - CFG["cost_roundtrip"]
        pnl_usd = pnl_net * pos.size_usd
        self.equity += pnl_usd
        self.trade_count += 1

        side_str = "LONG" if pos.side == 1 else "SHORT"
        log.info(f"CLOSE {side_str} {coin} @ ${exit_price:.4f} | {exit_type} | "
                 f"PnL={pnl_net:+.2%} (${pnl_usd:+.2f}) | "
                 f"bars={pos.bars_held} | equity=${self.equity:.2f}")

        with open(TRADES_FILE, "a") as f:
            f.write(json.dumps({
                "coin": coin, "side": side_str, "entry_price": pos.entry_price,
                "exit_price": exit_price, "exit_type": exit_type,
                "pnl_pct": pnl_pct, "pnl_net": pnl_net, "pnl_usd": pnl_usd,
                "bars_held": pos.bars_held, "entry_time": pos.entry_time,
                "exit_time": datetime.now(timezone.utc).isoformat(),
                "equity_after": self.equity, "trade_num": self.trade_count,
                "leverage": pos.leverage, "strategy_type": pos.strategy_type,
            }, default=str) + "\n")

        self.signal_logger.update_result(
            coin=coin, pnl_pct=pnl_net,
            exit_reason=exit_type, bars_held=pos.bars_held,
        )
        if coin not in self.coin_history:
            self.coin_history[coin] = []
        self.coin_history[coin].append(pnl_net)
        del self.positions[coin]

    # ── Main Loop ──

    async def run(self):
        log.info(f"v5.3 Paper Bot | equity=${self.equity:.2f} | "
                 f"A={CFG['leverage_a']}x(Q{CFG['cvd_quantile_a']:.0%}) "
                 f"B={CFG['leverage_b']}x(Q{CFG['cvd_quantile']:.0%}) | "
                 f"SL={CFG['k_lower']}×ATR | {len(COIN_LIST)} coins | "
                 f"source={DATA_SOURCE}")

        while True:
            try:
                now = datetime.now(timezone.utc)
                log.info(f"=== Cycle @ {now.strftime('%H:%M')} UTC | "
                         f"equity=${self.equity:.2f} | pos={len(self.positions)} ===")

                # Parallel fetch
                coin_data = {}
                with ThreadPoolExecutor(max_workers=min(len(COIN_LIST), N_WORKERS)) as pool:
                    futures = {pool.submit(self.fetch_4h, c): c for c in COIN_LIST}
                    for fut in futures:
                        coin = futures[fut]
                        try:
                            coin_data[coin] = fut.result()
                        except Exception as e:
                            log.warning(f"[{coin}] Fetch error: {e}")

                # Sequential signal processing
                for coin, df in coin_data.items():
                    try:
                        if df.empty:
                            continue
                        if coin in self.positions:
                            self._check_exit(coin, df)
                            continue
                        if len(self.positions) >= CFG["max_positions"]:
                            continue
                        signal, info = self.generate_signal(df, coin)
                        if signal != 0:
                            self._open_position(coin, signal, df, info)
                    except Exception as e:
                        log.warning(f"[{coin}] Error: {e}")

                self._save_state()

                # Sleep
                next_bar = now.replace(minute=0, second=0, microsecond=0)
                next_bar += timedelta(hours=(4 - next_bar.hour % 4) % 4 or 4)
                sleep_sec = max(60, min((next_bar - now).total_seconds(), BAR_SECONDS))
                log.info(f"Next cycle in {sleep_sec/60:.0f}min")
                await asyncio.sleep(sleep_sec)

            except KeyboardInterrupt:
                log.info("Shutting down...")
                self._save_state()
                self._release_lock()
                break
            except Exception as e:
                log.error(f"Error: {e}", exc_info=True)
                await asyncio.sleep(60)


if __name__ == "__main__":
    bot = PaperBot()
    asyncio.run(bot.run())
