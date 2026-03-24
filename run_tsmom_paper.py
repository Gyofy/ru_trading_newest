"""v6.0 Honest Portfolio Paper Trading Bot.

Strategy:
  1. BTC 7d return → Global Direction (LONG/SHORT)
  2. Alt 9 coins: relative strength vs BTC → top/bottom 2
  3. RSI + CVD Q75 + OI filters on selected coins
  4. BTC direction flip → immediate exit (DIR_FLIP)
  5. TP=5×ATR, SL=2×ATR, TTL=24 bars, max 2 positions

Data: yfinance (default) or Binance ccxt (DATA_SOURCE=binance)
"""

import sys, os, json, time, logging, asyncio, multiprocessing
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

DATA_SOURCE = os.environ.get("DATA_SOURCE", "yfinance")
try:
    import yfinance as yf
except ImportError:
    yf = None

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
ALTS = ["ETH", "SOL", "XRP", "ADA", "DOT", "LINK", "DOGE", "AVAX", "BNB"]
ALL_COINS = ["BTC"] + ALTS

CFG = {
    "lookback_bars": 42,       # 7d × 6 bars/day
    "cvd_quantile": 0.75,
    "cvd_roll_window": 120,
    "k_upper": 5.0,
    "k_lower": 2.0,
    "max_hold_bars": 24,
    "oi_zscore_max": 2.0,
    "cost_roundtrip": 0.0020,
    "leverage": 2,
    "equity_risk_pct": 0.02,
    "max_positions": 2,
    "top_n": 2,                # select top N by relative strength
}

STATE_FILE = STATE_DIR / "state.json"
TRADES_FILE = STATE_DIR / "trades.jsonl"
LOCK_FILE = STATE_DIR / "bot.lock"
INITIAL_EQUITY = 1000.0
BAR_SECONDS = 4 * 3600
N_WORKERS = max(1, int(multiprocessing.cpu_count() * 0.7))


@dataclass
class Position:
    coin: str
    side: int
    entry_price: float
    entry_time: str
    tp_price: float
    sl_price: float
    atr_at_entry: float
    size_usd: float
    leverage: int = 2
    bars_held: int = 0


class PaperBot:

    def __init__(self):
        self._acquire_lock()
        self.equity = INITIAL_EQUITY
        self.positions: dict[str, Position] = {}
        self.trade_count = 0
        self.signal_logger = SignalLogger(STATE_DIR / "signal_log.jsonl")
        self.coin_data: dict[str, pd.DataFrame] = {}
        self._load_state()

    def _acquire_lock(self):
        if LOCK_FILE.exists():
            try:
                import psutil
                old_pid = json.loads(LOCK_FILE.read_text()).get("pid", 0)
                if psutil.pid_exists(old_pid):
                    log.error(f"Another instance (PID {old_pid}). Exiting.")
                    sys.exit(1)
            except ImportError:
                if time.time() - LOCK_FILE.stat().st_mtime < 300:
                    log.error("Lock < 5min. Exiting.")
                    sys.exit(1)
        LOCK_FILE.write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))

    def _release_lock(self):
        LOCK_FILE.unlink(missing_ok=True)

    def _load_state(self):
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                s = json.load(f)
            self.equity = s.get("equity", INITIAL_EQUITY)
            self.trade_count = s.get("trade_count", 0)
            for coin, p in s.get("positions", {}).items():
                self.positions[coin] = Position(**p)
            log.info(f"State: ${self.equity:.2f}, {len(self.positions)} pos, {self.trade_count} trades")

    def _save_state(self):
        with open(STATE_FILE, "w") as f:
            json.dump({
                "equity": self.equity,
                "trade_count": self.trade_count,
                "positions": {k: asdict(v) for k, v in self.positions.items()},
                "updated": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)

    # ── Data ──

    def _fetch_4h(self, coin: str) -> pd.DataFrame:
        if DATA_SOURCE == "binance":
            return self._fetch_binance(coin)
        return self._fetch_yfinance(coin)

    def _fetch_yfinance(self, coin: str) -> pd.DataFrame:
        if yf is None: return pd.DataFrame()
        sym = COINS_YAHOO.get(coin, f"{coin}-USD")
        df = yf.download(sym, period="60d", interval="1h", progress=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
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
            ex = ccxt.binanceusdm({
                "apiKey": os.environ.get("BINANCE_API_KEY", ""),
                "secret": os.environ.get("BINANCE_API_SECRET", ""),
                "options": {"defaultType": "future"},
            })
            sym = COINS_BINANCE.get(coin, f"{coin}/USDT:USDT")
            ohlcv = ex.fetch_ohlcv(sym, "4h", limit=500)
            if not ohlcv: return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            return self._add_indicators(df)
        except Exception as e:
            log.warning(f"[{coin}] Binance fail: {e}")
            return self._fetch_yfinance(coin)

    @staticmethod
    def _add_indicators(df):
        if len(df) < 20: return df
        tr = np.maximum(df["high"]-df["low"], np.maximum(
            np.abs(df["high"]-df["close"].shift(1)),
            np.abs(df["low"]-df["close"].shift(1))))
        df["atr_14"] = tr.rolling(14, min_periods=1).mean()
        delta = df["close"].diff()
        gain = delta.where(delta>0,0).rolling(14).mean()
        loss = (-delta.where(delta<0,0)).rolling(14).mean()
        df["rsi_14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        hr = (df["high"]-df["low"]).replace(0, np.nan)
        bf = ((df["close"]-df["low"])/hr).fillna(0.5).clip(0,1)
        vd = (2*bf-1)*df["volume"]
        cvd = vd.cumsum()
        cvd_ma = cvd.rolling(24, min_periods=6).mean()
        df["cvd_ratio_24"] = ((cvd-cvd_ma)/cvd_ma.abs().replace(0,np.nan)).fillna(0)
        return df

    def _fetch_oi_zscore(self, coin):
        oi_dir = Path("data/raw/binance_public/metrics") / (coin+"USDT")
        csvs = sorted(oi_dir.glob("*.csv")) if oi_dir.exists() else []
        if not csvs: return 0.0
        dfs = []
        for f in csvs[-30:]:
            try: dfs.append(pd.read_csv(f))
            except: continue
        if not dfs: return 0.0
        oi = pd.concat(dfs, ignore_index=True)["sum_open_interest_value"].astype(float)
        if len(oi) < 10: return 0.0
        m = oi.rolling(len(oi), min_periods=10).mean().iloc[-1]
        s = oi.rolling(len(oi), min_periods=10).std().iloc[-1]
        return (oi.iloc[-1]-m)/s if s > 0 else 0.0

    # ── Signal ──

    def _get_global_direction(self) -> int:
        btc = self.coin_data.get("BTC")
        if btc is None or len(btc) < CFG["lookback_bars"] + 10:
            return 0
        ret = btc["close"].pct_change(CFG["lookback_bars"]).iloc[-1]
        return int(np.sign(ret)) if not np.isnan(ret) else 0

    def _rank_alts(self, direction: int) -> list[tuple[str, dict]]:
        btc = self.coin_data.get("BTC")
        if btc is None: return []
        btc_ret = btc["close"].pct_change(CFG["lookback_bars"]).iloc[-1]
        if np.isnan(btc_ret): return []

        candidates = []
        for coin in ALTS:
            if coin in self.positions: continue
            df = self.coin_data.get(coin)
            if df is None or len(df) < CFG["lookback_bars"] + 30: continue

            coin_ret = df["close"].pct_change(CFG["lookback_bars"]).iloc[-1]
            if np.isnan(coin_ret): continue
            relative_ret = coin_ret - btc_ret

            rsi = df["rsi_14"].iloc[-1] if "rsi_14" in df.columns else 50
            if np.isnan(rsi): continue
            if direction == 1 and rsi <= 50: continue
            if direction == -1 and rsi >= 50: continue

            cvd = df["cvd_ratio_24"]
            cw = CFG["cvd_roll_window"]
            q_hi = cvd.rolling(cw, min_periods=30).quantile(CFG["cvd_quantile"]).iloc[-1]
            q_lo = cvd.rolling(cw, min_periods=30).quantile(1-CFG["cvd_quantile"]).iloc[-1]
            cvd_now = cvd.iloc[-1]
            if direction == -1 and cvd_now <= q_hi: continue
            if direction == 1 and cvd_now >= q_lo: continue

            oi_z = self._fetch_oi_zscore(coin)
            if abs(oi_z) > CFG["oi_zscore_max"]: continue

            atr = df["atr_14"].iloc[-1] if "atr_14" in df.columns else 0
            if np.isnan(atr) or atr <= 0: continue

            candidates.append((coin, {
                "relative_ret": relative_ret, "rsi": rsi,
                "cvd": cvd_now, "atr": atr, "oi_z": oi_z,
                "close": df["close"].iloc[-1],
            }))

        if direction == 1:
            candidates.sort(key=lambda x: -x[1]["relative_ret"])
        else:
            candidates.sort(key=lambda x: x[1]["relative_ret"])

        return candidates[:CFG["top_n"]]

    # ── Position ──

    def _open(self, coin: str, side: int, info: dict):
        price = info["close"]
        atr = info["atr"]
        tp_d = max(CFG["k_upper"]*atr, price*0.002)
        sl_d = max(CFG["k_lower"]*atr, price*0.002)
        tp = price + tp_d*side
        sl = price - sl_d*side
        lev = CFG["leverage"]
        risk = self.equity * CFG["equity_risk_pct"]
        size = min(risk / (sl_d/price), self.equity*0.2) * lev

        self.positions[coin] = Position(
            coin=coin, side=side, entry_price=price,
            entry_time=datetime.now(timezone.utc).isoformat(),
            tp_price=tp, sl_price=sl, atr_at_entry=atr,
            size_usd=size, leverage=lev,
        )
        side_str = "LONG" if side == 1 else "SHORT"
        log.info(f"OPEN {side_str} {coin} @ ${price:.4f} | TP=${tp:.4f} SL=${sl:.4f} | "
                 f"${size:.0f} ({lev}x) rel_ret={info['relative_ret']:+.2%}")

    def _check_exit(self, coin: str, global_dir: int) -> bool:
        if coin not in self.positions: return False
        pos = self.positions[coin]
        df = self.coin_data.get(coin)
        if df is None: return False

        hi, lo, cl = df["high"].iloc[-1], df["low"].iloc[-1], df["close"].iloc[-1]
        pos.bars_held += 1

        ep, et = None, None
        if pos.side == 1:
            if lo <= pos.sl_price: ep, et = pos.sl_price, "SL"
            elif hi >= pos.tp_price: ep, et = pos.tp_price, "TP"
        else:
            if hi >= pos.sl_price: ep, et = pos.sl_price, "SL"
            elif lo <= pos.tp_price: ep, et = pos.tp_price, "TP"

        if ep is None and pos.bars_held >= CFG["max_hold_bars"]:
            ep, et = cl, "TTL"

        if ep is None and global_dir != pos.side and pos.bars_held >= 3:
            ep, et = cl, "DIR_FLIP"

        if ep is not None:
            self._close(coin, ep, et)
            return True
        return False

    def _close(self, coin: str, exit_price: float, exit_type: str):
        pos = self.positions[coin]
        pnl = ((exit_price-pos.entry_price)/pos.entry_price)*pos.side - CFG["cost_roundtrip"]
        pnl_usd = pnl * pos.size_usd
        self.equity += pnl_usd
        self.trade_count += 1

        side_str = "LONG" if pos.side == 1 else "SHORT"
        log.info(f"CLOSE {side_str} {coin} @ ${exit_price:.4f} | {exit_type} | "
                 f"PnL={pnl:+.2%} (${pnl_usd:+.2f}) bars={pos.bars_held} eq=${self.equity:.2f}")

        with open(TRADES_FILE, "a") as f:
            f.write(json.dumps({
                "coin": coin, "side": side_str, "entry_price": pos.entry_price,
                "exit_price": exit_price, "exit_type": exit_type,
                "pnl_pct": pnl, "pnl_usd": pnl_usd,
                "bars_held": pos.bars_held, "entry_time": pos.entry_time,
                "exit_time": datetime.now(timezone.utc).isoformat(),
                "equity_after": self.equity, "trade_num": self.trade_count,
                "leverage": pos.leverage,
            }, default=str) + "\n")

        self.signal_logger.update_result(
            coin=coin, pnl_pct=pnl, exit_reason=exit_type, bars_held=pos.bars_held)
        del self.positions[coin]

    # ── Loop ──

    async def run(self):
        log.info(f"v6.0 Paper Bot | ${self.equity:.2f} | max_pos={CFG['max_positions']} | "
                 f"lev={CFG['leverage']}x | source={DATA_SOURCE}")

        while True:
            try:
                now = datetime.now(timezone.utc)
                log.info(f"=== {now.strftime('%H:%M')} UTC | ${self.equity:.2f} | "
                         f"pos={len(self.positions)} ===")

                # Fetch all coins
                with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
                    futures = {pool.submit(self._fetch_4h, c): c for c in ALL_COINS}
                    for fut in futures:
                        coin = futures[fut]
                        try:
                            self.coin_data[coin] = fut.result()
                        except Exception as e:
                            log.warning(f"[{coin}] Fetch: {e}")

                # Global direction
                gdir = self._get_global_direction()
                gdir_str = "LONG" if gdir == 1 else ("SHORT" if gdir == -1 else "FLAT")
                log.info(f"BTC direction: {gdir_str}")

                # Exits
                for coin in list(self.positions.keys()):
                    try:
                        self._check_exit(coin, gdir)
                    except Exception as e:
                        log.warning(f"[{coin}] Exit check: {e}")

                # Entries
                if gdir != 0 and len(self.positions) < CFG["max_positions"]:
                    ranked = self._rank_alts(gdir)
                    slots = CFG["max_positions"] - len(self.positions)
                    for coin, info in ranked[:slots]:
                        try:
                            self._open(coin, gdir, info)
                        except Exception as e:
                            log.warning(f"[{coin}] Entry: {e}")
                elif gdir == 0:
                    log.info("No BTC momentum → FLAT")

                self._save_state()

                next_bar = now.replace(minute=0, second=0, microsecond=0)
                next_bar += timedelta(hours=(4 - next_bar.hour % 4) % 4 or 4)
                sleep = max(60, min((next_bar - now).total_seconds(), BAR_SECONDS))
                log.info(f"Next in {sleep/60:.0f}min")
                await asyncio.sleep(sleep)

            except KeyboardInterrupt:
                self._save_state()
                self._release_lock()
                break
            except Exception as e:
                log.error(f"Error: {e}", exc_info=True)
                await asyncio.sleep(60)


if __name__ == "__main__":
    bot = PaperBot()
    asyncio.run(bot.run())
