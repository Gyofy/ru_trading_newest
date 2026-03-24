"""v6.1 Paper Trading Bot — 1h resolution + continuous trading.

Strategy:
  1. BTC 7d return → Global Direction
  2. Alt relative strength → top 2
  3. RSI + CVD + OI filters
  4. DIR_FLIP exit on BTC reversal
  5. 1h cycle: check every hour, enter immediately when conditions met

Data: Binance ccxt (default) or yfinance (DATA_SOURCE=yfinance)
Logs: bar-by-bar trajectory for future RL training
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
TRAJ_FILE = STATE_DIR / "trajectories.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(STATE_DIR / "bot.log", encoding="utf-8"),
    ]
)
log = logging.getLogger("v6.1")

from src.rl.signal_logger import SignalLogger

# ══════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════

COINS_YAHOO = {
    "BTC":"BTC-USD","ETH":"ETH-USD","SOL":"SOL-USD","XRP":"XRP-USD",
    "ADA":"ADA-USD","DOT":"DOT-USD","LINK":"LINK-USD",
    "DOGE":"DOGE-USD","AVAX":"AVAX-USD","BNB":"BNB-USD",
}
COINS_BINANCE = {
    "BTC":"BTC/USDT:USDT","ETH":"ETH/USDT:USDT","SOL":"SOL/USDT:USDT",
    "XRP":"XRP/USDT:USDT","ADA":"ADA/USDT:USDT","DOT":"DOT/USDT:USDT",
    "LINK":"LINK/USDT:USDT","DOGE":"DOGE/USDT:USDT","AVAX":"AVAX/USDT:USDT",
    "BNB":"BNB/USDT:USDT",
}
ALTS = ["ETH","SOL","XRP","ADA","DOT","LINK","DOGE","AVAX","BNB"]
ALL_COINS = ["BTC"] + ALTS

CFG = {
    "lookback_bars": 168,       # 7d × 24h = 168 1h-bars
    "cvd_quantile": 0.75,
    "cvd_roll_window": 480,     # 120 4h-bars × 4 = 480 1h-bars
    "k_upper": 5.0,
    "k_lower": 2.0,
    "max_hold_bars": 96,        # 24 4h-bars × 4 = 96 1h-bars
    "oi_zscore_max": 2.0,
    "cost_roundtrip": 0.0020,
    "leverage": 2,
    "equity_risk_pct": 0.02,
    "max_positions": 2,
    "top_n": 2,
    "cycle_seconds": 3600,      # 1h cycle (check every hour)
    "cooldown_bars": 4,         # 4h cooldown after exit
}

STATE_FILE = STATE_DIR / "state.json"
TRADES_FILE = STATE_DIR / "trades.jsonl"
LOCK_FILE = STATE_DIR / "bot.lock"
INITIAL_EQUITY = 1000.0
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
        self.cooldowns: dict[str, int] = {}  # coin → cycle count when available
        self.cycle_count = 0
        self._load_state()

    def _acquire_lock(self):
        if LOCK_FILE.exists():
            try:
                import psutil
                pid = json.loads(LOCK_FILE.read_text()).get("pid", 0)
                if psutil.pid_exists(pid):
                    log.error(f"PID {pid} running. Exit.")
                    sys.exit(1)
            except ImportError:
                if time.time() - LOCK_FILE.stat().st_mtime < 300:
                    log.error("Lock fresh. Exit.")
                    sys.exit(1)
        LOCK_FILE.write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))

    def _release_lock(self):
        LOCK_FILE.unlink(missing_ok=True)

    def _load_state(self):
        if STATE_FILE.exists():
            s = json.loads(STATE_FILE.read_text())
            self.equity = s.get("equity", INITIAL_EQUITY)
            self.trade_count = s.get("trade_count", 0)
            for coin, p in s.get("positions", {}).items():
                self.positions[coin] = Position(**p)
            log.info(f"State: ${self.equity:.2f}, {len(self.positions)} pos, {self.trade_count} trades")

    def _save_state(self):
        STATE_FILE.write_text(json.dumps({
            "equity": self.equity,
            "trade_count": self.trade_count,
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "updated": datetime.now(timezone.utc).isoformat(),
        }, indent=2))

    # ── Data ──

    def _fetch_1h(self, coin: str) -> pd.DataFrame:
        if DATA_SOURCE == "binance":
            return self._fetch_binance(coin)
        return self._fetch_yfinance(coin)

    def _fetch_yfinance(self, coin: str) -> pd.DataFrame:
        if yf is None: return pd.DataFrame()
        sym = COINS_YAHOO.get(coin, f"{coin}-USD")
        df = yf.download(sym, period="30d", interval="1h", progress=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
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
            ohlcv = ex.fetch_ohlcv(sym, "1h", limit=500)
            if not ohlcv: return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            return self._add_indicators(df)
        except Exception as e:
            log.warning(f"[{coin}] Binance: {e}")
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
        df["rsi_14"] = 100-(100/(1+gain/loss.replace(0,np.nan)))
        hr = (df["high"]-df["low"]).replace(0,np.nan)
        bf = ((df["close"]-df["low"])/hr).fillna(0.5).clip(0,1)
        vd = (2*bf-1)*df["volume"]
        cvd = vd.cumsum()
        cvd_ma = cvd.rolling(24, min_periods=6).mean()
        df["cvd_ratio"] = ((cvd-cvd_ma)/cvd_ma.abs().replace(0,np.nan)).fillna(0)
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
        m, s = oi.rolling(len(oi),min_periods=10).mean().iloc[-1], oi.rolling(len(oi),min_periods=10).std().iloc[-1]
        return (oi.iloc[-1]-m)/s if s > 0 else 0.0

    # ── Signal ──

    def _get_global_direction(self) -> int:
        btc = self.coin_data.get("BTC")
        if btc is None or len(btc) < CFG["lookback_bars"]+10: return 0
        ret = btc["close"].pct_change(CFG["lookback_bars"]).iloc[-1]
        return int(np.sign(ret)) if not np.isnan(ret) else 0

    def _rank_alts(self, direction: int) -> list:
        btc = self.coin_data.get("BTC")
        if btc is None: return []
        btc_ret = btc["close"].pct_change(CFG["lookback_bars"]).iloc[-1]
        if np.isnan(btc_ret): return []

        cands = []
        for coin in ALTS:
            if coin in self.positions: continue
            if self.cooldowns.get(coin, 0) > self.cycle_count: continue
            df = self.coin_data.get(coin)
            if df is None or len(df) < CFG["lookback_bars"]+30: continue

            coin_ret = df["close"].pct_change(CFG["lookback_bars"]).iloc[-1]
            if np.isnan(coin_ret): continue
            rel = coin_ret - btc_ret

            rsi = df["rsi_14"].iloc[-1] if "rsi_14" in df.columns else 50
            if np.isnan(rsi): continue
            if direction==1 and rsi<=50: continue
            if direction==-1 and rsi>=50: continue

            cvd = df["cvd_ratio"]
            cw = CFG["cvd_roll_window"]
            q_hi = cvd.rolling(cw, min_periods=100).quantile(CFG["cvd_quantile"]).iloc[-1]
            q_lo = cvd.rolling(cw, min_periods=100).quantile(1-CFG["cvd_quantile"]).iloc[-1]
            cvd_now = cvd.iloc[-1]
            if np.isnan(q_hi) or np.isnan(q_lo): continue
            if direction==-1 and cvd_now<=q_hi: continue
            if direction==1 and cvd_now>=q_lo: continue

            oi_z = self._fetch_oi_zscore(coin)
            if abs(oi_z) > CFG["oi_zscore_max"]: continue

            atr = df["atr_14"].iloc[-1] if "atr_14" in df.columns else 0
            if np.isnan(atr) or atr<=0: continue

            cands.append((coin, {"relative_ret":rel, "rsi":rsi, "cvd":cvd_now,
                                  "atr":atr, "oi_z":oi_z, "close":df["close"].iloc[-1]}))

        if direction==1: cands.sort(key=lambda x: -x[1]["relative_ret"])
        else: cands.sort(key=lambda x: x[1]["relative_ret"])
        return cands[:CFG["top_n"]]

    # ── Position ──

    def _open(self, coin, side, info):
        price, atr = info["close"], info["atr"]
        tp_d = max(CFG["k_upper"]*atr, price*0.002)
        sl_d = max(CFG["k_lower"]*atr, price*0.002)
        lev = CFG["leverage"]
        risk = self.equity * CFG["equity_risk_pct"]
        size = min(risk/(sl_d/price), self.equity*0.2) * lev

        self.positions[coin] = Position(
            coin=coin, side=side, entry_price=price,
            entry_time=datetime.now(timezone.utc).isoformat(),
            tp_price=price+tp_d*side, sl_price=price-sl_d*side,
            atr_at_entry=atr, size_usd=size, leverage=lev,
        )
        side_str = "LONG" if side==1 else "SHORT"
        log.info(f"OPEN {side_str} {coin} @ ${price:.4f} | TP=${price+tp_d*side:.4f} "
                 f"SL=${price-sl_d*side:.4f} | ${size:.0f} ({lev}x) "
                 f"rel={info['relative_ret']:+.2%}")

        # Log entry snapshot (market state at entry)
        self._log_trade_analysis("ENTRY", coin, {
            "price": price, "side": side_str,
            "atr": atr, "atr_pct": atr/price,
            "rsi": info.get("rsi", 50),
            "cvd": info.get("cvd", 0),
            "oi_zscore": info.get("oi_z", 0),
            "relative_strength": info.get("relative_ret", 0),
            "btc_7d_return": self.coin_data["BTC"]["close"].pct_change(CFG["lookback_bars"]).iloc[-1]
                if "BTC" in self.coin_data else 0,
            "sl_distance_pct": sl_d/price,
            "tp_distance_pct": tp_d/price,
            "leverage": lev,
            "equity": self.equity,
        })

    def _check_exit(self, coin, gdir):
        if coin not in self.positions: return False
        pos = self.positions[coin]
        df = self.coin_data.get(coin)
        if df is None: return False

        hi, lo, cl = df["high"].iloc[-1], df["low"].iloc[-1], df["close"].iloc[-1]
        pos.bars_held += 1

        ep, et = None, None
        if pos.side==1:
            if lo<=pos.sl_price: ep, et = pos.sl_price, "SL"
            elif hi>=pos.tp_price: ep, et = pos.tp_price, "TP"
        else:
            if hi>=pos.sl_price: ep, et = pos.sl_price, "SL"
            elif lo<=pos.tp_price: ep, et = pos.tp_price, "TP"
        if ep is None and pos.bars_held>=CFG["max_hold_bars"]: ep, et = cl, "TTL"
        if ep is None and gdir!=pos.side and pos.bars_held>=6: ep, et = cl, "DIR_FLIP"  # 6 1h-bars = 6h delay

        if ep is not None:
            self._close(coin, ep, et)
            return True

        # Trajectory logging (every bar while position open)
        self._log_trajectory(coin, pos, cl)
        return False

    def _close(self, coin, exit_price, exit_type):
        pos = self.positions[coin]
        pnl = ((exit_price-pos.entry_price)/pos.entry_price)*pos.side - CFG["cost_roundtrip"]
        pnl_usd = pnl * pos.size_usd
        self.equity += pnl_usd
        self.trade_count += 1
        self.cooldowns[coin] = self.cycle_count + CFG["cooldown_bars"]

        side_str = "LONG" if pos.side==1 else "SHORT"
        log.info(f"CLOSE {side_str} {coin} @ ${exit_price:.4f} | {exit_type} | "
                 f"PnL={pnl:+.2%} (${pnl_usd:+.2f}) bars={pos.bars_held} eq=${self.equity:.2f}")

        with open(TRADES_FILE, "a") as f:
            f.write(json.dumps({
                "coin":coin, "side":side_str, "entry_price":pos.entry_price,
                "exit_price":exit_price, "exit_type":exit_type,
                "pnl_pct":pnl, "pnl_usd":pnl_usd,
                "bars_held":pos.bars_held, "entry_time":pos.entry_time,
                "exit_time":datetime.now(timezone.utc).isoformat(),
                "equity_after":self.equity, "trade_num":self.trade_count,
                "leverage":pos.leverage,
            }, default=str) + "\n")

        self.signal_logger.update_result(
            coin=coin, pnl_pct=pnl, exit_reason=exit_type, bars_held=pos.bars_held)

        # Log exit analysis (market state + cause diagnosis)
        df = self.coin_data.get(coin)
        exit_rsi = df["rsi_14"].iloc[-1] if df is not None and "rsi_14" in df.columns else 0
        exit_cvd = df["cvd_ratio"].iloc[-1] if df is not None and "cvd_ratio" in df.columns else 0
        exit_atr = df["atr_14"].iloc[-1] if df is not None and "atr_14" in df.columns else 0

        # Diagnose cause
        if exit_type == "SL":
            if exit_atr > pos.atr_at_entry * 1.5:
                cause = "volatility_spike"
            elif (pos.side == 1 and exit_rsi < 30) or (pos.side == -1 and exit_rsi > 70):
                cause = "trend_reversal"
            else:
                cause = "noise_stop"
        elif exit_type == "TP":
            cause = "trend_continuation"
        elif exit_type == "DIR_FLIP":
            cause = "btc_direction_change"
        elif exit_type == "TTL":
            if pnl > 0:
                cause = "slow_profit"
            else:
                cause = "no_momentum"
        else:
            cause = "unknown"

        self._log_trade_analysis("EXIT", coin, {
            "exit_price": exit_price,
            "exit_type": exit_type,
            "pnl_pct": pnl,
            "pnl_usd": pnl_usd,
            "bars_held": pos.bars_held,
            "cause": cause,
            "entry_price": pos.entry_price,
            "entry_atr": pos.atr_at_entry,
            "exit_atr": exit_atr,
            "atr_change": exit_atr / pos.atr_at_entry if pos.atr_at_entry > 0 else 1,
            "exit_rsi": exit_rsi,
            "exit_cvd": exit_cvd,
            "btc_7d_return": self.coin_data["BTC"]["close"].pct_change(CFG["lookback_bars"]).iloc[-1]
                if "BTC" in self.coin_data else 0,
            "equity_after": self.equity,
        })

        del self.positions[coin]

    # ── Trade Analysis Logging ──

    def _log_trade_analysis(self, event_type: str, coin: str, data: dict):
        """Log entry/exit analysis for post-trade review."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "coin": coin,
            "cycle": self.cycle_count,
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in data.items()},
        }
        analysis_file = STATE_DIR / "trade_analysis.jsonl"
        with open(analysis_file, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # ── Trajectory Logging (RL data collection) ──

    def _log_trajectory(self, coin, pos, current_price):
        """Log bar-by-bar position state for future RL training."""
        df = self.coin_data.get(coin)
        if df is None: return

        unrealized = ((current_price - pos.entry_price) / pos.entry_price) * pos.side
        atr_now = df["atr_14"].iloc[-1] if "atr_14" in df.columns else pos.atr_at_entry
        rsi_now = df["rsi_14"].iloc[-1] if "rsi_14" in df.columns else 50
        cvd_now = df["cvd_ratio"].iloc[-1] if "cvd_ratio" in df.columns else 0

        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "coin": coin,
            "bars_held": pos.bars_held,
            "unrealized_pnl": round(unrealized, 6),
            "price_vs_sl": round((current_price - pos.sl_price) / pos.atr_at_entry * pos.side, 4),
            "price_vs_tp": round((pos.tp_price - current_price) / pos.atr_at_entry * pos.side, 4),
            "atr_ratio": round(atr_now / pos.atr_at_entry, 4) if pos.atr_at_entry > 0 else 1.0,
            "rsi": round(rsi_now, 1),
            "cvd": round(cvd_now, 4),
            "side": pos.side,
            "action": "HOLD",  # will be backfilled with actual exit action
        }

        with open(TRAJ_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ── Loop ──

    async def run(self):
        log.info(f"v6.1 Bot | ${self.equity:.2f} | 1h cycle | max_pos={CFG['max_positions']} | "
                 f"lev={CFG['leverage']}x | source={DATA_SOURCE}")

        while True:
            try:
                self.cycle_count += 1
                now = datetime.now(timezone.utc)
                log.info(f"=== Cycle #{self.cycle_count} {now.strftime('%H:%M')} UTC | "
                         f"${self.equity:.2f} | pos={len(self.positions)} ===")

                # Parallel fetch (1h bars)
                with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
                    futs = {pool.submit(self._fetch_1h, c): c for c in ALL_COINS}
                    for fut in futs:
                        coin = futs[fut]
                        try: self.coin_data[coin] = fut.result()
                        except Exception as e: log.warning(f"[{coin}] {e}")

                # Direction
                gdir = self._get_global_direction()
                gdir_str = {1:"LONG", -1:"SHORT"}.get(gdir, "FLAT")
                log.info(f"BTC 7d: {gdir_str}")

                # Exits (check every cycle)
                for coin in list(self.positions.keys()):
                    try: self._check_exit(coin, gdir)
                    except Exception as e: log.warning(f"[{coin}] exit: {e}")

                # Entries (immediate when conditions met)
                if gdir != 0 and len(self.positions) < CFG["max_positions"]:
                    ranked = self._rank_alts(gdir)
                    for coin, info in ranked[:CFG["max_positions"]-len(self.positions)]:
                        try: self._open(coin, gdir, info)
                        except Exception as e: log.warning(f"[{coin}] entry: {e}")

                self._save_state()

                # Sleep until next 1h bar
                next_h = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                sleep = max(30, (next_h - now).total_seconds())
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
