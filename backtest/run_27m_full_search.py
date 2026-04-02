"""27-month full edge search: every strategy x every parameter x every validation.

Data: 7 coins x 27 months (2024-01-01 ~ 2026-03-31) parquet, 1h bars
Goal: Find strategies that survive ALL validation tests on 2+ years of data.

Strategies tested:
  - TSMOM variants (6h/12h/24h/48h/7d momentum)
  - Relative Strength vs BTC/ETH
  - EMA crossovers (8/21, 12/26, 20/50)
  - RSI reversals
  - Donchian breakout
  - MACD cross
  - Bollinger Band reversion
  - Dual momentum (absolute + relative)
  - Volume-weighted momentum

Parameters swept: SL=[0.5,0.75,1.0,1.5,2.0] x TP=[2,3,4,5,7] = 25 per strategy
"""

from __future__ import annotations
import json, logging, sys, time, math, uuid, random
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("search")

# ── Cost model ──
FEE_RATE = 0.0020   # 0.20% roundtrip
SLIP = 0.0003       # 0.03% per side

# ── Data loader ──
class ParquetHub:
    def __init__(self, data_dir: Path, coins: list[str], timeframes=("1h",)):
        self.data: dict[str, dict[str, pd.DataFrame]] = {}
        for coin in coins:
            self.data[coin] = {}
            for tf in timeframes:
                pattern = f"{coin}_{tf}_*.parquet"
                files = list(data_dir.glob(pattern))
                if files:
                    df = pd.read_parquet(files[0])
                    df.index = pd.to_datetime(df.index, utc=True)
                    df.sort_index(inplace=True)
                    for c in ["open","high","low","close","volume"]:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    self.data[coin][tf] = df
                    log.info(f"  {coin} {tf}: {len(df)} bars")

    def get(self, coin, tf, limit, ts):
        if coin not in self.data or tf not in self.data[coin]:
            return None
        df = self.data[coin][tf]
        avail = df.loc[:ts]
        if len(avail) < 2:
            return None
        return avail.tail(limit)

    def get_bar(self, coin, tf, ts):
        if coin not in self.data or tf not in self.data[coin]:
            return None
        df = self.data[coin][tf]
        if ts in df.index:
            r = df.loc[ts]
            return {"open": float(r["open"]), "high": float(r["high"]),
                    "low": float(r["low"]), "close": float(r["close"]),
                    "volume": float(r["volume"])}
        return None

    def timestamps(self, coins, tf="1h"):
        idx = pd.DatetimeIndex([])
        for c in coins:
            if c in self.data and tf in self.data[c]:
                idx = idx.union(self.data[c][tf].index)
        return idx.sort_values()

    @staticmethod
    def atr(df, period=14):
        if df is None or len(df) < period+1:
            return 0.0
        h,l,c = df["high"].values, df["low"].values, df["close"].values
        tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
        if len(tr) < period:
            return float(np.mean(tr))
        a = np.mean(tr[:period])
        for i in range(period, len(tr)):
            a = a*(1-1/period) + tr[i]/period
        return float(a)

    @staticmethod
    def rsi(close, period=14):
        if len(close) < period+1:
            return None
        d = np.diff(close)
        g = np.where(d>0,d,0)
        l = np.where(d<0,-d,0)
        ag, al = np.mean(g[-period:]), np.mean(l[-period:])
        if al == 0: return 100.0
        return 100 - 100/(1+ag/al)

    @staticmethod
    def ema(close, span):
        s = pd.Series(close)
        return s.ewm(span=span, adjust=False).mean().values

    @staticmethod
    def vwap(df, window=20):
        if len(df) < window:
            return None
        tp = (df["high"] + df["low"] + df["close"]) / 3
        v = df["volume"]
        return float((tp*v).rolling(window).sum().iloc[-1] / v.rolling(window).sum().iloc[-1])


# ── Strategies ──

def tsmom(hub, coin, ts, cfg):
    """Generalized TSMOM: long_lb + short_lb momentum alignment."""
    long_lb = cfg.get("long_lb", 24)
    short_lb = cfg.get("short_lb", 6)
    df = hub.get(coin, "1h", long_lb+5, ts)
    if df is None or len(df) < long_lb+1:
        return None
    c = df["close"].values
    mom_l = (c[-1]-c[-long_lb])/c[-long_lb]
    mom_s = (c[-1]-c[-short_lb])/c[-short_lb]
    if not ((mom_l>0 and mom_s>0) or (mom_l<0 and mom_s<0)):
        return None
    side = "BUY" if mom_l > 0 else "SELL"
    min_mv = cfg.get("min_move", 0.003)
    if abs(mom_l) < min_mv:
        return None
    rsi = hub.rsi(c, 14)
    if rsi and ((side=="BUY" and rsi>75) or (side=="SELL" and rsi<25)):
        return None
    return {"side": side, "str": abs(mom_l)}

def rel_strength(hub, coin, ts, cfg):
    """Coin vs BTC relative strength."""
    ref = cfg.get("ref", "BTC")
    lb = cfg.get("lookback", 12)
    df_c = hub.get(coin, "1h", lb+2, ts)
    df_r = hub.get(ref, "1h", lb+2, ts)
    if df_c is None or df_r is None or len(df_c)<lb+1 or len(df_r)<lb+1:
        return None
    cc, cr = df_c["close"].values, df_r["close"].values
    n = min(len(cc), len(cr))
    cc, cr = cc[-n:], cr[-n:]
    lb = min(lb, n-1)
    coin_r = (cc[-1]-cc[-lb])/cc[-lb]
    ref_r = (cr[-1]-cr[-lb])/cr[-lb]
    rs = coin_r - ref_r
    min_rs = cfg.get("min_rs", 0.008)
    if abs(rs) < min_rs:
        return None
    if ref_r > 0.001 and rs > min_rs:
        return {"side": "BUY", "str": abs(rs)}
    elif ref_r < -0.001 and rs < -min_rs:
        return {"side": "SELL", "str": abs(rs)}
    return None

def ema_cross(hub, coin, ts, cfg):
    """EMA crossover."""
    fast = cfg.get("fast", 8)
    slow = cfg.get("slow", 21)
    df = hub.get(coin, "1h", slow+10, ts)
    if df is None or len(df) < slow+5:
        return None
    c = df["close"].values
    ef = hub.ema(c, fast)
    es = hub.ema(c, slow)
    if ef[-1]>es[-1] and ef[-2]<=es[-2]:
        return {"side": "BUY", "str": abs(ef[-1]-es[-1])/c[-1]}
    elif ef[-1]<es[-1] and ef[-2]>=es[-2]:
        return {"side": "SELL", "str": abs(ef[-1]-es[-1])/c[-1]}
    return None

def macd_cross(hub, coin, ts, cfg):
    """MACD line crosses signal line with trend filter."""
    df = hub.get(coin, "1h", 40, ts)
    if df is None or len(df) < 30:
        return None
    c = df["close"]
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    sig = macd.ewm(span=9).mean()
    if macd.iloc[-1]>sig.iloc[-1] and macd.iloc[-2]<=sig.iloc[-2]:
        side = "BUY"
    elif macd.iloc[-1]<sig.iloc[-1] and macd.iloc[-2]>=sig.iloc[-2]:
        side = "SELL"
    else:
        return None
    # Trend filter
    mom = (float(c.iloc[-1])-float(c.iloc[-24]))/float(c.iloc[-24]) if len(c)>=25 else 0
    if side=="BUY" and mom < -0.01:
        return None
    if side=="SELL" and mom > 0.01:
        return None
    return {"side": side, "str": abs(float(macd.iloc[-1]-sig.iloc[-1]))/float(c.iloc[-1])}

def rsi_reversal(hub, coin, ts, cfg):
    """RSI extreme + turning + body confirmation."""
    df = hub.get(coin, "1h", 30, ts)
    if df is None or len(df) < 16:
        return None
    c = df["close"].values
    rsi = hub.rsi(c, 14)
    rsi_p = hub.rsi(c[:-1], 14)
    if rsi is None or rsi_p is None:
        return None
    ob = cfg.get("ob", 72)
    os_ = cfg.get("os", 28)
    if rsi_p < os_ and rsi > rsi_p + 2:
        side = "BUY"
    elif rsi_p > ob and rsi < rsi_p - 2:
        side = "SELL"
    else:
        return None
    body = (c[-1] - df["open"].values[-1]) / df["open"].values[-1]
    if (side=="BUY" and body<0) or (side=="SELL" and body>0):
        return None
    return {"side": side, "str": abs(rsi-50)}

def donchian(hub, coin, ts, cfg):
    """Donchian channel breakout with volume."""
    period = cfg.get("period", 20)
    df = hub.get(coin, "1h", period+5, ts)
    if df is None or len(df) < period+2:
        return None
    h,l,c,v = df["high"].values, df["low"].values, df["close"].values, df["volume"].values
    dc_h = np.max(h[-period-1:-1])
    dc_l = np.min(l[-period-1:-1])
    if c[-1] > dc_h:
        side = "BUY"
    elif c[-1] < dc_l:
        side = "SELL"
    else:
        return None
    va = np.mean(v[-period:-1])
    if va > 0 and v[-1] < va * 1.2:
        return None
    return {"side": side, "str": abs(c[-1]-(dc_h+dc_l)/2)/c[-1]}

def bollinger_rev(hub, coin, ts, cfg):
    """Bollinger Band mean reversion: close outside 2sigma + returning."""
    window = cfg.get("window", 20)
    df = hub.get(coin, "1h", window+5, ts)
    if df is None or len(df) < window+2:
        return None
    c = df["close"].values
    ma = np.mean(c[-window:])
    std = np.std(c[-window:])
    if std < 1e-10:
        return None
    z = (c[-1] - ma) / std
    z_prev = (c[-2] - ma) / std
    sigma = cfg.get("sigma", 2.0)
    # Was outside, now returning
    if z_prev < -sigma and z > z_prev + 0.3:
        return {"side": "BUY", "str": abs(z)}
    elif z_prev > sigma and z < z_prev - 0.3:
        return {"side": "SELL", "str": abs(z)}
    return None

def dual_momentum(hub, coin, ts, cfg):
    """Dual momentum: absolute + relative (vs BTC). Both must be positive."""
    abs_lb = cfg.get("abs_lb", 24)
    df_c = hub.get(coin, "1h", abs_lb+2, ts)
    df_b = hub.get("BTC", "1h", abs_lb+2, ts)
    if df_c is None or df_b is None or len(df_c)<abs_lb+1 or len(df_b)<abs_lb+1:
        return None
    cc, cb = df_c["close"].values, df_b["close"].values
    abs_mom = (cc[-1]-cc[-abs_lb])/cc[-abs_lb]
    btc_mom = (cb[-1]-cb[-abs_lb])/cb[-abs_lb]
    rel_mom = abs_mom - btc_mom
    # Both absolute and relative must agree
    if abs_mom > 0.003 and rel_mom > 0.005:
        return {"side": "BUY", "str": abs_mom + rel_mom}
    elif abs_mom < -0.003 and rel_mom < -0.005:
        return {"side": "SELL", "str": abs(abs_mom) + abs(rel_mom)}
    return None

def vol_weighted_mom(hub, coin, ts, cfg):
    """Volume-weighted momentum: high-volume bars count more."""
    lb = cfg.get("lookback", 24)
    df = hub.get(coin, "1h", lb+2, ts)
    if df is None or len(df) < lb+1:
        return None
    c, v = df["close"].values[-lb:], df["volume"].values[-lb:]
    rets = np.diff(c) / c[:-1]
    if len(rets) < 5:
        return None
    v_weights = v[1:] / (np.mean(v[1:]) + 1e-10)
    vw_mom = np.sum(rets * v_weights) / np.sum(v_weights)
    if abs(vw_mom) < cfg.get("min_move", 0.002):
        return None
    side = "BUY" if vw_mom > 0 else "SELL"
    return {"side": side, "str": abs(vw_mom)}


# ── All strategy variants ──
STRATEGIES = {
    "tsmom_24h_6h":   (tsmom, {"long_lb":24, "short_lb":6, "min_move":0.004}),
    "tsmom_48h_12h":  (tsmom, {"long_lb":48, "short_lb":12, "min_move":0.005}),
    "tsmom_12h_4h":   (tsmom, {"long_lb":12, "short_lb":4, "min_move":0.003}),
    "tsmom_168h_24h": (tsmom, {"long_lb":168, "short_lb":24, "min_move":0.01}),
    "tsmom_72h_12h":  (tsmom, {"long_lb":72, "short_lb":12, "min_move":0.007}),
    "rs_btc_12h":     (rel_strength, {"ref":"BTC", "lookback":12, "min_rs":0.008}),
    "rs_btc_24h":     (rel_strength, {"ref":"BTC", "lookback":24, "min_rs":0.01}),
    "rs_eth_12h":     (rel_strength, {"ref":"ETH", "lookback":12, "min_rs":0.008}),
    "ema_8_21":       (ema_cross, {"fast":8, "slow":21}),
    "ema_12_26":      (ema_cross, {"fast":12, "slow":26}),
    "ema_20_50":      (ema_cross, {"fast":20, "slow":50}),
    "macd_trend":     (macd_cross, {}),
    "rsi_rev":        (rsi_reversal, {"ob":72, "os":28}),
    "rsi_rev_tight":  (rsi_reversal, {"ob":75, "os":25}),
    "donchian_20":    (donchian, {"period":20}),
    "donchian_48":    (donchian, {"period":48}),
    "boll_rev_20":    (bollinger_rev, {"window":20, "sigma":2.0}),
    "boll_rev_48":    (bollinger_rev, {"window":48, "sigma":2.0}),
    "dual_mom_24":    (dual_momentum, {"abs_lb":24}),
    "dual_mom_48":    (dual_momentum, {"abs_lb":48}),
    "dual_mom_168":   (dual_momentum, {"abs_lb":168}),
    "vw_mom_24":      (vol_weighted_mom, {"lookback":24, "min_move":0.002}),
    "vw_mom_48":      (vol_weighted_mom, {"lookback":48, "min_move":0.003}),
}


# ── Backtest engine ──
@dataclass
class Pos:
    id: str; coin: str; strat: str; side: str
    entry: float; entry_ts: pd.Timestamp
    sl: float; tp: float; notional: float
    ttl: int; held: int = 0
    mfe: float = 0.0; mae: float = 0.0

class Engine:
    def __init__(self, hub, coins, sl_mult=1.5, tp_mult=5.0, capital=5000, lev=3, max_pos=8):
        self.hub = hub
        self.coins = coins
        self.sl_m = sl_mult
        self.tp_m = tp_mult
        self.capital = capital
        self.lev = lev
        self.max_pos = max_pos
        self.open: list[Pos] = []
        self.closed: list[dict] = []
        self.cooldowns: dict[str, pd.Timestamp] = {}

    def run(self, strats: dict, start_ts=None, end_ts=None) -> list[dict]:
        ts_all = self.hub.timestamps(self.coins, "1h")
        if start_ts: ts_all = ts_all[ts_all >= start_ts]
        if end_ts: ts_all = ts_all[ts_all <= end_ts]

        for ts in ts_all:
            self._check_exits(ts)
            if len(self.open) < self.max_pos:
                self._eval(ts, strats)

        for p in list(self.open):
            self._close_pos(p, "END", ts_all[-1] if len(ts_all) else pd.Timestamp.now(tz="UTC"))
        return self.closed

    def _check_exits(self, ts):
        for p in list(self.open):
            bar = self.hub.get_bar(p.coin, "1h", ts)
            if not bar:
                continue
            p.held += 1
            h, l = bar["high"], bar["low"]
            if p.side == "BUY":
                p.mfe = max(p.mfe, h) if p.mfe else h
                p.mae = min(p.mae, l) if p.mae else l
                if l <= p.sl: self._close_pos(p, "SL", ts)
                elif h >= p.tp: self._close_pos(p, "TP", ts)
                elif p.held >= p.ttl: self._close_pos(p, "TTL", ts)
            else:
                p.mfe = min(p.mfe, l) if p.mfe else l
                p.mae = max(p.mae, h) if p.mae else h
                if h >= p.sl: self._close_pos(p, "SL", ts)
                elif l <= p.tp: self._close_pos(p, "TP", ts)
                elif p.held >= p.ttl: self._close_pos(p, "TTL", ts)

    def _eval(self, ts, strats):
        for sname, (fn, cfg) in strats.items():
            if len(self.open) >= self.max_pos:
                break
            for coin in self.coins:
                if any(p.coin==coin for p in self.open):
                    continue
                ck = f"{sname}:{coin}"
                if ck in self.cooldowns and (ts-self.cooldowns[ck]).total_seconds() < 3600:
                    continue
                try:
                    sig = fn(self.hub, coin, ts, cfg)
                except:
                    continue
                if not sig:
                    continue
                self._open_pos(sname, coin, sig, ts)
                self.cooldowns[ck] = ts

    def _open_pos(self, sname, coin, sig, ts):
        df = self.hub.get(coin, "1h", 20, ts)
        if df is None or len(df) < 15:
            return
        atr = self.hub.atr(df)
        if atr <= 0:
            return
        price = float(df["close"].iloc[-1])
        side = sig["side"]
        fill = price*(1+SLIP) if side=="BUY" else price*(1-SLIP)
        sl_d = max(atr*self.sl_m, price*0.0045)
        tp_d = atr*self.tp_m
        if side=="BUY":
            sl, tp = fill-sl_d, fill+tp_d
        else:
            sl, tp = fill+sl_d, fill-tp_d
        notional = self.capital*0.08*self.lev
        self.open.append(Pos(str(uuid.uuid4())[:8], coin, sname, side, fill, ts, sl, tp, notional, ttl=48))

    def _close_pos(self, p, reason, ts):
        if p not in self.open:
            return
        self.open.remove(p)
        if reason=="SL": ep = p.sl
        elif reason=="TP": ep = p.tp
        else:
            bar = self.hub.get_bar(p.coin, "1h", ts)
            ep = bar["close"] if bar else p.entry
        if p.side=="BUY":
            ep *= (1-SLIP)
            pnl_pct = (ep-p.entry)/p.entry
            mfe_pct = (p.mfe-p.entry)/p.entry if p.mfe else 0
            mae_pct = (p.mae-p.entry)/p.entry if p.mae else 0
        else:
            ep *= (1+SLIP)
            pnl_pct = (p.entry-ep)/p.entry
            mfe_pct = (p.entry-p.mfe)/p.entry if p.mfe else 0
            mae_pct = (p.entry-p.mae)/p.entry if p.mae else 0
        fee = p.notional * FEE_RATE
        gross = pnl_pct * p.notional
        self.closed.append({
            "strat": p.strat, "coin": p.coin, "side": p.side, "reason": reason,
            "gross": round(gross,2), "net": round(gross-fee,2), "fee": round(fee,2),
            "pnl_pct": round(pnl_pct*100,3), "mfe": round(mfe_pct*100,3),
            "mae": round(mae_pct*100,3), "held": p.held,
            "entry_ts": str(p.entry_ts)[:19], "exit_ts": str(ts)[:19],
        })


def metrics(trades):
    if not trades: return {"n":0, "wr":0, "gross":0, "fee":0, "net":0, "pf":0, "tp%":0, "sl%":0, "avg_held":0}
    df = pd.DataFrame(trades)
    n = len(df)
    w = df[df["net"]>0]
    wr = len(w)/n*100
    return {
        "n": n, "wr": round(wr,1),
        "gross": round(df["gross"].sum(),2),
        "fee": round(df["fee"].sum(),2),
        "net": round(df["net"].sum(),2),
        "pf": round(w["net"].sum()/(abs(df[df["net"]<=0]["net"].sum())+0.01), 2),
        "tp%": round(len(df[df["reason"]=="TP"])/n*100, 1),
        "sl%": round(len(df[df["reason"]=="SL"])/n*100, 1),
        "avg_held": round(df["held"].mean(), 1),
    }


def main():
    t0 = time.time()
    data_dir = ROOT / "data" / "historical"
    trade_coins = ["SOL", "XRP", "ADA", "DOT", "DOGE"]
    ref_coins = ["BTC", "ETH"]
    all_coins = trade_coins + [c for c in ref_coins if c not in trade_coins] + ["BNB"]

    log.info("Loading 27-month parquet data...")
    hub = ParquetHub(data_dir, all_coins, timeframes=("1h",))

    # Time boundaries
    ts_all = hub.timestamps(trade_coins, "1h")
    split_70 = ts_all[int(len(ts_all)*0.70)]
    split_50 = ts_all[len(ts_all)//2]
    log.info(f"Data: {ts_all[0]} ~ {ts_all[-1]} ({len(ts_all)} bars)")
    log.info(f"70% split: {split_70} | 50% split: {split_50}")

    # ══════════════════════════════════════════════════════════
    # PHASE 1: Sweep all strategies x parameters on FULL period
    # ══════════════════════════════════════════════════════════
    log.info("PHASE 1: Full-period sweep (23 strategies x 25 params = 575 combos)")
    sl_mults = [0.5, 0.75, 1.0, 1.5, 2.0]
    tp_mults = [2.0, 3.0, 4.0, 5.0, 7.0]

    all_results = []
    for sname in STRATEGIES:
        best_net = -999999
        best_cfg = None
        for sl in sl_mults:
            for tp in tp_mults:
                if tp < sl: continue
                eng = Engine(hub, trade_coins, sl_mult=sl, tp_mult=tp, max_pos=6)
                strat = {sname: STRATEGIES[sname]}
                trades = eng.run(strat)
                m = metrics(trades)
                m.update({"strat": sname, "sl": sl, "tp": tp})
                all_results.append(m)
                if m.get("net", -999999) > best_net:
                    best_net = m["net"]
                    best_cfg = m

        if best_cfg and best_cfg["n"] > 0:
            status = "+" if best_net > 0 else "-"
            log.info(f"  {sname:<20} [{status}] N={best_cfg['n']:>5} WR={best_cfg['wr']:>5.1f}% "
                     f"Net=${best_net:>+10.2f} PF={best_cfg['pf']:>5.2f} "
                     f"SL={best_cfg['sl']} TP={best_cfg['tp']}")

    # Filter profitable
    profitable = [r for r in all_results if r["net"] > 0 and r["n"] >= 20]
    profitable.sort(key=lambda x: x["net"], reverse=True)

    print("\n" + "="*100)
    print(f"  PHASE 1 RESULTS: {len(profitable)} profitable / {len(all_results)} total")
    print("="*100)
    hdr = f"  {'Strategy':<20} {'SL':>5} {'TP':>5} {'N':>5} {'WR%':>6} {'Gross$':>10} {'Fee$':>8} {'Net$':>10} {'PF':>5} {'TP%':>5}"
    print(hdr)
    print("  "+"-"*(len(hdr)-2))
    for r in profitable[:30]:
        print(f"  {r['strat']:<20} {r['sl']:>5.2f} {r['tp']:>5.2f} {r['n']:>5} {r['wr']:>5.1f}% "
              f"{r['gross']:>+10.2f} {r['fee']:>8.2f} {r['net']:>+10.2f} {r['pf']:>5.2f} {r['tp%']:>4.1f}%")

    # ══════════════════════════════════════════════════════════
    # PHASE 2: Walk-Forward validation on top results
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*100)
    print("  PHASE 2: WALK-FORWARD VALIDATION (train 70%, test 30%)")
    print("="*100)

    # Take top 5 per strategy (unique strategies)
    seen_strats = set()
    top_configs = []
    for r in profitable:
        if r["strat"] not in seen_strats and r["n"] >= 30:
            top_configs.append(r)
            seen_strats.add(r["strat"])
        if len(top_configs) >= 15:
            break

    wf_survivors = []
    for cfg in top_configs:
        sname = cfg["strat"]
        strat = {sname: STRATEGIES[sname]}

        # Train period
        eng_tr = Engine(hub, trade_coins, sl_mult=cfg["sl"], tp_mult=cfg["tp"], max_pos=6)
        tr_trades = eng_tr.run(strat, end_ts=split_70)
        tr_m = metrics(tr_trades)

        # Test period (OOS)
        eng_te = Engine(hub, trade_coins, sl_mult=cfg["sl"], tp_mult=cfg["tp"], max_pos=6)
        te_trades = eng_te.run(strat, start_ts=split_70)
        te_m = metrics(te_trades)

        # Half 1 / Half 2
        eng_h1 = Engine(hub, trade_coins, sl_mult=cfg["sl"], tp_mult=cfg["tp"], max_pos=6)
        h1_trades = eng_h1.run(strat, end_ts=split_50)
        h1_m = metrics(h1_trades)

        eng_h2 = Engine(hub, trade_coins, sl_mult=cfg["sl"], tp_mult=cfg["tp"], max_pos=6)
        h2_trades = eng_h2.run(strat, start_ts=split_50)
        h2_m = metrics(h2_trades)

        # Reversed signals
        class RevEngine(Engine):
            def _eval(self, ts, strats):
                for sn, (fn, c) in strats.items():
                    if len(self.open) >= self.max_pos: break
                    for coin in self.coins:
                        if any(p.coin==coin for p in self.open): continue
                        ck = f"{sn}:{coin}"
                        if ck in self.cooldowns and (ts-self.cooldowns[ck]).total_seconds()<3600: continue
                        try: sig = fn(self.hub, coin, ts, c)
                        except: continue
                        if not sig: continue
                        sig["side"] = "SELL" if sig["side"]=="BUY" else "BUY"
                        self._open_pos(sn, coin, sig, ts)
                        self.cooldowns[ck] = ts

        eng_rev = RevEngine(hub, trade_coins, sl_mult=cfg["sl"], tp_mult=cfg["tp"], max_pos=6)
        rev_trades = eng_rev.run(strat)
        rev_m = metrics(rev_trades)

        # Neighborhood check
        neighbors_ok = 0
        for dsl in [-0.2, -0.1, 0, 0.1, 0.2]:
            for dtp in [-0.2, -0.1, 0, 0.1, 0.2]:
                eng_n = Engine(hub, trade_coins, sl_mult=cfg["sl"]*(1+dsl), tp_mult=cfg["tp"]*(1+dtp), max_pos=6)
                n_trades = eng_n.run(strat)
                if n_trades and sum(t["net"] for t in n_trades) > 0:
                    neighbors_ok += 1

        # Verdict
        passes = 0
        checks = []
        c1 = cfg["net"] > 0;                    checks.append(("Full profitable", c1)); passes += c1
        c2 = te_m.get("net",0) > 0;             checks.append(("OOS profitable", c2)); passes += c2
        c3 = h1_m.get("net",0)>0 and h2_m.get("net",0)>0; checks.append(("Both halves", c3)); passes += c3
        c4 = rev_m.get("net",0) < 0;            checks.append(("Reversed loses", c4)); passes += c4
        c5 = neighbors_ok >= 15;                 checks.append(("Param robust", c5)); passes += c5

        status = "PASS" if passes >= 4 else "PARTIAL" if passes >= 3 else "FAIL"
        print(f"\n  [{sname}] SL={cfg['sl']} TP={cfg['tp']} -> {passes}/5 [{status}]")
        print(f"    Full:     N={cfg['n']:>4} Net=${cfg['net']:>+9.2f}")
        print(f"    OOS:      N={te_m.get('n',0):>4} Net=${te_m.get('net',0):>+9.2f}")
        print(f"    Half1:    N={h1_m.get('n',0):>4} Net=${h1_m.get('net',0):>+9.2f}")
        print(f"    Half2:    N={h2_m.get('n',0):>4} Net=${h2_m.get('net',0):>+9.2f}")
        print(f"    Reversed: N={rev_m.get('n',0):>4} Net=${rev_m.get('net',0):>+9.2f}")
        print(f"    Neighbors: {neighbors_ok}/25 profitable")
        for name, ok in checks:
            print(f"    [{'OK' if ok else 'X ':>2}] {name}")

        if passes >= 4:
            wf_survivors.append({
                **cfg,
                "oos_net": te_m.get("net", 0),
                "h1_net": h1_m.get("net", 0),
                "h2_net": h2_m.get("net", 0),
                "rev_net": rev_m.get("net", 0),
                "neighbors": neighbors_ok,
                "passes": passes,
            })

    # ══════════════════════════════════════════════════════════
    # PHASE 3: Combined portfolio of survivors
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*100)
    print(f"  PHASE 3: SURVIVORS ({len(wf_survivors)} strategies) - COMBINED PORTFOLIO")
    print("="*100)

    if wf_survivors:
        combo_strats = {}
        for s in wf_survivors:
            combo_strats[s["strat"]] = STRATEGIES[s["strat"]]

        # Use best SL/TP per strategy
        best_sl = {s["strat"]: s["sl"] for s in wf_survivors}
        best_tp = {s["strat"]: s["tp"] for s in wf_survivors}
        avg_sl = np.mean(list(best_sl.values()))
        avg_tp = np.mean(list(best_tp.values()))

        eng_combo = Engine(hub, trade_coins, sl_mult=avg_sl, tp_mult=avg_tp, max_pos=10)
        combo_trades = eng_combo.run(combo_strats)
        combo_m = metrics(combo_trades)

        print(f"\n  Combined: N={combo_m.get('n',0)} WR={combo_m.get('wr',0):.1f}% "
              f"Gross=${combo_m.get('gross',0):+.2f} Net=${combo_m.get('net',0):+.2f} "
              f"PF={combo_m.get('pf',0):.2f}")

        # Per-strategy in combo
        if combo_trades:
            df_c = pd.DataFrame(combo_trades)
            for s in sorted(df_c["strat"].unique()):
                sdf = df_c[df_c["strat"]==s]
                sm = metrics(sdf.to_dict("records"))
                print(f"    {s:<20} N={sm['n']:>4} WR={sm['wr']:>5.1f}% Net=${sm['net']:>+9.2f} PF={sm['pf']:>5.2f}")

        # Per-year breakdown
        if combo_trades:
            df_c = pd.DataFrame(combo_trades)
            df_c["year"] = pd.to_datetime(df_c["entry_ts"]).dt.year
            for yr in sorted(df_c["year"].unique()):
                ydf = df_c[df_c["year"]==yr]
                ym = metrics(ydf.to_dict("records"))
                print(f"    Year {yr}: N={ym['n']:>4} WR={ym['wr']:>5.1f}% Net=${ym['net']:>+9.2f}")
    else:
        print("\n  No survivors. All strategies failed validation.")

    # Save
    output = ROOT / "backtest" / "output"
    output.mkdir(exist_ok=True)
    with open(output / "27m_all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    with open(output / "27m_survivors.json", "w") as f:
        json.dump(wf_survivors, f, indent=2, default=str)
    if wf_survivors and combo_trades:
        with open(output / "27m_combo_trades.jsonl", "w") as f:
            for t in combo_trades:
                f.write(json.dumps(t, default=str) + "\n")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} min")
    print(f"  Strategies tested: {len(STRATEGIES)}")
    print(f"  Total combos: {len(all_results)}")
    print(f"  Profitable: {len(profitable)}")
    print(f"  Validated survivors: {len(wf_survivors)}")


if __name__ == "__main__":
    main()
