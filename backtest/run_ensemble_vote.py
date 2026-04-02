"""Ensemble Voting + Regime Filter on 27-month 1h data.

5 signals vote, need N+ agreement. Regime filter blocks low-vol entries.
"""
import numpy as np, pandas as pd, sys, time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FEE = 0.0020
SLIP = 0.0003

class Hub:
    def __init__(s, d, coins):
        s.data = {}
        for c in coins:
            fs = list(d.glob(f"{c}_1h_*.parquet"))
            if fs:
                df = pd.read_parquet(fs[0])
                df.index = pd.to_datetime(df.index, utc=True)
                df.sort_index(inplace=True)
                for col in ["open","high","low","close","volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                s.data[c] = df
    def get(s,c,n,ts):
        if c not in s.data: return None
        a = s.data[c].loc[:ts]
        return a.tail(n) if len(a)>=2 else None
    def bar(s,c,ts):
        if c not in s.data or ts not in s.data[c].index: return None
        r = s.data[c].loc[ts]
        return {"h":float(r["high"]),"l":float(r["low"]),"c":float(r["close"])}
    def timestamps(s, coins):
        idx = pd.DatetimeIndex([])
        for c in coins:
            if c in s.data: idx = idx.union(s.data[c].index)
        return idx.sort_values()

def signals_at(hub, c, ts):
    df = hub.get(c, 80, ts)
    if df is None or len(df) < 75: return None
    cl = df["close"].values
    vol = df["volume"].values
    votes = []
    # 1. TSMOM 24h
    if len(cl) >= 25:
        m = (cl[-1]-cl[-24])/cl[-24]
        if m > 0.003: votes.append(1)
        elif m < -0.003: votes.append(-1)
    # 2. EMA 12/26
    if len(cl) >= 28:
        s = pd.Series(cl)
        e12 = s.ewm(span=12).mean().values[-1]
        e26 = s.ewm(span=26).mean().values[-1]
        votes.append(1 if e12 > e26 else -1)
    # 3. MACD histogram
    if len(cl) >= 30:
        s = pd.Series(cl)
        macd = s.ewm(span=12).mean() - s.ewm(span=26).mean()
        sig = macd.ewm(span=9).mean()
        votes.append(1 if macd.iloc[-1] > sig.iloc[-1] else -1)
    # 4. RSI zone
    if len(cl) >= 15:
        d = np.diff(cl)
        g = np.where(d>0,d,0)
        lo = np.where(d<0,-d,0)
        ag, al = np.mean(g[-14:]), np.mean(lo[-14:])
        rsi = 100 if al==0 else 100-100/(1+ag/al)
        if rsi > 55: votes.append(1)
        elif rsi < 45: votes.append(-1)
    # 5. Volume trend
    if len(vol) >= 13:
        vr = np.mean(vol[-6:]) / (np.mean(vol[-12:-6]) + 1e-10)
        price_dir = 1 if cl[-1] > cl[-6] else -1
        if vr > 1.1: votes.append(price_dir)
    if not votes: return None
    return sum(votes), len(votes)

def vol_regime(hub, c, ts):
    df = hub.get(c, 50, ts)
    if df is None or len(df) < 25: return "low"
    rets = np.diff(np.log(df["close"].values))
    rv = np.std(rets[-24:])
    if rv > 0.015: return "high"
    elif rv > 0.008: return "medium"
    return "low"

def run_config(hub, coins, ts_all, min_votes, sl_m, tp_m, regime_filter, cooldown_sec=7200):
    open_pos = []
    closed = []
    cooldowns = {}
    for ts in ts_all:
        for p in list(open_pos):
            bar = hub.bar(p["coin"], ts)
            if not bar: continue
            p["held"] += 1
            h, l, c = bar["h"], bar["l"], bar["c"]
            if p["side"] == "BUY":
                if l <= p["sl"]:
                    pnl = (p["sl"]*(1-SLIP)-p["entry"])/p["entry"]*p["not"]
                    closed.append({"net":pnl-p["not"]*FEE, "reason":"SL"})
                    open_pos.remove(p)
                elif h >= p["tp"]:
                    pnl = (p["tp"]*(1-SLIP)-p["entry"])/p["entry"]*p["not"]
                    closed.append({"net":pnl-p["not"]*FEE, "reason":"TP"})
                    open_pos.remove(p)
                elif p["held"] >= 48:
                    pnl = (c*(1-SLIP)-p["entry"])/p["entry"]*p["not"]
                    closed.append({"net":pnl-p["not"]*FEE, "reason":"TTL"})
                    open_pos.remove(p)
            else:
                if h >= p["sl"]:
                    pnl = (p["entry"]-p["sl"]*(1+SLIP))/p["entry"]*p["not"]
                    closed.append({"net":pnl-p["not"]*FEE, "reason":"SL"})
                    open_pos.remove(p)
                elif l <= p["tp"]:
                    pnl = (p["entry"]-p["tp"]*(1+SLIP))/p["entry"]*p["not"]
                    closed.append({"net":pnl-p["not"]*FEE, "reason":"TP"})
                    open_pos.remove(p)
                elif p["held"] >= 48:
                    pnl = (p["entry"]-c*(1+SLIP))/p["entry"]*p["not"]
                    closed.append({"net":pnl-p["not"]*FEE, "reason":"TTL"})
                    open_pos.remove(p)

        if len(open_pos) >= 4: continue
        for coin in coins:
            if any(p["coin"]==coin for p in open_pos): continue
            if coin in cooldowns and (ts-cooldowns[coin]).total_seconds() < cooldown_sec: continue
            if regime_filter and vol_regime(hub, coin, ts) == "low": continue
            r = signals_at(hub, coin, ts)
            if r is None: continue
            total, n_v = r
            if abs(total) < min_votes: continue
            side = "BUY" if total > 0 else "SELL"
            df = hub.get(coin, 20, ts)
            if df is None or len(df) < 15: continue
            cl = df["close"].values
            hi, lo = df["high"].values, df["low"].values
            tr = np.maximum(hi[1:]-lo[1:], np.maximum(np.abs(hi[1:]-cl[:-1]),np.abs(lo[1:]-cl[:-1])))
            atr = float(np.mean(tr[-14:])) if len(tr)>=14 else float(np.mean(tr))
            if atr <= 0: continue
            price = cl[-1]
            fill = price*(1+SLIP) if side=="BUY" else price*(1-SLIP)
            sl_d = max(atr*sl_m, price*0.0045)
            tp_d = atr*tp_m
            if side=="BUY": sl,tp = fill-sl_d, fill+tp_d
            else: sl,tp = fill+sl_d, fill-tp_d
            notional = 5000*0.08*3
            open_pos.append({"coin":coin,"side":side,"entry":fill,"sl":sl,"tp":tp,"not":notional,"held":0})
            cooldowns[coin] = ts
    return closed

def main():
    t0 = time.time()
    hub = Hub(ROOT / "data" / "historical", ["SOL","XRP","ADA","DOT","DOGE","ETH","BNB"])
    coins = ["SOL","XRP","ADA","DOT","DOGE"]
    ts_all = hub.timestamps(coins)
    print(f"Loaded {len(ts_all)} bars", flush=True)

    print(f"\n{'Config':<30} {'N':>5} {'WR%':>6} {'Net$':>10} {'TP':>4} {'SL':>4} {'TTL':>4}", flush=True)
    print("-"*70, flush=True)

    for mv in [3, 4, 5]:
        for sl in [1.5, 2.0]:
            for tp in [5.0, 7.0]:
                for rf in [True, False]:
                    for cd in [7200, 14400]:
                        closed = run_config(hub, coins, ts_all, mv, sl, tp, rf, cd)
                        n = len(closed)
                        if n > 0:
                            net = sum(t["net"] for t in closed)
                            wr = sum(1 for t in closed if t["net"]>0)/n*100
                            tp_ct = sum(1 for t in closed if t["reason"]=="TP")
                            sl_ct = sum(1 for t in closed if t["reason"]=="SL")
                            ttl_ct = sum(1 for t in closed if t["reason"]=="TTL")
                            key = f"V{mv}_SL{sl}_TP{tp}_R{'Y' if rf else 'N'}_CD{cd//3600}h"
                            mark = " ***" if net > 0 else ""
                            print(f"{key:<30} {n:>5} {wr:>5.1f}% {net:>+10.2f} {tp_ct:>4} {sl_ct:>4} {ttl_ct:>4}{mark}", flush=True)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
