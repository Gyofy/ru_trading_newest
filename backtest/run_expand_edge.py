"""Expand validated 1D TSMOM 28d/7d edge: coin combos, adaptive lookback, combined portfolio."""
import numpy as np, pandas as pd, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FEE = 0.0020; SLIP = 0.0003
data_dir = ROOT / "data" / "historical"
all_coins = ["SOL","XRP","ADA","DOT","DOGE","BNB","ETH"]

data_1d, data_4h = {}, {}
for c in all_coins:
    fs = list(data_dir.glob(f"{c}_1h_*.parquet"))
    if fs:
        df = pd.read_parquet(fs[0])
        df.index = pd.to_datetime(df.index, utc=True)
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        data_1d[c] = df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
        data_4h[c] = df.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

ref_1d = data_1d["SOL"].index
sp70 = ref_1d[int(len(ref_1d)*0.70)]
sp50 = ref_1d[len(ref_1d)//2]


def run(data, coins, ts_range, fn, sl_m, tp_m, ttl, cd_sec, max_pos=4):
    op, cl, cd = [], [], {}
    for ts in ts_range:
        for p in list(op):
            if ts not in data[p["c"]].index: continue
            bar = data[p["c"]].loc[ts]; p["h"] += 1
            h,l,c = float(bar["high"]),float(bar["low"]),float(bar["close"])
            pnl = None
            if p["s"]=="BUY":
                if l<=p["sl"]: pnl=(p["sl"]*(1-SLIP)-p["e"])/p["e"]*p["n"]-p["n"]*FEE
                elif h>=p["tp"]: pnl=(p["tp"]*(1-SLIP)-p["e"])/p["e"]*p["n"]-p["n"]*FEE
                elif p["h"]>=ttl: pnl=(c*(1-SLIP)-p["e"])/p["e"]*p["n"]-p["n"]*FEE
            else:
                if h>=p["sl"]: pnl=(p["e"]-p["sl"]*(1+SLIP))/p["e"]*p["n"]-p["n"]*FEE
                elif l<=p["tp"]: pnl=(p["e"]-p["tp"]*(1+SLIP))/p["e"]*p["n"]-p["n"]*FEE
                elif p["h"]>=ttl: pnl=(p["e"]-c*(1+SLIP))/p["e"]*p["n"]-p["n"]*FEE
            if pnl is not None:
                cl.append(pnl); op.remove(p)
        if len(op) >= max_pos: continue
        for coin in coins:
            if any(p["c"]==coin for p in op): continue
            if coin in cd and (ts-cd[coin]).total_seconds() < cd_sec: continue
            if ts not in data[coin].index: continue
            sig = fn(data[coin].loc[:ts])
            if sig is None: continue
            df2 = data[coin].loc[:ts]; cl2 = df2["close"].values
            h2, l2 = df2["high"].values, df2["low"].values
            tr = np.maximum(h2[1:]-l2[1:], np.maximum(np.abs(h2[1:]-cl2[:-1]),np.abs(l2[1:]-cl2[:-1])))
            atr = float(np.mean(tr[-14:])) if len(tr)>=14 else float(np.mean(tr))
            if atr <= 0: continue
            price = cl2[-1]
            fill = price*(1+SLIP) if sig=="BUY" else price*(1-SLIP)
            sd = max(atr*sl_m, price*0.0045); td = atr*tp_m
            if sig=="BUY": sl,tp = fill-sd, fill+td
            else: sl,tp = fill+sd, fill-td
            op.append({"c":coin,"s":sig,"e":fill,"sl":sl,"tp":tp,"n":5000*0.06*3,"h":0})
            cd[coin] = ts
    return cl


def tsmom_28d7d(df):
    if len(df) < 30: return None
    cl = df["close"].values
    if len(cl) < 29: return None
    m28 = (cl[-1]-cl[-28])/cl[-28]
    m7 = (cl[-1]-cl[-7])/cl[-7]
    if not ((m28>0 and m7>0) or (m28<0 and m7<0)): return None
    if abs(m28) < 0.03: return None
    return "BUY" if m28 > 0 else "SELL"


def adaptive_tsmom(df):
    if len(df) < 50: return None
    cl = df["close"].values
    best_score, best_side = 0, None
    for lb in [14, 21, 28, 42]:
        if len(cl) < lb+1: continue
        mom = (cl[-1]-cl[-lb])/cl[-lb]
        m7 = (cl[-1]-cl[-7])/cl[-7]
        if not ((mom>0 and m7>0) or (mom<0 and m7<0)): continue
        if abs(mom) < 0.02: continue
        if abs(mom) > best_score:
            best_score = abs(mom)
            best_side = "BUY" if mom > 0 else "SELL"
    return best_side


def mr_4h(df):
    if len(df) < 25: return None
    cl = df["close"].values
    ma = np.mean(cl[-20:]); std = np.std(cl[-20:])
    if std < 1e-10: return None
    z = (cl[-1]-ma)/std; z_prev = (cl[-2]-ma)/std
    if z_prev < -2.0 and z > z_prev + 0.3: return "BUY"
    elif z_prev > 2.0 and z < z_prev - 0.3: return "SELL"
    return None


def report(name, trades, trades_oos=None, trades_h1=None, trades_h2=None):
    n = len(trades); net = sum(trades)
    wr = sum(1 for x in trades if x>0)/n*100 if n else 0
    line = f"  {name:<35} N={n:>4} WR={wr:>5.1f}% Net=${net:>+9.2f}"
    if trades_oos is not None:
        no = len(trades_oos); neto = sum(trades_oos) if trades_oos else 0
        line += f" | OOS N={no:>3} ${neto:>+8.2f}"
        line += " PASS" if neto > 0 else " FAIL"
    print(line, flush=True)


coins5 = ["SOL","XRP","ADA","DOT","DOGE"]

# ═══════════════════════════════════════
# TEST 1: Best 6-coin combo
# ═══════════════════════════════════════
print("=== COIN COMBINATION SEARCH ===", flush=True)
for exclude in all_coins:
    coins6 = [c for c in all_coins if c != exclude]
    t_full = run(data_1d, coins6, ref_1d, tsmom_28d7d, 2.0, 3.0, 14, 86400)
    t_oos = run(data_1d, coins6, ref_1d[ref_1d>sp70], tsmom_28d7d, 2.0, 3.0, 14, 86400)
    report(f"Without {exclude}", t_full, t_oos)

# ═══════════════════════════════════════
# TEST 2: Adaptive lookback
# ═══════════════════════════════════════
print("\n=== ADAPTIVE LOOKBACK (best of 14/21/28/42d) ===", flush=True)
for sl, tp in [(2.0, 3.0), (1.5, 3.0), (2.0, 5.0)]:
    t_full = run(data_1d, coins5, ref_1d, adaptive_tsmom, sl, tp, 14, 86400)
    t_oos = run(data_1d, coins5, ref_1d[ref_1d>sp70], adaptive_tsmom, sl, tp, 14, 86400)
    report(f"Adaptive SL={sl} TP={tp}", t_full, t_oos)

# ═══════════════════════════════════════
# TEST 3: Combined 1D TSMOM + 4H MeanRev
# ═══════════════════════════════════════
print("\n=== COMBINED PORTFOLIO (1D TSMOM + 4H MeanRev) ===", flush=True)
t_1d_full = run(data_1d, coins5, ref_1d, tsmom_28d7d, 2.0, 3.0, 14, 86400)
t_4h_full = run(data_4h, coins5, data_4h["SOL"].index, mr_4h, 2.0, 7.0, 24, 14400)
t_1d_oos = run(data_1d, coins5, ref_1d[ref_1d>sp70], tsmom_28d7d, 2.0, 3.0, 14, 86400)

sp70_4h = data_4h["SOL"].index[int(len(data_4h["SOL"])*0.70)]
t_4h_oos = run(data_4h, coins5, data_4h["SOL"].index[data_4h["SOL"].index>sp70_4h], mr_4h, 2.0, 7.0, 24, 14400)

report("1D TSMOM 28d/7d alone", t_1d_full, t_1d_oos)
report("4H MeanRev alone", t_4h_full, t_4h_oos)
combined = t_1d_full + t_4h_full
combined_oos = t_1d_oos + t_4h_oos
report("COMBINED", combined, combined_oos)

print("\nDone.", flush=True)
