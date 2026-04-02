"""5-Round Brainstorming: systematically test improvements to 1D TSMOM 28d/7d."""
import numpy as np, pandas as pd, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FEE = 0.0020; SLIP = 0.0003

data_dir = ROOT / "data" / "historical"
coins = ["SOL","XRP","ADA","DOT","DOGE"]

data_1d = {}
for c in coins:
    fs = list(data_dir.glob(f"{c}_1h_*.parquet"))
    if fs:
        df = pd.read_parquet(fs[0])
        df.index = pd.to_datetime(df.index, utc=True)
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        data_1d[c] = df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

ref = data_1d["SOL"].index
sp70 = ref[int(len(ref)*0.70)]
sp50 = ref[len(ref)//2]


def run(coins, ts_range, fn, sl_m, tp_m, ttl=14):
    op, cl, cd = [], [], {}
    for ts in ts_range:
        for p in list(op):
            if ts not in data_1d[p["c"]].index: continue
            bar = data_1d[p["c"]].loc[ts]; p["h"] += 1
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
            if pnl is not None: cl.append(pnl); op.remove(p)
        if len(op)>=4: continue
        for coin in coins:
            if any(p["c"]==coin for p in op): continue
            if coin in cd and (ts-cd[coin]).total_seconds()<86400: continue
            if ts not in data_1d[coin].index: continue
            sig = fn(data_1d[coin].loc[:ts])
            if sig is None: continue
            df2 = data_1d[coin].loc[:ts]; cl2 = df2["close"].values
            h2,l2 = df2["high"].values, df2["low"].values
            tr = np.maximum(h2[1:]-l2[1:], np.maximum(np.abs(h2[1:]-cl2[:-1]),np.abs(l2[1:]-cl2[:-1])))
            atr = float(np.mean(tr[-14:])) if len(tr)>=14 else float(np.mean(tr))
            if atr<=0: continue
            price = cl2[-1]; fill = price*(1+SLIP) if sig=="BUY" else price*(1-SLIP)
            sd = max(atr*sl_m, price*0.0045); td = atr*tp_m
            if sig=="BUY": sl,tp = fill-sd, fill+td
            else: sl,tp = fill+sd, fill-td
            op.append({"c":coin,"s":sig,"e":fill,"sl":sl,"tp":tp,"n":5000*0.06*3,"h":0})
            cd[coin] = ts
    return cl


def rpt(name, fn, sl=2.0, tp=3.0, ttl=14):
    t_full = run(coins, ref, fn, sl, tp, ttl)
    t_oos = run(coins, ref[ref>sp70], fn, sl, tp, ttl)
    t_h1 = run(coins, ref[ref<=sp50], fn, sl, tp, ttl)
    t_h2 = run(coins, ref[ref>sp50], fn, sl, tp, ttl)
    def s(t): n=len(t); return (n, sum(1 for x in t if x>0)/n*100 if n else 0, sum(t))
    nf,wf,netf = s(t_full)
    no,wo,neto = s(t_oos)
    n1,w1,net1 = s(t_h1)
    n2,w2,net2 = s(t_h2)
    p1,p2 = neto>0, net1>0 and net2>0
    mark = " ***" if p1 and p2 else " *" if p1 else ""
    print(f"  {name:<40} N={nf:>4} WR={wf:>5.1f}% ${netf:>+8.2f} | OOS ${neto:>+8.2f}[{'P' if p1 else 'F'}] H={net1:>+7.0f}/{net2:>+7.0f}[{'P' if p2 else 'F'}]{mark}", flush=True)
    return {"name":name,"n":nf,"wr":wf,"net":netf,"oos":neto,"h1":net1,"h2":net2,"oos_pass":p1,"both_pass":p2}


def tsmom(df, lb1=28, lb2=7, minm=0.03):
    if len(df) < lb1+2: return None
    cl = df["close"].values
    m1 = (cl[-1]-cl[-lb1])/cl[-lb1]
    m2 = (cl[-1]-cl[-lb2])/cl[-lb2]
    if not ((m1>0 and m2>0) or (m1<0 and m2<0)): return None
    if abs(m1) < minm: return None
    return "BUY" if m1>0 else "SELL"


# ═══════════════════════════════════════════════════════════
# ROUND 1: Signal Filters
# ═══════════════════════════════════════════════════════════
print("=" * 90, flush=True)
print("  ROUND 1: Signal Filters on 1D TSMOM 28d/7d", flush=True)
print("=" * 90, flush=True)

results = []

# Baseline
results.append(rpt("Baseline", lambda df: tsmom(df)))

# + Volume filter
def tsmom_vol(df):
    sig = tsmom(df)
    if sig is None: return None
    vol = df["volume"].values
    if len(vol)<21: return sig
    if np.mean(vol[-21:-1])>0 and vol[-1] < np.mean(vol[-21:-1])*1.2: return None
    return sig
results.append(rpt("+ Volume > 1.2x avg", tsmom_vol))

# + RSI
def tsmom_rsi(df):
    sig = tsmom(df)
    if sig is None: return None
    cl = df["close"].values
    if len(cl)<15: return sig
    d = np.diff(cl[-15:])
    g,lo = np.where(d>0,d,0), np.where(d<0,-d,0)
    ag,al = np.mean(g),np.mean(lo)
    rsi = 100 if al==0 else 100-100/(1+ag/al)
    if sig=="BUY" and rsi>70: return None
    if sig=="SELL" and rsi<30: return None
    return sig
results.append(rpt("+ RSI filter (70/30)", tsmom_rsi))

# + Body confirmation
def tsmom_body(df):
    sig = tsmom(df)
    if sig is None: return None
    body = (float(df["close"].iloc[-1])-float(df["open"].iloc[-1]))/float(df["open"].iloc[-1])
    if sig=="BUY" and body<0: return None
    if sig=="SELL" and body>0: return None
    return sig
results.append(rpt("+ Body confirmation", tsmom_body))

# + Strict 5% min
results.append(rpt("+ Strict 5% min move", lambda df: tsmom(df, 28, 7, 0.05)))

# + ATR expansion
def tsmom_atr_expand(df):
    sig = tsmom(df)
    if sig is None: return None
    cl = df["close"].values; h,l = df["high"].values, df["low"].values
    if len(cl)<25: return sig
    atr_recent = np.mean(h[-5:]-l[-5:])
    atr_prior = np.mean(h[-20:-5]-l[-20:-5])
    if atr_prior>0 and atr_recent/atr_prior < 1.0: return None  # Only in expanding vol
    return sig
results.append(rpt("+ ATR expansion only", tsmom_atr_expand))


# ═══════════════════════════════════════════════════════════
# ROUND 2: SL/TP Optimization
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("  ROUND 2: SL/TP Optimization", flush=True)
print("=" * 90, flush=True)

base_fn = lambda df: tsmom(df)
for sl, tp in [(1.5, 2.0), (1.5, 2.5), (1.5, 3.0), (2.0, 2.0), (2.0, 2.5), (2.0, 3.0), (2.0, 4.0), (2.5, 3.0), (2.5, 4.0), (3.0, 3.0)]:
    results.append(rpt(f"SL={sl} TP={tp}", base_fn, sl, tp))


# ═══════════════════════════════════════════════════════════
# ROUND 3: Momentum Period Variations
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("  ROUND 3: Momentum Period Variations", flush=True)
print("=" * 90, flush=True)

for lb1, lb2, mm in [(14,3,0.02), (21,5,0.025), (21,7,0.025), (28,7,0.03), (28,14,0.03), (35,7,0.035), (42,7,0.04), (42,14,0.04), (56,14,0.05)]:
    fn = lambda df, _l1=lb1, _l2=lb2, _m=mm: tsmom(df, _l1, _l2, _m)
    results.append(rpt(f"Mom {lb1}d/{lb2}d (min {mm*100:.1f}%)", fn))


# ═══════════════════════════════════════════════════════════
# ROUND 4: TTL (holding period) Optimization
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("  ROUND 4: TTL (Max Hold Days) Optimization", flush=True)
print("=" * 90, flush=True)

for ttl in [7, 10, 14, 21, 28]:
    results.append(rpt(f"TTL={ttl} days", base_fn, 2.0, 3.0, ttl))


# ═══════════════════════════════════════════════════════════
# ROUND 5: Combined Best (best filter + best SL/TP + best period)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 90, flush=True)
print("  ROUND 5: Combined Best Improvements", flush=True)
print("=" * 90, flush=True)

# Find OOS-passing variants from each round
oos_pass = [r for r in results if r["oos_pass"]]
both_pass = [r for r in results if r["oos_pass"] and r["both_pass"]]

print(f"\n  OOS-passing configs: {len(oos_pass)}/{len(results)}", flush=True)
print(f"  Both-halves-passing: {len(both_pass)}/{len(results)}", flush=True)

if both_pass:
    both_pass.sort(key=lambda x: x["net"], reverse=True)
    print("\n  TOP OOS+BOTH passing:", flush=True)
    for r in both_pass[:5]:
        print(f"    {r['name']:<40} N={r['n']:>4} Net=${r['net']:>+8.2f} OOS=${r['oos']:>+8.2f}", flush=True)

if oos_pass:
    oos_pass.sort(key=lambda x: x["oos"], reverse=True)
    print("\n  TOP by OOS Net:", flush=True)
    for r in oos_pass[:5]:
        print(f"    {r['name']:<40} N={r['n']:>4} Net=${r['net']:>+8.2f} OOS=${r['oos']:>+8.2f}", flush=True)

# Try combinations of best findings
print("\n  Trying best combinations:", flush=True)

# Best filter + Best period
def combo_rsi_35d7d(df):
    sig = tsmom(df, 35, 7, 0.035)
    if sig is None: return None
    cl = df["close"].values
    if len(cl)<15: return sig
    d = np.diff(cl[-15:])
    g,lo = np.where(d>0,d,0), np.where(d<0,-d,0)
    ag,al = np.mean(g),np.mean(lo)
    rsi = 100 if al==0 else 100-100/(1+ag/al)
    if sig=="BUY" and rsi>70: return None
    if sig=="SELL" and rsi<30: return None
    return sig

def combo_body_42d14d(df):
    sig = tsmom(df, 42, 14, 0.04)
    if sig is None: return None
    body = (float(df["close"].iloc[-1])-float(df["open"].iloc[-1]))/float(df["open"].iloc[-1])
    if sig=="BUY" and body<0: return None
    if sig=="SELL" and body>0: return None
    return sig

results.append(rpt("RSI + 35d/7d SL=2 TP=3", combo_rsi_35d7d, 2.0, 3.0))
results.append(rpt("Body + 42d/14d SL=2 TP=3", combo_body_42d14d, 2.0, 3.0))
results.append(rpt("RSI + 28d/7d SL=1.5 TP=3", tsmom_rsi, 1.5, 3.0))
results.append(rpt("RSI + 28d/7d SL=2 TP=2.5", tsmom_rsi, 2.0, 2.5))
results.append(rpt("Body + 28d/7d SL=1.5 TP=2.5", tsmom_body, 1.5, 2.5))

# Final summary
print("\n" + "=" * 90, flush=True)
print("  FINAL: All OOS-passing configs ranked by OOS Net", flush=True)
print("=" * 90, flush=True)

all_oos = [r for r in results if r["oos_pass"]]
all_oos.sort(key=lambda x: x["oos"], reverse=True)
for r in all_oos:
    bp = " BOTH" if r["both_pass"] else ""
    print(f"  {r['name']:<40} Full=${r['net']:>+8.2f} OOS=${r['oos']:>+8.2f} N={r['n']:>4} WR={r['wr']:>5.1f}%{bp}", flush=True)

print(f"\n  Total tested: {len(results)}", flush=True)
print(f"  OOS pass: {len(all_oos)}", flush=True)
print(f"  OOS + Both halves: {len([r for r in all_oos if r['both_pass']])}", flush=True)
print("Done.", flush=True)
