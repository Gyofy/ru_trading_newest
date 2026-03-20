"""EXHAUSTIVE SEARCH: every combination, every variable, find ALL positive edges."""
import numpy as np, pandas as pd, sys, itertools
sys.path.insert(0, '.')
import yfinance as yf

cost = 0.0018
np.random.seed(42)

# Load 180d 1h data
print("Loading data...")
coins_raw = {}
for coin, sym in [("SOL","SOL-USD"),("XRP","XRP-USD"),("ADA","ADA-USD"),
                   ("BTC","BTC-USD"),("ETH","ETH-USD"),("LINK","LINK-USD"),
                   ("DOGE","DOGE-USD"),("AVAX","AVAX-USD"),("DOT","DOT-USD")]:
    try:
        df = yf.Ticker(sym).history(period="180d", interval="1h")
        df.columns = [c.lower() for c in df.columns]
        c=df["close"].values; h=df["high"].values; l=df["low"].values
        o=df["open"].values; v=df["volume"].values.astype(float)
        ret=pd.Series(c).pct_change().values
        tr=np.maximum(h-l,np.maximum(np.abs(h-np.roll(c,1)),np.abs(l-np.roll(c,1))))
        tr[0]=h[0]-l[0]
        atr=pd.Series(tr).rolling(14,min_periods=1).mean().values
        v_sma=pd.Series(v).rolling(24,min_periods=5).mean().values
        ema8=pd.Series(c).ewm(span=8,adjust=False).mean().values
        ema21=pd.Series(c).ewm(span=21,adjust=False).mean().values
        delta=pd.Series(c).diff()
        gain=delta.clip(lower=0).ewm(span=14,adjust=False).mean().values
        loss_v=(-delta.clip(upper=0)).ewm(span=14,adjust=False).mean().values
        rsi=100-100/(1+gain/(loss_v+1e-10))
        coins_raw[coin]={"df":df,"c":c,"h":h,"l":l,"o":o,"v":v,"ret":ret,
                         "atr":atr,"v_sma":v_sma,"ema8":ema8,"ema21":ema21,"rsi":rsi,"n":len(c)}
        print("  %s: %d bars" % (coin, len(c)))
    except: pass

btc = coins_raw["BTC"]
btc_ret = btc["ret"]
n_global = min(d["n"] for d in coins_raw.values())

def sim(coin_data, entries, tp_m, sl_m, hold, n):
    c=coin_data["c"][-n:]; h=coin_data["h"][-n:]; l=coin_data["l"][-n:]
    o=coin_data["o"][-n:]; atr=coin_data["atr"][-n:]
    results=[]; pe=0
    for i, side in entries:
        if i<pe or i>=n-hold-1: continue
        eb=i+1; entry=o[eb]
        ai=atr[i] if not np.isnan(atr[i]) else entry*0.01
        td=ai*tp_m; sd=ai*sl_m
        if side=="BUY": tp,sl=entry+td,entry-sd
        else: tp,sl=entry-td,entry+sd
        pnl=0
        for j in range(eb,min(eb+hold,n)):
            if side=="BUY":
                if l[j]<=sl: pnl=(sl-entry)/entry; break
                if h[j]>=tp: pnl=(tp-entry)/entry; break
            else:
                if h[j]>=sl: pnl=(entry-sl)/entry; break
                if l[j]<=tp: pnl=(entry-tp)/entry; break
        else:
            if eb+hold-1<n:
                ep=c[eb+hold-1]
                pnl=(ep-entry)/entry if side=="BUY" else (entry-ep)/entry
        results.append(pnl-cost); pe=eb+hold+1
    return np.array(results) if results else np.array([])

# ================================================================
print()
print("=" * 90)
print("  EXHAUSTIVE SEARCH: %d coins x all strategies x all params" % len(coins_raw))
print("=" * 90)
print()

positive_strategies = []
total_tested = 0

# ENTRY SIGNALS
entry_generators = {
    # BTC spike
    "btc_spike_both": lambda d, cn, br, thresh: [(i,"BUY" if br[i]>0 else "SELL") for i in range(cn//3,cn) if not np.isnan(br[i]) and abs(br[i])>thresh],
    "btc_spike_long": lambda d, cn, br, thresh: [(i,"BUY") for i in range(cn//3,cn) if not np.isnan(br[i]) and br[i]>thresh],
    "btc_spike_short": lambda d, cn, br, thresh: [(i,"SELL") for i in range(cn//3,cn) if not np.isnan(br[i]) and br[i]<-thresh],
    # Momentum
    "mom_follow": lambda d, cn, br, lb: [(i,"BUY" if d["c"][i]/d["c"][i-lb]-1>0 else "SELL") for i in range(max(cn//3,lb+1),cn) if abs(d["c"][i]/d["c"][i-lb]-1)>0.003],
    "mom_counter": lambda d, cn, br, lb: [(i,"SELL" if d["c"][i]/d["c"][i-lb]-1>0 else "BUY") for i in range(max(cn//3,lb+1),cn) if abs(d["c"][i]/d["c"][i-lb]-1)>0.003],
    # EMA
    "ema_follow": lambda d, cn, br, _: [(i,"BUY" if d["ema8"][i]>d["ema21"][i] else "SELL") for i in range(max(cn//3,26),cn)],
    # RSI
    "rsi_revert": lambda d, cn, br, thresh: [(i,"BUY") if d["rsi"][i]<thresh else (i,"SELL") for i in range(max(cn//3,15),cn) if d["rsi"][i]<thresh or d["rsi"][i]>(100-thresh)],
    # Volume spike
    "vol_spike": lambda d, cn, br, mult: [(i,"BUY" if d["c"][i]>d["o"][i] else "SELL") for i in range(max(cn//3,25),cn) if not np.isnan(d["v_sma"][i]) and d["v_sma"][i]>0 and d["v"][i]>d["v_sma"][i]*mult and abs((d["c"][i]-d["o"][i])/d["o"][i])>0.002],
    # Always long / always short (baseline)
    "always_long": lambda d, cn, br, _: [(i,"BUY") for i in range(cn//3,cn,3)],
    "always_short": lambda d, cn, br, _: [(i,"SELL") for i in range(cn//3,cn,3)],
    # BTC spike + vol confirmation
    "btc_spike_vol": lambda d, cn, br, thresh: [(i,"BUY" if br[i]>0 else "SELL") for i in range(max(cn//3,25),cn) if not np.isnan(br[i]) and abs(br[i])>thresh and not np.isnan(d["v_sma"][i]) and d["v_sma"][i]>0 and d["v"][i]>d["v_sma"][i]*1.5],
    # BTC spike + RSI confirmation
    "btc_spike_rsi": lambda d, cn, br, thresh: [(i,"BUY") for i in range(max(cn//3,15),cn) if not np.isnan(br[i]) and br[i]>thresh and d["rsi"][i]<60] + [(i,"SELL") for i in range(max(cn//3,15),cn) if not np.isnan(br[i]) and br[i]<-thresh and d["rsi"][i]>40],
    # Multi-coin: enter weakest coin when BTC drops (relative strength)
}

# Parameter grids
param_grids = {
    "btc_spike_both": [0.008, 0.010, 0.012, 0.015, 0.020],
    "btc_spike_long": [0.008, 0.010, 0.012, 0.015, 0.020],
    "btc_spike_short": [0.008, 0.010, 0.012, 0.015, 0.020],
    "mom_follow": [3, 6, 12, 24],
    "mom_counter": [3, 6, 12, 24],
    "ema_follow": [0],
    "rsi_revert": [20, 25, 30],
    "vol_spike": [1.5, 2.0, 3.0],
    "always_long": [0],
    "always_short": [0],
    "btc_spike_vol": [0.008, 0.010, 0.012, 0.015],
    "btc_spike_rsi": [0.008, 0.010, 0.012, 0.015],
}

# Barrier grids
tp_sl_grids = [(0.7,0.5), (1.0,0.7), (1.0,1.0), (1.5,1.0), (2.0,1.0), (2.0,1.5), (3.0,1.0)]
hold_grids = [2, 4, 6, 8, 12, 24]

target_coins = ["SOL","XRP","ADA","ETH","LINK","DOT","DOGE","AVAX","BTC"]

for strat_name, gen_fn in entry_generators.items():
    params = param_grids[strat_name]
    for param in params:
        for coin in target_coins:
            if coin not in coins_raw: continue
            d = coins_raw[coin]
            cn = min(d["n"], n_global)
            br = btc_ret[-cn:]

            try:
                entries = gen_fn({"c":d["c"][-cn:],"o":d["o"][-cn:],"v":d["v"][-cn:],
                                  "v_sma":d["v_sma"][-cn:],"ema8":d["ema8"][-cn:],
                                  "ema21":d["ema21"][-cn:],"rsi":d["rsi"][-cn:]},
                                 cn, br, param)
            except: continue

            if len(entries) < 5: continue

            for tp_m, sl_m in tp_sl_grids:
                for hold in hold_grids:
                    total_tested += 1
                    arr = sim(d, entries, tp_m, sl_m, hold, cn)
                    if len(arr) < 10: continue

                    avg = np.mean(arr); wr = np.mean(arr>0); nn = len(arr)
                    if avg > 0.0005:  # positive after cost
                        eq = np.cumprod(1+arr)
                        mdd = np.min(eq/np.maximum.accumulate(eq)-1)
                        positive_strategies.append({
                            "strat": strat_name, "param": param, "coin": coin,
                            "tp": tp_m, "sl": sl_m, "hold": hold,
                            "n": nn, "wr": wr, "avg": avg, "mdd": mdd,
                        })

print("Total combinations tested: %d" % total_tested)
print("Positive strategies (avg > 0.05%%): %d" % len(positive_strategies))
print("Hit rate: %.1f%%" % (len(positive_strategies)/max(total_tested,1)*100))
print()

# Expected by random chance (at 50% base rate, some will be positive)
# Binomial: P(avg > 0) ≈ 50% minus cost drag
expected_random = total_tested * 0.30  # rough: 30% of random strategies look positive
print("Expected positive by chance: ~%.0f (%.1f%%)" % (expected_random, expected_random/max(total_tested,1)*100))
print()

if positive_strategies:
    # Sort by avg PnL
    positive_strategies.sort(key=lambda x: x["avg"], reverse=True)

    print("=" * 90)
    print("  TOP 30 POSITIVE STRATEGIES")
    print("=" * 90)
    print()
    print("%-20s %-5s %-5s %-4s %-4s %-3s | %4s %5s %8s %6s" % (
        "Strategy","Param","Coin","TP","SL","H","N","WR","Avg","MDD"))
    print("-" * 85)

    for s in positive_strategies[:30]:
        print("%-20s %-5s %-5s %-4.1f %-4.1f %-3d | %4d %4.1f%% %+7.4f%% %5.1f%%" % (
            s["strat"], str(s["param"])[:5], s["coin"],
            s["tp"], s["sl"], s["hold"],
            s["n"], s["wr"]*100, s["avg"]*100, s["mdd"]*100))

    # Group by strategy type
    print()
    print("=" * 90)
    print("  STRATEGY TYPE SUMMARY")
    print("=" * 90)
    print()

    strat_groups = {}
    for s in positive_strategies:
        key = s["strat"]
        if key not in strat_groups: strat_groups[key] = []
        strat_groups[key].append(s)

    for key in sorted(strat_groups, key=lambda k: -len(strat_groups[k])):
        group = strat_groups[key]
        avg_avg = np.mean([s["avg"] for s in group])
        avg_n = np.mean([s["n"] for s in group])
        coins_hit = set(s["coin"] for s in group)
        print("  %-20s: %3d positive combos, avg_pnl=%+.4f%%, coins=%s" % (
            key, len(group), avg_avg*100, ",".join(sorted(coins_hit))))

    # Robustness check: strategies that work across multiple coins
    print()
    print("=" * 90)
    print("  MULTI-COIN ROBUST STRATEGIES (same strat+param works on 3+ coins)")
    print("=" * 90)
    print()

    from collections import defaultdict
    multi = defaultdict(list)
    for s in positive_strategies:
        key = (s["strat"], s["param"], s["tp"], s["sl"], s["hold"])
        multi[key].append(s)

    robust = [(k, v) for k, v in multi.items() if len(set(s["coin"] for s in v)) >= 3]
    robust.sort(key=lambda x: np.mean([s["avg"] for s in x[1]]), reverse=True)

    if robust:
        for (strat, param, tp, sl, hold), group in robust[:15]:
            coins_str = ",".join(sorted(set(s["coin"] for s in group)))
            avg = np.mean([s["avg"] for s in group])
            avg_n = np.mean([s["n"] for s in group])
            print("  %-18s p=%-5s tp%.1f sl%.1f h%d: %s avg=%+.4f%% n=%.0f" % (
                strat, str(param)[:5], tp, sl, hold, coins_str, avg*100, avg_n))
    else:
        print("  No strategy works consistently across 3+ coins")

    print()
    print("=" * 90)
    print("  CRITICAL: MULTIPLE COMPARISON CORRECTION")
    print("=" * 90)
    print()
    print("  Tested: %d combinations" % total_tested)
    print("  Positive: %d (%.1f%%)" % (len(positive_strategies), len(positive_strategies)/max(total_tested,1)*100))
    bonferroni = 0.05 / total_tested
    print("  Bonferroni threshold (p=0.05): p < %.6f per test" % bonferroni)
    print("  -> Most positives are likely noise from multiple testing")
    print()

    # Estimate: how many would be positive with random entry?
    print("  Random baseline test (shuffle direction):")
    n_random_positive = 0
    n_random_total = 0
    for coin in ["SOL","XRP","ADA"]:
        if coin not in coins_raw: continue
        d = coins_raw[coin]
        cn = min(d["n"], n_global)
        # Random entries every 3 bars
        rand_entries = [(i, np.random.choice(["BUY","SELL"])) for i in range(cn//3, cn, 3)]
        for tp_m, sl_m in [(1.5, 1.0)]:
            for hold in [6]:
                n_random_total += 1
                arr = sim(d, rand_entries, tp_m, sl_m, hold, cn)
                if len(arr) > 10 and np.mean(arr) > 0.0005:
                    n_random_positive += 1

    print("    Random positive: %d / %d" % (n_random_positive, n_random_total))
