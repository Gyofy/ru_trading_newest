"""SOL / XRP / ADA Deep Dive: coin-specific edge hunting."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, '.')
import yfinance as yf

cost = 0.0018

print("=" * 80)
print("  SOL / XRP / ADA DEEP DIVE")
print("=" * 80)
print()

# Fetch all data
coins = {}
for coin, sym in [("SOL","SOL-USD"),("XRP","XRP-USD"),("ADA","ADA-USD"),
                   ("BTC","BTC-USD"),("ETH","ETH-USD")]:
    df = yf.Ticker(sym).history(period="180d", interval="1h")
    df.columns = [c.lower() for c in df.columns]
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    o = df["open"].values; v = df["volume"].values.astype(float)
    ret = pd.Series(c).pct_change().values
    tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)),np.abs(l-np.roll(c,1))))
    tr[0] = h[0]-l[0]
    atr = pd.Series(tr).rolling(14,min_periods=1).mean().values
    coins[coin] = {"df":df,"c":c,"h":h,"l":l,"o":o,"v":v,"ret":ret,"atr":atr,"n":len(c)}

btc_ret = coins["BTC"]["ret"]
n = min(coins["SOL"]["n"], coins["BTC"]["n"])

# ================================================================
print("[1] COIN PERSONALITY: 변동성/상관/베타 프로파일")
print("=" * 80)
print()

for coin in ["SOL","XRP","ADA"]:
    d = coins[coin]
    cn = min(d["n"], n)
    cr = d["ret"][-cn:]
    br = btc_ret[-cn:]
    valid = ~np.isnan(cr) & ~np.isnan(br)

    vol_1h = np.nanstd(cr) * 100
    vol_daily = vol_1h * np.sqrt(24)
    avg_atr_pct = np.nanmean(d["atr"][-cn:] / d["c"][-cn:]) * 100
    corr_btc = np.corrcoef(cr[valid], br[valid])[0,1]
    beta = np.cov(cr[valid], br[valid])[0,1] / (np.var(br[valid]) + 1e-10)
    max_1h_up = np.nanmax(cr) * 100
    max_1h_dn = np.nanmin(cr) * 100
    total_ret = (d["c"][-1] / d["c"][0] - 1) * 100

    print("  %s:" % coin)
    print("    180d return: %+.1f%%" % total_ret)
    print("    1h vol: %.2f%%, daily vol: %.1f%%, ATR/price: %.2f%%" % (vol_1h, vol_daily, avg_atr_pct))
    print("    BTC corr: %.3f, beta: %.2f" % (corr_btc, beta))
    print("    Max 1h: +%.1f%% / %.1f%%" % (max_1h_up, max_1h_dn))
    print()

# ================================================================
print("[2] TIME-OF-DAY PATTERN: 시간대별 수익률")
print("=" * 80)
print()

for coin in ["SOL","XRP","ADA"]:
    d = coins[coin]
    df = d["df"]
    hours = df.index.hour
    cr = d["ret"]

    print("  %s hourly avg return (UTC):" % coin)
    best_h = -1; best_ret = -999; worst_h = -1; worst_ret = 999
    for h in range(0, 24, 2):
        mask = (hours >= h) & (hours < h+2)
        avg = np.nanmean(cr[mask]) * 100
        if avg > best_ret: best_ret = avg; best_h = h
        if avg < worst_ret: worst_ret = avg; worst_h = h
        bar = "#" * int(abs(avg) * 500) if not np.isnan(avg) else ""
        sign = "+" if avg > 0 else "-"
        print("    %02d-%02d: %+.4f%% %s%s" % (h, h+2, avg, sign, bar))
    print("    Best: %02d-%02d (%+.4f%%), Worst: %02d-%02d (%+.4f%%)" % (
        best_h, best_h+2, best_ret, worst_h, worst_h+2, worst_ret))
    print()

# ================================================================
print("[3] DAY-OF-WEEK PATTERN")
print("=" * 80)
print()

for coin in ["SOL","XRP","ADA"]:
    d = coins[coin]
    df = d["df"]
    dows = df.index.dayofweek
    cr = d["ret"]
    print("  %s:" % coin)
    for dow, name in enumerate(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]):
        mask = dows == dow
        avg = np.nanmean(cr[mask]) * 100
        vol = np.nanstd(cr[mask]) * 100
        print("    %s: ret=%+.4f%% vol=%.3f%%" % (name, avg, vol))
    print()

# ================================================================
print("[4] CROSS-COIN RELATIVE STRENGTH")
print("=" * 80)
print()
print("  SOL vs XRP vs ADA: when one outperforms, does it continue?")
print()

cn = min(coins["SOL"]["n"], coins["XRP"]["n"], coins["ADA"]["n"])
sol_r = coins["SOL"]["ret"][-cn:]
xrp_r = coins["XRP"]["ret"][-cn:]
ada_r = coins["ADA"]["ret"][-cn:]

# Rolling 12h relative performance
for lookback in [6, 12, 24]:
    sol_cum = pd.Series(sol_r).rolling(lookback).sum().values
    xrp_cum = pd.Series(xrp_r).rolling(lookback).sum().values
    ada_cum = pd.Series(ada_r).rolling(lookback).sum().values

    # When SOL outperforms both -> does it continue?
    split = cn // 3
    for leader in ["SOL","XRP","ADA"]:
        l_cum = {"SOL":sol_cum,"XRP":xrp_cum,"ADA":ada_cum}[leader]
        others = [c for c in ["SOL","XRP","ADA"] if c != leader]
        o_cums = [{"SOL":sol_cum,"XRP":xrp_cum,"ADA":ada_cum}[c] for c in others]

        # Leader outperforms both others
        nexts = []
        for i in range(max(split, lookback+1), cn-6):
            if np.isnan(l_cum[i]): continue
            if all(not np.isnan(oc[i]) and l_cum[i] > oc[i] for oc in o_cums):
                # Leader's next 6h return
                leader_ret = {"SOL":sol_r,"XRP":xrp_r,"ADA":ada_r}[leader]
                next_6h = np.nansum(leader_ret[i+1:i+7])
                nexts.append(next_6h)

        if len(nexts) > 10:
            arr = np.array(nexts)
            print("  %s leads (last %dh) -> next 6h: n=%d avg=%+.3f%% WR=%.0f%%" % (
                leader, lookback, len(arr), np.mean(arr)*100, np.mean(arr>0)*100))
    print()

# ================================================================
print("[5] VOLUME PROFILE: 각 코인의 거래량 패턴")
print("=" * 80)
print()

for coin in ["SOL","XRP","ADA"]:
    d = coins[coin]
    v = d["v"][-cn:]
    cr = d["ret"][-cn:]
    v_sma = pd.Series(v).rolling(24).mean().values

    # High volume bars: do they predict direction?
    split = cn // 3
    for vol_thresh in [2.0, 3.0]:
        hi_vol = []
        for i in range(max(split, 25), cn-6):
            if np.isnan(v_sma[i]) or v_sma[i] < 1: continue
            if v[i] > v_sma[i] * vol_thresh:
                bar_dir = 1 if cr[i] > 0 else -1
                next_6 = np.nansum(cr[i+1:i+7]) * bar_dir
                hi_vol.append(next_6)
        if len(hi_vol) > 5:
            arr = np.array(hi_vol)
            print("  %s vol>%.0fx: n=%d next_6h=%+.3f%% WR=%.0f%%" % (
                coin, vol_thresh, len(arr), np.mean(arr)*100, np.mean(arr>0)*100))
    print()

# ================================================================
print("[6] MEAN REVERSION vs MOMENTUM: 어떤 코인이 어떤 성격?")
print("=" * 80)
print()

for coin in ["SOL","XRP","ADA"]:
    d = coins[coin]
    cr = d["ret"][-cn:]
    split = cn // 3

    for lookback in [3, 6, 12]:
        cum = pd.Series(cr).rolling(lookback).sum().values
        # Momentum: past up -> future up?
        mom_results = []
        # Mean reversion: past up -> future down?
        rev_results = []
        for i in range(max(split, lookback+1), cn-6):
            if np.isnan(cum[i]): continue
            next_6 = np.nansum(cr[i+1:i+7])
            if cum[i] > 0:
                mom_results.append(next_6)      # momentum: expect +
                rev_results.append(-next_6)     # reversion: expect -
            else:
                mom_results.append(-next_6)     # momentum: expect -
                rev_results.append(next_6)      # reversion: expect +

        mom = np.array(mom_results)
        print("  %s lb=%d: momentum=%+.4f%% reversion=%+.4f%% (%s tendency)" % (
            coin, lookback, np.mean(mom)*100, np.mean(-mom)*100,
            "MOMENTUM" if np.mean(mom) > 0 else "MEAN-REVERT"))
    print()

# ================================================================
print("[7] COIN-SPECIFIC EDGE: BTC spike별 코인 반응 차이")
print("=" * 80)
print()

br = btc_ret[-cn:]
for coin in ["SOL","XRP","ADA"]:
    cr = coins[coin]["ret"][-cn:]
    d = coins[coin]
    atr_c = d["atr"][-cn:]
    close_c = d["c"][-cn:]
    open_c = d["o"][-cn:]
    high_c = d["h"][-cn:]
    low_c = d["l"][-cn:]
    split = cn // 3

    # 6h hold simulation per coin
    results = []; pe = 0
    for i in range(max(split,2), cn-7):
        if i < pe: continue
        if np.isnan(br[i]) or abs(br[i]) < 0.012: continue
        side = "BUY" if br[i] > 0 else "SELL"
        eb = i+1
        if eb >= cn-6: continue
        entry = open_c[eb]
        ai = atr_c[i] if not np.isnan(atr_c[i]) else entry*0.01
        td=ai*1.5; sd=ai*1.0
        if side=="BUY": tp,sl = entry+td, entry-sd
        else: tp,sl = entry-td, entry+sd
        pnl = 0
        for j in range(eb, min(eb+6, cn)):
            if side=="BUY":
                if low_c[j]<=sl: pnl=(sl-entry)/entry; break
                if high_c[j]>=tp: pnl=(tp-entry)/entry; break
            else:
                if high_c[j]>=sl: pnl=(entry-sl)/entry; break
                if low_c[j]<=tp: pnl=(entry-tp)/entry; break
        else:
            if eb+5<cn:
                ep=close_c[eb+5]
                pnl=(ep-entry)/entry if side=="BUY" else (entry-ep)/entry
        results.append({"pnl": pnl-cost, "side": side, "btc_ret": br[i]})
        pe = eb + 7

    arr = np.array([r["pnl"] for r in results])
    up_arr = np.array([r["pnl"] for r in results if r["side"]=="BUY"])
    dn_arr = np.array([r["pnl"] for r in results if r["side"]=="SELL"])

    print("  %s (btc>1.2%%, 6h hold):" % coin)
    if len(arr) > 5:
        print("    ALL:  n=%d WR=%.1f%% avg=%+.4f%%" % (len(arr), np.mean(arr>0)*100, np.mean(arr)*100))
    if len(up_arr) > 3:
        print("    LONG: n=%d WR=%.1f%% avg=%+.4f%%" % (len(up_arr), np.mean(up_arr>0)*100, np.mean(up_arr)*100))
    if len(dn_arr) > 3:
        print("    SHORT:n=%d WR=%.1f%% avg=%+.4f%%" % (len(dn_arr), np.mean(dn_arr>0)*100, np.mean(dn_arr)*100))

    # Beta-adjusted: stronger spikes -> stronger follow?
    large = [r for r in results if abs(r["btc_ret"]) > 0.015]
    small = [r for r in results if abs(r["btc_ret"]) <= 0.015]
    if len(large) > 3:
        la = np.array([r["pnl"] for r in large])
        print("    BIG spike(>1.5%%):  n=%d avg=%+.4f%%" % (len(la), np.mean(la)*100))
    if len(small) > 3:
        sa = np.array([r["pnl"] for r in small])
        print("    SMALL spike(1.2-1.5%%): n=%d avg=%+.4f%%" % (len(sa), np.mean(sa)*100))
    print()

print("=" * 80)
print("  SUMMARY: coin-specific insights for strategy design")
print("=" * 80)
