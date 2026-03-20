"""SOL/ADA 3h 단기: 캔들 이상변화 + 주문량 이상 기반 edge hunting."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, '.')
import yfinance as yf

cost = 0.0018

print("=" * 80)
print("  SOL / ADA 3H MICROSTRUCTURE EDGE HUNTING")
print("  캔들 이상변화 + 주문량 이상 + 모든 가능성")
print("=" * 80)
print()

# Load 1h data
data = {}
for coin, sym in [("SOL","SOL-USD"),("ADA","ADA-USD"),("BTC","BTC-USD")]:
    df = yf.Ticker(sym).history(period="180d", interval="1h")
    df.columns = [c.lower() for c in df.columns]
    c=df["close"].values; h=df["high"].values; l=df["low"].values
    o=df["open"].values; v=df["volume"].values.astype(float)
    n = len(c)
    tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)),np.abs(l-np.roll(c,1))))
    tr[0]=h[0]-l[0]
    atr = pd.Series(tr).rolling(14,min_periods=1).mean().values
    ret = pd.Series(c).pct_change().values
    v_sma = pd.Series(v).rolling(24,min_periods=5).mean().values

    # === MICROSTRUCTURE FEATURES (causal, no leakage) ===
    # 1. Bar structure
    body = np.abs(c - o)
    wick_upper = h - np.maximum(c, o)
    wick_lower = np.minimum(c, o) - l
    bar_range = h - l
    body_ratio = body / (bar_range + 1e-10)  # 0=doji, 1=marubozu
    upper_wick_ratio = wick_upper / (bar_range + 1e-10)
    lower_wick_ratio = wick_lower / (bar_range + 1e-10)

    # 2. Volume anomaly
    vol_ratio = v / (v_sma + 1e-10)

    # 3. Bar size anomaly (ATR 대비)
    bar_size_ratio = bar_range / (atr + 1e-10)

    # 4. Consecutive direction
    bar_dir = np.sign(c - o)
    consec = np.zeros(n)
    for i in range(1, n):
        if bar_dir[i] == bar_dir[i-1] and bar_dir[i] != 0:
            consec[i] = consec[i-1] + bar_dir[i]
        else:
            consec[i] = bar_dir[i]

    # 5. Gap (open vs prev close)
    gap = np.zeros(n)
    gap[1:] = (o[1:] - c[:-1]) / c[:-1]

    # 6. Volume-price divergence
    # Price up but volume down vs avg = bearish divergence
    vp_div = np.zeros(n)
    for i in range(1, n):
        price_up = ret[i] > 0 if not np.isnan(ret[i]) else False
        vol_below = vol_ratio[i] < 0.8 if not np.isnan(vol_ratio[i]) else False
        vol_above = vol_ratio[i] > 1.5 if not np.isnan(vol_ratio[i]) else False
        if price_up and vol_below: vp_div[i] = -1  # bearish div
        elif not price_up and vol_below: vp_div[i] = 1  # bullish div (sell on low vol)
        elif price_up and vol_above: vp_div[i] = 2  # strong bull
        elif not price_up and vol_above: vp_div[i] = -2  # strong bear

    data[coin] = {
        "df":df, "c":c, "h":h, "l":l, "o":o, "v":v, "n":n,
        "atr":atr, "ret":ret, "v_sma":v_sma, "vol_ratio":vol_ratio,
        "body_ratio":body_ratio, "upper_wick_ratio":upper_wick_ratio,
        "lower_wick_ratio":lower_wick_ratio, "bar_size_ratio":bar_size_ratio,
        "consec":consec, "gap":gap, "vp_div":vp_div, "bar_dir":bar_dir,
    }
    print("  %s: %d bars" % (coin, n))

print()
btc_ret = data["BTC"]["ret"]
n_global = min(d["n"] for d in data.values())

def sim3h(d, entries, tp_m, sl_m, n):
    c=d["c"]; h=d["h"]; l=d["l"]; o=d["o"]; atr=d["atr"]
    results=[]; pe=0
    for i, side in entries:
        if i<pe or i>=n-4: continue
        eb=i+1; entry=o[eb]
        ai=atr[i] if not np.isnan(atr[i]) else entry*0.01
        td=ai*tp_m; sd=ai*sl_m
        if side=="BUY": tp,sl=entry+td,entry-sd
        else: tp,sl=entry-td,entry+sd
        pnl=0
        for j in range(eb,min(eb+3,n)):  # 3h hold
            if side=="BUY":
                if l[j]<=sl: pnl=(sl-entry)/entry; break
                if h[j]>=tp: pnl=(tp-entry)/entry; break
            else:
                if h[j]>=sl: pnl=(entry-sl)/entry; break
                if l[j]<=tp: pnl=(entry-tp)/entry; break
        else:
            if eb+2<n:
                ep=c[eb+2]; pnl=(ep-entry)/entry if side=="BUY" else (entry-ep)/entry
        results.append(pnl-cost); pe=eb+4
    return np.array(results) if results else np.array([])

def rpt(name, arr):
    if len(arr)<5: return
    nn=len(arr); wr=np.mean(arr>0); avg=np.mean(arr)
    m = "***" if avg>0.002 else " **" if avg>0.0005 else "  *" if avg>0 else "   "
    print("  %-40s n=%3d WR=%.1f%% avg=%+.5f%% %s" % (name, nn, wr*100, avg*100, m))

# ================================================================
print("=" * 80)
print("  [1] CANDLE ANOMALY SIGNALS")
print("=" * 80)
print()

for coin in ["SOL","ADA"]:
    d = data[coin]; cn = min(d["n"], n_global)
    split = cn // 3
    print("--- %s ---" % coin)

    # A. Doji (body < 20% of range) -> reversal?
    for tp_m, sl_m in [(0.7,0.5), (1.0,0.7), (1.5,1.0)]:
        entries = []
        for i in range(max(split,3), cn):
            if d["bar_size_ratio"][i] < 0.5: continue  # skip tiny bars
            if d["body_ratio"][i] < 0.2:  # doji
                # Direction: reverse of last 3 bars
                prev_dir = np.sign(d["c"][i] - d["c"][max(0,i-3)])
                if prev_dir > 0: entries.append((i, "SELL"))
                elif prev_dir < 0: entries.append((i, "BUY"))
        rpt("%s doji_reversal tp%.1f/sl%.1f" % (coin, tp_m, sl_m), sim3h(d, entries, tp_m, sl_m, cn))

    # B. Marubozu (body > 80% of range) -> continuation
    for tp_m, sl_m in [(0.7,0.5), (1.0,0.7), (1.5,1.0)]:
        entries = []
        for i in range(max(split,1), cn):
            if d["body_ratio"][i] > 0.8 and d["bar_size_ratio"][i] > 1.0:
                side = "BUY" if d["c"][i] > d["o"][i] else "SELL"
                entries.append((i, side))
        rpt("%s marubozu_cont tp%.1f/sl%.1f" % (coin, tp_m, sl_m), sim3h(d, entries, tp_m, sl_m, cn))

    # C. Long upper wick (>50% of range) -> bearish
    for tp_m, sl_m in [(0.7,0.5), (1.0,0.7)]:
        entries = []
        for i in range(max(split,1), cn):
            if d["upper_wick_ratio"][i] > 0.5 and d["bar_size_ratio"][i] > 0.8:
                entries.append((i, "SELL"))
        rpt("%s long_upper_wick tp%.1f/sl%.1f" % (coin, tp_m, sl_m), sim3h(d, entries, tp_m, sl_m, cn))

    # D. Long lower wick (>50%) -> bullish
    for tp_m, sl_m in [(0.7,0.5), (1.0,0.7)]:
        entries = []
        for i in range(max(split,1), cn):
            if d["lower_wick_ratio"][i] > 0.5 and d["bar_size_ratio"][i] > 0.8:
                entries.append((i, "BUY"))
        rpt("%s long_lower_wick tp%.1f/sl%.1f" % (coin, tp_m, sl_m), sim3h(d, entries, tp_m, sl_m, cn))

    # E. Engulfing (current bar engulfs previous)
    for tp_m, sl_m in [(1.0,0.7), (1.5,1.0)]:
        entries = []
        for i in range(max(split,2), cn):
            prev_body = abs(d["c"][i-1] - d["o"][i-1])
            curr_body = abs(d["c"][i] - d["o"][i])
            if curr_body > prev_body * 1.5 and d["bar_size_ratio"][i] > 0.8:
                if d["c"][i] > d["o"][i] and d["c"][i-1] < d["o"][i-1]:
                    entries.append((i, "BUY"))  # bullish engulfing
                elif d["c"][i] < d["o"][i] and d["c"][i-1] > d["o"][i-1]:
                    entries.append((i, "SELL"))  # bearish engulfing
        rpt("%s engulfing tp%.1f/sl%.1f" % (coin, tp_m, sl_m), sim3h(d, entries, tp_m, sl_m, cn))

    # F. Consecutive bars (3+ same direction)
    for tp_m, sl_m in [(1.0,0.7), (1.5,1.0)]:
        # Continuation
        entries_cont = [(i, "BUY" if d["consec"][i]>0 else "SELL") for i in range(max(split,4), cn) if abs(d["consec"][i]) >= 3]
        rpt("%s consec3_cont tp%.1f/sl%.1f" % (coin, tp_m, sl_m), sim3h(d, entries_cont, tp_m, sl_m, cn))
        # Reversal
        entries_rev = [(i, "SELL" if d["consec"][i]>0 else "BUY") for i in range(max(split,4), cn) if abs(d["consec"][i]) >= 3]
        rpt("%s consec3_rev tp%.1f/sl%.1f" % (coin, tp_m, sl_m), sim3h(d, entries_rev, tp_m, sl_m, cn))

    print()

# ================================================================
print("=" * 80)
print("  [2] VOLUME ANOMALY SIGNALS")
print("=" * 80)
print()

for coin in ["SOL","ADA"]:
    d = data[coin]; cn = min(d["n"], n_global)
    split = cn // 3
    print("--- %s ---" % coin)

    # A. Volume spike + bar direction
    for vol_mult in [2.0, 3.0, 5.0]:
        for tp_m, sl_m in [(0.7,0.5), (1.0,0.7), (1.5,1.0)]:
            entries = []
            for i in range(max(split,25), cn):
                if np.isnan(d["vol_ratio"][i]) or d["vol_ratio"][i] < vol_mult: continue
                side = "BUY" if d["c"][i] > d["o"][i] else "SELL"
                entries.append((i, side))
            rpt("%s volx%.0f_%s tp%.1f/sl%.1f" % (coin, vol_mult, "dir", tp_m, sl_m), sim3h(d, entries, tp_m, sl_m, cn))

    # B. Volume spike + reversal (contrarian)
    for vol_mult in [2.0, 3.0, 5.0]:
        for tp_m, sl_m in [(0.7,0.5), (1.0,0.7)]:
            entries = []
            for i in range(max(split,25), cn):
                if np.isnan(d["vol_ratio"][i]) or d["vol_ratio"][i] < vol_mult: continue
                side = "SELL" if d["c"][i] > d["o"][i] else "BUY"  # counter
                entries.append((i, side))
            rpt("%s volx%.0f_rev tp%.1f/sl%.1f" % (coin, vol_mult, tp_m, sl_m), sim3h(d, entries, tp_m, sl_m, cn))

    # C. Volume dry-up then spike (compression -> expansion)
    for tp_m, sl_m in [(1.0,0.7), (1.5,1.0)]:
        entries = []
        for i in range(max(split,25), cn):
            if i < 5: continue
            # Previous 3 bars: low volume
            prev_avg_ratio = np.mean(d["vol_ratio"][i-3:i])
            if np.isnan(prev_avg_ratio): continue
            if prev_avg_ratio < 0.7 and d["vol_ratio"][i] > 2.0:
                side = "BUY" if d["c"][i] > d["o"][i] else "SELL"
                entries.append((i, side))
        rpt("%s vol_compress_expand tp%.1f/sl%.1f" % (coin, tp_m, sl_m), sim3h(d, entries, tp_m, sl_m, cn))

    # D. Volume-price divergence
    for tp_m, sl_m in [(0.7,0.5), (1.0,0.7)]:
        # Strong bull (price up + high vol)
        entries_bull = [(i,"BUY") for i in range(max(split,2),cn) if d["vp_div"][i]==2]
        rpt("%s vp_strong_bull tp%.1f/sl%.1f" % (coin, tp_m, sl_m), sim3h(d, entries_bull, tp_m, sl_m, cn))
        # Strong bear
        entries_bear = [(i,"SELL") for i in range(max(split,2),cn) if d["vp_div"][i]==-2]
        rpt("%s vp_strong_bear tp%.1f/sl%.1f" % (coin, tp_m, sl_m), sim3h(d, entries_bear, tp_m, sl_m, cn))
        # Bearish div (price up, vol low) -> sell
        entries_bd = [(i,"SELL") for i in range(max(split,2),cn) if d["vp_div"][i]==-1]
        rpt("%s vp_bearish_div tp%.1f/sl%.1f" % (coin, tp_m, sl_m), sim3h(d, entries_bd, tp_m, sl_m, cn))
        # Bullish div (price down, vol low) -> buy
        entries_bld = [(i,"BUY") for i in range(max(split,2),cn) if d["vp_div"][i]==1]
        rpt("%s vp_bullish_div tp%.1f/sl%.1f" % (coin, tp_m, sl_m), sim3h(d, entries_bld, tp_m, sl_m, cn))

    print()

# ================================================================
print("=" * 80)
print("  [3] BAR SIZE ANOMALY (ATR 대비 비정상 크기)")
print("=" * 80)
print()

for coin in ["SOL","ADA"]:
    d = data[coin]; cn = min(d["n"], n_global)
    split = cn // 3
    print("--- %s ---" % coin)

    # A. Oversized bar (>2x ATR) -> continuation
    for size_thresh in [1.5, 2.0, 3.0]:
        for tp_m, sl_m in [(1.0,0.7), (1.5,1.0)]:
            entries = []
            for i in range(max(split,1), cn):
                if d["bar_size_ratio"][i] > size_thresh:
                    side = "BUY" if d["c"][i] > d["o"][i] else "SELL"
                    entries.append((i, side))
            rpt("%s bigbar>%.1f_cont tp%.1f/sl%.1f" % (coin, size_thresh, tp_m, sl_m), sim3h(d, entries, tp_m, sl_m, cn))

    # B. Oversized bar -> reversal
    for size_thresh in [2.0, 3.0]:
        for tp_m, sl_m in [(0.7,0.5), (1.0,0.7)]:
            entries = []
            for i in range(max(split,1), cn):
                if d["bar_size_ratio"][i] > size_thresh:
                    side = "SELL" if d["c"][i] > d["o"][i] else "BUY"
                    entries.append((i, side))
            rpt("%s bigbar>%.1f_rev tp%.1f/sl%.1f" % (coin, size_thresh, tp_m, sl_m), sim3h(d, entries, tp_m, sl_m, cn))

    # C. Tiny bar (<0.3x ATR) after big bar -> breakout direction
    for tp_m, sl_m in [(1.0,0.7), (1.5,1.0)]:
        entries = []
        for i in range(max(split,2), cn):
            if d["bar_size_ratio"][i] < 0.3 and d["bar_size_ratio"][i-1] > 1.5:
                # Breakout: direction of previous big bar
                side = "BUY" if d["c"][i-1] > d["o"][i-1] else "SELL"
                entries.append((i, side))
        rpt("%s squeeze_breakout tp%.1f/sl%.1f" % (coin, tp_m, sl_m), sim3h(d, entries, tp_m, sl_m, cn))

    print()

# ================================================================
print("=" * 80)
print("  [4] COMBINED: BTC spike + candle/volume anomaly")
print("=" * 80)
print()

br = btc_ret[-n_global:]

for coin in ["SOL","ADA"]:
    d = data[coin]; cn = n_global
    c=d["c"][-cn:]; h=d["h"][-cn:]; l=d["l"][-cn:]; o=d["o"][-cn:]
    atr=d["atr"][-cn:]; vr=d["vol_ratio"][-cn:]
    bsr=d["bar_size_ratio"][-cn:]; bdr=d["body_ratio"][-cn:]
    split = cn // 3
    print("--- %s ---" % coin)

    # A. BTC spike + volume spike on alt
    for btc_t in [0.008, 0.010, 0.012]:
        for tp_m, sl_m in [(1.0,0.7), (1.5,1.0)]:
            entries = []
            for i in range(max(split,25), cn):
                if np.isnan(br[i]) or abs(br[i]) < btc_t: continue
                if np.isnan(vr[i]) or vr[i] < 1.5: continue
                side = "BUY" if br[i] > 0 else "SELL"
                entries.append((i, side))
            d2 = {"c":c,"h":h,"l":l,"o":o,"atr":atr}
            rpt("%s btc%.1f+vol tp%.1f/sl%.1f" % (coin,btc_t*100,tp_m,sl_m), sim3h(d2,entries,tp_m,sl_m,cn))

    # B. BTC spike + big bar on alt
    for btc_t in [0.008, 0.012]:
        for tp_m, sl_m in [(1.0,0.7), (1.5,1.0)]:
            entries = []
            for i in range(max(split,2), cn):
                if np.isnan(br[i]) or abs(br[i]) < btc_t: continue
                if bsr[i] < 1.5: continue  # alt must also be moving
                side = "BUY" if br[i] > 0 else "SELL"
                entries.append((i, side))
            d2 = {"c":c,"h":h,"l":l,"o":o,"atr":atr}
            rpt("%s btc%.1f+bigbar tp%.1f/sl%.1f" % (coin,btc_t*100,tp_m,sl_m), sim3h(d2,entries,tp_m,sl_m,cn))

    # C. BTC spike + marubozu on alt (strong conviction bar)
    for btc_t in [0.008, 0.012]:
        for tp_m, sl_m in [(1.0,0.7), (1.5,1.0)]:
            entries = []
            for i in range(max(split,2), cn):
                if np.isnan(br[i]) or abs(br[i]) < btc_t: continue
                if bdr[i] < 0.7: continue  # strong body
                if bsr[i] < 1.0: continue  # decent size
                side = "BUY" if br[i] > 0 else "SELL"
                entries.append((i, side))
            d2 = {"c":c,"h":h,"l":l,"o":o,"atr":atr}
            rpt("%s btc%.1f+maru tp%.1f/sl%.1f" % (coin,btc_t*100,tp_m,sl_m), sim3h(d2,entries,tp_m,sl_m,cn))

    print()

# ================================================================
print("=" * 80)
print("  [5] GAP + ANOMALY SIGNALS")
print("=" * 80)
print()

for coin in ["SOL","ADA"]:
    d = data[coin]; cn = min(d["n"], n_global)
    split = cn // 3
    print("--- %s ---" % coin)

    # Gap > 0.3% -> continuation
    for gap_t in [0.002, 0.003, 0.005]:
        for tp_m, sl_m in [(0.7,0.5), (1.0,0.7)]:
            entries = [(i, "BUY" if d["gap"][i]>0 else "SELL") for i in range(max(split,2), cn) if abs(d["gap"][i]) > gap_t]
            rpt("%s gap>%.1f%%_cont tp%.1f/sl%.1f" % (coin, gap_t*100, tp_m, sl_m), sim3h(d, entries, tp_m, sl_m, cn))

    # Gap > 0.3% -> reversal (gap fill)
    for gap_t in [0.003, 0.005]:
        for tp_m, sl_m in [(0.5,0.5), (0.7,0.5)]:
            entries = [(i, "SELL" if d["gap"][i]>0 else "BUY") for i in range(max(split,2), cn) if abs(d["gap"][i]) > gap_t]
            rpt("%s gap>%.1f%%_fill tp%.1f/sl%.1f" % (coin, gap_t*100, tp_m, sl_m), sim3h(d, entries, tp_m, sl_m, cn))

    print()

print("=" * 80)
print("  ** or *** = positive strategies (cost-adjusted)")
print("=" * 80)
