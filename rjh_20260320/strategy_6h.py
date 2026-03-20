"""6h hold strategy - comprehensive search with bias corrections."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, '.')
import yfinance as yf

equity = 500.0
cost = 0.0018

print("=" * 80)
print("  6H HOLD STRATEGY: COMPREHENSIVE SEARCH")
print("  Corrections applied: next-bar entry, proper cost, no cherry-pick")
print("=" * 80)
print()

# Fetch 180d 1h data
coins_yf = {"SOL":"SOL-USD","BTC":"BTC-USD","ETH":"ETH-USD",
            "XRP":"XRP-USD","ADA":"ADA-USD","LINK":"LINK-USD"}
data = {}
for coin, sym in coins_yf.items():
    try:
        df = yf.Ticker(sym).history(period="180d", interval="1h")
        df.columns = [c.lower() for c in df.columns]
        c = df["close"].values; h = df["high"].values
        l = df["low"].values; o = df["open"].values
        v = df["volume"].values.astype(float)
        n = len(c)
        tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)),np.abs(l-np.roll(c,1))))
        tr[0] = h[0]-l[0]
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
        vol_sma = pd.Series(v).rolling(20, min_periods=5).mean().values
        # EMA
        ema8 = pd.Series(c).ewm(span=8, adjust=False).mean().values
        ema21 = pd.Series(c).ewm(span=21, adjust=False).mean().values
        # RSI
        delta = pd.Series(c).diff()
        gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean().values
        loss_v = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean().values
        rsi = 100 - 100 / (1 + gain / (loss_v + 1e-10))
        # Daily
        df_d = yf.Ticker(sym).history(period="1y", interval="1d")
        df_d.columns = [c2.lower() for c2 in df_d.columns]
        d_close = df_d["close"].values
        d_sma10 = pd.Series(d_close).rolling(10).mean().values
        d_sma20 = pd.Series(d_close).rolling(20).mean().values

        data[coin] = {"df":df, "df_d":df_d, "c":c, "h":h, "l":l, "o":o,
                      "v":v, "atr":atr, "vol_sma":vol_sma, "n":n,
                      "ema8":ema8, "ema21":ema21, "rsi":rsi,
                      "d_close":d_close, "d_sma10":d_sma10, "d_sma20":d_sma20}
        print("  %s: %d bars (%.0f days)" % (coin, n, n/24))
    except Exception as e:
        print("  %s: FAIL (%s)" % (coin, str(e)[:40]))
print()

btc = data.get("BTC")
if not btc: sys.exit(1)
btc_ret = pd.Series(btc["c"]).pct_change().values
btc_n = btc["n"]

def sim_6h(close, high, low, opn, atr, entries, n, tp_m, sl_m, hold_bars=6):
    """Simulate trades with NEXT BAR OPEN entry, hold_bars hold."""
    results = []
    pe = 0
    for i, side in entries:
        if i < pe or i >= n - hold_bars - 1: continue
        eb = i + 1  # enter NEXT bar open
        entry = opn[eb]
        # Use ATR from SIGNAL bar (not entry bar) to avoid lookahead
        ai = atr[i] if not np.isnan(atr[i]) else entry * 0.01
        td = ai * tp_m; sd = ai * sl_m
        if side == "BUY": tp, sl = entry + td, entry - sd
        else: tp, sl = entry - td, entry + sd
        pnl = 0
        for j in range(eb, min(eb + hold_bars, n)):
            if side == "BUY":
                if low[j] <= sl: pnl = (sl - entry) / entry; break
                if high[j] >= tp: pnl = (tp - entry) / entry; break
            else:
                if high[j] >= sl: pnl = (entry - sl) / entry; break
                if low[j] <= tp: pnl = (entry - tp) / entry; break
        else:
            if eb + hold_bars - 1 < n:
                ep = close[eb + hold_bars - 1]
                pnl = (ep - entry) / entry if side == "BUY" else (entry - ep) / entry
        results.append(pnl - cost)
        pe = eb + hold_bars + 1  # non-overlapping
    return np.array(results) if results else np.array([])

def report(name, arr, lev=1):
    if len(arr) < 10: return False
    nn = len(arr); wr = np.mean(arr > 0); avg = np.mean(arr)
    la = avg * lev
    eq = np.cumprod(1 + arr * lev)
    mdd = np.min(eq / np.maximum.accumulate(eq) - 1) if len(eq) > 0 else 0
    mt = nn / (btc_n / 24 / 30)
    dollar = equity * la * mt
    m = "***" if avg > 0.002 else " **" if avg > 0.0005 else "  *" if avg > 0 else "   "
    print("  %-35s n=%3d WR=%.1f%% avg=%+.4f%% %dx:$%+.0f/mo MDD=%.0f%% %s" % (
        name, nn, wr*100, avg*100, lev, dollar, mdd*100, m))
    return avg > 0

# ============================================================
# STRATEGY 1: Momentum (various lookbacks)
# ============================================================
print("=" * 80)
print("  [1] MOMENTUM STRATEGIES (6h hold)")
print("=" * 80)
print()

for coin in ["SOL", "ETH", "XRP", "BTC"]:
    if coin not in data: continue
    d = data[coin]
    split = d["n"] // 3
    for lb in [3, 6, 12, 24]:
        for tp_m, sl_m in [(1.0, 0.7), (1.5, 1.0), (2.0, 1.0), (2.0, 1.5)]:
            entries = []
            for i in range(max(split, lb+1), d["n"]):
                mom = d["c"][i] / d["c"][i-lb] - 1
                if abs(mom) < 0.003: continue
                entries.append((i, "BUY" if mom > 0 else "SELL"))
            arr = sim_6h(d["c"], d["h"], d["l"], d["o"], d["atr"], entries, d["n"], tp_m, sl_m)
            report("%s mom%d tp%.1f/sl%.1f" % (coin, lb, tp_m, sl_m), arr, 3)
    print()

# ============================================================
# STRATEGY 2: Daily Trend + 1h Momentum
# ============================================================
print("=" * 80)
print("  [2] DAILY TREND + 1H MOMENTUM (6h hold)")
print("=" * 80)
print()

for coin in ["SOL", "ETH", "XRP", "ADA"]:
    if coin not in data: continue
    d = data[coin]
    split = d["n"] // 3
    for mom_lb in [3, 6, 12]:
        for tp_m, sl_m in [(1.5, 1.0), (2.0, 1.0), (2.0, 1.5)]:
            entries = []
            for i in range(max(split, 24), d["n"]):
                # Daily trend
                ts = d["df"].index[i]
                dm = d["df_d"].index <= ts
                if dm.sum() < 21: continue
                di = dm.sum() - 1
                if np.isnan(d["d_sma10"][di]) or np.isnan(d["d_sma20"][di]): continue
                daily_up = d["d_sma10"][di] > d["d_sma20"][di]
                # 1h momentum
                mom = d["c"][i] / d["c"][i-mom_lb] - 1
                if abs(mom) < 0.003: continue
                if daily_up and mom > 0: entries.append((i, "BUY"))
                elif not daily_up and mom < 0: entries.append((i, "SELL"))
            arr = sim_6h(d["c"], d["h"], d["l"], d["o"], d["atr"], entries, d["n"], tp_m, sl_m)
            report("%s daily+mom%d tp%.1f/sl%.1f" % (coin, mom_lb, tp_m, sl_m), arr, 3)
    print()

# ============================================================
# STRATEGY 3: BTC Spike -> Alt (UP only vs DOWN only)
# ============================================================
print("=" * 80)
print("  [3] BTC SPIKE -> ALT (UP/DOWN separated, 6h hold)")
print("=" * 80)
print()

for thresh in [0.01, 0.012, 0.015]:
    for direction in ["UP_ONLY", "DOWN_ONLY", "BOTH"]:
        all_r = []
        for coin in ["SOL", "ETH", "XRP", "ADA"]:
            if coin not in data: continue
            d = data[coin]
            cn = min(d["n"], btc_n)
            br = btc_ret[-cn:]
            split = cn // 3
            entries = []
            for i in range(max(split, 2), cn):
                if np.isnan(br[i]): continue
                if direction == "UP_ONLY" and br[i] > thresh:
                    entries.append((i, "BUY"))
                elif direction == "DOWN_ONLY" and br[i] < -thresh:
                    entries.append((i, "SELL"))
                elif direction == "BOTH" and abs(br[i]) > thresh:
                    entries.append((i, "BUY" if br[i] > 0 else "SELL"))
            c2 = d["c"][-cn:]; h2 = d["h"][-cn:]; l2 = d["l"][-cn:]
            o2 = d["o"][-cn:]; a2 = d["atr"][-cn:]
            arr = sim_6h(c2, h2, l2, o2, a2, entries, cn, 1.5, 1.0)
            if len(arr) > 0: all_r.extend(arr.tolist())

        if len(all_r) > 5:
            report("btc>%.1f%% %s 4coins" % (thresh*100, direction), np.array(all_r), 3)
    print()

# ============================================================
# STRATEGY 4: Volume Spike + Momentum
# ============================================================
print("=" * 80)
print("  [4] VOLUME SPIKE + MOMENTUM (6h hold)")
print("=" * 80)
print()

for coin in ["SOL", "ETH", "XRP", "BTC"]:
    if coin not in data: continue
    d = data[coin]
    split = d["n"] // 3
    for vol_mult in [1.5, 2.0, 3.0]:
        for tp_m, sl_m in [(1.5, 1.0), (2.0, 1.0)]:
            entries = []
            for i in range(max(split, 21), d["n"]):
                if np.isnan(d["vol_sma"][i]) or d["vol_sma"][i] < 1: continue
                if d["v"][i] < d["vol_sma"][i] * vol_mult: continue
                bar_ret = (d["c"][i] - d["o"][i]) / d["o"][i]
                if abs(bar_ret) < 0.002: continue
                entries.append((i, "BUY" if bar_ret > 0 else "SELL"))
            arr = sim_6h(d["c"], d["h"], d["l"], d["o"], d["atr"], entries, d["n"], tp_m, sl_m)
            report("%s vol>%.1fx tp%.1f/sl%.1f" % (coin, vol_mult, tp_m, sl_m), arr, 3)
    print()

# ============================================================
# STRATEGY 5: EMA Cross + ADX Filter
# ============================================================
print("=" * 80)
print("  [5] EMA CROSS + TREND FILTER (6h hold)")
print("=" * 80)
print()

for coin in ["SOL", "ETH", "XRP", "BTC"]:
    if coin not in data: continue
    d = data[coin]
    split = d["n"] // 3
    for tp_m, sl_m in [(1.5, 1.0), (2.0, 1.0), (2.0, 1.5)]:
        # Plain EMA cross
        entries = []
        for i in range(max(split, 26), d["n"]):
            entries.append((i, "BUY" if d["ema8"][i] > d["ema21"][i] else "SELL"))
        arr = sim_6h(d["c"], d["h"], d["l"], d["o"], d["atr"], entries, d["n"], tp_m, sl_m)
        report("%s ema tp%.1f/sl%.1f" % (coin, tp_m, sl_m), arr, 3)
    print()

# ============================================================
# STRATEGY 6: Mean Reversion (RSI)
# ============================================================
print("=" * 80)
print("  [6] RSI MEAN REVERSION (6h hold)")
print("=" * 80)
print()

for coin in ["SOL", "ETH", "XRP", "BTC"]:
    if coin not in data: continue
    d = data[coin]
    split = d["n"] // 3
    for rlo, rhi in [(20, 80), (25, 75), (30, 70)]:
        for tp_m, sl_m in [(0.7, 1.0), (1.0, 1.0), (1.0, 1.5)]:
            entries = []
            for i in range(max(split, 15), d["n"]):
                if d["rsi"][i] < rlo: entries.append((i, "BUY"))
                elif d["rsi"][i] > rhi: entries.append((i, "SELL"))
            arr = sim_6h(d["c"], d["h"], d["l"], d["o"], d["atr"], entries, d["n"], tp_m, sl_m)
            report("%s rsi<%d/>%d tp%.1f/sl%.1f" % (coin, rlo, rhi, tp_m, sl_m), arr, 3)
    print()

# ============================================================
# SUMMARY
# ============================================================
print("=" * 80)
print("  ** or *** = positive avg PnL strategies")
print("  All use NEXT BAR OPEN entry (no lookahead)")
print("  All use SIGNAL BAR ATR (no future ATR)")
print("  6h hold = 6 bars on 1h data")
print("=" * 80)
