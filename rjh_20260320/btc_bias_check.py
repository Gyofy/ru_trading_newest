"""Is the BTC spike strategy just 'buy BTC in a bull market'?"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, '.')
import yfinance as yf

cost = 0.0018

print("=" * 80)
print("  BIAS CHECK: BTC spike 전략 = 그냥 비트코인 롱?")
print("=" * 80)
print()

# Get data
btc = yf.Ticker("BTC-USD").history(period="180d", interval="1h")
btc.columns = [c.lower() for c in btc.columns]
btc_close = btc["close"].values
btc_ret = pd.Series(btc_close).pct_change().values
n = len(btc_close)

# Get daily for context
btc_d = yf.Ticker("BTC-USD").history(period="1y", interval="1d")
btc_d.columns = [c.lower() for c in btc_d.columns]

print("[1] 180일간 BTC 방향성")
print()
total_ret = (btc_close[-1] / btc_close[0] - 1) * 100
print("  BTC 180d return: %+.1f%%" % total_ret)
print("  BTC start: $%.0f, end: $%.0f" % (btc_close[0], btc_close[-1]))
print()

# Monthly breakdown
monthly_bars = n // 6  # ~30 days each
for m in range(6):
    s = m * monthly_bars
    e = min((m+1) * monthly_bars, n)
    mret = (btc_close[e-1] / btc_close[s] - 1) * 100
    print("  Month %d: %+.1f%%" % (m+1, mret))
print()

# [2] UP spikes vs DOWN spikes count
print("[2] BTC 1h spike 방향 분포")
print()
for thresh in [0.012, 0.015]:
    up = np.sum(btc_ret > thresh)
    dn = np.sum(btc_ret < -thresh)
    print("  |ret|>%.1f%%: UP=%d DOWN=%d ratio=%.1f:1" % (thresh*100, up, dn, up/max(dn,1)))
print()
print("  -> DOWN이 UP보다 많으면 bearish bias 아님")
print()

# [3] BTC itself after spike (not alts)
print("[3] BTC spike 후 BTC 자신의 다음 바 수익률")
print()
for thresh in [0.012, 0.015]:
    for direction in ["UP", "DOWN", "BOTH"]:
        nexts = []
        split = n // 3
        for i in range(split, n-1):
            if np.isnan(btc_ret[i]): continue
            if direction == "UP" and btc_ret[i] > thresh:
                nexts.append(btc_ret[i+1] if not np.isnan(btc_ret[i+1]) else 0)
            elif direction == "DOWN" and btc_ret[i] < -thresh:
                nexts.append(btc_ret[i+1] if not np.isnan(btc_ret[i+1]) else 0)
            elif direction == "BOTH" and abs(btc_ret[i]) > thresh:
                side_mult = 1 if btc_ret[i] > 0 else -1
                nexts.append(btc_ret[i+1] * side_mult if not np.isnan(btc_ret[i+1]) else 0)
        if nexts:
            arr = np.array(nexts)
            print("  btc>%.1f%% %s: n=%d next_bar=%+.3f%% WR=%.0f%%" % (
                thresh*100, direction, len(arr), np.mean(arr)*100,
                np.mean(arr>0)*100 if direction!="BOTH" else np.mean(arr>0)*100))
print()

# [4] Simple BTC buy-and-hold vs strategy
print("[4] 비교: BTC Buy & Hold vs BTC Spike Strategy")
print()

# Buy and hold (same period)
split = n // 3
bnh_ret = (btc_close[-1] / btc_close[split] - 1) * 100
bnh_days = (n - split) / 24
bnh_monthly = bnh_ret / (bnh_days / 30)
print("  BTC Buy & Hold (%.0f days): %+.1f%% total, %+.1f%%/month" % (bnh_days, bnh_ret, bnh_monthly))
print()

# Strategy: enter BTC after BTC spike, hold 6h
print("  BTC Spike -> BTC itself (not alts):")
tr = np.maximum(btc["high"].values-btc["low"].values,
    np.maximum(np.abs(btc["high"].values-np.roll(btc_close,1)),
               np.abs(btc["low"].values-np.roll(btc_close,1))))
tr[0] = btc["high"].values[0] - btc["low"].values[0]
atr = pd.Series(tr).rolling(14,min_periods=1).mean().values

for thresh in [0.012, 0.015]:
    results = []; pe = 0
    for i in range(split, n-7):
        if i < pe: continue
        if np.isnan(btc_ret[i]) or abs(btc_ret[i]) < thresh: continue
        side = "BUY" if btc_ret[i] > 0 else "SELL"
        eb = i + 1
        entry = btc["open"].values[eb]
        ai = atr[i] if not np.isnan(atr[i]) else entry*0.01
        tp = entry + ai*1.5 if side=="BUY" else entry - ai*1.5
        sl = entry - ai*1.0 if side=="BUY" else entry + ai*1.0
        pnl = 0
        for j in range(eb, min(eb+6, n)):
            if side=="BUY":
                if btc["low"].values[j] <= sl: pnl=(sl-entry)/entry; break
                if btc["high"].values[j] >= tp: pnl=(tp-entry)/entry; break
            else:
                if btc["high"].values[j] >= sl: pnl=(entry-sl)/entry; break
                if btc["low"].values[j] <= tp: pnl=(entry-tp)/entry; break
        else:
            if eb+5 < n:
                ep = btc_close[eb+5]
                pnl = (ep-entry)/entry if side=="BUY" else (entry-ep)/entry
        results.append(pnl-cost); pe = eb+7

    arr = np.array(results)
    if len(arr) > 5:
        wr = np.mean(arr>0); avg = np.mean(arr)
        print("    btc>%.1f%% -> BTC: n=%d WR=%.1f%% avg=%+.4f%%" % (thresh*100, len(arr), wr*100, avg*100))
print()

# [5] Critical: what if we ONLY go LONG (ignore shorts)?
print("[5] LONG ONLY vs SHORT ONLY vs BOTH")
print()

sol = yf.Ticker("SOL-USD").history(period="180d", interval="1h")
sol.columns = [c.lower() for c in sol.columns]
sol_c = sol["close"].values; sol_h = sol["high"].values
sol_l = sol["low"].values; sol_o = sol["open"].values
cn = min(len(sol_c), n)
sol_c=sol_c[-cn:]; sol_h=sol_h[-cn:]; sol_l=sol_l[-cn:]; sol_o=sol_o[-cn:]
br = btc_ret[-cn:]
tr_sol = np.maximum(sol_h-sol_l, np.maximum(np.abs(sol_h-np.roll(sol_c,1)),np.abs(sol_l-np.roll(sol_c,1))))
tr_sol[0]=sol_h[0]-sol_l[0]
atr_sol = pd.Series(tr_sol).rolling(14,min_periods=1).mean().values
split2 = cn // 3

for label, condition in [
    ("LONG only (BTC up)", lambda br_i: br_i > 0.012),
    ("SHORT only (BTC down)", lambda br_i: br_i < -0.012),
    ("ALWAYS LONG (ignore BTC dir)", lambda br_i: abs(br_i) > 0.012),
    ("ALWAYS SHORT (ignore BTC dir)", lambda br_i: abs(br_i) > 0.012),
    ("FOLLOW BTC (current)", lambda br_i: abs(br_i) > 0.012),
    ("COUNTER BTC (contrarian)", lambda br_i: abs(br_i) > 0.012),
]:
    results = []; pe = 0
    for i in range(max(split2,2), cn-7):
        if i < pe: continue
        if np.isnan(br[i]) or not condition(br[i]): continue

        if "LONG only" in label:
            side = "BUY"
        elif "SHORT only" in label:
            side = "SELL"
        elif "ALWAYS LONG" in label:
            side = "BUY"
        elif "ALWAYS SHORT" in label:
            side = "SELL"
        elif "FOLLOW" in label:
            side = "BUY" if br[i] > 0 else "SELL"
        elif "COUNTER" in label:
            side = "SELL" if br[i] > 0 else "BUY"

        eb = i+1
        if eb >= cn-6: continue
        entry = sol_o[eb]
        ai = atr_sol[i] if not np.isnan(atr_sol[i]) else entry*0.01
        tp = entry+ai*1.5 if side=="BUY" else entry-ai*1.5
        sl = entry-ai*1.0 if side=="BUY" else entry+ai*1.0
        pnl = 0
        for j in range(eb, min(eb+6, cn)):
            if side=="BUY":
                if sol_l[j]<=sl: pnl=(sl-entry)/entry; break
                if sol_h[j]>=tp: pnl=(tp-entry)/entry; break
            else:
                if sol_h[j]>=sl: pnl=(entry-sl)/entry; break
                if sol_l[j]<=tp: pnl=(entry-tp)/entry; break
        else:
            if eb+5<cn:
                ep=sol_c[eb+5]
                pnl=(ep-entry)/entry if side=="BUY" else (entry-ep)/entry
        results.append(pnl-cost); pe=eb+7

    if len(results) < 5: continue
    arr = np.array(results)
    wr = np.mean(arr>0); avg = np.mean(arr)
    m = "***" if avg>0.002 else " **" if avg>0.0005 else "  *" if avg>0 else "   "
    print("  %-35s n=%3d WR=%.1f%% avg=%+.4f%% %s" % (label, len(arr), wr*100, avg*100, m))

print()

# [6] Period split: first half vs second half
print("[6] 기간 분할: 전반 90일 vs 후반 90일")
print()

half = cn // 2
for period_label, start, end in [("First 90d", cn//3, half), ("Last 90d", half, cn)]:
    results = []; pe = start
    for i in range(max(start,2), min(end, cn-7)):
        if i < pe: continue
        if np.isnan(br[i]) or abs(br[i]) < 0.012: continue
        side = "BUY" if br[i] > 0 else "SELL"
        eb = i+1
        if eb >= cn-6: continue
        entry = sol_o[eb]
        ai = atr_sol[i] if not np.isnan(atr_sol[i]) else entry*0.01
        tp = entry+ai*1.5 if side=="BUY" else entry-ai*1.5
        sl = entry-ai*1.0 if side=="BUY" else entry+ai*1.0
        pnl = 0
        for j in range(eb, min(eb+6, cn)):
            if side=="BUY":
                if sol_l[j]<=sl: pnl=(sl-entry)/entry; break
                if sol_h[j]>=tp: pnl=(tp-entry)/entry; break
            else:
                if sol_h[j]>=sl: pnl=(entry-sl)/entry; break
                if sol_l[j]<=tp: pnl=(entry-tp)/entry; break
        else:
            if eb+5<cn:
                ep=sol_c[eb+5]
                pnl=(ep-entry)/entry if side=="BUY" else (entry-ep)/entry
        results.append(pnl-cost); pe=eb+7

    if len(results) < 5: continue
    arr = np.array(results)
    sol_period_ret = (sol_c[min(end,cn-1)] / sol_c[start] - 1) * 100
    print("  %s: n=%d WR=%.1f%% avg=%+.4f%% | SOL period return: %+.1f%%" % (
        period_label, len(arr), np.mean(arr>0)*100, np.mean(arr)*100, sol_period_ret))

print()
print("=" * 80)
print("  VERDICT")
print("=" * 80)
