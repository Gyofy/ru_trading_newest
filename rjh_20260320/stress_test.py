"""BTC Spike Strategy - Cold Hard Stress Test."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, '.')
import yfinance as yf

equity = 500.0
cost = 0.0018

btc = yf.Ticker("BTC-USD").history(period="180d", interval="1h")
btc.columns = [c.lower() for c in btc.columns]
btc_close = btc["close"].values
btc_ret = pd.Series(btc_close).pct_change().values
n_btc = len(btc_close)

sol = yf.Ticker("SOL-USD").history(period="180d", interval="1h")
sol.columns = [c.lower() for c in sol.columns]
sol_close = sol["close"].values
sol_ret = pd.Series(sol_close).pct_change().values

cn = min(n_btc, len(sol_close))
br = btc_ret[-cn:]
sr = sol_ret[-cn:]

print("=" * 80)
print("  [1] BTC SPIKE 후 알트가 '이미' 움직였는가?")
print("=" * 80)
print()
print("  BTC bar close 기준 spike -> SOL 같은 바 vs 다음 바")
print()

for thresh in [0.015, 0.02]:
    up_idx = [i for i in range(cn-1) if br[i] > thresh]
    dn_idx = [i for i in range(cn-1) if br[i] < -thresh]

    same_up = [sr[i] for i in up_idx if not np.isnan(sr[i])]
    next_up = [sr[i+1] for i in up_idx if not np.isnan(sr[i+1])]
    same_dn = [sr[i] for i in dn_idx if not np.isnan(sr[i])]
    next_dn = [sr[i+1] for i in dn_idx if not np.isnan(sr[i+1])]

    print("  BTC >+%.1f%% (%d events):" % (thresh*100, len(up_idx)))
    if same_up:
        print("    SOL same bar: %+.3f%% (WR %.0f%%)" % (np.mean(same_up)*100, np.mean(np.array(same_up)>0)*100))
    if next_up:
        print("    SOL next bar: %+.3f%% (WR %.0f%%) <- 실제 먹을 수 있는 양" % (np.mean(next_up)*100, np.mean(np.array(next_up)>0)*100))

    print("  BTC <-%.1f%% (%d events):" % (thresh*100, len(dn_idx)))
    if same_dn:
        print("    SOL same bar: %+.3f%% (WR %.0f%%)" % (np.mean(same_dn)*100, np.mean(np.array(same_dn)<0)*100))
    if next_dn:
        print("    SOL next bar: %+.3f%% (WR %.0f%%) <- 실제 먹을 수 있는 양" % (np.mean(next_dn)*100, np.mean(np.array(next_dn)>0)*100))
    print()

print("=" * 80)
print("  [2] 수정 시뮬: NEXT BAR OPEN 진입 (현실적)")
print("=" * 80)
print()

coins_yf = {"SOL":"SOL-USD", "ETH":"ETH-USD", "XRP":"XRP-USD", "ADA":"ADA-USD"}

for thresh in [0.015, 0.02]:
    for tp_m, sl_m in [(1.0, 0.7), (1.5, 1.0)]:
        all_r = []
        for coin, sym in coins_yf.items():
            try:
                df = yf.Ticker(sym).history(period="180d", interval="1h")
                df.columns = [c.lower() for c in df.columns]
            except: continue
            c = df["close"].values; h = df["high"].values
            l = df["low"].values; o = df["open"].values
            cn2 = min(len(c), n_btc)
            c=c[-cn2:]; h=h[-cn2:]; l=l[-cn2:]; o=o[-cn2:]
            br2 = btc_ret[-cn2:]
            tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)),np.abs(l-np.roll(c,1))))
            tr[0]=h[0]-l[0]
            atr = pd.Series(tr).rolling(14,min_periods=1).mean().values
            split = cn2//3; pe=0
            for i in range(max(split,2), cn2-3):
                if i<pe: continue
                if np.isnan(br2[i]) or abs(br2[i])<thresh: continue
                side = "BUY" if br2[i]>0 else "SELL"
                eb = i+1  # enter NEXT bar
                if eb >= cn2-2: continue
                entry = o[eb]
                ai = atr[eb] if not np.isnan(atr[eb]) else entry*0.01
                td=ai*tp_m; sd=ai*sl_m
                if side=="BUY": tp,sl = entry+td, entry-sd
                else: tp,sl = entry-td, entry+sd
                pnl=0
                for j in range(eb, min(eb+2, cn2)):
                    if side=="BUY":
                        if l[j]<=sl: pnl=(sl-entry)/entry; break
                        if h[j]>=tp: pnl=(tp-entry)/entry; break
                    else:
                        if h[j]>=sl: pnl=(entry-sl)/entry; break
                        if l[j]<=tp: pnl=(entry-tp)/entry; break
                else:
                    if eb+1<cn2:
                        ep=c[eb+1]
                        pnl=(ep-entry)/entry if side=="BUY" else (entry-ep)/entry
                all_r.append(pnl-cost); pe=eb+3

        if len(all_r)<5: continue
        arr = np.array(all_r)
        nn=len(arr); wr=np.mean(arr>0); avg=np.mean(arr)
        for lev in [3, 5]:
            la = avg*lev
            eq_curve = np.cumprod(1 + arr*lev)
            mdd = np.min(eq_curve / np.maximum.accumulate(eq_curve) - 1)
            mt = nn / (n_btc/24/30)
            dollar = equity * la * mt
            m = "***" if avg>0.001 else " **" if avg>0 else "   "
            print("  btc>%.1f%% tp%.1f/sl%.1f %dx: n=%3d WR=%.1f%% avg=%+.4f%% mo=$%+.0f MDD=%.1f%% %s" % (
                thresh*100, tp_m, sl_m, lev, nn, wr*100, la*100, dollar, mdd*100, m))
    print()

print("=" * 80)
print("  [3] TAIL RISK: 최악의 역방향 움직임")
print("=" * 80)
print()

for coin, sym in coins_yf.items():
    try:
        df = yf.Ticker(sym).history(period="180d", interval="1h")
        df.columns = [c.lower() for c in df.columns]
    except: continue
    c = df["close"].values
    cn2 = min(len(c), n_btc)
    c = c[-cn2:]
    br2 = btc_ret[-cn2:]

    adverses = []
    for i in range(cn2//3, cn2-2):
        if np.isnan(br2[i]) or abs(br2[i]) < 0.015: continue
        btc_dir = 1 if br2[i] > 0 else -1
        # Max adverse in next 2 bars
        for j in range(i+1, min(i+3, cn2)):
            move = (c[j] / c[i] - 1) * (-btc_dir)  # adverse = opposite direction
            adverses.append(move)

    if not adverses: continue
    adv = np.array(adverses)
    print("  %s (n=%d events):" % (coin, len(adv)//2))
    print("    mean adverse: %.2f%%" % (np.mean(adv)*100))
    print("    p95: %.2f%%, p99: %.2f%%, max: %.2f%%" % (
        np.percentile(adv, 95)*100, np.percentile(adv, 99)*100, np.max(adv)*100))

    for lev in [5, 10]:
        max_loss_lev = np.max(adv) * lev
        p99_loss_lev = np.percentile(adv, 99) * lev
        liq_dist = 1/lev - 0.004
        liq_events = np.sum(adv > liq_dist)
        print("    %2dx: p99 loss=%.1f%%, max loss=%.1f%%, liq events=%d/%d" % (
            lev, p99_loss_lev*100, max_loss_lev*100, liq_events, len(adv)))
    print()

print("=" * 80)
print("  [4] CONSECUTIVE LOSSES + EQUITY DRAWDOWN")
print("=" * 80)
print()

# Rebuild best config trades: btc>1.5%, tp1.5, sl1.0, next-bar entry
all_trades = []
for coin, sym in coins_yf.items():
    try:
        df = yf.Ticker(sym).history(period="180d", interval="1h")
        df.columns = [c.lower() for c in df.columns]
    except: continue
    c=df["close"].values; h=df["high"].values; l=df["low"].values; o=df["open"].values
    cn2=min(len(c), n_btc)
    c=c[-cn2:]; h=h[-cn2:]; l=l[-cn2:]; o=o[-cn2:]
    br2=btc_ret[-cn2:]
    tr=np.maximum(h-l,np.maximum(np.abs(h-np.roll(c,1)),np.abs(l-np.roll(c,1))))
    tr[0]=h[0]-l[0]
    atr=pd.Series(tr).rolling(14,min_periods=1).mean().values
    split=cn2//3; pe=0
    for i in range(max(split,2), cn2-3):
        if i<pe: continue
        if np.isnan(br2[i]) or abs(br2[i])<0.015: continue
        eb=i+1
        if eb>=cn2-2: continue
        entry=o[eb]; ai=atr[eb] if not np.isnan(atr[eb]) else entry*0.01
        side="BUY" if br2[i]>0 else "SELL"
        td=ai*1.5; sd=ai*1.0
        if side=="BUY": tp,sl=entry+td,entry-sd
        else: tp,sl=entry-td,entry+sd
        pnl=0
        for j in range(eb,min(eb+2,cn2)):
            if side=="BUY":
                if l[j]<=sl: pnl=(sl-entry)/entry; break
                if h[j]>=tp: pnl=(tp-entry)/entry; break
            else:
                if h[j]>=sl: pnl=(entry-sl)/entry; break
                if l[j]<=tp: pnl=(entry-tp)/entry; break
        else:
            if eb+1<cn2:
                ep=c[eb+1]
                pnl=(ep-entry)/entry if side=="BUY" else (entry-ep)/entry
        all_trades.append(pnl-cost); pe=eb+3

arr = np.array(all_trades)
print("  Total trades: %d, Losses: %d (%.1f%%)" % (len(arr), np.sum(arr<0), np.mean(arr<0)*100))
print("  avg: %+.4f%%, WR: %.1f%%" % (np.mean(arr)*100, np.mean(arr>0)*100))
print()

# Max consecutive losses
losses = arr < 0
max_streak = 0; cur = 0; streak_sum = 0; max_streak_sum = 0
for i in range(len(arr)):
    if losses[i]:
        cur += 1
        streak_sum += arr[i]
        if cur > max_streak:
            max_streak = cur
            max_streak_sum = streak_sum
    else:
        cur = 0; streak_sum = 0

print("  Max consecutive losses: %d (total %.2f%%)" % (max_streak, max_streak_sum*100))
print()

for lev in [3, 5, 10]:
    eq = equity
    min_eq = equity
    for pnl in arr:
        eq *= (1 + pnl * lev)
        min_eq = min(min_eq, eq)
    dd = (min_eq / equity - 1) * 100
    print("  %2dx: $%.0f start -> $%.0f end, min $%.0f (DD %.1f%%)" % (lev, equity, eq, min_eq, dd))

print()
print("=" * 80)
print("  [5] FINAL VERDICT")
print("=" * 80)
