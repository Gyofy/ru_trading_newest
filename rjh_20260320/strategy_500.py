"""$500 equity strategy analysis - extended data validation."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, '.')
import yfinance as yf

equity = 500.0
cost = 0.0018

print("=" * 80)
print("  $500 EQUITY STRATEGY ANALYSIS")
print("=" * 80)
print()

# Fetch data
coins_data = {}
for coin, sym in [("SOL","SOL-USD"),("BTC","BTC-USD"),("ETH","ETH-USD"),
                   ("XRP","XRP-USD"),("ADA","ADA-USD")]:
    try:
        df = yf.Ticker(sym).history(period="180d", interval="1h")
        df.columns = [c.lower() for c in df.columns]
        close = df["close"].values; high = df["high"].values; low = df["low"].values
        n = len(close)
        tr = np.maximum(high-low, np.maximum(np.abs(high-np.roll(close,1)),np.abs(low-np.roll(close,1))))
        tr[0] = high[0]-low[0]
        atr = pd.Series(tr).rolling(14,min_periods=1).mean().values
        coins_data[coin] = {"df":df,"close":close,"high":high,"low":low,"atr":atr,"n":n}
        print("  %s: %d bars (%.0f days)" % (coin, n, n/24))
    except Exception as e:
        print("  %s: FAIL" % coin)
print()

btc = coins_data.get("BTC")
if not btc: sys.exit(1)
btc_ret = pd.Series(btc["close"]).pct_change().values
btc_n = btc["n"]
total_days = btc_n / 24

# ========== STRATEGY A: BTC SPIKE ==========
print("=" * 80)
print("  [A] BTC SPIKE -> ALT FOLLOW")
print("=" * 80)
print()

for btc_thresh in [0.01, 0.012, 0.015, 0.02]:
    for tp_m, sl_m in [(0.8, 0.6), (1.0, 0.7), (1.5, 1.0)]:
        all_r = []
        for coin in ["SOL","ETH","XRP","ADA"]:
            if coin not in coins_data: continue
            d = coins_data[coin]
            cn = min(d["n"], btc_n)
            c=d["close"][-cn:]; h=d["high"][-cn:]; l=d["low"][-cn:]; a=d["atr"][-cn:]
            br = btc_ret[-cn:]
            split = cn // 3
            pe = 0
            for i in range(max(split,2), cn-2):
                if i<pe: continue
                if np.isnan(br[i]) or abs(br[i])<btc_thresh: continue
                side = "BUY" if br[i]>0 else "SELL"
                entry=c[i]; ai=a[i] if not np.isnan(a[i]) else entry*0.01
                td=ai*tp_m; sd=ai*sl_m
                if side=="BUY": tp,sl=entry+td,entry-sd
                else: tp,sl=entry-td,entry+sd
                pnl=0
                for j in range(i+1,min(i+3,cn)):
                    if side=="BUY":
                        if l[j]<=sl: pnl=(sl-entry)/entry; break
                        if h[j]>=tp: pnl=(tp-entry)/entry; break
                    else:
                        if h[j]>=sl: pnl=(entry-sl)/entry; break
                        if l[j]<=tp: pnl=(entry-tp)/entry; break
                else:
                    if i+2<cn:
                        ep=c[min(i+2,cn-1)]
                        pnl=(ep-entry)/entry if side=="BUY" else (entry-ep)/entry
                all_r.append(pnl-cost); pe=i+3

        if len(all_r)<5: continue
        arr=np.array(all_r); nn=len(arr); wr=np.mean(arr>0); avg=np.mean(arr)
        mt = nn / (total_days/30)  # trades per month
        for lev in [5]:
            la = avg*lev; mp = la*mt; dollar = equity*mp
            m = "***" if avg>0.001 else " **" if avg>0 else "   "
            print("  btc>%.1f%% tp%.1f/sl%.1f %dx: n=%3d WR=%.1f%% avg=%+.4f%% mo=$%+.0f %s" % (
                btc_thresh*100, tp_m, sl_m, lev, nn, wr*100, la*100, dollar, m))
    print()

# ========== STRATEGY B: DAILY TREND + HOLD ==========
print("=" * 80)
print("  [B] DAILY TREND + 12-24h HOLD")
print("=" * 80)
print()

for coin in ["SOL","ETH","XRP"]:
    if coin not in coins_data: continue
    d = coins_data[coin]
    sym = coin + "-USD"
    try:
        df_d = yf.Ticker(sym).history(period="1y", interval="1d")
        df_d.columns = [c.lower() for c in df_d.columns]
    except: continue

    d_close = df_d["close"].values
    d_sma10 = pd.Series(d_close).rolling(10).mean().values
    d_sma20 = pd.Series(d_close).rolling(20).mean().values

    close=d["close"]; high=d["high"]; low=d["low"]; atr=d["atr"]; n=d["n"]
    df = d["df"]
    split = n // 3

    for hold in [12, 24]:
        for tp_m, sl_m in [(1.5, 1.0), (2.0, 1.0), (2.5, 1.0)]:
            results=[]; pe=0
            for i in range(max(split,24), n-hold):
                if i<pe: continue
                hour_ts = df.index[i]
                dm = df_d.index <= hour_ts
                if dm.sum()<21: continue
                di = dm.sum()-1
                if np.isnan(d_sma10[di]) or np.isnan(d_sma20[di]): continue
                daily_up = d_sma10[di] > d_sma20[di]
                mom = close[i]/close[i-6]-1
                if abs(mom)<0.003: continue
                if daily_up and mom>0: side="BUY"
                elif not daily_up and mom<0: side="SELL"
                else: continue
                entry=close[i]; a=atr[i] if not np.isnan(atr[i]) else entry*0.01
                td=a*tp_m; sd=a*sl_m
                if side=="BUY": tp,sl=entry+td,entry-sd
                else: tp,sl=entry-td,entry+sd
                pnl=0
                for j in range(i+1,min(i+hold+1,n)):
                    if side=="BUY":
                        if low[j]<=sl: pnl=(sl-entry)/entry; break
                        if high[j]>=tp: pnl=(tp-entry)/entry; break
                    else:
                        if high[j]>=sl: pnl=(entry-sl)/entry; break
                        if low[j]<=tp: pnl=(entry-tp)/entry; break
                else:
                    if i+hold<n:
                        ep=close[i+hold]
                        pnl=(ep-entry)/entry if side=="BUY" else (entry-ep)/entry
                results.append(pnl-cost); pe=i+hold+1

            if len(results)<10: continue
            arr=np.array(results); nn=len(arr); wr=np.mean(arr>0); avg=np.mean(arr)
            mt = nn/(n/24/30)
            for lev in [3]:
                la=avg*lev; mp=la*mt; dollar=equity*mp
                m = "***" if avg>0.001 else " **" if avg>0 else "   "
                print("  %s %dh tp%.1f/sl%.1f %dx: n=%3d WR=%.1f%% avg=%+.4f%% mo=$%+.0f %s" % (
                    coin, hold, tp_m, sl_m, lev, nn, wr*100, la*100, dollar, m))
    print()

# ========== STRATEGY C: PURE MOMENTUM MULTI-TIMEFRAME ==========
print("=" * 80)
print("  [C] MULTI-SIGNAL CONVICTION (BTC+MOM+VOL)")
print("=" * 80)
print()

for coin in ["SOL","ETH","XRP","ADA"]:
    if coin not in coins_data: continue
    d = coins_data[coin]
    close=d["close"]; high=d["high"]; low=d["low"]; atr=d["atr"]; n=d["n"]
    vol = d["df"]["volume"].values.astype(float)
    vol_sma = pd.Series(vol).rolling(20,min_periods=5).mean().values

    cn = min(n, btc_n)
    br = btc_ret[-cn:]
    c=close[-cn:]; h=high[-cn:]; l=low[-cn:]; a=atr[-cn:]
    v=vol[-cn:]; vs=vol_sma[-cn:]
    split = cn // 3

    for min_signals in [2, 3]:
        for hold in [4, 8, 12]:
            results=[]; pe=0
            for i in range(max(split,24), cn-hold):
                if i<pe: continue

                # Signal 1: BTC momentum (6h)
                btc_mom = btc["close"][-cn:][i]/btc["close"][-cn:][max(0,i-6)]-1 if i>=6 else 0
                # Signal 2: Coin momentum (6h)
                coin_mom = c[i]/c[max(0,i-6)]-1 if i>=6 else 0
                # Signal 3: Volume spike
                vol_spike = v[i] > vs[i]*1.5 if not np.isnan(vs[i]) and vs[i]>0 else False

                signals_long = 0; signals_short = 0
                if btc_mom > 0.003: signals_long += 1
                elif btc_mom < -0.003: signals_short += 1
                if coin_mom > 0.003: signals_long += 1
                elif coin_mom < -0.003: signals_short += 1
                if vol_spike:
                    bar_ret = (c[i]-c[i-1])/c[i-1] if i>0 else 0
                    if bar_ret > 0: signals_long += 1
                    else: signals_short += 1

                if signals_long >= min_signals: side = "BUY"
                elif signals_short >= min_signals: side = "SELL"
                else: continue

                entry=c[i]; ai=a[i] if not np.isnan(a[i]) else entry*0.01
                td=ai*1.5; sd=ai*1.0
                if side=="BUY": tp,sl=entry+td,entry-sd
                else: tp,sl=entry-td,entry+sd
                pnl=0
                for j in range(i+1,min(i+hold+1,cn)):
                    if side=="BUY":
                        if l[j]<=sl: pnl=(sl-entry)/entry; break
                        if h[j]>=tp: pnl=(tp-entry)/entry; break
                    else:
                        if h[j]>=sl: pnl=(entry-sl)/entry; break
                        if l[j]<=tp: pnl=(entry-tp)/entry; break
                else:
                    if i+hold<cn:
                        ep=c[i+hold]
                        pnl=(ep-entry)/entry if side=="BUY" else (entry-ep)/entry
                results.append(pnl-cost); pe=i+hold+1

            if len(results)<5: continue
            arr=np.array(results); nn=len(arr); wr=np.mean(arr>0); avg=np.mean(arr)
            mt=nn/(cn/24/30)
            for lev in [3, 5]:
                la=avg*lev; mp=la*mt; dollar=equity*mp
                m = "***" if avg>0.002 else " **" if avg>0.0005 else "  *" if avg>0 else "   "
                print("  %s sig>=%d %dh %dx: n=%3d WR=%.1f%% avg=%+.4f%% mo=$%+.0f %s" % (
                    coin, min_signals, hold, lev, nn, wr*100, la*100, dollar, m))
    print()

print("=" * 80)
print("  ACTIONABLE NEXT STEPS")
print("=" * 80)
print()
print("  Look for rows with ** or *** above.")
print("  Those are the only strategies worth pursuing.")
