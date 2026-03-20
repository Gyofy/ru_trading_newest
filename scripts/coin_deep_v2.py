"""SOL/XRP/ADA: Round 2 - stress test every finding."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, '.')
import yfinance as yf

cost = 0.0018

# Load data
coins = {}
for coin, sym in [("SOL","SOL-USD"),("XRP","XRP-USD"),("ADA","ADA-USD"),("BTC","BTC-USD"),("ETH","ETH-USD")]:
    df = yf.Ticker(sym).history(period="180d", interval="1h")
    df.columns = [c.lower() for c in df.columns]
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    o = df["open"].values; v = df["volume"].values.astype(float)
    ret = pd.Series(c).pct_change().values
    tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)),np.abs(l-np.roll(c,1))))
    tr[0] = h[0]-l[0]
    atr = pd.Series(tr).rolling(14,min_periods=1).mean().values
    coins[coin] = {"df":df,"c":c,"h":h,"l":l,"o":o,"v":v,"ret":ret,"atr":atr,"n":len(c)}

n = min(*(d["n"] for d in coins.values()))
btc_ret = coins["BTC"]["ret"][-n:]

def sim(close, high, low, opn, atr, entries, nn, tp_m, sl_m, hold=6):
    results = []; pe = 0
    for i, side in entries:
        if i<pe or i>=nn-hold-1: continue
        eb = i+1; entry = opn[eb]
        ai = atr[i] if not np.isnan(atr[i]) else entry*0.01
        td=ai*tp_m; sd=ai*sl_m
        if side=="BUY": tp,sl = entry+td,entry-sd
        else: tp,sl = entry-td,entry+sd
        pnl=0
        for j in range(eb, min(eb+hold, nn)):
            if side=="BUY":
                if low[j]<=sl: pnl=(sl-entry)/entry; break
                if high[j]>=tp: pnl=(tp-entry)/entry; break
            else:
                if high[j]>=sl: pnl=(entry-sl)/entry; break
                if low[j]<=tp: pnl=(entry-tp)/entry; break
        else:
            if eb+hold-1<nn:
                ep=close[eb+hold-1]
                pnl=(ep-entry)/entry if side=="BUY" else (entry-ep)/entry
        results.append(pnl-cost); pe=eb+hold+1
    return np.array(results) if results else np.array([])

# ================================================================
print("=" * 80)
print("  ROUND 2: 각 발견의 비판적 검증")
print("=" * 80)
print()

# ================================================================
print("[A] 시간대 패턴: 진짜인가 노이즈인가?")
print("=" * 80)
print()
print("  00-02 UTC가 최고 → 이 시간만 트레이딩하면?")
print()

for coin in ["SOL","XRP","ADA"]:
    d = coins[coin]
    cn = min(d["n"], n)
    df = d["df"]; hours = df.index.hour
    c=d["c"][-cn:]; h=d["h"][-cn:]; l=d["l"][-cn:]; o=d["o"][-cn:]
    atr=d["atr"][-cn:]; cr=d["ret"][-cn:]
    split = cn // 3

    for time_range, label in [((0,2),"00-02 best"), ((16,18),"16-18 worst"), ((0,24),"all hours")]:
        entries = []
        for i in range(max(split,13), cn):
            hr = df.index[-cn:][i].hour
            if not (time_range[0] <= hr < time_range[1]): continue
            # Direction: 12bar momentum
            mom = c[i] / c[max(0,i-12)] - 1
            if abs(mom) < 0.003: continue
            entries.append((i, "BUY" if mom > 0 else "SELL"))

        arr = sim(c,h,l,o,atr,entries,cn,1.5,1.0,6)
        if len(arr) < 5: continue
        wr = np.mean(arr>0); avg = np.mean(arr)
        m = " **" if avg > 0.0005 else "   "
        print("  %s %-15s n=%3d WR=%.1f%% avg=%+.4f%% %s" % (coin, label, len(arr), wr*100, avg*100, m))
    print()

# ================================================================
print("[B] XRP 모멘텀 성격: 정말 모멘텀인가?")
print("=" * 80)
print()
print("  XRP가 3/6/12h 모멘텀이면 → 과거 방향으로 6h 진입 시 수익?")
print()

for coin in ["SOL","XRP","ADA"]:
    d = coins[coin]
    cn = min(d["n"], n)
    c=d["c"][-cn:]; h=d["h"][-cn:]; l=d["l"][-cn:]; o=d["o"][-cn:]; atr=d["atr"][-cn:]
    split = cn // 3

    for lb in [3, 6, 12]:
        # Momentum: enter in direction of past lb bars
        entries = []
        for i in range(max(split, lb+1), cn):
            mom = c[i] / c[i-lb] - 1
            if abs(mom) < 0.003: continue
            entries.append((i, "BUY" if mom > 0 else "SELL"))

        arr = sim(c,h,l,o,atr,entries,cn,1.5,1.0,6)
        if len(arr) < 10: continue
        wr = np.mean(arr>0); avg = np.mean(arr)

        # Also test contrarian
        entries_c = [(i, "SELL" if s=="BUY" else "BUY") for i,s in entries]
        arr_c = sim(c,h,l,o,atr,entries_c,cn,1.5,1.0,6)
        avg_c = np.mean(arr_c) if len(arr_c) > 0 else 0

        m = " **" if avg > 0.0005 else "   "
        mc = " **" if avg_c > 0.0005 else "   "
        print("  %s mom%d: FOLLOW=%+.4f%%%s  COUNTER=%+.4f%%%s (n=%d)" % (
            coin, lb, avg*100, m, avg_c*100, mc, len(arr)))
    print()

# ================================================================
print("[C] XRP 볼륨 스파이크: trade-level 시뮬 (비용 포함)")
print("=" * 80)
print()

for coin in ["SOL","XRP","ADA"]:
    d = coins[coin]
    cn = min(d["n"], n)
    c=d["c"][-cn:]; h=d["h"][-cn:]; l=d["l"][-cn:]; o=d["o"][-cn:]
    atr=d["atr"][-cn:]; v=d["v"][-cn:]
    v_sma = pd.Series(v).rolling(24,min_periods=5).mean().values
    split = cn // 3

    for vol_mult in [2.0, 3.0]:
        for tp_m, sl_m in [(1.0, 0.7), (1.5, 1.0), (2.0, 1.0)]:
            entries = []
            for i in range(max(split, 25), cn):
                if np.isnan(v_sma[i]) or v_sma[i] < 1: continue
                if v[i] < v_sma[i] * vol_mult: continue
                bar_ret = (c[i] - o[i]) / o[i] if o[i] > 0 else 0
                if abs(bar_ret) < 0.002: continue
                entries.append((i, "BUY" if bar_ret > 0 else "SELL"))

            arr = sim(c,h,l,o,atr,entries,cn,tp_m,sl_m,6)
            if len(arr) < 10: continue
            wr = np.mean(arr>0); avg = np.mean(arr)
            m = "***" if avg>0.002 else " **" if avg>0.0005 else "  *" if avg>0 else "   "
            print("  %s vol>%.0fx tp%.1f/sl%.1f: n=%3d WR=%.1f%% avg=%+.4f%% %s" % (
                coin, vol_mult, tp_m, sl_m, len(arr), wr*100, avg*100, m))
    print()

# ================================================================
print("[D] SHORT만 따로: 하락장에서 SHORT edge 검증")
print("=" * 80)
print()

for coin in ["SOL","XRP","ADA"]:
    d = coins[coin]
    cn = min(d["n"], n)
    c=d["c"][-cn:]; h=d["h"][-cn:]; l=d["l"][-cn:]; o=d["o"][-cn:]; atr=d["atr"][-cn:]
    br = btc_ret[-cn:]
    split = cn // 3

    # SHORT only when BTC drops
    for thresh in [0.008, 0.010, 0.012, 0.015]:
        entries = [(i, "SELL") for i in range(max(split,2), cn) if not np.isnan(br[i]) and br[i] < -thresh]
        arr = sim(c,h,l,o,atr,entries,cn,1.5,1.0,6)
        if len(arr) < 5: continue
        wr = np.mean(arr>0); avg = np.mean(arr)
        m = "***" if avg>0.002 else " **" if avg>0.0005 else "  *" if avg>0 else "   "
        print("  %s SHORT btc<-%.1f%%: n=%3d WR=%.1f%% avg=%+.4f%% %s" % (coin, thresh*100, len(arr), wr*100, avg*100, m))
    print()

# ================================================================
print("[E] 기간 안정성: 60일씩 3구간 분할")
print("=" * 80)
print()

for coin in ["SOL","XRP","ADA"]:
    d = coins[coin]
    cn = min(d["n"], n)
    c=d["c"][-cn:]; h=d["h"][-cn:]; l=d["l"][-cn:]; o=d["o"][-cn:]; atr=d["atr"][-cn:]
    br = btc_ret[-cn:]
    third = cn // 3

    for period, (start,end) in enumerate([
        (0, third), (third, 2*third), (2*third, cn)
    ]):
        coin_ret_period = (c[min(end-1,cn-1)] / c[start] - 1) * 100

        # BTC spike strategy
        entries = []
        for i in range(max(start,2), end):
            if np.isnan(br[i]) or abs(br[i]) < 0.012: continue
            entries.append((i, "BUY" if br[i] > 0 else "SELL"))
        arr = sim(c,h,l,o,atr,entries,cn,1.5,1.0,6)
        if len(arr) < 3:
            print("  %s P%d: n<3" % (coin, period+1))
            continue
        wr = np.mean(arr>0); avg = np.mean(arr)
        m = " **" if avg > 0 else "   "
        print("  %s P%d (coin %+.0f%%): n=%2d WR=%.0f%% avg=%+.4f%% %s" % (
            coin, period+1, coin_ret_period, len(arr), wr*100, avg*100, m))
    print()

# ================================================================
print("[F] COMBINED SIGNAL: BTC spike + Volume + Momentum 복합")
print("=" * 80)
print()

for coin in ["SOL","XRP","ADA"]:
    d = coins[coin]
    cn = min(d["n"], n)
    c=d["c"][-cn:]; h=d["h"][-cn:]; l=d["l"][-cn:]; o=d["o"][-cn:]
    atr=d["atr"][-cn:]; v=d["v"][-cn:]
    v_sma = pd.Series(v).rolling(24,min_periods=5).mean().values
    br = btc_ret[-cn:]
    split = cn // 3

    for min_signals in [2, 3]:
        entries = []
        for i in range(max(split,13), cn):
            signals = 0; direction = 0
            # Signal 1: BTC spike
            if not np.isnan(br[i]) and abs(br[i]) > 0.008:
                signals += 1
                direction += 1 if br[i] > 0 else -1
            # Signal 2: Volume spike
            if not np.isnan(v_sma[i]) and v_sma[i] > 0 and v[i] > v_sma[i] * 1.5:
                bar_d = 1 if c[i] > o[i] else -1
                signals += 1; direction += bar_d
            # Signal 3: 6h momentum
            if i >= 6:
                mom = c[i] / c[i-6] - 1
                if abs(mom) > 0.003:
                    signals += 1; direction += 1 if mom > 0 else -1

            if signals >= min_signals and abs(direction) >= min_signals:
                entries.append((i, "BUY" if direction > 0 else "SELL"))

        arr = sim(c,h,l,o,atr,entries,cn,1.5,1.0,6)
        if len(arr) < 5: continue
        wr = np.mean(arr>0); avg = np.mean(arr)
        m = "***" if avg>0.002 else " **" if avg>0.0005 else "  *" if avg>0 else "   "
        print("  %s combined>=%d: n=%3d WR=%.1f%% avg=%+.4f%% %s" % (coin, min_signals, len(arr), wr*100, avg*100, m))
    print()

# ================================================================
print("=" * 80)
print("  ROUND 2 COMPLETE: ** marks positive strategies")
print("=" * 80)
