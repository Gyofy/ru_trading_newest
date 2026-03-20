"""2h Strategy Brainstorm - 5 paradigms on real 1h data."""
import pandas as pd, numpy as np, sys
sys.path.insert(0, '.')
import yfinance as yf

cost = 0.0018

def sim_2h(close, high, low, atr, entries, n, tp_m, sl_m):
    """Simulate 2h hold trades."""
    results = []
    pe = 0
    for i, side in entries:
        if i < pe or i >= n-2: continue
        entry = close[i]
        a = atr[i] if not np.isnan(atr[i]) else entry*0.01
        tp_d = a * tp_m; sl_d = a * sl_m
        if side == "BUY": tp, sl = entry+tp_d, entry-sl_d
        else: tp, sl = entry-tp_d, entry+sl_d
        pnl = 0
        for j in range(i+1, min(i+3, n)):
            if side == "BUY":
                if low[j] <= sl: pnl = (sl-entry)/entry; break
                if high[j] >= tp: pnl = (tp-entry)/entry; break
            else:
                if high[j] >= sl: pnl = (entry-sl)/entry; break
                if low[j] <= tp: pnl = (entry-tp)/entry; break
        else:
            if i+2 < n:
                ep = close[min(i+2,n-1)]
                pnl = (ep-entry)/entry if side=="BUY" else (entry-ep)/entry
        results.append(pnl - cost); pe = i + 3
    return np.array(results) if results else np.array([0.0])

def get_data(sym):
    df = yf.Ticker(sym).history(period="60d", interval="1h")
    df.columns = [c.lower() for c in df.columns]
    close = df["close"].values; high = df["high"].values; low = df["low"].values
    n = len(close)
    tr = np.maximum(high-low, np.maximum(np.abs(high-np.roll(close,1)), np.abs(low-np.roll(close,1))))
    tr[0] = high[0]-low[0]
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
    return df, close, high, low, atr, n

def report(name, arr):
    if len(arr) < 3: return
    nn = len(arr); wr = np.mean(arr>0); avg = np.mean(arr)
    m = "***" if avg > 0.002 else " **" if avg > 0.0005 else "  *" if avg > 0 else "   "
    print("  %-30s n=%3d WR=%.1f%% avg=%+.4f%% %s" % (name, nn, wr*100, avg*100, m))

coins = {"SOL":"SOL-USD", "BTC":"BTC-USD", "ETH":"ETH-USD", "XRP":"XRP-USD", "ADA":"ADA-USD"}

# Load all data
data = {}
for coin, sym in coins.items():
    try:
        data[coin] = get_data(sym)
    except Exception as e:
        print(f"Skip {coin}: {e}")

btc_df, btc_close, _, _, _, btc_n = data.get("BTC", (None,)*6)

print("=" * 70)
print("  5 Strategy Paradigms - 2h Hold - Real 1h Data")
print("=" * 70)
print()

# === 1. Daily trend + 1h momentum alignment ===
print("[1] Daily Trend + 1h Momentum Alignment")
for coin in ["SOL", "ETH", "XRP", "ADA"]:
    if coin not in data: continue
    df, close, high, low, atr, n = data[coin]
    try:
        df_d = yf.Ticker(coins[coin]).history(period="120d", interval="1d")
        df_d.columns = [c.lower() for c in df_d.columns]
    except: continue
    d_close = df_d["close"].values
    d_sma20 = pd.Series(d_close).rolling(20).mean().values
    split = n // 2
    entries = []
    for i in range(max(split, 24), n):
        hour_ts = df.index[i]
        dm = df_d.index <= hour_ts
        if dm.sum() < 21: continue
        di = dm.sum() - 1
        trend = "UP" if d_close[di] > d_sma20[di] else "DOWN"
        mom3 = close[i] / close[i-3] - 1
        if abs(mom3) < 0.002: continue
        if trend == "UP" and mom3 > 0: entries.append((i, "BUY"))
        elif trend == "DOWN" and mom3 < 0: entries.append((i, "SELL"))
    arr = sim_2h(close, high, low, atr, entries, n, 1.0, 0.7)
    report(f"{coin} daily+mom3", arr)
print()

# === 2. Range Breakout ===
print("[2] Range Breakout")
for coin in ["SOL", "BTC", "ETH", "XRP"]:
    if coin not in data: continue
    _, close, high, low, atr, n = data[coin]
    split = n // 2
    for lb in [6, 12, 24]:
        entries = []
        for i in range(max(split, lb+1), n):
            rh = np.max(high[i-lb:i]); rl = np.min(low[i-lb:i])
            rp = (rh-rl)/((rh+rl)/2)
            if rp < 0.005: continue
            if close[i] > rh: entries.append((i, "BUY"))
            elif close[i] < rl: entries.append((i, "SELL"))
        arr = sim_2h(close, high, low, atr, entries, n, 1.5, 0.8)
        report(f"{coin} breakout_lb{lb}", arr)
print()

# === 3. Volume Spike + Direction ===
print("[3] Volume Spike + Bar Direction")
for coin in ["SOL", "BTC", "ETH", "XRP"]:
    if coin not in data: continue
    df, close, high, low, atr, n = data[coin]
    vol = df["volume"].values.astype(float)
    vol_sma = pd.Series(vol).rolling(20, min_periods=5).mean().values
    split = n // 2
    for vm in [1.5, 2.0, 3.0]:
        entries = []
        for i in range(max(split, 21), n):
            if np.isnan(vol_sma[i]) or vol_sma[i] < 1: continue
            if vol[i] < vol_sma[i] * vm: continue
            bar_ret = (close[i] - df["open"].iloc[i]) / df["open"].iloc[i]
            if abs(bar_ret) < 0.002: continue
            entries.append((i, "BUY" if bar_ret > 0 else "SELL"))
        arr = sim_2h(close, high, low, atr, entries, n, 1.0, 0.7)
        report(f"{coin} vol>{vm:.1f}x", arr)
print()

# === 4. Mean Reversion (RSI extremes) ===
print("[4] Mean Reversion (RSI extremes)")
for coin in ["SOL", "BTC", "ETH", "XRP"]:
    if coin not in data: continue
    _, close, high, low, atr, n = data[coin]
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean().values
    loss_v = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean().values
    rsi = 100 - 100 / (1 + gain / (loss_v + 1e-10))
    split = n // 2
    for rlo, rhi in [(20,80), (25,75), (30,70)]:
        entries = []
        for i in range(max(split, 15), n):
            if rsi[i] < rlo: entries.append((i, "BUY"))
            elif rsi[i] > rhi: entries.append((i, "SELL"))
        arr = sim_2h(close, high, low, atr, entries, n, 0.7, 1.0)
        report(f"{coin} RSI<{rlo}/>{rhi}", arr)
print()

# === 5. BTC Lead -> Alt Lag ===
print("[5] BTC Lead -> Alt Follow")
if btc_close is not None:
    btc_ret = pd.Series(btc_close).pct_change().values
    for coin in ["SOL", "ETH", "XRP", "ADA"]:
        if coin not in data: continue
        _, close, high, low, atr, n = data[coin]
        cn = min(n, btc_n)
        c2 = close[-cn:]; h2 = high[-cn:]; l2 = low[-cn:]
        a2 = atr[-cn:]
        br = btc_ret[-cn:]
        split2 = cn // 2
        for bt in [0.005, 0.008, 0.01, 0.015, 0.02]:
            entries = []
            for i in range(max(split2, 2), cn):
                if np.isnan(br[i]) or abs(br[i]) < bt: continue
                entries.append((i, "BUY" if br[i] > 0 else "SELL"))
            arr = sim_2h(c2, h2, l2, a2, entries, cn, 0.8, 0.6)
            report(f"{coin} btc>{bt*100:.1f}%", arr)
print()

# === 6. Funding Rate Proxy (high funding = mean reversion) ===
print("[6] Extreme Momentum Reversal (proxy for funding/crowding)")
for coin in ["SOL", "BTC", "ETH"]:
    if coin not in data: continue
    _, close, high, low, atr, n = data[coin]
    ret_6h = pd.Series(close).pct_change(6).values
    split = n // 2
    for thresh in [0.03, 0.04, 0.05]:
        entries = []
        for i in range(max(split, 7), n):
            if np.isnan(ret_6h[i]): continue
            if ret_6h[i] > thresh: entries.append((i, "SELL"))  # overbought reversal
            elif ret_6h[i] < -thresh: entries.append((i, "BUY"))  # oversold reversal
        arr = sim_2h(close, high, low, atr, entries, n, 0.7, 0.7)
        report(f"{coin} rev>{thresh*100:.0f}%", arr)

print()
print("=" * 70)
print("Strategies with ** or *** have positive avg PnL")
print("=" * 70)
