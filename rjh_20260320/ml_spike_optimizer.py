"""ML-Enhanced BTC Spike Strategy: maximize win rate on confirmed edge.

Core edge: BTC 1h >1.2% + alt confirmation -> alt follow (3-6h)
ML role: NOT direction prediction, but SIGNAL QUALITY SCORING
  - "This particular spike, should we enter or skip?"
  - Features: spike characteristics, alt reaction, market context
  - Target: trade outcome (win/loss)
"""
import numpy as np, pandas as pd, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import yfinance as yf
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import balanced_accuracy_score, classification_report

cost = 0.0018

print("=" * 80)
print("  ML SPIKE OPTIMIZER: BTC spike edge에 ML 필터 적용")
print("=" * 80)
print()

# Load data
data = {}
for coin, sym in [("SOL","SOL-USD"),("ADA","ADA-USD"),("BTC","BTC-USD"),("ETH","ETH-USD")]:
    df = yf.Ticker(sym).history(period="180d", interval="1h")
    df.columns = [c.lower() for c in df.columns]
    c=df["close"].values; h=df["high"].values; l=df["low"].values
    o=df["open"].values; v=df["volume"].values.astype(float)
    n=len(c)
    tr=np.maximum(h-l,np.maximum(np.abs(h-np.roll(c,1)),np.abs(l-np.roll(c,1))))
    tr[0]=h[0]-l[0]
    atr=pd.Series(tr).rolling(14,min_periods=1).mean().values
    ret=pd.Series(c).pct_change().values
    v_sma=pd.Series(v).rolling(24,min_periods=5).mean().values
    data[coin]={"df":df,"c":c,"h":h,"l":l,"o":o,"v":v,"n":n,
                "atr":atr,"ret":ret,"v_sma":v_sma}
    print("  %s: %d bars" % (coin, n))

btc = data["BTC"]
btc_ret = btc["ret"]
n_global = min(d["n"] for d in data.values())
print()

# ================================================================
# Step 1: Build feature matrix for each spike event
# ================================================================
print("[1] Building spike feature matrix...")
print()

def build_spike_features(coin_data, btc_data, n, btc_threshold=0.012):
    """Build features for each BTC spike event."""
    cd = coin_data; bd = btc_data
    c=cd["c"][-n:]; h=cd["h"][-n:]; l=cd["l"][-n:]; o=cd["o"][-n:]
    v=cd["v"][-n:]; atr=cd["atr"][-n:]; v_sma=cd["v_sma"][-n:]
    bc=bd["c"][-n:]; br=bd["ret"][-n:]
    cr=cd["ret"][-n:]

    features = []; labels = []; indices = []

    for i in range(max(30, n//3), n-7):
        if np.isnan(br[i]) or abs(br[i]) < btc_threshold: continue

        side = "BUY" if br[i] > 0 else "SELL"

        # === FEATURES (all causal, computed at bar i) ===
        f = {}

        # 1. BTC spike characteristics
        f["btc_ret"] = br[i]
        f["btc_abs_ret"] = abs(br[i])
        f["btc_bar_range"] = (bd["h"][-n:][i] - bd["l"][-n:][i]) / bd["c"][-n:][i]
        f["btc_body_ratio"] = abs(bc[i]-bd["o"][-n:][i]) / (bd["h"][-n:][i]-bd["l"][-n:][i]+1e-10)
        # BTC momentum before spike
        if i >= 3: f["btc_mom3"] = bc[i-1]/bc[i-4]-1 if bc[i-4]>0 else 0
        else: f["btc_mom3"] = 0
        if i >= 12: f["btc_mom12"] = bc[i-1]/bc[i-13]-1 if bc[i-13]>0 else 0
        else: f["btc_mom12"] = 0

        # 2. Alt coin same-bar reaction
        f["alt_ret"] = cr[i] if not np.isnan(cr[i]) else 0
        f["alt_abs_ret"] = abs(f["alt_ret"])
        f["alt_bar_range"] = (h[i]-l[i])/(c[i]+1e-10)
        f["alt_body_ratio"] = abs(c[i]-o[i])/(h[i]-l[i]+1e-10)
        f["alt_range_vs_atr"] = (h[i]-l[i])/(atr[i]+1e-10) if not np.isnan(atr[i]) else 1
        # Alt reaction strength vs BTC
        f["alt_btc_ratio"] = f["alt_ret"] / (br[i]+1e-10) if abs(br[i])>0.001 else 0
        # Wick analysis
        if c[i] >= o[i]:
            f["alt_upper_wick"] = (h[i]-c[i])/(h[i]-l[i]+1e-10)
            f["alt_lower_wick"] = (o[i]-l[i])/(h[i]-l[i]+1e-10)
        else:
            f["alt_upper_wick"] = (h[i]-o[i])/(h[i]-l[i]+1e-10)
            f["alt_lower_wick"] = (c[i]-l[i])/(h[i]-l[i]+1e-10)

        # 3. Volume context
        f["alt_vol_ratio"] = v[i]/(v_sma[i]+1e-10) if not np.isnan(v_sma[i]) and v_sma[i]>0 else 1
        f["alt_vol_vs_prev"] = v[i]/(v[i-1]+1e-10) if i>0 and v[i-1]>0 else 1
        # Volume trend (last 6 bars)
        if i >= 6:
            v_recent = np.mean(v[i-6:i])
            v_older = np.mean(v[max(0,i-12):i-6])
            f["vol_trend"] = v_recent/(v_older+1e-10) - 1
        else:
            f["vol_trend"] = 0

        # 4. Alt momentum before spike
        if i >= 3: f["alt_mom3"] = c[i-1]/c[i-4]-1 if c[i-4]>0 else 0
        else: f["alt_mom3"] = 0
        if i >= 6: f["alt_mom6"] = c[i-1]/c[i-7]-1 if c[i-7]>0 else 0
        else: f["alt_mom6"] = 0
        if i >= 12: f["alt_mom12"] = c[i-1]/c[i-13]-1 if c[i-13]>0 else 0
        else: f["alt_mom12"] = 0

        # 5. Volatility context
        if i >= 24:
            f["alt_vol_24h"] = np.nanstd(cr[i-24:i])
            f["btc_vol_24h"] = np.nanstd(br[i-24:i])
        else:
            f["alt_vol_24h"] = 0; f["btc_vol_24h"] = 0

        # 6. ATR context
        f["atr_pct"] = atr[i]/c[i] if not np.isnan(atr[i]) and c[i]>0 else 0.01

        # 7. Direction alignment (does alt agree with BTC?)
        f["direction_agree"] = 1 if (br[i]>0 and cr[i]>0) or (br[i]<0 and cr[i]<0) else 0

        # 8. Time features
        hour = data[list(data.keys())[0]]["df"].index[-n:][i].hour
        f["hour_sin"] = np.sin(2*np.pi*hour/24)
        f["hour_cos"] = np.cos(2*np.pi*hour/24)
        dow = data[list(data.keys())[0]]["df"].index[-n:][i].dayofweek
        f["dow_sin"] = np.sin(2*np.pi*dow/7)
        f["dow_cos"] = np.cos(2*np.pi*dow/7)

        # 9. Recent trade performance (if we had traded last N spikes)
        # Can't compute without historical trades, skip

        # === LABEL: did this trade win? ===
        eb = i + 1
        if eb >= n - 3: continue
        entry = o[eb]
        ai = atr[i] if not np.isnan(atr[i]) else entry*0.01
        tp_d = ai*1.5; sl_d = ai*1.0
        if side=="BUY": tp,sl = entry+tp_d, entry-sl_d
        else: tp,sl = entry-tp_d, entry+sl_d

        pnl = 0
        for j in range(eb, min(eb+6, n)):
            if side=="BUY":
                if l[j]<=sl: pnl=(sl-entry)/entry; break
                if h[j]>=tp: pnl=(tp-entry)/entry; break
            else:
                if h[j]>=sl: pnl=(entry-sl)/entry; break
                if l[j]<=tp: pnl=(entry-tp)/entry; break
        else:
            if eb+5<n:
                ep=c[eb+5]
                pnl=(ep-entry)/entry if side=="BUY" else (entry-ep)/entry

        net_pnl = pnl - cost
        label = 1 if net_pnl > 0 else 0

        features.append(f)
        labels.append(label)
        indices.append(i)

    return pd.DataFrame(features), np.array(labels), indices

# Build for each coin
for coin in ["SOL", "ADA"]:
    print("--- %s ---" % coin)
    X_df, y, idx = build_spike_features(data[coin], btc, n_global)
    print("  Spike events: %d (win=%d, loss=%d, WR=%.1f%%)" % (
        len(y), np.sum(y==1), np.sum(y==0), np.mean(y)*100))
    print("  Features: %d" % X_df.shape[1])
    print("  Feature columns: %s" % list(X_df.columns))
    print()

    if len(y) < 30:
        print("  Too few events, skip")
        continue

    X = X_df.fillna(0).values

    # ================================================================
    # Step 2: ML filter - can we predict which spikes will win?
    # ================================================================
    print("[2] ML Filter Training (TimeSeriesSplit)...")

    tscv = TimeSeriesSplit(n_splits=3, gap=5)
    models = [
        ("ExtraTrees", ExtraTreesClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42)),
        ("GBM", GradientBoostingClassifier(n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=42)),
    ]

    for model_name, model in models:
        fold_scores = []; fold_filtered_pnl = []; fold_all_pnl = []

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            if len(tr_idx) < 20 or len(val_idx) < 10: continue

            model.fit(X[tr_idx], y[tr_idx])
            pred = model.predict(X[val_idx])
            proba = model.predict_proba(X[val_idx])[:, 1]
            bacc = balanced_accuracy_score(y[val_idx], pred)
            fold_scores.append(bacc)

            # Simulate: only enter when model says "win"
            # Reconstruct PnL for validation period
            cd = data[coin]; bd = btc
            cn = n_global
            c=cd["c"][-cn:]; h=cd["h"][-cn:]; l=cd["l"][-cn:]; o=cd["o"][-cn:]
            atr_c=cd["atr"][-cn:]; br=bd["ret"][-cn:]

            all_pnl = []; filtered_pnl = []
            for vi, v_idx in enumerate(val_idx):
                event_i = idx[v_idx]
                side = "BUY" if br[event_i] > 0 else "SELL"
                eb = event_i + 1
                if eb >= cn - 6: continue
                entry = o[eb]
                ai = atr_c[event_i] if not np.isnan(atr_c[event_i]) else entry*0.01
                tp_d=ai*1.5; sl_d=ai*1.0
                if side=="BUY": tp,sl=entry+tp_d,entry-sl_d
                else: tp,sl=entry-tp_d,entry+sl_d
                pnl=0
                for j in range(eb,min(eb+6,cn)):
                    if side=="BUY":
                        if l[j]<=sl: pnl=(sl-entry)/entry; break
                        if h[j]>=tp: pnl=(tp-entry)/entry; break
                    else:
                        if h[j]>=sl: pnl=(entry-sl)/entry; break
                        if l[j]<=tp: pnl=(entry-tp)/entry; break
                else:
                    if eb+5<cn:
                        ep=c[eb+5]
                        pnl=(ep-entry)/entry if side=="BUY" else (entry-ep)/entry

                net = pnl - cost
                all_pnl.append(net)
                if pred[vi] == 1:  # model says enter
                    filtered_pnl.append(net)

            fold_all_pnl.extend(all_pnl)
            fold_filtered_pnl.extend(filtered_pnl)

        if not fold_scores: continue
        avg_bacc = np.mean(fold_scores)

        all_arr = np.array(fold_all_pnl) if fold_all_pnl else np.array([0])
        filt_arr = np.array(fold_filtered_pnl) if fold_filtered_pnl else np.array([0])

        # Threshold sweep: vary confidence threshold
        print()
        print("  %s %s:" % (coin, model_name))
        print("    CV bacc: %.3f" % avg_bacc)
        print("    ALL trades:      n=%d WR=%.1f%% avg=%+.4f%%" % (
            len(all_arr), np.mean(all_arr>0)*100, np.mean(all_arr)*100))
        print("    ML filtered:     n=%d WR=%.1f%% avg=%+.4f%%" % (
            len(filt_arr), np.mean(filt_arr>0)*100 if len(filt_arr)>0 else 0, np.mean(filt_arr)*100 if len(filt_arr)>0 else 0))

        improvement = np.mean(filt_arr) - np.mean(all_arr) if len(filt_arr)>0 else 0
        print("    Improvement: %+.4f%%" % (improvement*100))
        if improvement > 0:
            print("    -> ML FILTER IMPROVES WIN RATE")
        else:
            print("    -> ML filter does NOT help")

    # ================================================================
    # Step 3: Feature importance
    # ================================================================
    print()
    print("[3] Feature Importance (full dataset):")
    et = ExtraTreesClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42)
    et.fit(X, y)
    imp = pd.Series(et.feature_importances_, index=X_df.columns).sort_values(ascending=False)
    for feat, val in imp.head(10).items():
        print("    %.4f  %s" % (val, feat))
    print()

    # ================================================================
    # Step 4: Probability threshold sweep
    # ================================================================
    print("[4] Probability Threshold Sweep:")
    # Retrain on first 60%, test on last 40%
    split = int(len(X) * 0.6)
    if split > 20 and len(X) - split > 10:
        et2 = ExtraTreesClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42)
        et2.fit(X[:split], y[:split])
        proba_test = et2.predict_proba(X[split:])[:, 1]

        # Reconstruct PnL for test period
        test_pnls = []
        for vi in range(split, len(X)):
            event_i = idx[vi]
            side = "BUY" if br[event_i] > 0 else "SELL"
            eb = event_i + 1
            if eb >= cn - 6: test_pnls.append(0); continue
            entry = o[eb]
            ai = atr_c[event_i] if not np.isnan(atr_c[event_i]) else entry*0.01
            tp_d=ai*1.5; sl_d=ai*1.0
            if side=="BUY": tp,sl=entry+tp_d,entry-sl_d
            else: tp,sl=entry-tp_d,entry+sl_d
            pnl=0
            for j in range(eb,min(eb+6,cn)):
                if side=="BUY":
                    if l[j]<=sl: pnl=(sl-entry)/entry; break
                    if h[j]>=tp: pnl=(tp-entry)/entry; break
                else:
                    if h[j]>=sl: pnl=(entry-sl)/entry; break
                    if l[j]<=tp: pnl=(entry-tp)/entry; break
            else:
                if eb+5<cn:
                    ep=c[eb+5]
                    pnl=(ep-entry)/entry if side=="BUY" else (entry-ep)/entry
            test_pnls.append(pnl-cost)

        test_pnls = np.array(test_pnls)

        print("    Threshold  Trades  WR      Avg PnL")
        for thresh in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]:
            mask = proba_test >= thresh
            if mask.sum() < 3: continue
            sel = test_pnls[mask]
            print("    %.2f       %3d     %.1f%%   %+.4f%%" % (thresh, len(sel), np.mean(sel>0)*100, np.mean(sel)*100))

    print()
    print("-" * 80)
    print()

print("=" * 80)
print("  DONE")
print("=" * 80)
