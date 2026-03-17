"""Paper Trading Simulation -- 5M KRW, v3.4, 8-week OOS replay."""
import sys
sys.path.insert(0, "C:/Users/RJ/Desktop/CLAUDE_CRYPTO_AGENT")
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import json, yaml, warnings, numpy as np, pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from src.data.crawlers.crypto_ohlcv import fetch_all_top10
from src.data.crawlers.macro_commodity_crawler import crawl_all_macro_data
from src.models.masking_loop import create_labels_triple_barrier, LABEL_MAP, HORIZONS
from src.models.enhanced_ensemble import EnhancedEnsemble
from src.execution.cost_model import CostModel, FeeSchedule, FundingConfig, MissFillConfig, ExitType
from src.utils.config import bar_minutes as cfg_bar_minutes
from sklearn.feature_selection import mutual_info_classif
warnings.filterwarnings("ignore")

with open("config/frozen_params_v3_4.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

COMMON = CFG["common"]
COINS = CFG["coins"]
CC = CFG["cost_model"]
EXK = CFG["excluded_feature_keywords"]
BM = cfg_bar_minutes()
MH = COMMON["max_horizon"]
RF = COMMON["risk_frac"]
SEED = 5_000_000
USDKRW = 1450

CM = CostModel(
    fee_schedule=FeeSchedule(maker_fee=CC["maker_fee"], taker_fee=CC["taker_fee"],
        slippage_entry=CC["slippage_entry"], slippage_exit_limit=CC["slippage_exit_limit"],
        slippage_exit_market=CC["slippage_exit_market"]),
    funding_config=FundingConfig(interval_hours=CC["funding_interval_hours"],
        default_rate=CC["funding_default_rate"]),
    miss_fill_config=MissFillConfig(reject_prob=CC["miss_fill_reject_prob"],
        missed_ev_pct=CC["miss_fill_missed_ev"]))

def ck(coin, key):
    return COINS[coin].get(f"{key}_override", COMMON[key])

def isex(c):
    return any(kw in c.lower() for kw in EXK)

def regime(df, i):
    if i < 42: return "UNKNOWN"
    c = df["close"].values[max(0,i-41):i+1]
    ef = pd.Series(c).ewm(span=10).mean().iloc[-1]
    es = pd.Series(c).ewm(span=30).mean().iloc[-1]
    rs = np.std(np.diff(c)/c[:-1]) if len(c)>1 else 0
    if ef > es*1.005: return "TREND_UP"
    elif ef < es*0.995: return "TREND_DOWN"
    else: return "RANGE_HIGH" if rs > 0.02 else "RANGE_LOW"

print("Fetching data...")
ohlcv = fetch_all_top10("365d", "1h")
first = list(ohlcv.values())[0]
macro = crawl_all_macro_data(first.index)
ma = macro.get("aligned", pd.DataFrame())
data = {}
for coin in COINS:
    df = ohlcv[coin].copy()
    if len(ma) > 0:
        for col in ma.columns:
            df[col] = ma[col].reindex(df.index).ffill().bfill().fillna(0)
    data[coin] = df

all_trades = []
for coin in COINS:
    df = data[coin]
    idx = df.index; end = idx[-1]
    oos_start = end - timedelta(days=56)
    purge = timedelta(hours=(MH*2+6)*(BM//60))
    train_df = df[idx <= oos_start - purge]
    oos_df = df[idx >= oos_start]

    params = COINS[coin]
    h = HORIZONS[-1]; hl = f"label_{h*BM}min"
    ku, kl = ck(coin,"k_upper"), ck(coin,"k_lower")

    labeled = create_labels_triple_barrier(train_df.copy(), h, k_upper_override=ku, k_lower_override=kl, verbose=False)
    if hl not in labeled.columns: continue

    exclude = {"label","future_return","open","high","low","close","volume"}
    for hh in HORIZONS: exclude.add(f"label_{hh*BM}min"); exclude.add(f"return_{hh*BM}min")
    lk = ["future","target","label_","return_","fwd_","forward_"]
    fcols = [c for c in labeled.columns if c not in exclude and not any(k in c.lower() for k in lk)
             and not isex(c) and labeled[c].dtype in [np.float64,np.float32,np.int64,np.int32,float,int]]

    mf = params["max_features"]
    clean = labeled.replace([np.inf,-np.inf], np.nan).ffill().bfill()
    X = clean[fcols].fillna(0).values
    y = clean[hl].fillna(1).values.astype(int)
    if len(fcols) > mf:
        mi = mutual_info_classif(X[:min(2000,len(X))], y[:min(2000,len(X))],
                                 discrete_features=False, random_state=42, n_neighbors=5)
        top = np.argsort(mi)[-mf:]
        fcols = [fcols[i] for i in sorted(top)]
        X = clean[fcols].fillna(0).values

    ys1 = (y != LABEL_MAP["HOLD"]).astype(int)
    if len(np.unique(ys1)) < 2: continue
    s1c = np.bincount(ys1, minlength=2)
    s1w = np.where(s1c>0, len(ys1)/(2*s1c+1e-10), 1.0)[ys1]
    s1 = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=6, verbose=False)
    s1.fit(X, ys1, sample_weight=s1w)
    print(f"  {coin}: S1 trained")

    s2 = None; tm = y != LABEL_MAP["HOLD"]
    if tm.sum() >= 30:
        ys2 = (y[tm] == LABEL_MAP["UP"]).astype(int)
        if len(np.unique(ys2)) >= 2:
            s2c = np.bincount(ys2, minlength=2)
            s2w = np.where(s2c>0, len(ys2)/(2*s2c+1e-10), 1.0)[ys2]
            s2 = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=6, verbose=False)
            s2.fit(X[tm], ys2, sample_weight=s2w)
            print(f"  {coin}: S2 trained")

    oc = oos_df.replace([np.inf,-np.inf], np.nan).ffill().bfill()
    Xo = oc[fcols].fillna(0).values
    s1p = s1.predict_proba(Xo); s1pred = (s1p[:,1] >= params["stage1_threshold"]).astype(int)
    s2pred = np.zeros(len(Xo), dtype=int); s2prob = np.full(len(Xo), 0.5)
    if s2 is not None:
        s2p = s2.predict_proba(Xo); s2pred = np.argmax(s2p, axis=1); s2prob = s2p[:,1]

    close = oos_df["close"].values; high = oos_df["high"].values; low = oos_df["low"].values
    times = oos_df.index; n = len(close)
    atr = oos_df["atr_14"].values if "atr_14" in oos_df.columns else np.full(n, 0.01)
    blocked = COINS[coin].get("blocked_regimes_override", CFG["blocked_regimes"])

    na = 0
    for i in range(n - MH):
        if i < na: continue
        if s1pred[i] != 1: continue
        r = regime(oos_df, i)
        if r in blocked: continue
        entry = close[i]; side = "BUY" if s2pred[i]==1 else "SELL"
        ca = atr[i] if not np.isnan(atr[i]) else entry*0.01
        ud = max(ku*ca, 0.002*entry); ld = max(kl*ca, 0.002*entry)
        tp = entry+ud if side=="BUY" else entry-ud
        sl = entry-ld if side=="BUY" else entry+ld
        et, eb, ep = None, -1, 0.0
        for j in range(i+1, min(i+MH+1, n)):
            if side=="BUY": ht,hs = high[j]>=tp, low[j]<=sl
            else: ht,hs = low[j]<=tp, high[j]>=sl
            if ht and hs: et,eb,ep="stop_loss",j,sl; break
            elif ht: et,eb,ep="take_profit",j,tp; break
            elif hs: et,eb,ep="stop_loss",j,sl; break
        if et is None: eb=min(i+MH,n-1); et="time_stop"; ep=close[eb]
        hold = eb-i
        gp = (ep-entry)/entry if side=="BUY" else (entry-ep)/entry
        sd = max(ld/entry, 0.003); nr = RF/sd; gpeq = gp*nr
        ee = {"take_profit":ExitType.TAKE_PROFIT,"stop_loss":ExitType.STOP_LOSS,"time_stop":ExitType.TIME_STOP}[et]
        cost = CM.estimate_trade_cost(entry_price=entry, sl_price=sl, tp_price=tp,
            risk_frac=RF, exit_type=ee, holding_bars=hold, bar_minutes=BM, entry_is_maker=True)
        net = gpeq - cost.total_eq
        all_trades.append({"coin":coin,"side":side,"entry_time":str(times[i])[:16],
            "exit_time":str(times[eb])[:16],"exit_type":et,"hold":hold,
            "gross_eq":round(gpeq,6),"cost_eq":round(cost.total_eq,6),"net_eq":round(net,6),"regime":r})
        na = eb+1

# Sort by time
all_trades.sort(key=lambda x: x["entry_time"])

# Print
print()
print("="*75)
print(f"  PAPER TRADING REPORT -- v3.4 Frozen Params")
print(f"  Seed: {SEED:,} KRW (${SEED/USDKRW:,.0f} USD)")
print(f"  Period: 8 weeks OOS | Coins: {list(COINS.keys())}")
print("="*75)
print()
print(f"  {'#':>3s} {'Coin':>4s} {'Side':>5s} {'Entry Date':>12s} {'Exit Date':>12s} "
      f"{'Type':>12s} {'Hold':>4s} {'Net%':>8s} {'PnL KRW':>12s} {'Equity':>14s} {'':>3s}")
print(f"  {'-'*95}")

eq = 1.0
peak_eq = SEED
low_eq = SEED
for i, t in enumerate(all_trades):
    eq *= (1 + t["net_eq"])
    eq_krw = SEED * eq
    pnl_krw = t["net_eq"] * SEED * (eq / (1 + t["net_eq"]))
    peak_eq = max(peak_eq, eq_krw)
    low_eq = min(low_eq, eq_krw)
    tag = "W" if t["net_eq"] > 0 else "L"
    print(f"  {i+1:3d} {t['coin']:>4s} {t['side']:>5s} {t['entry_time'][5:]:>12s} {t['exit_time'][5:]:>12s} "
          f"{t['exit_type']:>12s} {t['hold']:>4d} {t['net_eq']*100:>+7.3f}% {pnl_krw:>+12,.0f} {eq_krw:>14,.0f} [{tag}]")

final = SEED * eq
profit = final - SEED
pnls = [t["net_eq"] for t in all_trades]
wins = sum(1 for p in pnls if p > 0)
ea = np.cumsum(pnls); pk = np.maximum.accumulate(ea); mdd = abs(np.min(ea-pk))
costs = sum(t["cost_eq"] for t in all_trades)

print()
print("="*75)
print(f"  FINAL RESULTS")
print("="*75)
print(f"  Initial equity:  {SEED:>14,} KRW")
print(f"  Final equity:    {final:>14,.0f} KRW")
print(f"  Profit/Loss:     {profit:>+14,.0f} KRW ({(eq-1)*100:+.2f}%)")
print(f"  Peak equity:     {peak_eq:>14,.0f} KRW")
print(f"  Lowest equity:   {low_eq:>14,.0f} KRW")
print()
print(f"  Total trades:    {len(all_trades)}")
print(f"  Win / Loss:      {wins}W / {len(all_trades)-wins}L ({wins/len(all_trades)*100:.1f}%)")
print(f"  Avg PnL/trade:   {np.mean(pnls)*100:+.3f}%")
print(f"  Max drawdown:    {mdd*100:.2f}%")
print(f"  Total cost:      {costs*100:.3f}%")
print()

for coin in COINS:
    ct = [t for t in all_trades if t["coin"]==coin]
    cp = [t["net_eq"] for t in ct]
    if not cp: continue
    cw = sum(1 for p in cp if p > 0)
    print(f"  {coin}: {len(ct)} trades ({cw}W/{len(ct)-cw}L), "
          f"PnL {sum(cp)*SEED:+,.0f} KRW ({sum(cp)*100:+.2f}%), "
          f"avg {np.mean(cp)*100:+.3f}%/trade")

print()
print(f"  Annualized (est): {(eq-1)/8*52*100:+.1f}%")
print("="*75)
