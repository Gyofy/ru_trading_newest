"""A/B Test: v3.4 (baseline) vs v3.5 (sample weighting + 1h barrier).

Changes in v3.5:
  A. Sample weighting: uniqueness + recency (no model structure change)
  B. 1h-resolution barrier simulator (4x granularity for TP/SL order)

Both use frozen_params_v3_4.yaml -- NO parameter changes.
"""

import sys
sys.path.insert(0, "C:/Users/RJ/Desktop/CLAUDE_CRYPTO_AGENT")
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import json, yaml, warnings, time
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import asdict

warnings.filterwarnings("ignore")

from src.data.crawlers.crypto_ohlcv import fetch_all_top10, fetch_ohlcv, TOP10_YAHOO
from src.data.crawlers.macro_commodity_crawler import crawl_all_macro_data
from src.models.masking_loop import create_labels_triple_barrier, LABEL_MAP, HORIZONS
from src.models.enhanced_ensemble import EnhancedEnsemble
from src.execution.cost_model import CostModel, FeeSchedule, FundingConfig, MissFillConfig, ExitType
from src.evaluation.hires_barrier import simulate_hires, HiResTrade
from src.utils.config import bar_minutes as cfg_bar_minutes
from src.utils.sample_weight import compute_sample_weights
from sklearn.feature_selection import mutual_info_classif

REPORT_DIR = Path("data/reports/ab_test_v3_5")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open("config/frozen_params_v3_4.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

COMMON = CFG["common"]
COINS = CFG["coins"]
CC = CFG["cost_model"]
EXK = CFG["excluded_feature_keywords"]
BM = cfg_bar_minutes()
MH = COMMON["max_horizon"]
RF = COMMON["risk_frac"]
N_JOBS = 6

CM = CostModel(
    fee_schedule=FeeSchedule(maker_fee=CC["maker_fee"], taker_fee=CC["taker_fee"],
        slippage_entry=CC["slippage_entry"], slippage_exit_limit=CC["slippage_exit_limit"],
        slippage_exit_market=CC["slippage_exit_market"]),
    funding_config=FundingConfig(interval_hours=CC["funding_interval_hours"],
        default_rate=CC["funding_default_rate"]),
    miss_fill_config=MissFillConfig(reject_prob=CC["miss_fill_reject_prob"],
        missed_ev_pct=CC["miss_fill_missed_ev"]))

PURGE_BARS = MH * 2
EMBARGO_BARS = 6
OOS_DAYS = 56  # 8 weeks


def ck(coin, key):
    return COINS[coin].get(f"{key}_override", COMMON[key])


def isex(c):
    return any(kw in c.lower() for kw in EXK)


def get_regime(df, i):
    if i < 42: return "UNKNOWN"
    c = df["close"].values[max(0, i-41):i+1]
    ef = pd.Series(c).ewm(span=10).mean().iloc[-1]
    es = pd.Series(c).ewm(span=30).mean().iloc[-1]
    rs = np.std(np.diff(c)/c[:-1]) if len(c)>1 else 0
    if ef > es*1.005: return "TREND_UP"
    elif ef < es*0.995: return "TREND_DOWN"
    else: return "RANGE_HIGH" if rs > 0.02 else "RANGE_LOW"


def metrics(trades, label):
    n = len(trades)
    if n == 0:
        return {"label": label, "trades": 0}
    pnls = [t.net_pnl_eq for t in trades]
    tp = sum(1 for t in trades if t.exit_type == "take_profit")
    eq = np.cumsum(pnls)
    pk = np.maximum.accumulate(eq)
    mdd = abs(np.min(eq - pk))
    tc = sum(t.cost_eq for t in trades)
    tg = sum(abs(t.gross_pnl_eq) for t in trades)
    return {
        "label": label, "trades": n,
        "win_rate": round(tp/n, 4),
        "avg_pnl": round(np.mean(pnls), 6),
        "total_pnl": round(np.sum(pnls), 6),
        "mdd": round(mdd, 6),
        "cost_share": round(tc/(tg+1e-10), 4),
        "sharpe": round(np.mean(pnls)/(np.std(pnls)+1e-10) * np.sqrt(252*6/max(np.mean([t.holding_bars_4h for t in trades]),1)), 2) if n > 1 else 0,
    }


def main():
    start = datetime.now()
    print(f"\n{'='*70}")
    print(f"  A/B TEST: v3.4 (baseline) vs v3.5 (weighting + 1h barrier)")
    print(f"  {start.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")

    # Fetch both 4h and 1h data
    print("\n  Fetching 4h data...")
    ohlcv_4h = fetch_all_top10("365d", "1h")  # fetches 1h, resamples to 4h internally

    print("  Fetching 1h data (for hi-res barrier)...")
    ohlcv_1h = {}
    for ticker, yahoo_sym in TOP10_YAHOO.items():
        if ticker in [c for c in COINS]:
            df_1h = fetch_ohlcv(ticker, yahoo_sym, "365d", "1h")
            if not df_1h.empty:
                ohlcv_1h[ticker] = df_1h
                print(f"    {ticker}: {len(df_1h)} 1h bars")

    # Macro
    first = list(ohlcv_4h.values())[0]
    macro = crawl_all_macro_data(first.index)
    ma = macro.get("aligned", pd.DataFrame())

    data_4h = {}
    data_1h = {}
    for coin in COINS:
        if coin not in ohlcv_4h:
            continue
        df4 = ohlcv_4h[coin].copy()
        if len(ma) > 0:
            for col in ma.columns:
                df4[col] = ma[col].reindex(df4.index).ffill().bfill().fillna(0)
        data_4h[coin] = df4

        if coin in ohlcv_1h:
            data_1h[coin] = ohlcv_1h[coin]

    results = {}

    for coin in COINS:
        if coin not in data_4h:
            continue

        print(f"\n{'='*50}")
        print(f"  {coin}")
        print(f"{'='*50}")

        df4 = data_4h[coin]
        idx = df4.index
        end = idx[-1]
        oos_start = end - timedelta(days=OOS_DAYS)
        purge_td = timedelta(hours=(PURGE_BARS + EMBARGO_BARS) * (BM // 60))
        train_end = oos_start - purge_td

        train_df = df4[idx <= train_end]
        oos_df = df4[idx >= oos_start]

        # 1h OOS
        oos_1h = None
        if coin in data_1h:
            h1_idx = data_1h[coin].index
            oos_1h = data_1h[coin][h1_idx >= oos_start]
            print(f"  1h OOS bars: {len(oos_1h)}")

        print(f"  Train: {len(train_df)} bars, OOS: {len(oos_df)} bars (4h)")

        params = COINS[coin]
        h = HORIZONS[-1]
        hl = f"label_{h*BM}min"
        ku, kl = ck(coin, "k_upper"), ck(coin, "k_lower")

        labeled = create_labels_triple_barrier(
            train_df.copy(), h, k_upper_override=ku, k_lower_override=kl, verbose=False)
        if hl not in labeled.columns:
            continue

        exclude = {"label", "future_return", "open", "high", "low", "close", "volume"}
        for hh in HORIZONS:
            exclude.add(f"label_{hh*BM}min"); exclude.add(f"return_{hh*BM}min")
        lk = ["future", "target", "label_", "return_", "fwd_", "forward_"]
        fcols = [c for c in labeled.columns
                 if c not in exclude and not any(k in c.lower() for k in lk)
                 and not isex(c) and labeled[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]

        mf = params["max_features"]
        clean = labeled.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        X_all = clean[fcols].fillna(0).values
        y_all = clean[hl].fillna(1).values.astype(int)

        if len(fcols) > mf:
            mi = mutual_info_classif(X_all[:min(2000, len(X_all))], y_all[:min(2000, len(X_all))],
                                     discrete_features=False, random_state=42, n_neighbors=5)
            top = np.argsort(mi)[-mf:]
            fcols = [fcols[i] for i in sorted(top)]
            X_all = clean[fcols].fillna(0).values

        # ==================== ARM A: Baseline (v3.4, no weighting, 4h barrier) ====================
        print(f"\n  --- ARM A: v3.4 baseline ---")
        t0 = time.time()

        y_s1 = (y_all != LABEL_MAP["HOLD"]).astype(int)
        s1c = np.bincount(y_s1, minlength=2)
        s1w_base = np.where(s1c > 0, len(y_s1) / (2 * s1c + 1e-10), 1.0)[y_s1]

        s1_a = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
        s1_a.fit(X_all, y_s1, sample_weight=s1w_base)

        s2_a = None
        tm = y_all != LABEL_MAP["HOLD"]
        if tm.sum() >= 30:
            y_s2 = (y_all[tm] == LABEL_MAP["UP"]).astype(int)
            if len(np.unique(y_s2)) >= 2:
                s2c = np.bincount(y_s2, minlength=2)
                s2w_base = np.where(s2c > 0, len(y_s2) / (2 * s2c + 1e-10), 1.0)[y_s2]
                s2_a = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
                s2_a.fit(X_all[tm], y_s2, sample_weight=s2w_base)

        print(f"  Trained in {time.time()-t0:.1f}s")

        # Predict OOS
        oc = oos_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        Xo = oc[fcols].fillna(0).values
        s1p_a = s1_a.predict_proba(Xo)
        s1pred_a = (s1p_a[:, 1] >= params["stage1_threshold"]).astype(int)
        s2pred_a = np.zeros(len(Xo), dtype=int)
        if s2_a is not None:
            s2p_a = s2_a.predict_proba(Xo)
            s2pred_a = np.argmax(s2p_a, axis=1)

        blocked = COINS[coin].get("blocked_regimes_override", CFG["blocked_regimes"])

        # ARM A: 4h barrier (current)
        trades_a_4h = simulate_hires(
            oos_df, oos_df, s1pred_a, s2pred_a,  # signal=4h, hires=4h (same = current behavior)
            k_upper=ku, k_lower=kl, max_hold_4h=MH,
            risk_frac=RF, cost_model=CM, bar_minutes=BM,
            coin=coin, blocked_regimes=blocked,
            regime_fn=get_regime)
        ma_4h = metrics(trades_a_4h, f"{coin}_A_4h")
        print(f"  A(4h): {ma_4h['trades']}T avg={ma_4h.get('avg_pnl',0):+.4%} "
              f"total={ma_4h.get('total_pnl',0):+.4%} mdd={ma_4h.get('mdd',0):.4%}")

        # ARM A: 1h barrier (improvement B only)
        trades_a_1h = []
        if oos_1h is not None and len(oos_1h) > 0:
            trades_a_1h = simulate_hires(
                oos_df, oos_1h, s1pred_a, s2pred_a,
                k_upper=ku, k_lower=kl, max_hold_4h=MH,
                risk_frac=RF, cost_model=CM, bar_minutes=BM,
                coin=coin, blocked_regimes=blocked,
                regime_fn=get_regime)
            ma_1h = metrics(trades_a_1h, f"{coin}_A_1h")
            print(f"  A(1h): {ma_1h['trades']}T avg={ma_1h.get('avg_pnl',0):+.4%} "
                  f"total={ma_1h.get('total_pnl',0):+.4%} mdd={ma_1h.get('mdd',0):.4%}")

        # ==================== ARM B: v3.5 (weighting + 1h barrier) ====================
        print(f"\n  --- ARM B: v3.5 (weighting + 1h) ---")
        t0 = time.time()

        # Compute sample weights
        sw = compute_sample_weights(
            clean, y_all, horizon=MH, recency_halflife=45.0)

        # Combine with class balance
        s1w_v35 = s1w_base * sw
        s1_b = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
        s1_b.fit(X_all, y_s1, sample_weight=s1w_v35)

        s2_b = None
        if tm.sum() >= 30 and s2_a is not None:
            s2w_v35 = s2w_base * sw[tm]
            s2_b = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
            s2_b.fit(X_all[tm], y_s2, sample_weight=s2w_v35)

        print(f"  Trained in {time.time()-t0:.1f}s")

        # Predict OOS
        s1p_b = s1_b.predict_proba(Xo)
        s1pred_b = (s1p_b[:, 1] >= params["stage1_threshold"]).astype(int)
        s2pred_b = np.zeros(len(Xo), dtype=int)
        if s2_b is not None:
            s2p_b = s2_b.predict_proba(Xo)
            s2pred_b = np.argmax(s2p_b, axis=1)

        # ARM B: 1h barrier + weighting
        trades_b = []
        if oos_1h is not None and len(oos_1h) > 0:
            trades_b = simulate_hires(
                oos_df, oos_1h, s1pred_b, s2pred_b,
                k_upper=ku, k_lower=kl, max_hold_4h=MH,
                risk_frac=RF, cost_model=CM, bar_minutes=BM,
                coin=coin, blocked_regimes=blocked,
                regime_fn=get_regime)
        else:
            trades_b = simulate_hires(
                oos_df, oos_df, s1pred_b, s2pred_b,
                k_upper=ku, k_lower=kl, max_hold_4h=MH,
                risk_frac=RF, cost_model=CM, bar_minutes=BM,
                coin=coin, blocked_regimes=blocked,
                regime_fn=get_regime)

        mb = metrics(trades_b, f"{coin}_B")
        print(f"  B(w+1h): {mb['trades']}T avg={mb.get('avg_pnl',0):+.4%} "
              f"total={mb.get('total_pnl',0):+.4%} mdd={mb.get('mdd',0):.4%}")

        # Signal difference
        diff_s1 = (s1pred_a != s1pred_b).sum()
        diff_s2 = (s2pred_a != s2pred_b).sum()
        print(f"  Signal diff: S1 {diff_s1}/{len(s1pred_a)} changed, "
              f"S2 {diff_s2}/{len(s2pred_a)} changed")

        results[coin] = {
            "A_4h": ma_4h,
            "A_1h": metrics(trades_a_1h, f"{coin}_A_1h") if trades_a_1h else None,
            "B_w1h": mb,
            "signal_diff_s1": int(diff_s1),
            "signal_diff_s2": int(diff_s2),
        }

    # Summary
    print(f"\n{'='*70}")
    print(f"  A/B TEST SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Coin':>5s} | {'ARM':>8s} | {'Trades':>6s} | {'Avg PnL':>10s} | {'Total':>10s} | "
          f"{'MDD':>8s} | {'Sharpe':>7s}")
    print(f"  {'-'*70}")

    for coin in COINS:
        r = results.get(coin, {})
        for arm_key, arm_label in [("A_4h", "A(4h)"), ("A_1h", "A(1h)"), ("B_w1h", "B(w+1h)")]:
            m = r.get(arm_key)
            if m and m.get("trades", 0) > 0:
                print(f"  {coin:>5s} | {arm_label:>8s} | {m['trades']:>6d} | {m['avg_pnl']:>+10.4%} | "
                      f"{m['total_pnl']:>+10.4%} | {m['mdd']:>8.4%} | {m.get('sharpe',0):>7.1f}")

    with open(REPORT_DIR / "ab_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = (datetime.now() - start).total_seconds() / 60
    print(f"\n  Completed in {elapsed:.1f} min")
    print(f"  Report: {REPORT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
