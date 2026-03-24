"""Mega Search v3 -- Fresh model search on clean pipeline.

Changes from v2:
  - Symmetric labeling (both barriers use k_upper=3.0)
  - No bfill anywhere (ffill + fillna(0))
  - Rolling VWAP (not cumulative)
  - center=False in decomposition
  - max_horizon=12 (was 18)
  - Train/test split: 70/30 walk-forward (no look-ahead)
  - LONG + SHORT enabled

Tests 10 model configs x 5 coins x 4 OOS windows.
"""

import sys, json, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
warnings.filterwarnings("ignore")

from src.data.crawlers.crypto_ohlcv import add_technical_indicators, _add_decomposition, _add_cross_asset_correlation
from src.data.crawlers.signal_features import add_signal_features
from src.data.crawlers.microstructure_rollup import add_microstructure_rollup
from src.utils.feature_policy import is_excluded_feature
from src.execution.live_predictor import train_combo, predict_2stage
from src.execution.risk_engine import RiskEngine

COINS = ["DOT", "ADA", "XRP", "SOL", "LINK"]
REPORT_DIR = PROJECT / "experiments" / "mega_search_v3_results"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = REPORT_DIR / "search_log.jsonl"

with open(PROJECT / "config" / "frozen_params_v4_3.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

COMMON = CFG["common"]
K_UPPER = COMMON["k_upper"]
K_LOWER = COMMON["k_lower"]
MAX_H = COMMON["max_horizon"]

# Model configs to test
CONFIGS = [
    {"name": "et",         "combo": "et",       "weights": {"et": 1.0}},
    {"name": "cb",         "combo": "cb",       "weights": {"cb": 1.0}},
    {"name": "xgb",        "combo": "xgb",      "weights": {"xgb": 1.0}},
    {"name": "et+cb_50",   "combo": "et+cb",    "weights": {"et": 0.5, "cb": 0.5}},
    {"name": "et+cb_70",   "combo": "et+cb",    "weights": {"et": 0.7, "cb": 0.3}},
    {"name": "et+xgb_50",  "combo": "et+xgb",   "weights": {"et": 0.5, "xgb": 0.5}},
    {"name": "et+xgb_70",  "combo": "et+xgb",   "weights": {"et": 0.7, "xgb": 0.3}},
    {"name": "et+tabm_70", "combo": "et+tabm",  "weights": {"et": 0.7, "tabm": 0.3}},
    {"name": "cb+xgb_50",  "combo": "cb+xgb",   "weights": {"cb": 0.5, "xgb": 0.5}},
    {"name": "et+cb+xgb",  "combo": "et+cb+xgb","weights": {"et": 0.4, "cb": 0.3, "xgb": 0.3}},
]

# OOS windows (walk-forward)
OOS_WINDOWS = [
    {"train_end_pct": 0.50, "test_pct": 0.125},  # train 0-50%, test 50-62.5%
    {"train_end_pct": 0.55, "test_pct": 0.125},  # train 0-55%, test 55-67.5%
    {"train_end_pct": 0.60, "test_pct": 0.125},  # train 0-60%, test 60-72.5%
    {"train_end_pct": 0.70, "test_pct": 0.125},  # train 0-70%, test 70-82.5%
]


def fetch_data() -> dict:
    import yfinance as yf
    tm = {"BTC":"BTC-USD","ETH":"ETH-USD","DOT":"DOT-USD",
          "ADA":"ADA-USD","XRP":"XRP-USD","SOL":"SOL-USD","LINK":"LINK-USD"}
    raw = {}
    for coin in set(COINS + ["BTC", "ETH"]):
        df = yf.Ticker(tm[coin]).history(period="180d", interval="1h")
        if df.empty: continue
        df.columns = [c.lower() for c in df.columns]
        df = df[["open","high","low","close","volume"]]
        df.index = df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
        raw[coin] = df.resample("4h").agg(
            {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
        ).dropna()
        print(f"  {coin}: {len(raw[coin])} bars")
    return raw


def compute_features(raw: dict) -> dict:
    featured = {}
    for coin, df in raw.items():
        if len(df) < 100: continue
        df = df.copy()
        df = add_technical_indicators(df)
        df = _add_decomposition(df, period=42)
        df = add_signal_features(df, verbose=False)
        df = add_microstructure_rollup(df, verbose=False)
        df.ffill(inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        cols_drop = [c for c in df.columns if is_excluded_feature(c)]
        if cols_drop:
            df.drop(columns=cols_drop, inplace=True, errors="ignore")
        featured[coin] = df
    # Cross-asset correlation
    if len(featured) >= 2:
        for coin in COINS:
            if coin in featured:
                try:
                    featured[coin] = _add_cross_asset_correlation(featured, coin, window=20)
                except Exception:
                    pass
    return featured


def simulate_trades(df, combo, s1_thresh, test_start, test_end):
    """Walk-forward trade simulation on test period."""
    trades = []
    i = test_start
    while i < min(test_end, len(df) - MAX_H):
        pred = predict_2stage(combo, df.iloc[:i+1], s1_thresh)
        if pred.side == "HOLD":
            i += 1
            continue

        entry = df.iloc[i]["close"]
        atr = df["atr_14"].iloc[i] if "atr_14" in df.columns else entry * 0.01
        if np.isnan(atr) or atr < 1e-10:
            atr = entry * 0.01
        sl, tp = RiskEngine.compute_barriers(entry, atr, pred.side, K_UPPER, K_LOWER, 0.002)

        exit_p = entry
        bars = 0
        for j in range(1, MAX_H + 1):
            if i + j >= len(df):
                break
            bar = df.iloc[i + j]
            bars = j
            if pred.side == "BUY":
                if bar["high"] >= tp: exit_p = tp; break
                if bar["low"] <= sl: exit_p = sl; break
            else:
                if bar["low"] <= tp: exit_p = tp; break
                if bar["high"] >= sl: exit_p = sl; break
        else:
            exit_p = df.iloc[min(i + MAX_H, len(df) - 1)]["close"]

        if pred.side == "BUY":
            pnl = (exit_p - entry) / entry
        else:
            pnl = (entry - exit_p) / entry
        pnl -= 0.002  # cost

        trades.append({"pnl": pnl, "bars": bars, "side": pred.side})
        i += max(1, bars)

    return trades


def main():
    print(f"\n{'='*70}")
    print(f"  MEGA SEARCH v3 -- Clean Pipeline Fresh Search")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Configs: {len(CONFIGS)} | Coins: {COINS} | Windows: {len(OOS_WINDOWS)}")
    print(f"  max_horizon={MAX_H} | k_upper={K_UPPER} | k_lower={K_LOWER}")
    print(f"{'='*70}\n")

    # Fetch & feature
    print("[1] Fetching 180d data...")
    raw = fetch_data()
    print("[2] Computing features...")
    featured = compute_features(raw)
    print(f"    Ready: {[c for c in COINS if c in featured]}\n")

    best = {}
    total = 0

    for cfg_idx, model_cfg in enumerate(CONFIGS):
        cfg_name = model_cfg["name"]
        print(f"\n[{cfg_idx+1}/{len(CONFIGS)}] {cfg_name}")

        for coin in COINS:
            if coin not in featured:
                continue
            df = featured[coin]
            n = len(df)
            coin_cfg = {
                "model_combo": model_cfg["combo"],
                "model_weights": model_cfg["weights"],
                "stage1_threshold": CFG["coins"].get(coin, {}).get("stage1_threshold", 0.45),
                "max_features": 120,
                "n_estimators": 300,
                "max_depth_tree": 8,
                "min_child_samples": 10,
            }
            s1_thresh = coin_cfg["stage1_threshold"]

            window_results = []
            t0 = time.time()

            for w in OOS_WINDOWS:
                train_end = int(n * w["train_end_pct"])
                test_end = int(n * (w["train_end_pct"] + w["test_pct"]))
                if train_end < 200 or test_end > n:
                    continue

                # Train on train portion only
                train_df = df.iloc[:train_end]
                combo = train_combo(train_df.copy(), coin, COMMON, coin_cfg)
                if combo is None:
                    continue

                # Test on OOS portion
                trades = simulate_trades(df, combo, s1_thresh, train_end, test_end)
                if not trades:
                    continue

                pnls = [t["pnl"] for t in trades]
                window_results.append({
                    "n_trades": len(trades),
                    "avg_pnl": np.mean(pnls),
                    "total_pnl": np.sum(pnls),
                    "win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
                    "max_dd": np.max(np.maximum.accumulate(np.cumsum(pnls)) - np.cumsum(pnls)),
                })

            if not window_results:
                continue

            result = {
                "n_windows": len(window_results),
                "avg_pnl_mean": round(np.mean([r["avg_pnl"] for r in window_results]), 6),
                "total_pnl_sum": round(np.sum([r["total_pnl"] for r in window_results]), 6),
                "avg_trades": round(np.mean([r["n_trades"] for r in window_results]), 1),
                "avg_wr": round(np.mean([r["win_rate"] for r in window_results]), 4),
                "avg_dd": round(np.mean([r["max_dd"] for r in window_results]), 6),
            }

            elapsed = time.time() - t0
            is_best = False
            prev = best.get(coin, {}).get("avg_pnl_mean", -999)
            if result["avg_pnl_mean"] > prev:
                best[coin] = {"config": cfg_name, **result}
                is_best = True

            total += 1
            star = " ***BEST***" if is_best else ""
            print(f"  {coin}: avg={result['avg_pnl_mean']:+.4%} wr={result['avg_wr']:.0%} "
                  f"trades={result['avg_trades']:.0f} dd={result['avg_dd']:.2%} [{elapsed:.1f}s]{star}")

            entry = {"coin": coin, "config": cfg_name, **result,
                     "ts": datetime.now().isoformat()}
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    # Final report
    print(f"\n{'='*70}")
    print(f"  MEGA SEARCH v3 COMPLETE ({total} evaluations)")
    print(f"{'='*70}")
    for coin in COINS:
        b = best.get(coin)
        if b:
            print(f"  {coin:>5s}: {b['config']:>15s} | avg={b['avg_pnl_mean']:+.4%} "
                  f"wr={b['avg_wr']:.0%} trades={b['avg_trades']:.0f} dd={b['avg_dd']:.2%}")

    with open(REPORT_DIR / "best_per_coin.json", "w") as f:
        json.dump(best, f, indent=2, default=str)
    print(f"\n  Results: {REPORT_DIR}")


if __name__ == "__main__":
    main()
