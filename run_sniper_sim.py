#!/usr/bin/env python3
"""4h Direction + 5m Sniper Entry Simulator.

4h 모델이 방향을 잡으면, 5분봉에서 pullback을 기다려 진입.
SL은 local extreme (타이트), TP는 4h ATR×1.0 (빠른 도달).

Usage:
    python run_sniper_sim.py --days 60
"""

from __future__ import annotations

import argparse, json, logging, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.crawlers.crypto_ohlcv import add_technical_indicators, _add_decomposition
from src.data.crawlers.signal_features import add_signal_features
from src.utils.feature_policy import is_excluded_feature
from src.execution.live_predictor import train_combo, predict_2stage
from src.execution.sniper_entry import detect_pullback_entry, simulate_sniper_trade

COINS = ["DOT", "ADA", "XRP", "SOL", "LINK", "BTC", "ETH"]
LOG_DIR = ROOT / "data" / "reports" / "sniper_sim"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler()])
logger = logging.getLogger("sniper")


def fetch(coins, days):
    import yfinance as yf
    tm = {"BTC":"BTC-USD","ETH":"ETH-USD","DOT":"DOT-USD","ADA":"ADA-USD",
          "XRP":"XRP-USD","SOL":"SOL-USD","LINK":"LINK-USD"}
    result = {}
    for coin in coins:
        sym = tm.get(coin)
        if not sym: continue
        try:
            # 4h 봉 (모델 학습 + 방향)
            df_1h = yf.Ticker(sym).history(period=f"{days}d", interval="1h")
            if df_1h.empty: continue
            df_1h.columns = [c.lower() for c in df_1h.columns]
            df_1h = df_1h[["open","high","low","close","volume"]]
            df_1h.index = df_1h.index.tz_convert("UTC") if df_1h.index.tz else df_1h.index.tz_localize("UTC")
            df_4h = df_1h.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()

            # 5분봉 (진입 타이밍) — yfinance 최대 60일
            df_5m = yf.Ticker(sym).history(period=f"{min(days,59)}d", interval="5m")
            if df_5m.empty: continue
            df_5m.columns = [c.lower() for c in df_5m.columns]
            df_5m = df_5m[["open","high","low","close","volume"]]
            df_5m.index = df_5m.index.tz_convert("UTC") if df_5m.index.tz else df_5m.index.tz_localize("UTC")

            result[coin] = {"4h": df_4h, "5m": df_5m}
            logger.info(f"[Data] {coin}: 4h={len(df_4h)} 5m={len(df_5m)}")
        except Exception as e:
            logger.error(f"[Data] {coin}: {e}")
    return result


def add_feat(df):
    df = df.copy()
    df = add_technical_indicators(df)
    df = _add_decomposition(df, period=42)
    try:
        df = add_signal_features(df, verbose=False)
    except: pass
    df.ffill(inplace=True)
    df.replace([np.inf,-np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    cols = [c for c in df.columns if is_excluded_feature(c)]
    if cols: df.drop(columns=cols, inplace=True, errors="ignore")
    return df


def run(days=60, equity=10000.0):
    t0 = time.time()

    with open(ROOT / "config" / "frozen_params_v4_3.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    common = cfg["common"]

    logger.info(f"{'='*60}")
    logger.info(f"  4H DIRECTION + 5M SNIPER ENTRY")
    logger.info(f"  Equity: {equity} | Days: {days}")
    logger.info(f"  4h: k_u={common['k_upper']} k_l={common['k_lower']}")
    logger.info(f"  Sniper: SL=local extreme (<0.5%), TP=ATR×1.0")
    logger.info(f"{'='*60}")

    # Fetch
    data = fetch(COINS, days)

    # Feature + Train 4h models
    logger.info("[1] Training 4h direction models...")
    models = {}
    for coin in COINS:
        if coin not in data: continue
        df4h = add_feat(data[coin]["4h"])
        n = len(df4h)
        te = int(n * 0.7)
        if te < 100: continue
        coin_cfg = cfg["coins"].get(coin, {})
        coin_cfg.update({"max_features": 120, "n_estimators": 300,
                         "max_depth_tree": 8, "min_child_samples": 10})
        combo = train_combo(df4h.iloc[:te].copy(), coin, common, coin_cfg)
        if combo:
            models[coin] = {"combo": combo, "df4h": df4h, "train_end": te}
            logger.info(f"  {coin}: CV={combo.cv_score:.4f}")

    # Simulation
    logger.info("[2] Simulating sniper entries...")
    results = {}

    for coin in COINS:
        if coin not in models or coin not in data: continue

        m = models[coin]
        df4h = m["df4h"]
        combo = m["combo"]
        te = m["train_end"]
        df5m = data[coin]["5m"]
        s1_th = cfg["coins"].get(coin, {}).get("stage1_threshold", 0.45)

        coin_equity = equity
        trades = []

        # 4h bar 단위로 순회 (test period)
        for bar_idx in range(te, len(df4h)):
            # 4h 모델 예측
            pred = predict_2stage(combo, df4h.iloc[:bar_idx+1], s1_th)
            if pred.side == "HOLD":
                continue

            # 이 4h bar의 시간 범위
            bar_time = df4h.index[bar_idx]
            bar_end = bar_time + pd.Timedelta(hours=4)

            # 4h ATR
            atr_4h = df4h["atr_14"].iloc[bar_idx] if "atr_14" in df4h.columns else 0
            if np.isnan(atr_4h) or atr_4h < 1e-10:
                atr_4h = df4h["close"].iloc[bar_idx] * 0.015

            # 이 4h 구간의 5분봉 데이터
            mask = (df5m.index >= bar_time) & (df5m.index < bar_end)
            window_5m = df5m[mask]
            if len(window_5m) < 20:
                continue

            # 5분봉에서 pullback 감지 (10분 간격으로 스캔)
            entry_found = False
            for scan_idx in range(20, len(window_5m), 2):
                scan_slice = window_5m.iloc[:scan_idx]
                setup = detect_pullback_entry(
                    scan_slice, pred.side, atr_4h,
                    df4h["close"].iloc[bar_idx],
                    max_sl_pct=0.005,   # SL 최대 0.5%
                    min_rr=1.5,
                )
                if setup is None:
                    continue

                setup.coin = coin

                # 진입 후 시뮬레이션 (남은 5분봉)
                remaining = window_5m.iloc[scan_idx:]
                if len(remaining) < 3:
                    continue

                result = simulate_sniper_trade(
                    remaining, setup,
                    max_bars=30,  # 최대 150분 (30 × 5m)
                    cost_pct=0.0015,
                )

                # Equity 계산
                conf = pred.confidence
                rf = 0.010 if conf > 0.65 else (0.007 if conf > 0.50 else 0.005)
                sl_dist = setup.sl_pct + 1e-10
                coin_equity += result["pnl"] * coin_equity * rf / sl_dist

                trades.append({
                    "bar": bar_idx, "side": pred.side,
                    "pnl": round(result["pnl"], 6),
                    "exit": result["exit"], "bars_5m": result["bars"],
                    "sl_pct": round(setup.sl_pct * 100, 3),
                    "tp_pct": round(setup.tp_pct * 100, 3),
                    "rr": round(setup.rr, 2),
                    "trigger": setup.trigger,
                })
                entry_found = True
                break  # 1 entry per 4h bar

        results[coin] = {"trades": trades, "equity": coin_equity}

    # Report
    logger.info(f"\n{'='*60}")
    logger.info(f"  SNIPER RESULTS")
    logger.info(f"{'='*60}")

    total_t = 0; total_pnl = 0; equities = {}
    for coin in COINS:
        r = results.get(coin)
        if not r or not r["trades"]: continue
        trades = r["trades"]
        equities[coin] = r["equity"]
        pnls = [t["pnl"] for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        total_t += len(trades); total_pnl += sum(pnls)
        cum = np.cumsum(pnls)
        mdd = np.max(np.maximum.accumulate(cum) - cum) if len(cum) > 0 else 0
        exits = {}
        for t in trades: exits[t["exit"]] = exits.get(t["exit"], 0) + 1
        avg_sl = np.mean([t["sl_pct"] for t in trades])
        avg_rr = np.mean([t["rr"] for t in trades])

        logger.info(
            f"  {coin:>5s}: {len(trades):>3d} trades | WR={wins/len(trades):.0%} | "
            f"avg={np.mean(pnls):+.3%} | total={sum(pnls):+.2%} | "
            f"MDD={mdd:.2%} | avg_SL={avg_sl:.2f}% R:R={avg_rr:.1f} | "
            f"TP:{exits.get('TP',0)} SL:{exits.get('SL',0)} TTL:{exits.get('TTL',0)}"
        )

    port = equity + sum(ce - equity for ce in equities.values())
    logger.info(f"  {'TOTAL':>5s}: {total_t:>3d} trades | PnL: {total_pnl:+.2%}")
    logger.info(f"  Portfolio: ${port:,.2f} | Elapsed: {time.time()-t0:.1f}s")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--equity", type=float, default=10000.0)
    a = p.parse_args()
    run(days=a.days, equity=a.equity)
