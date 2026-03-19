#!/usr/bin/env python3
"""Multi-Timeframe Paper Simulation.

5분봉 → 방향 결정, 1분봉 → 진입 타이밍.
SL/TP는 5분봉 ATR 기준 (1분봉 노이즈 필터).

Usage:
    python run_mtf_sim.py --days 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.crawlers.crypto_ohlcv import add_technical_indicators, _add_decomposition
from src.data.crawlers.signal_features import add_signal_features
from src.data.crawlers.microstructure_rollup import add_microstructure_rollup
from src.utils.feature_policy import is_excluded_feature
from src.execution.live_predictor import train_combo, predict_2stage
from src.execution.mtf_engine import (
    MTFDirectionManager, DirectionSignal, TimingSignal,
    compute_mtf_barriers, estimate_leverage_risk,
)

COINS = ["TAO", "DOT", "ADA", "XRP", "SOL", "LINK", "BTC", "ETH"]
LOG_DIR = PROJECT_ROOT / "data" / "reports" / "mtf_sim"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "mtf_sim.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("mtf_sim")


def fetch_multi_tf(coins: list, days: int = 30) -> dict:
    """Fetch 5m and 1m data."""
    import yfinance as yf
    tm = {"BTC":"BTC-USD","ETH":"ETH-USD","DOT":"DOT-USD","ADA":"ADA-USD",
          "XRP":"XRP-USD","SOL":"SOL-USD","LINK":"LINK-USD"}

    result = {}
    for coin in coins:
        # TAO: local CSV (5m not available via yfinance)
        if coin == "TAO":
            tao_path = PROJECT_ROOT / "data" / "microstructure" / "TAO_4h_ohlcv.csv"
            if tao_path.exists():
                df = pd.read_csv(tao_path, index_col=0, parse_dates=True)
                df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index
                result[coin] = {"5m": df, "1m": None}  # TAO는 4h만 가능
                logger.info(f"[Data] TAO: {len(df)} bars (4h, local)")
            continue

        sym = tm.get(coin, f"{coin}-USD")
        try:
            # 5분봉
            df5 = yf.Ticker(sym).history(period=f"{days}d", interval="5m")
            if df5.empty:
                continue
            df5.columns = [c.lower() for c in df5.columns]
            df5 = df5[["open", "high", "low", "close", "volume"]]
            df5.index = df5.index.tz_convert("UTC") if df5.index.tz else df5.index.tz_localize("UTC")

            result[coin] = {"5m": df5, "1m": None}
            logger.info(f"[Data] {coin}: 5m={len(df5)} bars")
        except Exception as e:
            logger.error(f"[Data] {coin}: {e}")

    return result


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature pipeline (same as bot)."""
    df = df.copy()
    df = add_technical_indicators(df)
    df = _add_decomposition(df, period=42)
    try:
        df = add_signal_features(df, verbose=False)
    except Exception:
        pass
    try:
        df = add_microstructure_rollup(df, verbose=False)
    except Exception:
        pass
    df.ffill(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    cols_drop = [c for c in df.columns if is_excluded_feature(c)]
    if cols_drop:
        df.drop(columns=cols_drop, inplace=True, errors="ignore")
    return df


def run_mtf_sim(days: int = 30, equity: float = 10000.0):
    t0 = time.time()

    with open(PROJECT_ROOT / "config" / "frozen_params_v4_3_mtf.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tf5_cfg = cfg["tf_5m"]
    exec_cfg = cfg["execution"]
    entry_cfg = cfg["entry_rules"]
    sizing_cfg = cfg["sizing_tiers"]
    leverage = exec_cfg["leverage"]

    logger.info(f"{'='*60}")
    logger.info(f"  MTF SIMULATION (5m direction + entry)")
    logger.info(f"  Equity: {equity} | Leverage: {leverage}x | Days: {days}")
    logger.info(f"  SL: {exec_cfg['sl_atr_mult']}x ATR(5m)")
    logger.info(f"  TP: {exec_cfg['tp1_atr_mult']}/{exec_cfg['tp2_atr_mult']}/{exec_cfg['tp3_atr_mult']}x ATR")
    logger.info(f"{'='*60}")

    # Fetch data
    logger.info("[1] Fetching 5m data...")
    raw = fetch_multi_tf(COINS, days=days)
    if not raw:
        logger.error("No data!")
        return

    # Feature engineering on 5m
    logger.info("[2] Computing features...")
    featured = {}
    for coin, data in raw.items():
        df5 = data.get("5m")
        if df5 is None or len(df5) < 200:
            continue
        featured[coin] = add_features(df5)
        logger.info(f"  {coin}: {len(featured[coin])} bars, {len(featured[coin].columns)} features")

    # Train on first 70%
    logger.info("[3] Training models...")
    common_cfg = {
        "k_upper": tf5_cfg["k_upper"],
        "k_lower": tf5_cfg["k_lower"],
        "max_horizon": tf5_cfg["max_horizon"],
        "bar_minutes": tf5_cfg["bar_minutes"],
        "min_barrier_pct": tf5_cfg["min_barrier_pct"],
        "risk_frac": 0.005,
    }
    models = {}
    for coin in COINS:
        if coin not in featured:
            continue
        df = featured[coin]
        n = len(df)
        train_end = int(n * 0.7)
        coin_cfg = cfg["coins"].get(coin, {"model_combo": "et+xgb", "model_weights": {"et": 0.7, "xgb": 0.3}})
        coin_cfg.update({"stage1_threshold": tf5_cfg["stage1_threshold"],
                         "max_features": 120, "n_estimators": 300,
                         "max_depth_tree": 8, "min_child_samples": 10})

        combo = train_combo(df.iloc[:train_end].copy(), coin, common_cfg, coin_cfg)
        if combo:
            models[coin] = {"combo": combo, "train_end": train_end}
            logger.info(f"  {coin}: {combo.model_names} CV={combo.cv_score:.4f}")
    logger.info(f"[3] Trained: {list(models.keys())}")

    # Walk-forward simulation
    logger.info("[4] Running MTF simulation...")
    results = {}
    direction_mgr = MTFDirectionManager(entry_window_bars=entry_cfg["entry_window_5m_bars"])

    for coin in COINS:
        if coin not in models or coin not in featured:
            continue

        df = featured[coin]
        combo = models[coin]["combo"]
        train_end = models[coin]["train_end"]
        n = len(df)
        max_h = tf5_cfg["max_horizon"]
        s1_thresh = tf5_cfg["stage1_threshold"]

        coin_equity = equity
        trades = []
        i = train_end

        while i < n - max_h:
            # 5분봉 예측
            pred = predict_2stage(combo, df.iloc[:i+1], s1_thresh)
            if pred.side == "HOLD":
                i += 1
                continue

            # 5분봉 ATR (SL/TP 기준)
            atr_5m = df["atr_14"].iloc[i] if "atr_14" in df.columns else 0
            if np.isnan(atr_5m) or atr_5m < 1e-10:
                atr_5m = df["close"].iloc[i] * 0.003  # fallback 0.3%

            entry_price = df["close"].iloc[i]

            # 배리어 계산 (5분봉 ATR 기준)
            barriers = compute_mtf_barriers(entry_price, atr_5m, pred.side, exec_cfg)

            # 레버리지 리스크 체크
            risk_info = estimate_leverage_risk(
                barriers["sl_dist_pct"], leverage, sizing_cfg.get("tier_mid", 0.007))

            if not risk_info["safe"]:
                i += 1
                continue

            # Triple barrier exit simulation
            sl = barriers["sl"]
            tp1 = barriers["tp1"]
            tp2 = barriers["tp2"]
            tp3 = barriers["tp3"]

            exit_price = entry_price
            exit_reason = "TIME_STOP"
            bars_held = 0
            price_high = entry_price
            price_low = entry_price
            partial_pnl = 0.0
            remaining_frac = 1.0

            for j in range(1, max_h + 1):
                if i + j >= n:
                    break
                bar = df.iloc[i + j]
                bars_held = j
                price_high = max(price_high, bar["high"])
                price_low = min(price_low, bar["low"])

                # SL check
                if pred.side == "BUY" and bar["low"] <= sl:
                    exit_price = sl
                    exit_reason = "SL_HIT"
                    break
                if pred.side == "SELL" and bar["high"] >= sl:
                    exit_price = sl
                    exit_reason = "SL_HIT"
                    break

                # TP3 (final)
                tp3_hit = (pred.side == "BUY" and bar["high"] >= tp3) or \
                          (pred.side == "SELL" and bar["low"] <= tp3)
                if tp3_hit:
                    exit_price = tp3
                    exit_reason = "TP3_HIT"
                    break

                # TP1 partial (33%)
                tp1_hit = (pred.side == "BUY" and bar["high"] >= tp1) or \
                          (pred.side == "SELL" and bar["low"] <= tp1)
                if tp1_hit and remaining_frac > 0.67:
                    pnl_tp1 = (tp1 - entry_price) / entry_price if pred.side == "BUY" \
                              else (entry_price - tp1) / entry_price
                    partial_pnl += pnl_tp1 * 0.33
                    remaining_frac -= 0.33
                    sl = entry_price  # SL → breakeven

                # TP2 partial (33%)
                tp2_hit = (pred.side == "BUY" and bar["high"] >= tp2) or \
                          (pred.side == "SELL" and bar["low"] <= tp2)
                if tp2_hit and remaining_frac > 0.34 and remaining_frac <= 0.67:
                    pnl_tp2 = (tp2 - entry_price) / entry_price if pred.side == "BUY" \
                              else (entry_price - tp2) / entry_price
                    partial_pnl += pnl_tp2 * 0.33
                    remaining_frac -= 0.33
                    sl = tp1  # SL → TP1
            else:
                exit_price = df.iloc[min(i + max_h, n - 1)]["close"]

            # Final PnL
            if pred.side == "BUY":
                final_pnl = (exit_price - entry_price) / entry_price * remaining_frac
            else:
                final_pnl = (entry_price - exit_price) / entry_price * remaining_frac

            total_pnl = partial_pnl + final_pnl - 0.002  # cost
            mfe = (price_high - entry_price) / entry_price if pred.side == "BUY" \
                  else (entry_price - price_low) / entry_price
            mae = (price_low - entry_price) / entry_price if pred.side == "BUY" \
                  else (entry_price - price_high) / entry_price

            # Equity update
            conf = pred.p_trade * pred.p_direction
            risk_f = sizing_cfg.get("tier_high", 0.010) if conf > 0.65 \
                     else (sizing_cfg.get("tier_mid", 0.007) if conf > 0.50
                           else sizing_cfg.get("tier_low", 0.005))
            stop_dist = barriers["sl_dist_pct"] + 1e-10
            coin_equity += total_pnl * coin_equity * risk_f / stop_dist

            trades.append({
                "bar": i, "side": pred.side, "pnl_pct": round(total_pnl, 6),
                "exit_reason": exit_reason, "bars_held": bars_held,
                "mfe": round(mfe, 6), "mae": round(mae, 6),
                "sl_dist": round(barriers["sl_dist_pct"], 6),
                "partial_pnl": round(partial_pnl, 6),
            })
            i += max(1, bars_held)

        results[coin] = {"trades": trades, "equity": coin_equity}

    # Report
    logger.info(f"\n{'='*60}")
    logger.info(f"  MTF SIMULATION RESULTS (5m TF)")
    logger.info(f"{'='*60}")

    total_trades = 0
    total_pnl = 0
    coin_equities = {}

    for coin in COINS:
        r = results.get(coin)
        if not r or not r["trades"]:
            logger.info(f"  {coin:>5s}: no trades")
            continue
        trades = r["trades"]
        coin_equities[coin] = r["equity"]
        pnls = [t["pnl_pct"] for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        total_trades += len(trades)
        total_pnl += sum(pnls)

        cum = np.cumsum(pnls)
        mdd = np.max(np.maximum.accumulate(cum) - cum) if len(cum) > 0 else 0

        logger.info(
            f"  {coin:>5s}: {len(trades):>3d} trades | "
            f"WR={wins/len(trades):.0%} | avg={np.mean(pnls):+.3%} | "
            f"total={sum(pnls):+.3%} | MDD={mdd:.2%} | "
            f"avg_sl_dist={np.mean([t['sl_dist'] for t in trades]):.3%} | "
            f"TP:{sum(1 for t in trades if 'TP' in t['exit_reason'])} "
            f"SL:{sum(1 for t in trades if t['exit_reason']=='SL_HIT')} "
            f"TTL:{sum(1 for t in trades if t['exit_reason']=='TIME_STOP')}"
        )

    portfolio_eq = equity + sum(ce - equity for ce in coin_equities.values())
    logger.info(f"  {'TOTAL':>5s}: {total_trades:>3d} trades | PnL: {total_pnl:+.3%}")
    logger.info(f"  Portfolio equity: {portfolio_eq:.2f} USDT")
    logger.info(f"  Elapsed: {time.time()-t0:.1f}s")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTF Paper Sim")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--equity", type=float, default=10000.0)
    args = parser.parse_args()
    run_mtf_sim(days=args.days, equity=args.equity)
