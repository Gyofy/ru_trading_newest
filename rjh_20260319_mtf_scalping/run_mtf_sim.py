#!/usr/bin/env python3
"""Multi-Timeframe Scalping Simulator.

5분봉: 방향 결정 (LONG/SHORT) — 보수적 threshold
1분봉: 진입 타이밍 — 5분봉 방향과 일치할 때만 진입
SL/TP: 5분봉 ATR 기준 — 빠른 손절 + 빠른 익절
보유시간: 최대 1시간 — 안 움직이면 나감

Usage:
    python run_mtf_sim.py --days 7
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
from src.utils.feature_policy import is_excluded_feature
from src.execution.live_predictor import train_combo, predict_2stage
from src.execution.mtf_engine import compute_mtf_barriers

COINS = ["DOT", "ADA", "XRP", "SOL", "LINK", "BTC", "ETH"]
LOG_DIR = PROJECT_ROOT / "data" / "reports" / "mtf_sim"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(LOG_DIR / "mtf_sim.log", encoding="utf-8")])
logger = logging.getLogger("mtf_sim")


def fetch_data(coins: list, days: int) -> dict:
    import yfinance as yf
    tm = {"BTC":"BTC-USD","ETH":"ETH-USD","DOT":"DOT-USD","ADA":"ADA-USD",
          "XRP":"XRP-USD","SOL":"SOL-USD","LINK":"LINK-USD"}
    result = {}
    for coin in coins:
        sym = tm.get(coin)
        if not sym:
            continue
        try:
            # 5분봉 (방향 모델용)
            df5 = yf.Ticker(sym).history(period=f"{days}d", interval="5m")
            if df5.empty:
                continue
            df5.columns = [c.lower() for c in df5.columns]
            df5 = df5[["open","high","low","close","volume"]]
            df5.index = df5.index.tz_convert("UTC") if df5.index.tz else df5.index.tz_localize("UTC")
            result[coin] = {"5m": df5}
            logger.info(f"[Data] {coin}: 5m={len(df5)} bars")
        except Exception as e:
            logger.error(f"[Data] {coin}: {e}")
    return result


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_technical_indicators(df)
    df = _add_decomposition(df, period=42)
    try:
        df = add_signal_features(df, verbose=False)
    except Exception:
        pass
    df.ffill(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    cols_drop = [c for c in df.columns if is_excluded_feature(c)]
    if cols_drop:
        df.drop(columns=cols_drop, inplace=True, errors="ignore")
    return df


def run_sim(days: int = 7, equity: float = 10000.0):
    t0 = time.time()

    with open(PROJECT_ROOT / "config" / "frozen_params_v4_3_mtf.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tf5 = cfg["tf_5m"]
    tf1 = cfg["tf_1m"]
    ex = cfg["execution"]
    entry = cfg["entry_rules"]
    sizing = cfg["sizing_tiers"]
    leverage = ex["leverage"]
    max_hold = ex["max_hold_minutes"]  # 60분

    logger.info(f"{'='*60}")
    logger.info(f"  MTF SCALPING SIM")
    logger.info(f"  5m direction (th={tf5['stage1_threshold']}) + 5m entry timing")
    logger.info(f"  SL: max(ATR*{ex['sl_atr_mult']}, {ex['min_sl_pct']*100}%)")
    logger.info(f"  TP: 2-stage {ex['tp1_atr_mult']}/{ex['tp2_atr_mult']}x ATR")
    logger.info(f"  Max hold: {max_hold}min | Leverage: {leverage}x")
    logger.info(f"{'='*60}")

    # Fetch 5분봉 data
    logger.info("[1] Fetching data...")
    raw = fetch_data(COINS, days=days)

    # Feature engineering
    logger.info("[2] Features...")
    featured = {}
    for coin, data in raw.items():
        df5 = data.get("5m")
        if df5 is None or len(df5) < 500:
            continue
        featured[coin] = add_features(df5)
        logger.info(f"  {coin}: {len(featured[coin])} bars")

    # 5분봉 모델 학습 (방향용)
    logger.info("[3] Training 5m direction models...")
    common_5m = {
        "k_upper": tf5["k_upper"], "k_lower": tf5["k_lower"],
        "max_horizon": tf5["max_horizon"], "bar_minutes": tf5["bar_minutes"],
        "min_barrier_pct": tf5["min_barrier_pct"], "risk_frac": 0.005,
    }
    models = {}
    for coin in COINS:
        if coin not in featured:
            continue
        df = featured[coin]
        n = len(df)
        train_end = int(n * 0.7)
        coin_cfg = cfg["coins"].get(coin, {"model_combo": "et+xgb", "model_weights": {"et": 0.7, "xgb": 0.3}})
        coin_cfg.update({"stage1_threshold": tf5["stage1_threshold"],
                         "max_features": 120, "n_estimators": 300,
                         "max_depth_tree": 8, "min_child_samples": 10})
        combo = train_combo(df.iloc[:train_end].copy(), coin, common_5m, coin_cfg)
        if combo:
            models[coin] = {"combo": combo, "train_end": train_end}
            logger.info(f"  {coin}: {combo.model_names} CV={combo.cv_score:.4f}")

    # Simulation: 5분봉 bar 단위로 진행
    # 5분봉 시그널이 나면, 그 5분봉 내에서 진입 (시뮬에서는 같은 bar close로 진입)
    # SL/TP는 5분봉 ATR 기반, max hold = 60분 = 12 bars(5m)
    logger.info("[4] Simulating...")
    max_h_bars = max_hold // tf5["bar_minutes"]  # 60min / 5min = 12 bars
    results = {}

    for coin in COINS:
        if coin not in models or coin not in featured:
            continue

        df = featured[coin]
        combo = models[coin]["combo"]
        train_end = models[coin]["train_end"]
        n = len(df)

        coin_equity = equity
        trades = []
        i = train_end
        direction_side = None
        direction_conf = 0.0
        direction_bar = 0
        window = entry["entry_window_5m_bars"]

        while i < n - max_h_bars:
            # 5분봉 예측
            pred = predict_2stage(combo, df.iloc[:i+1], tf5["stage1_threshold"])

            # 방향 업데이트
            if pred.side != "HOLD" and pred.confidence >= entry["min_5m_confidence"]:
                direction_side = pred.side
                direction_conf = pred.confidence
                direction_bar = i

            # 진입 가능 여부: 방향이 있고 window 내
            if direction_side is None or (i - direction_bar) > window:
                i += 1
                continue

            # 현재 bar에서 진입 조건 확인
            # (실제로는 1분봉 시그널이지만, 시뮬에서는 5분봉 방향 확인 후 즉시 진입)
            entry_price = df["close"].iloc[i]
            atr_5m = df["atr_14"].iloc[i] if "atr_14" in df.columns else entry_price * 0.003
            if np.isnan(atr_5m) or atr_5m < 1e-10:
                atr_5m = entry_price * 0.003

            side = direction_side
            barriers = compute_mtf_barriers(entry_price, atr_5m, side, ex)

            # SL 거리 체크
            sl_pct = barriers["sl_dist_pct"]
            max_sl = ex.get("max_sl_pct", 0.015)
            if sl_pct > max_sl:
                # SL 거리 과도 → 스킵
                i += 1
                continue

            sl = barriers["sl"]
            tp1 = barriers["tp1"]
            tp2 = barriers["tp2"]

            # Triple barrier (2단계: TP1=50%, TP2=나머지)
            exit_price = entry_price
            exit_reason = "TIME_STOP"
            bars_held = 0
            price_high = entry_price
            price_low = entry_price
            partial_pnl = 0.0
            remaining = 1.0
            tp1_hit = False

            for j in range(1, max_h_bars + 1):
                if i + j >= n:
                    break
                bar = df.iloc[i + j]
                bars_held = j
                price_high = max(price_high, bar["high"])
                price_low = min(price_low, bar["low"])

                # SL check
                sl_triggered = (side == "BUY" and bar["low"] <= sl) or \
                               (side == "SELL" and bar["high"] >= sl)
                if sl_triggered:
                    exit_price = sl
                    exit_reason = "SL_HIT"
                    break

                # TP2 (final) check
                tp2_triggered = (side == "BUY" and bar["high"] >= tp2) or \
                                (side == "SELL" and bar["low"] <= tp2)
                if tp2_triggered:
                    exit_price = tp2
                    exit_reason = "TP2_HIT"
                    break

                # TP1 (partial) check
                if not tp1_hit:
                    tp1_triggered = (side == "BUY" and bar["high"] >= tp1) or \
                                    (side == "SELL" and bar["low"] <= tp1)
                    if tp1_triggered:
                        tp1_pnl = (tp1 - entry_price) / entry_price if side == "BUY" \
                                  else (entry_price - tp1) / entry_price
                        partial_pnl += tp1_pnl * 0.5
                        remaining = 0.5
                        sl = entry_price  # SL → breakeven
                        tp1_hit = True
            else:
                exit_price = df.iloc[min(i + max_h_bars, n - 1)]["close"]

            # PnL
            if side == "BUY":
                final_pnl = (exit_price - entry_price) / entry_price * remaining
            else:
                final_pnl = (entry_price - exit_price) / entry_price * remaining
            total_pnl = partial_pnl + final_pnl - 0.0015  # taker * 2

            # Sizing
            conf = direction_conf
            rf = sizing.get("tier_high", 0.010) if conf > 0.65 \
                 else (sizing.get("tier_mid", 0.007) if conf > 0.50
                       else sizing.get("tier_low", 0.005))
            stop_dist = sl_pct + 1e-10
            coin_equity += total_pnl * coin_equity * rf / stop_dist

            trades.append({
                "bar": i, "side": side, "pnl": round(total_pnl, 6),
                "exit": exit_reason, "bars": bars_held,
                "sl_pct": round(sl_pct * 100, 3),
                "partial": round(partial_pnl, 6),
            })

            # 방향은 window 내에서 계속 유효 (소비하지 않음)
            i += max(1, bars_held)

        results[coin] = {"trades": trades, "equity": coin_equity}

    # Report
    logger.info(f"\n{'='*60}")
    logger.info(f"  MTF SCALPING RESULTS")
    logger.info(f"{'='*60}")

    total_trades = 0
    total_pnl = 0
    equities = {}

    for coin in COINS:
        r = results.get(coin)
        if not r or not r["trades"]:
            continue
        trades = r["trades"]
        equities[coin] = r["equity"]
        pnls = [t["pnl"] for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        total_trades += len(trades)
        total_pnl += sum(pnls)

        cum = np.cumsum(pnls)
        mdd = np.max(np.maximum.accumulate(cum) - cum) if len(cum) > 0 else 0

        exits = {}
        for t in trades:
            exits[t["exit"]] = exits.get(t["exit"], 0) + 1

        logger.info(
            f"  {coin:>5s}: {len(trades):>3d} trades | "
            f"WR={wins/len(trades):.0%} | avg={np.mean(pnls):+.3%} | "
            f"total={sum(pnls):+.2%} | MDD={mdd:.2%} | "
            f"SL:{exits.get('SL_HIT',0)} TP1+2:{exits.get('TP2_HIT',0)} TTL:{exits.get('TIME_STOP',0)}"
        )

    port_eq = equity + sum(ce - equity for ce in equities.values())
    logger.info(f"  {'TOTAL':>5s}: {total_trades:>3d} trades | PnL: {total_pnl:+.2%}")
    logger.info(f"  Portfolio: ${port_eq:,.2f} | Elapsed: {time.time()-t0:.1f}s")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--equity", type=float, default=10000.0)
    args = parser.parse_args()
    run_sim(days=args.days, equity=args.equity)
