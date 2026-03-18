#!/usr/bin/env python3
"""Offline Paper Trading Simulator -- v4.3.

Runs the full v4.3 pipeline (features → train → predict → RL gate → entry/exit)
on historical data without exchange connection. Uses yfinance for OHLCV.

Usage:
    python run_paper_sim.py
    python run_paper_sim.py --equity 10000 --days 60
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
from src.execution.live_predictor import train_combo, predict_2stage, save_combo
from src.execution.risk_engine import RiskEngine, RiskConfig
from src.rl.state_builder import build_rl_state
from src.rl.signal_logger import SignalLogger, SIZING_MAP
from src.rl.rl_gate import RLGate

COINS = ["DOT", "ADA", "XRP", "SOL", "LINK"]
LOG_DIR = PROJECT_ROOT / "data" / "reports" / "paper_sim"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "sim.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("paper_sim")


def fetch_historical(coins: list, days: int = 90) -> dict[str, pd.DataFrame]:
    """Fetch historical 4h OHLCV via yfinance."""
    import yfinance as yf

    ticker_map = {
        "BTC": "BTC-USD", "ETH": "ETH-USD", "DOT": "DOT-USD",
        "ADA": "ADA-USD", "XRP": "XRP-USD", "SOL": "SOL-USD",
        "LINK": "LINK-USD",
    }
    results = {}
    fetch_list = list(set(coins + ["BTC", "ETH"]))

    for coin in fetch_list:
        sym = ticker_map.get(coin, f"{coin}-USD")
        try:
            df = yf.Ticker(sym).history(period=f"{days}d", interval="1h")
            if df.empty:
                logger.warning(f"[Data] {coin}: empty")
                continue
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
            # Resample to 4h
            df = df.resample("4h").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()
            results[coin] = df
            logger.info(f"[Data] {coin}: {len(df)} bars (4h)")
        except Exception as e:
            logger.error(f"[Data] {coin}: {e}")

    return results


def compute_features(raw_data: dict) -> dict:
    featured = {}
    for coin, df in raw_data.items():
        if len(df) < 100:
            continue
        try:
            df = df.copy()
            df = add_technical_indicators(df)
            df = _add_decomposition(df, period=42)
            df = add_signal_features(df, verbose=False)
            df = add_microstructure_rollup(df, verbose=False)
            df.ffill(inplace=True)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)  # no bfill — prevents future data leakage
            cols_drop = [c for c in df.columns if is_excluded_feature(c)]
            if cols_drop:
                df.drop(columns=cols_drop, inplace=True, errors="ignore")
            featured[coin] = df
        except Exception as e:
            logger.error(f"[Features] {coin}: {e}")
    return featured


def detect_regime(df, lookback=24) -> str:
    if len(df) < lookback:
        return "UNKNOWN"
    adx = df["adx_14"].iloc[-1] if "adx_14" in df.columns else 20.0
    # Consistent with live bot: check di_diff first, then fall back
    if "di_diff" in df.columns:
        di_diff = df["di_diff"].iloc[-1]
    elif "plus_di_14" in df.columns and "minus_di_14" in df.columns:
        di_diff = df["plus_di_14"].iloc[-1] - df["minus_di_14"].iloc[-1]
    else:
        di_diff = 0.0
    if adx > 25:
        return "TREND_UP" if di_diff > 0 else "TREND_DOWN"
    if "atr_14" in df.columns:
        atr_pct = df["atr_14"].iloc[-1] / (df["close"].iloc[-1] + 1e-10)
        median_pct = (df["atr_14"].iloc[-96:] / (df["close"].iloc[-96:] + 1e-10)).median()
    else:
        return "RANGE_HIGH"
    return "RANGE_HIGH" if atr_pct > median_pct else "RANGE_LOW"


def run_simulation(equity: float = 10000.0, days: int = 90):
    t0 = time.time()

    # Load config
    cfg_path = PROJECT_ROOT / "config" / "frozen_params_v4_3.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    common = cfg["common"]
    coin_cfgs = cfg["coins"]
    blocked_regimes = cfg.get("blocked_regimes", ["RANGE_LOW"])

    # Risk engine
    risk = RiskEngine(RiskConfig(risk_frac=common["risk_frac"]))
    risk.set_initial_equity(equity)

    # RL gate (shadow mode)
    rl_gate = RLGate(warmup=200, shadow_mode=True)
    sig_logger = SignalLogger(LOG_DIR / "signal_log.jsonl")

    logger.info(f"{'='*60}")
    logger.info(f"  PAPER SIMULATION v4.3")
    logger.info(f"  Equity: {equity:.0f} USDT | Days: {days}")
    logger.info(f"  Coins: {COINS}")
    logger.info(f"{'='*60}")

    # 1. Fetch data
    logger.info("[Step 1] Fetching historical data...")
    raw_data = fetch_historical(COINS, days=days)
    if not raw_data:
        logger.error("No data fetched!")
        return

    # 2. Features
    logger.info("[Step 2] Computing features...")
    featured = compute_features(raw_data)
    logger.info(f"[Step 2] Featured: {list(featured.keys())}")

    # 3. Train models (train on first 70% ONLY — no look-ahead)
    logger.info("[Step 3] Training 2-stage models...")
    models = {}
    train_splits = {}
    for coin in COINS:
        if coin not in featured:
            continue
        df = featured[coin]
        n = len(df)
        test_start = int(n * 0.7)
        train_df = df.iloc[:test_start].copy()
        combo = train_combo(train_df, coin, common, coin_cfgs.get(coin, {}))
        if combo:
            models[coin] = combo
            train_splits[coin] = test_start
            save_combo(combo)
            logger.info(f"  {coin}: {combo.model_names} CV={combo.cv_score:.4f} "
                        f"samples={combo.train_samples} (train={test_start}, test={n-test_start})")
    logger.info(f"[Step 3] Trained: {list(models.keys())}")

    # 4. Walk-forward simulation (test on last 30% — model never saw this data)
    logger.info("[Step 4] Running walk-forward simulation...")
    results = {}

    for coin in COINS:
        if coin not in models or coin not in featured:
            continue

        df = featured[coin]
        combo = models[coin]
        coin_cfg = coin_cfgs.get(coin, {})
        s1_thresh = coin_cfg.get("stage1_threshold", 0.50)
        coin_blocked = coin_cfg.get("blocked_regimes_override", blocked_regimes)

        n = len(df)
        test_start = train_splits.get(coin, int(n * 0.7))
        max_horizon = common["max_horizon"]
        coin_equity = equity  # independent per-coin equity (no cross-contamination)

        trades = []
        i = test_start

        while i < n - max_horizon:
            # Slice up to current bar (no future leakage)
            df_slice = df.iloc[:i + 1]

            # Regime
            regime = detect_regime(df_slice)
            if regime in coin_blocked:
                i += 1
                continue

            # Predict
            pred = predict_2stage(combo, df_slice, s1_thresh)
            if pred.side == "HOLD":
                i += 1
                continue

            # SHORT disabled: R:R = 0.2:1 with k_upper=3.0/k_lower=0.6
            if pred.side == "SELL":
                i += 1
                continue

            # RL gate — use actual per-coin equity + recent PnL
            btc_df = featured.get("BTC", df_slice)
            recent_pnls = [t["pnl_pct"] for t in trades[-6:]]  # ~1 day of trades
            daily_pnl_est = sum(recent_pnls) * coin_equity if recent_pnls else 0.0
            recent_wins = sum(1 for p in recent_pnls if p > 0)
            recent_wr = recent_wins / len(recent_pnls) if recent_pnls else 0.5
            streak = 0
            for p in reversed([t["pnl_pct"] for t in trades]):
                if streak == 0:
                    streak = 1 if p > 0 else -1
                elif p > 0 and streak > 0:
                    streak += 1
                elif p <= 0 and streak < 0:
                    streak -= 1
                else:
                    break
            state = build_rl_state(
                df=df_slice, pred_side=pred.side,
                p_trade=pred.p_trade, p_direction=pred.p_direction,
                s1_threshold=s1_thresh, coin=coin,
                equity=coin_equity, daily_pnl=daily_pnl_est,
                weekly_pnl=daily_pnl_est,  # approx
                dd_ratio=max(0, -sum(recent_pnls)) / 0.02 if recent_pnls else 0.0,
                open_count=0,
                coin_win_rate_5=recent_wr,
                coin_avg_pnl_5=np.mean(recent_pnls) if recent_pnls else 0.0,
                coin_streak=streak,
                bars_since_last=trades[-1]["bars_held"] if trades else max_horizon,
                btc_df=btc_df,
            )
            rl_action, rl_score = rl_gate.decide(state)

            # Entry
            entry_price = df.iloc[i]["close"]
            atr = df_slice["atr_14"].iloc[-1] if "atr_14" in df_slice.columns else entry_price * 0.01
            if np.isnan(atr) or atr < 1e-10:
                atr = entry_price * 0.01
            sl, tp = RiskEngine.compute_barriers(
                entry_price, atr, pred.side,
                common["k_upper"], common["k_lower"], common["min_barrier_pct"],
            )

            # Triple barrier exit
            exit_price = entry_price
            exit_reason = "TIME_STOP"
            bars_held = 0
            price_high = entry_price
            price_low = entry_price

            for j in range(1, max_horizon + 1):
                if i + j >= n:
                    break
                bar = df.iloc[i + j]
                price = bar["close"]
                price_high = max(price_high, bar["high"])
                price_low = min(price_low, bar["low"])
                bars_held = j

                if pred.side == "BUY":
                    if bar["high"] >= tp:
                        exit_price = tp
                        exit_reason = "TP_HIT"
                        break
                    if bar["low"] <= sl:
                        exit_price = sl
                        exit_reason = "SL_HIT"
                        break
                else:
                    if bar["low"] <= tp:
                        exit_price = tp
                        exit_reason = "TP_HIT"
                        break
                    if bar["high"] >= sl:
                        exit_price = sl
                        exit_reason = "SL_HIT"
                        break
            else:
                exit_price = df.iloc[min(i + max_horizon, n - 1)]["close"]

            # PnL
            if pred.side == "BUY":
                pnl_pct = (exit_price - entry_price) / entry_price
                mfe = (price_high - entry_price) / entry_price
                mae = (price_low - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price
                mfe = (entry_price - price_low) / entry_price
                mae = (entry_price - price_high) / entry_price

            pnl_net = pnl_pct - 0.002  # cost deduction

            # Confidence-tiered risk + DD brake
            confidence = pred.p_trade * pred.p_direction
            daily_dd = max(0, -sum(t["pnl_pct"] for t in trades[-12:])) if trades else 0
            sizing_cfg = cfg.get("sizing_tiers", {})
            if confidence > 0.65:
                risk_f = sizing_cfg.get("tier_high", 0.015)
            elif confidence > 0.50:
                risk_f = sizing_cfg.get("tier_mid", 0.010)
            else:
                risk_f = sizing_cfg.get("tier_low", 0.005)
            if daily_dd > sizing_cfg.get("dd_brake_threshold", 0.015):
                risk_f *= 0.5

            stop_dist = abs(entry_price - sl) / entry_price + 1e-10
            coin_equity += pnl_net * coin_equity * risk_f / stop_dist

            trades.append({
                "bar": i, "side": pred.side, "regime": regime,
                "p_trade": round(pred.p_trade, 4),
                "p_dir": round(pred.p_direction, 4),
                "entry": round(entry_price, 6),
                "exit": round(exit_price, 6),
                "pnl_pct": round(pnl_net, 6),
                "exit_reason": exit_reason,
                "bars_held": bars_held,
                "mfe": round(mfe, 6),
                "mae": round(mae, 6),
                "rl_action": rl_action,
                "rl_score": round(rl_score, 4),
            })

            # Log to signal logger
            sig_logger.log(
                coin=coin, side=pred.side, regime=regime,
                state=state.tolist(),
                p_trade=pred.p_trade, p_direction=pred.p_direction,
                action=rl_action, rl_score=rl_score,
                entry_price=entry_price, sl_price=sl, tp_price=tp,
            )
            sig_logger.update_result(coin, pnl_net, exit_reason, bars_held, mfe, mae)

            # Skip forward past this trade
            # Re-entry on next bar after exit (matches live bot behavior)
            i += max(1, bars_held)

        results[coin] = {"trades": trades, "equity": coin_equity}

    # 5. Report
    logger.info(f"\n{'='*60}")
    logger.info(f"  SIMULATION RESULTS")
    logger.info(f"{'='*60}")

    total_trades = 0
    total_pnl = 0.0
    report = {}

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
        mfes = [t["mfe"] for t in trades]
        maes = [t["mae"] for t in trades]

        # Max drawdown
        cum = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        max_dd = dd.max() if len(dd) > 0 else 0

        coin_report = {
            "trades": len(trades),
            "wins": wins,
            "win_rate": round(wins / len(trades), 3),
            "avg_pnl": round(np.mean(pnls), 5),
            "total_pnl": round(np.sum(pnls), 5),
            "max_dd": round(max_dd, 5),
            "avg_mfe": round(np.mean(mfes), 5),
            "avg_mae": round(np.mean(maes), 5),
            "avg_bars": round(np.mean([t["bars_held"] for t in trades]), 1),
            "exits": {r: sum(1 for t in trades if t["exit_reason"] == r)
                      for r in ["TP_HIT", "SL_HIT", "TIME_STOP"]},
        }
        report[coin] = coin_report
        total_trades += len(trades)
        total_pnl += np.sum(pnls)

        logger.info(
            f"  {coin:>5s}: {len(trades):>3d} trades | "
            f"WR={coin_report['win_rate']:.0%} | "
            f"avg={coin_report['avg_pnl']:+.3%} | "
            f"total={coin_report['total_pnl']:+.3%} | "
            f"MDD={coin_report['max_dd']:.3%} | "
            f"MFE={coin_report['avg_mfe']:+.3%} MAE={coin_report['avg_mae']:+.3%} | "
            f"TP:{coin_report['exits']['TP_HIT']} SL:{coin_report['exits']['SL_HIT']} TTL:{coin_report['exits']['TIME_STOP']}"
        )

    # Portfolio equity = initial + sum of per-coin gains
    portfolio_equity = equity + sum(ce - equity for ce in coin_equities.values())
    logger.info(f"  {'TOTAL':>5s}: {total_trades:>3d} trades | total PnL: {total_pnl:+.3%}")
    logger.info(f"  Portfolio equity: {portfolio_equity:.2f} USDT (per-coin independent)")
    logger.info(f"  Elapsed: {time.time() - t0:.1f}s")

    # Save report
    report_path = LOG_DIR / f"sim_result_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"report": report, "equity": round(equity, 2),
                    "total_trades": total_trades, "total_pnl": round(total_pnl, 5)},
                  f, indent=2)
    logger.info(f"  Report: {report_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper Trading Simulator v4.3")
    parser.add_argument("--equity", type=float, default=10000.0)
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    run_simulation(equity=args.equity, days=args.days)
