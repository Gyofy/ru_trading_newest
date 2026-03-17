"""Trade-level EV Calculator -- 요약식 EV 대체 모듈.

기존 compute_net_ev_score()의 요약식:
  ev = (S2_accuracy - BEP) * R:R * risk_frac

이를 실제 bar-by-bar barrier hit simulation으로 대체.
각 테스트 윈도우에서 실제 trade sequence를 시뮬레이션하여
realized PnL distribution을 산출한다.

Usage:
    from src.evaluation.trade_level_ev import compute_trade_level_ev

    result = compute_trade_level_ev(
        test_df=test_df,          # OOS dataframe with OHLCV + features
        s1_pred=s1_pred,          # Stage1 predictions (0/1)
        s1_prob=s1_probs[:, 1],   # Stage1 probabilities
        s2_pred=s2_pred,          # Stage2 predictions (0=short, 1=long)
        s2_prob=s2_probs[:, 1],   # Stage2 probabilities
        k_upper=3.0, k_lower=0.6,
        risk_frac=0.005,
        cost_model=cost_model,
    )
    # result["avg_net_pnl"], result["total_net_pnl"], result["trade_count"], etc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.execution.cost_model import CostModel, FeeSchedule, FundingConfig, MissFillConfig, ExitType
from src.utils.config import bar_minutes as cfg_bar_minutes


def compute_trade_level_ev(
    test_df: pd.DataFrame,
    s1_pred: np.ndarray,
    s1_prob: np.ndarray,
    s2_pred: np.ndarray,
    s2_prob: np.ndarray,
    k_upper: float = 3.0,
    k_lower: float = 0.6,
    max_hold: int = 18,
    risk_frac: float = 0.005,
    cost_model: CostModel = None,
    min_barrier_pct: float = 0.002,
) -> dict:
    """Compute trade-level EV from bar-by-bar simulation.

    Returns dict with:
        trade_count, avg_net_pnl, total_net_pnl, win_rate,
        loss_rate, time_exit_rate, avg_holding, max_dd,
        cost_share, pnl_list, score
    """
    if cost_model is None:
        cost_model = CostModel()

    bm = cfg_bar_minutes()
    close = test_df["close"].values
    high = test_df["high"].values
    low = test_df["low"].values
    n = len(close)

    # ATR
    if "atr_14" in test_df.columns:
        atr = test_df["atr_14"].values
    else:
        tr = np.maximum(high - low,
                        np.maximum(np.abs(high - np.roll(close, 1)),
                                   np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

    pnl_list = []
    trade_count = 0
    tp_count = 0
    sl_count = 0
    ts_count = 0
    total_cost = 0.0
    total_gross = 0.0
    holdings = []
    next_available = 0

    for i in range(n - max_hold):
        if i < next_available:
            continue
        if s1_pred[i] != 1:
            continue

        entry_price = close[i]
        side = "BUY" if s2_pred[i] == 1 else "SELL"
        cur_atr = atr[i] if not np.isnan(atr[i]) else entry_price * 0.01

        upper_dist = max(k_upper * cur_atr, min_barrier_pct * entry_price)
        lower_dist = max(k_lower * cur_atr, min_barrier_pct * entry_price)

        if side == "BUY":
            tp_price = entry_price + upper_dist
            sl_price = entry_price - lower_dist
        else:
            tp_price = entry_price - upper_dist
            sl_price = entry_price + lower_dist

        # Bar-by-bar barrier check
        exit_type = None
        exit_bar = -1
        exit_price = 0.0

        for j in range(i + 1, min(i + max_hold + 1, n)):
            if side == "BUY":
                hit_tp = high[j] >= tp_price
                hit_sl = low[j] <= sl_price
            else:
                hit_tp = low[j] <= tp_price
                hit_sl = high[j] >= sl_price

            if hit_tp and hit_sl:
                exit_type = "stop_loss"
                exit_bar = j
                exit_price = sl_price
                break
            elif hit_tp:
                exit_type = "take_profit"
                exit_bar = j
                exit_price = tp_price
                break
            elif hit_sl:
                exit_type = "stop_loss"
                exit_bar = j
                exit_price = sl_price
                break

        if exit_type is None:
            exit_type = "time_stop"
            exit_bar = min(i + max_hold, n - 1)
            exit_price = close[exit_bar]

        holding_bars = exit_bar - i

        # PnL
        if side == "BUY":
            gross_pnl_pct = (exit_price - entry_price) / entry_price
        else:
            gross_pnl_pct = (entry_price - exit_price) / entry_price

        stop_dist_pct = max(lower_dist / entry_price, 0.003)
        notional_ratio = risk_frac / stop_dist_pct
        gross_pnl_eq = gross_pnl_pct * notional_ratio

        # Cost
        exit_enum = {
            "take_profit": ExitType.TAKE_PROFIT,
            "stop_loss": ExitType.STOP_LOSS,
            "time_stop": ExitType.TIME_STOP,
        }[exit_type]

        cost = cost_model.estimate_trade_cost(
            entry_price=entry_price, sl_price=sl_price,
            tp_price=tp_price, risk_frac=risk_frac,
            exit_type=exit_enum, holding_bars=holding_bars,
            bar_minutes=bm, entry_is_maker=True,
        )

        net_pnl_eq = gross_pnl_eq - cost.total_eq

        pnl_list.append(net_pnl_eq)
        trade_count += 1
        total_cost += cost.total_eq
        total_gross += abs(gross_pnl_eq)
        holdings.append(holding_bars)

        if exit_type == "take_profit":
            tp_count += 1
        elif exit_type == "stop_loss":
            sl_count += 1
        else:
            ts_count += 1

        next_available = exit_bar + 1

    # Metrics
    if trade_count == 0:
        return {
            "trade_count": 0,
            "avg_net_pnl": 0.0,
            "total_net_pnl": 0.0,
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "time_exit_rate": 0.0,
            "avg_holding": 0.0,
            "max_dd": 0.0,
            "cost_share": 0.0,
            "score": -100.0,
            "pnl_list": [],
        }

    avg_net = np.mean(pnl_list)
    total_net = np.sum(pnl_list)

    equity = np.cumsum(pnl_list)
    peak = np.maximum.accumulate(equity)
    max_dd = abs(np.min(equity - peak)) if len(equity) > 0 else 0

    cost_share = total_cost / (total_gross + 1e-10)

    # Score: trade-level based
    # avg_net_pnl * sqrt(trade_count) * stability_factor
    if avg_net > 0 and trade_count >= 3:
        std_pnl = np.std(pnl_list) if len(pnl_list) > 1 else abs(avg_net)
        sharpe_like = avg_net / (std_pnl + 1e-10)
        score = avg_net * 10000 * np.sqrt(trade_count) * (1 - 0.3 * max_dd * 100)
    else:
        score = avg_net * 10000 - 50

    return {
        "trade_count": trade_count,
        "avg_net_pnl": round(avg_net, 6),
        "total_net_pnl": round(total_net, 6),
        "win_rate": round(tp_count / trade_count, 4),
        "loss_rate": round(sl_count / trade_count, 4),
        "time_exit_rate": round(ts_count / trade_count, 4),
        "avg_holding": round(np.mean(holdings), 2),
        "max_dd": round(max_dd, 6),
        "cost_share": round(cost_share, 4),
        "score": round(score, 4),
        "pnl_list": [round(p, 6) for p in pnl_list],
    }
