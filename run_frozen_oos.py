"""Frozen OOS Validation -- Trade-level PnL Simulation.

v3.1_netev 최종 파라미터를 완전 동결하고,
최적화에 단 한 번도 쓰지 않은 구간에서 trade-level PnL을 계산한다.

검증 순서:
1. 파라미터 완전 동결 (재튜닝 금지)
2. OOS 구간 분리 (최근 6주, purge gap 포함)
3. Train on pre-OOS data -> Predict on OOS
4. Bar-by-bar barrier hit simulation -> realized PnL
5. 비용 차감 후 net PnL distribution 산출
6. Parameter perturbation stability check
"""

import sys
sys.path.insert(0, "C:/Users/RJ/Desktop/CLAUDE_CRYPTO_AGENT")
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import json
import time
import warnings
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

warnings.filterwarnings("ignore")

from src.data.crawlers.crypto_ohlcv import fetch_all_top10
from src.data.crawlers.macro_commodity_crawler import crawl_all_macro_data
from src.models.masking_loop import (
    create_labels_triple_barrier, compute_extended_metrics,
    LABEL_MAP, LABEL_NAMES, HORIZONS, HORIZON_LABELS,
    STAGE1_NAMES, STAGE2_NAMES,
)
from src.models.enhanced_ensemble import EnhancedEnsemble
from src.models.regime_filter import RegimeFilter, Regime4
from src.execution.cost_model import CostModel, FeeSchedule, FundingConfig, MissFillConfig, ExitType
from src.utils.config import load_settings, bar_minutes as cfg_bar_minutes
from sklearn.feature_selection import mutual_info_classif
from src.utils.feature_policy import is_excluded_feature, is_blocked_regime

# ==================== 설정 ====================
REPORT_DIR = Path("data/reports/frozen_oos")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS = load_settings()
BM = cfg_bar_minutes()
MAX_HORIZON = max(HORIZONS)
PURGE_BARS = MAX_HORIZON * 2
EMBARGO_BARS = 6

ACTIVE_COINS = ["XRP", "DOT", "ADA"]
OOS_DAYS = 42  # 6주
RISK_FRAC = 0.005  # 0.5% equity per trade

N_JOBS = 6

# 비용 모델 (Bybit VIP0 -- 동결)
COST_MODEL = CostModel(
    fee_schedule=FeeSchedule(
        maker_fee=0.0002,
        taker_fee=0.00055,
        slippage_entry=0.0003,
        slippage_exit_limit=0.0001,
        slippage_exit_market=0.0005,
    ),
    funding_config=FundingConfig(
        interval_hours=8.0,
        default_rate=0.0001,
    ),
    miss_fill_config=MissFillConfig(
        reject_prob=0.15,
        missed_ev_pct=0.0015,
    ),
)

REGIME_FILTER = RegimeFilter()

# ==================== 동결 파라미터 ====================
# R38 checkpoint에서 추출 -- 절대 수정 금지

FROZEN_PARAMS = {
    "XRP": {
        "k_upper": 3.0, "k_lower": 0.6,
        "stage1_threshold": 0.46, "max_features": 120,
        "num_leaves": 47, "learning_rate": 0.02,
        "n_estimators": 100, "max_depth_tree": 6,
        "subsample": 0.8, "colsample": 0.6,
        "min_child_samples": 30,
    },
    "ADA": {
        "k_upper": 3.0, "k_lower": 0.6,
        "stage1_threshold": 0.5, "max_features": 120,
        "num_leaves": 15, "learning_rate": 0.02,
        "n_estimators": 400, "max_depth_tree": 8,
        "subsample": 0.7, "colsample": 0.9,
        "min_child_samples": 5,
    },
    "DOT": {
        "k_upper": 3.0, "k_lower": 0.6,
        "stage1_threshold": 0.5, "max_features": 80,
        "num_leaves": 47, "learning_rate": 0.1,
        "n_estimators": 300, "max_depth_tree": 8,
        "subsample": 0.7, "colsample": 0.8,
        "min_child_samples": 10,
    },
}


# ==================== TradeRecord ====================

@dataclass
class TradeRecord:
    entry_bar_idx: int
    entry_time: str
    entry_price: float
    side: str  # "BUY" or "SELL"
    tp_price: float
    sl_price: float
    atr_at_entry: float

    exit_bar_idx: int = -1
    exit_time: str = ""
    exit_price: float = 0.0
    exit_type: str = ""  # "take_profit", "stop_loss", "time_stop"
    holding_bars: int = 0
    gross_pnl_pct: float = 0.0
    gross_pnl_eq: float = 0.0
    cost_total_eq: float = 0.0
    net_pnl_eq: float = 0.0
    regime: str = "UNKNOWN"

    # S1/S2 probabilities at entry
    s1_prob: float = 0.0
    s2_prob: float = 0.0


# ==================== Bar-by-Bar Simulator ====================

class BarByBarSimulator:
    """Trade-level PnL simulation with actual barrier hit sequences."""

    def __init__(self, cost_model: CostModel, risk_frac: float = 0.005,
                 max_hold: int = 18, bar_minutes: int = 240):
        self.cost_model = cost_model
        self.risk_frac = risk_frac
        self.max_hold = max_hold
        self.bm = bar_minutes

    def simulate(self, oos_df: pd.DataFrame, predictions: dict,
                 params: dict, regime_series: pd.Series = None) -> list:
        """Run bar-by-bar simulation.

        Args:
            oos_df: OOS OHLCV dataframe with 'close', 'high', 'low', 'atr_14'
            predictions: dict with 's1_pred', 's1_prob', 's2_pred', 's2_prob' arrays
            params: frozen params (k_upper, k_lower)
            regime_series: optional regime classification per bar

        Returns:
            list of TradeRecord
        """
        close = oos_df["close"].values
        high = oos_df["high"].values
        low = oos_df["low"].values
        times = oos_df.index
        n = len(close)

        # ATR
        if "atr_14" in oos_df.columns:
            atr = oos_df["atr_14"].values
        else:
            tr = np.maximum(high - low,
                            np.maximum(np.abs(high - np.roll(close, 1)),
                                       np.abs(low - np.roll(close, 1))))
            tr[0] = high[0] - low[0]
            atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

        k_upper = params["k_upper"]
        k_lower = params["k_lower"]
        min_barrier_pct = 0.002

        s1_pred = predictions["s1_pred"]
        s1_prob = predictions["s1_prob"]
        s2_pred = predictions["s2_pred"]
        s2_prob = predictions["s2_prob"]

        trades = []
        next_available_bar = 0  # non-overlapping constraint

        for i in range(n - self.max_hold):
            if i < next_available_bar:
                continue
            if s1_pred[i] != 1:
                continue

            # Regime filter: block RANGE_LOW
            if regime_series is not None and i < len(regime_series):
                try:
                    r = str(regime_series.iloc[i]) if hasattr(regime_series, 'iloc') else str(regime_series[i])
                    if is_blocked_regime(r):
                        continue
                except Exception:
                    pass

            # Entry
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

            for j in range(i + 1, min(i + self.max_hold + 1, n)):
                if side == "BUY":
                    hit_tp = high[j] >= tp_price
                    hit_sl = low[j] <= sl_price
                else:
                    hit_tp = low[j] <= tp_price
                    hit_sl = high[j] >= sl_price

                if hit_tp and hit_sl:
                    # Conservative: SL wins when both hit same bar
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
                # Time barrier
                exit_type = "time_stop"
                exit_bar = min(i + self.max_hold, n - 1)
                exit_price = close[exit_bar]

            holding_bars = exit_bar - i

            # Gross PnL
            if side == "BUY":
                gross_pnl_pct = (exit_price - entry_price) / entry_price
            else:
                gross_pnl_pct = (entry_price - exit_price) / entry_price

            # Position sizing -> notional ratio
            stop_dist_pct = lower_dist / entry_price
            if stop_dist_pct < 1e-6:
                stop_dist_pct = 0.003
            notional_ratio = self.risk_frac / stop_dist_pct
            gross_pnl_eq = gross_pnl_pct * notional_ratio

            # Cost
            exit_type_enum = {
                "take_profit": ExitType.TAKE_PROFIT,
                "stop_loss": ExitType.STOP_LOSS,
                "time_stop": ExitType.TIME_STOP,
            }[exit_type]

            cost = self.cost_model.estimate_trade_cost(
                entry_price=entry_price,
                sl_price=sl_price if side == "BUY" else sl_price,
                tp_price=tp_price,
                risk_frac=self.risk_frac,
                exit_type=exit_type_enum,
                holding_bars=holding_bars,
                bar_minutes=self.bm,
                entry_is_maker=True,
            )

            net_pnl_eq = gross_pnl_eq - cost.total_eq

            # Regime at entry
            regime = "UNKNOWN"
            if regime_series is not None and i < len(regime_series):
                try:
                    regime = str(regime_series.iloc[i])
                except Exception:
                    pass

            trade = TradeRecord(
                entry_bar_idx=i,
                entry_time=str(times[i]),
                entry_price=round(entry_price, 6),
                side=side,
                tp_price=round(tp_price, 6),
                sl_price=round(sl_price, 6),
                atr_at_entry=round(cur_atr, 6),
                exit_bar_idx=exit_bar,
                exit_time=str(times[exit_bar]) if exit_bar < n else "",
                exit_price=round(exit_price, 6),
                exit_type=exit_type,
                holding_bars=holding_bars,
                gross_pnl_pct=round(gross_pnl_pct, 6),
                gross_pnl_eq=round(gross_pnl_eq, 6),
                cost_total_eq=round(cost.total_eq, 6),
                net_pnl_eq=round(net_pnl_eq, 6),
                regime=regime,
                s1_prob=round(float(s1_prob[i]), 4),
                s2_prob=round(float(s2_prob[i]), 4),
            )
            trades.append(trade)
            next_available_bar = exit_bar + 1  # non-overlapping

        return trades


# ==================== Metrics ====================

def compute_oos_metrics(trades: list, coin: str) -> dict:
    """Compute comprehensive OOS metrics from trade records."""
    n = len(trades)
    if n == 0:
        return {"coin": coin, "trade_count": 0, "verdict": "NO TRADES"}

    pnls = [t.net_pnl_eq for t in trades]
    gross_pnls = [t.gross_pnl_eq for t in trades]
    costs = [t.cost_total_eq for t in trades]
    holdings = [t.holding_bars for t in trades]

    tp_count = sum(1 for t in trades if t.exit_type == "take_profit")
    sl_count = sum(1 for t in trades if t.exit_type == "stop_loss")
    ts_count = sum(1 for t in trades if t.exit_type == "time_stop")

    avg_net = np.mean(pnls)
    total_net = np.sum(pnls)
    total_gross = np.sum(np.abs(gross_pnls))
    total_cost = np.sum(costs)

    # Equity curve + max drawdown
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    drawdowns = equity - peak
    max_dd = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0

    # Sharpe-like
    if n > 1 and np.std(pnls) > 0:
        avg_hold = np.mean(holdings)
        trades_per_year = (252 * 6) / max(avg_hold, 1)
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(trades_per_year)
    else:
        sharpe = 0

    # 95% CI
    se = np.std(pnls) / np.sqrt(n) if n > 0 else 0
    ci_lo = avg_net - 1.96 * se
    ci_hi = avg_net + 1.96 * se

    # PnL percentiles
    pcts = np.percentile(pnls, [5, 25, 50, 75, 95]).tolist() if n > 0 else [0]*5

    # Regime breakdown
    regime_stats = {}
    for regime in set(t.regime for t in trades):
        r_trades = [t for t in trades if t.regime == regime]
        r_pnls = [t.net_pnl_eq for t in r_trades]
        regime_stats[regime] = {
            "count": len(r_trades),
            "avg_net_pnl": round(np.mean(r_pnls), 6) if r_pnls else 0,
            "total_net_pnl": round(np.sum(r_pnls), 6) if r_pnls else 0,
        }

    # Cost share
    cost_share = total_cost / (total_gross + 1e-10)

    # Verdict
    if avg_net > 0 and ci_lo > -0.001:
        verdict = "PASS" if n >= 15 else "PASS (low N, cautious)"
    elif avg_net > 0:
        verdict = "MARGINAL (CI includes zero)"
    else:
        verdict = "FAIL"

    return {
        "coin": coin,
        "trade_count": n,
        "tp_count": tp_count,
        "sl_count": sl_count,
        "time_stop_count": ts_count,
        "win_rate": round(tp_count / n, 4),
        "loss_rate": round(sl_count / n, 4),
        "time_exit_rate": round(ts_count / n, 4),
        "avg_net_pnl": round(avg_net, 6),
        "total_net_pnl": round(total_net, 6),
        "total_gross_pnl": round(np.sum(gross_pnls), 6),
        "total_cost": round(total_cost, 6),
        "cost_share": round(cost_share, 4),
        "avg_holding_bars": round(np.mean(holdings), 2),
        "max_drawdown": round(max_dd, 6),
        "sharpe_like": round(sharpe, 4),
        "ci_95_lo": round(ci_lo, 6),
        "ci_95_hi": round(ci_hi, 6),
        "pnl_p5": round(pcts[0], 6),
        "pnl_p25": round(pcts[1], 6),
        "pnl_p50": round(pcts[2], 6),
        "pnl_p75": round(pcts[3], 6),
        "pnl_p95": round(pcts[4], 6),
        "regime_breakdown": regime_stats,
        "verdict": verdict,
    }


# ==================== Frozen OOS Validator ====================

class FrozenOOSValidator:
    """Frozen parameter OOS validation with trade-level PnL."""

    def __init__(self):
        self.ohlcv = None
        self.macro = None
        self.feature_data = {}
        self.simulator = BarByBarSimulator(
            cost_model=COST_MODEL,
            risk_frac=RISK_FRAC,
            max_hold=MAX_HORIZON,
            bar_minutes=BM,
        )

    def fetch_data(self):
        """Data collection (same as optimizer)."""
        print(f"\n{'='*70}")
        print(f"  DATA COLLECTION")
        print(f"{'='*70}")

        self.ohlcv = fetch_all_top10("365d", "1h")
        print(f"  => {len(self.ohlcv)} coins loaded")

        first_coin = list(self.ohlcv.values())[0]
        self.macro = crawl_all_macro_data(first_coin.index)
        macro_aligned = self.macro.get("aligned", pd.DataFrame())

        for coin in ACTIVE_COINS:
            if coin not in self.ohlcv:
                print(f"  [SKIP] {coin}: no data")
                continue
            df = self.ohlcv[coin].copy()
            if len(macro_aligned) > 0:
                for col in macro_aligned.columns:
                    df[col] = macro_aligned[col].reindex(df.index).ffill().bfill().fillna(0)
            self.feature_data[coin] = df
            print(f"  {coin}: {len(df)} bars, {len(df.columns)} features")

    def split_oos(self, coin_df: pd.DataFrame) -> tuple:
        """Split data into train and OOS periods."""
        idx = coin_df.index
        data_end = idx[-1]
        oos_start = data_end - timedelta(days=OOS_DAYS)
        purge_td = timedelta(hours=(PURGE_BARS + EMBARGO_BARS) * (BM // 60))
        train_end = oos_start - purge_td

        train_mask = idx <= train_end
        oos_mask = idx >= oos_start

        train_df = coin_df[train_mask]
        oos_df = coin_df[oos_mask]

        return train_df, oos_df

    def train_and_predict(self, train_df: pd.DataFrame, oos_df: pd.DataFrame,
                          params: dict) -> dict:
        """Train with frozen params, predict on OOS."""
        h = HORIZONS[-1]
        h_label = f"label_{h*BM}min"
        k_upper = params["k_upper"]
        k_lower = params["k_lower"]

        # Label
        train_labeled = create_labels_triple_barrier(
            train_df.copy(), h, k_upper_override=k_upper,
            k_lower_override=k_lower, verbose=False)

        if h_label not in train_labeled.columns:
            return None

        # Feature selection (frozen max_features)
        exclude = {"label", "future_return", "open", "high", "low", "close", "volume"}
        for hh in HORIZONS:
            exclude.add(f"label_{hh*BM}min")
            exclude.add(f"return_{hh*BM}min")
        leak_keywords = ["future", "target", "label_", "return_", "fwd_", "forward_"]

        feature_cols = []
        for c in train_labeled.columns:
            if c in exclude:
                continue
            if any(kw in c.lower() for kw in leak_keywords):
                continue
            if is_excluded_feature(c):
                continue
            if train_labeled[c].dtype not in [np.float64, np.float32, np.int64, np.int32, float, int]:
                continue
            feature_cols.append(c)

        max_feat = params["max_features"]
        if len(feature_cols) > max_feat:
            X_mi = train_labeled[feature_cols].replace([np.inf, -np.inf], 0).fillna(0).values
            y_mi = train_labeled[h_label].values
            n_mi = min(2000, len(X_mi))
            mi = mutual_info_classif(X_mi[:n_mi], y_mi[:n_mi],
                                     discrete_features=False, random_state=42, n_neighbors=5)
            top_idx = np.argsort(mi)[-max_feat:]
            feature_cols = [feature_cols[i] for i in sorted(top_idx)]

        # Clean
        train_clean = train_labeled.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        oos_clean = oos_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

        X_train = train_clean[feature_cols].fillna(0).values
        X_oos = oos_clean[feature_cols].fillna(0).values
        y_train_3c = train_clean[h_label].fillna(1).values.astype(int)

        if len(X_train) < 60:
            return None

        # Stage 1: Trade/NoTrade
        y_s1_train = (y_train_3c != LABEL_MAP["HOLD"]).astype(int)
        if len(np.unique(y_s1_train)) < 2:
            return None

        s1_counts = np.bincount(y_s1_train, minlength=2)
        s1_sw = np.where(s1_counts > 0, len(y_s1_train) / (2 * s1_counts + 1e-10), 1.0)[y_s1_train]

        ens_s1 = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
        ens_s1.fit(X_train, y_s1_train, sample_weight=s1_sw)

        s1_probs = ens_s1.predict_proba(X_oos)
        threshold = params["stage1_threshold"]
        s1_pred = (s1_probs[:, 1] >= threshold).astype(int)

        # Stage 2: Long/Short
        trade_mask_train = y_train_3c != LABEL_MAP["HOLD"]
        s2_pred = np.zeros(len(X_oos), dtype=int)
        s2_prob = np.full(len(X_oos), 0.5)

        if trade_mask_train.sum() >= 30:
            X_s2_train = X_train[trade_mask_train]
            y_s2_train = (y_train_3c[trade_mask_train] == LABEL_MAP["UP"]).astype(int)

            if len(np.unique(y_s2_train)) >= 2:
                s2_counts = np.bincount(y_s2_train, minlength=2)
                s2_sw = np.where(s2_counts > 0, len(y_s2_train) / (2 * s2_counts + 1e-10), 1.0)[y_s2_train]

                ens_s2 = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
                ens_s2.fit(X_s2_train, y_s2_train, sample_weight=s2_sw)

                s2_probs_all = ens_s2.predict_proba(X_oos)
                s2_pred = np.argmax(s2_probs_all, axis=1)
                s2_prob = s2_probs_all[:, 1]

        return {
            "s1_pred": s1_pred,
            "s1_prob": s1_probs[:, 1],
            "s2_pred": s2_pred,
            "s2_prob": s2_prob,
            "feature_cols": feature_cols,
            "n_train": len(X_train),
            "n_oos": len(X_oos),
            "s1_trade_rate": float(s1_pred.mean()),
        }

    def run_coin(self, coin: str) -> dict:
        """Full frozen OOS pipeline for one coin."""
        print(f"\n  >> {coin}")
        params = FROZEN_PARAMS[coin]
        coin_df = self.feature_data[coin]

        # Split
        train_df, oos_df = self.split_oos(coin_df)
        print(f"    Train: {len(train_df)} bars ({train_df.index[0]} ~ {train_df.index[-1]})")
        print(f"    OOS:   {len(oos_df)} bars ({oos_df.index[0]} ~ {oos_df.index[-1]})")
        print(f"    Purge gap: {PURGE_BARS + EMBARGO_BARS} bars")

        # Train + predict
        t0 = time.time()
        predictions = self.train_and_predict(train_df, oos_df, params)
        if predictions is None:
            print(f"    [FAIL] Training failed")
            return {"coin": coin, "verdict": "TRAIN FAILED"}

        train_time = time.time() - t0
        print(f"    Trained in {train_time:.1f}s | n_train={predictions['n_train']} "
              f"n_oos={predictions['n_oos']} | S1 trade_rate={predictions['s1_trade_rate']:.1%}")

        # Regime classification
        regime_series = None
        try:
            regimes = REGIME_FILTER.classify_series(oos_df)
            if regimes is not None and len(regimes) == len(oos_df):
                regime_series = regimes
        except Exception:
            pass

        # Simulate trades
        trades = self.simulator.simulate(oos_df, predictions, params, regime_series)
        print(f"    Trades: {len(trades)}")

        # Metrics
        metrics = compute_oos_metrics(trades, coin)
        metrics["train_time_sec"] = round(train_time, 1)
        metrics["params"] = params
        metrics["oos_period"] = f"{oos_df.index[0]} ~ {oos_df.index[-1]}"
        metrics["n_train_bars"] = len(train_df)
        metrics["n_oos_bars"] = len(oos_df)

        # Print summary
        if metrics["trade_count"] > 0:
            print(f"    Win rate: {metrics['win_rate']:.1%} | "
                  f"Time exit: {metrics['time_exit_rate']:.1%}")
            print(f"    Avg net PnL: {metrics['avg_net_pnl']:+.4%} | "
                  f"Total: {metrics['total_net_pnl']:+.4%}")
            print(f"    Cost share: {metrics['cost_share']:.1%} | "
                  f"Max DD: {metrics['max_drawdown']:.4%}")
            print(f"    95% CI: [{metrics['ci_95_lo']:+.4%}, {metrics['ci_95_hi']:+.4%}]")
            print(f"    Sharpe-like: {metrics['sharpe_like']:.2f}")
            print(f"    Verdict: {metrics['verdict']}")
        else:
            print(f"    [WARN] No trades generated in OOS period")

        # Save per-trade CSV
        if trades:
            trades_df = pd.DataFrame([asdict(t) for t in trades])
            trades_df.to_csv(REPORT_DIR / f"frozen_oos_trades_{coin}.csv", index=False)

        return metrics


# ==================== Perturbation Tester ====================

class PerturbationTester:
    """Test parameter stability around frozen best."""

    def __init__(self, validator: FrozenOOSValidator):
        self.validator = validator

    def test_coin(self, coin: str) -> dict:
        """Test perturbation grid for one coin."""
        base = FROZEN_PARAMS[coin].copy()
        results = []

        # Only perturb k and threshold (model hyperparams stay frozen)
        k_upper_grid = [2.5, 3.0, 3.5]
        k_lower_grid = [0.5, 0.6, 0.7]
        th_grid = [max(0.35, base["stage1_threshold"] - 0.05),
                   base["stage1_threshold"],
                   min(0.65, base["stage1_threshold"] + 0.05)]

        coin_df = self.validator.feature_data[coin]
        train_df, oos_df = self.validator.split_oos(coin_df)

        for ku in k_upper_grid:
            for kl in k_lower_grid:
                rr = ku / kl
                if rr < 0.6:
                    continue

                for th in th_grid:
                    perturbed = base.copy()
                    perturbed["k_upper"] = ku
                    perturbed["k_lower"] = kl
                    perturbed["stage1_threshold"] = th

                    try:
                        preds = self.validator.train_and_predict(train_df, oos_df, perturbed)
                        if preds is None:
                            continue
                        trades = self.validator.simulator.simulate(oos_df, preds, perturbed)
                        metrics = compute_oos_metrics(trades, coin)

                        results.append({
                            "k_upper": ku, "k_lower": kl, "threshold": th,
                            "rr": round(rr, 2),
                            "trade_count": metrics["trade_count"],
                            "avg_net_pnl": metrics["avg_net_pnl"],
                            "total_net_pnl": metrics["total_net_pnl"],
                            "win_rate": metrics["win_rate"],
                            "verdict": metrics["verdict"],
                        })
                    except Exception as e:
                        results.append({
                            "k_upper": ku, "k_lower": kl, "threshold": th,
                            "error": str(e),
                        })

        return {"coin": coin, "perturbation_results": results}


# ==================== Report ====================

def generate_report(results: dict, perturbation: dict):
    """Generate markdown report."""
    lines = [
        "# Frozen OOS Validation Report",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"OOS Period: {OOS_DAYS} days (last 6 weeks)",
        f"Parameters: FROZEN (v3.1_netev R38)",
        f"",
        "---",
        "",
        "## 1. Trade-level OOS Results",
        "",
        "| Coin | Trades | Win% | TimExit% | Avg Net PnL | Total | MaxDD | Sharpe | 95% CI | Verdict |",
        "|------|--------|------|----------|-------------|-------|-------|--------|--------|---------|",
    ]

    for coin in ACTIVE_COINS:
        m = results.get(coin, {})
        if m.get("trade_count", 0) == 0:
            lines.append(f"| {coin} | 0 | - | - | - | - | - | - | - | NO TRADES |")
            continue
        lines.append(
            f"| {coin} | {m['trade_count']} | {m['win_rate']:.1%} | "
            f"{m['time_exit_rate']:.1%} | {m['avg_net_pnl']:+.4%} | "
            f"{m['total_net_pnl']:+.4%} | {m['max_drawdown']:.4%} | "
            f"{m['sharpe_like']:.2f} | [{m['ci_95_lo']:+.4%}, {m['ci_95_hi']:+.4%}] | "
            f"{m['verdict']} |"
        )

    lines += ["", "## 2. PnL Distribution", ""]
    for coin in ACTIVE_COINS:
        m = results.get(coin, {})
        if m.get("trade_count", 0) == 0:
            continue
        lines.append(f"**{coin}**: P5={m['pnl_p5']:+.4%} P25={m['pnl_p25']:+.4%} "
                      f"P50={m['pnl_p50']:+.4%} P75={m['pnl_p75']:+.4%} P95={m['pnl_p95']:+.4%}")

    lines += ["", "## 3. Cost Analysis", ""]
    lines.append("| Coin | Gross PnL | Total Cost | Cost Share | Net PnL |")
    lines.append("|------|-----------|------------|------------|---------|")
    for coin in ACTIVE_COINS:
        m = results.get(coin, {})
        if m.get("trade_count", 0) == 0:
            continue
        lines.append(f"| {coin} | {m['total_gross_pnl']:+.4%} | {m['total_cost']:.4%} | "
                      f"{m['cost_share']:.1%} | {m['total_net_pnl']:+.4%} |")

    lines += ["", "## 4. Regime Breakdown", ""]
    for coin in ACTIVE_COINS:
        m = results.get(coin, {})
        rb = m.get("regime_breakdown", {})
        if not rb:
            continue
        lines.append(f"**{coin}**:")
        for regime, stats in sorted(rb.items()):
            lines.append(f"  - {regime}: {stats['count']} trades, "
                          f"avg={stats['avg_net_pnl']:+.4%}, total={stats['total_net_pnl']:+.4%}")
        lines.append("")

    # Perturbation
    lines += ["## 5. Parameter Perturbation Stability", ""]
    for coin in ACTIVE_COINS:
        p = perturbation.get(coin, {})
        pr = p.get("perturbation_results", [])
        if not pr:
            continue
        lines.append(f"**{coin}** ({len(pr)} configs tested):")
        lines.append("| k_u | k_l | R:R | th | Trades | Avg Net PnL | Verdict |")
        lines.append("|-----|-----|-----|-----|--------|-------------|---------|")
        for r in sorted(pr, key=lambda x: x.get("avg_net_pnl", -999), reverse=True):
            if "error" in r:
                continue
            lines.append(f"| {r['k_upper']} | {r['k_lower']} | {r['rr']} | "
                          f"{r['threshold']} | {r['trade_count']} | "
                          f"{r['avg_net_pnl']:+.4%} | {r['verdict']} |")
        lines.append("")

    # Overall verdict
    verdicts = [results.get(c, {}).get("verdict", "N/A") for c in ACTIVE_COINS]
    pass_count = sum(1 for v in verdicts if "PASS" in v)
    lines += [
        "---",
        "",
        "## Overall Verdict",
        "",
        f"- PASS: {pass_count}/{len(ACTIVE_COINS)} coins",
        f"- Recommendation: {'Proceed to Paper Trading' if pass_count >= 2 else 'Review model / EV calculation'}",
    ]

    report_text = "\n".join(lines)
    with open(REPORT_DIR / "frozen_oos_report.md", "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


# ==================== MAIN ====================

def main():
    start_time = datetime.now()
    print(f"\n{'='*70}")
    print(f"  FROZEN OOS VALIDATION")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  OOS window: last {OOS_DAYS} days")
    print(f"  Coins: {ACTIVE_COINS}")
    print(f"  Parameters: FROZEN (v3.1_netev R38)")
    print(f"  Cost model: Bybit VIP0")
    print(f"  Mode: Trade-level bar-by-bar simulation")
    print(f"{'='*70}")

    try:
        validator = FrozenOOSValidator()
        validator.fetch_data()

        # Phase 1: Frozen OOS for each coin
        print(f"\n{'='*70}")
        print(f"  PHASE 1: FROZEN OOS VALIDATION")
        print(f"{'='*70}")

        results = {}
        for coin in ACTIVE_COINS:
            if coin not in validator.feature_data:
                print(f"  [SKIP] {coin}: no data")
                continue
            metrics = validator.run_coin(coin)
            results[coin] = metrics

        # Phase 2: Perturbation stability
        print(f"\n{'='*70}")
        print(f"  PHASE 2: PARAMETER PERTURBATION STABILITY")
        print(f"{'='*70}")

        perturbation = {}
        tester = PerturbationTester(validator)
        for coin in ACTIVE_COINS:
            if coin not in validator.feature_data:
                continue
            print(f"\n  >> {coin} perturbation test")
            p_results = tester.test_coin(coin)
            perturbation[coin] = p_results
            n_configs = len(p_results.get("perturbation_results", []))
            profitable = sum(1 for r in p_results.get("perturbation_results", [])
                             if r.get("avg_net_pnl", -1) > 0)
            print(f"    {profitable}/{n_configs} configs profitable")

        # Phase 3: Report
        print(f"\n{'='*70}")
        print(f"  GENERATING REPORT")
        print(f"{'='*70}")

        # Save JSON
        full_results = {
            "version": "frozen_oos_v1",
            "generated_at": datetime.now().isoformat(),
            "oos_days": OOS_DAYS,
            "frozen_params": FROZEN_PARAMS,
            "cost_model": {
                "maker_fee": COST_MODEL.fees.maker_fee,
                "taker_fee": COST_MODEL.fees.taker_fee,
                "slippage_entry": COST_MODEL.fees.slippage_entry,
                "slippage_exit_market": COST_MODEL.fees.slippage_exit_market,
                "funding_rate": COST_MODEL.funding.default_rate,
                "miss_fill_prob": COST_MODEL.miss_fill.reject_prob,
            },
            "coin_results": results,
            "perturbation": perturbation,
        }

        with open(REPORT_DIR / "frozen_oos_results.json", "w") as f:
            json.dump(full_results, f, indent=2, default=str)

        # Generate markdown
        report = generate_report(results, perturbation)

        elapsed = (datetime.now() - start_time).total_seconds() / 60
        print(f"\n{'='*70}")
        print(f"  FROZEN OOS COMPLETE ({elapsed:.1f} min)")
        print(f"{'='*70}")

        for coin in ACTIVE_COINS:
            m = results.get(coin, {})
            v = m.get("verdict", "N/A")
            tc = m.get("trade_count", 0)
            net = m.get("avg_net_pnl", 0)
            print(f"  {coin}: {tc} trades, avg_net={net:+.4%}, verdict={v}")

        print(f"\n  Reports saved to: {REPORT_DIR}/")
        print(f"{'='*70}")

    except Exception as e:
        print(f"\n[FATAL] {type(e).__name__}: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
