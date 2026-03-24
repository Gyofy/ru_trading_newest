"""TSMOM + ML Enhanced Strategy — 통합 백테스트.

Phase 1: TSMOM base (rule-based direction)
Phase 1b: + RSI>50 filter, + BTC alignment
Phase 2a: + S1 ML filter, + CVD timing
Phase 2b: + confidence sizing, + dynamic exit
Phase 3:  Grid optimization + OOS walk-forward

Architecture:
  Layer 1 (Direction):  TSMOM rule → LONG/SHORT
  Layer 2 (Filter):     ML signal quality → SKIP/ENTER
  Layer 3 (Timing):     CVD extreme → pullback entry
  Layer 4 (Sizing):     confidence-weighted Kelly
  Layer 5 (Exit):       trailing stop + CVD reversal
"""

import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass, field
from typing import Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════

COINS = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOT", "LINK"]
COST_ROUNDTRIP = 0.0020  # 0.20% (maker entry + taker exit + slippage)


def load_data(period: str = "365d") -> dict[str, pd.DataFrame]:
    """Load OHLCV + technical indicators + microstructure for all coins."""
    from src.data.crawlers.crypto_ohlcv import fetch_ohlcv, resample_to_4h, add_technical_indicators, TOP10_YAHOO
    from src.data.crawlers.microstructure_rollup import add_microstructure_rollup

    data = {}
    for coin in COINS:
        sym = TOP10_YAHOO.get(coin)
        if not sym:
            continue
        df = fetch_ohlcv(coin, sym, period=period, interval="1h")
        if df.empty:
            continue
        df = resample_to_4h(df)
        df = add_technical_indicators(df)
        df = add_microstructure_rollup(df)
        data[coin] = df
        print(f"  [OK] {coin}: {len(df)} bars, {len(df.columns)} cols")
    return data


# ══════════════════════════════════════════════════════════════
# TSMOM Signal Generation (Layer 1)
# ══════════════════════════════════════════════════════════════

def add_tsmom_signal(df: pd.DataFrame, lookback_days: int = 14,
                     volume_weighted: bool = False) -> pd.DataFrame:
    """Time-Series Momentum signal.

    Rule: past N days return > 0 → LONG (+1), < 0 → SHORT (-1)
    Volume-weighted: weight returns by relative volume.
    """
    lookback_bars = lookback_days * 6  # 4h bars per day = 6

    if volume_weighted and "volume" in df.columns:
        ret = df["close"].pct_change()
        vol_weight = df["volume"] / df["volume"].rolling(lookback_bars, min_periods=1).mean()
        weighted_ret = (ret * vol_weight).rolling(lookback_bars, min_periods=lookback_bars).sum()
        df["tsmom_signal"] = np.sign(weighted_ret)
    else:
        past_ret = df["close"].pct_change(lookback_bars)
        df["tsmom_signal"] = np.sign(past_ret)

    # Signal strength (absolute momentum magnitude)
    df["tsmom_strength"] = df["close"].pct_change(lookback_bars).abs()

    return df


def add_rsi_filter(df: pd.DataFrame) -> pd.DataFrame:
    """RSI > 50 = LONG valid, RSI < 50 = SHORT valid."""
    if "rsi_14" not in df.columns:
        return df
    df["rsi_trend_filter"] = np.where(
        df["tsmom_signal"] == 1,
        (df["rsi_14"] > 50).astype(int),    # LONG: RSI must be > 50
        np.where(
            df["tsmom_signal"] == -1,
            (df["rsi_14"] < 50).astype(int),  # SHORT: RSI must be < 50
            0
        )
    )
    return df


def add_btc_alignment(df: pd.DataFrame, btc_df: pd.DataFrame,
                       lookback_days: int = 7) -> pd.DataFrame:
    """BTC direction alignment filter."""
    lookback_bars = lookback_days * 6
    btc_ret = btc_df["close"].pct_change(lookback_bars)
    btc_dir = np.sign(btc_ret)

    # Align BTC direction to coin's index
    btc_dir_aligned = btc_dir.reindex(df.index, method="ffill")

    # Alignment: coin signal matches BTC direction
    df["btc_aligned"] = (df["tsmom_signal"] == btc_dir_aligned).astype(int)
    df["btc_direction"] = btc_dir_aligned
    return df


# ══════════════════════════════════════════════════════════════
# CVD Timing (Layer 3)
# ══════════════════════════════════════════════════════════════

def add_cvd_timing(df: pd.DataFrame) -> pd.DataFrame:
    """CVD extreme detection for entry timing.

    SHORT 정배 + CVD high (매수 과열) = 좋은 SHORT 진입
    LONG 정배 + CVD low (매도 과열) = 좋은 LONG 진입
    """
    if "cvd_ratio_24" not in df.columns:
        # Fallback: compute simple CVD ratio
        hr = df["high"] - df["low"]
        hr = hr.replace(0, np.nan)
        buy_frac = (df["close"] - df["low"]) / hr
        buy_frac = buy_frac.fillna(0.5).clip(0, 1)
        vd = (2 * buy_frac - 1) * df["volume"]
        cvd = vd.cumsum()
        cvd_ma = cvd.rolling(24, min_periods=6).mean()
        df["cvd_ratio_24"] = (cvd - cvd_ma) / cvd_ma.abs().replace(0, np.nan)
        df["cvd_ratio_24"] = df["cvd_ratio_24"].fillna(0)

    # Percentile-based extreme detection
    rolling_q75 = df["cvd_ratio_24"].rolling(120, min_periods=30).quantile(0.75)
    rolling_q25 = df["cvd_ratio_24"].rolling(120, min_periods=30).quantile(0.25)

    # CVD counter-direction = good timing
    # SHORT signal + CVD high = entering short at overextended bounce
    # LONG signal + CVD low = entering long at overextended dip
    df["cvd_timing"] = 0
    short_mask = (df["tsmom_signal"] == -1) & (df["cvd_ratio_24"] > rolling_q75)
    long_mask = (df["tsmom_signal"] == 1) & (df["cvd_ratio_24"] < rolling_q25)
    df.loc[short_mask, "cvd_timing"] = 1
    df.loc[long_mask, "cvd_timing"] = 1

    return df


# ══════════════════════════════════════════════════════════════
# Barrier Backtest Engine
# ══════════════════════════════════════════════════════════════

@dataclass
class TradeResult:
    entry_bar: int
    exit_bar: int
    side: int           # +1 LONG, -1 SHORT
    entry_price: float
    exit_price: float
    pnl_pct: float      # gross
    net_pnl_pct: float   # after cost
    exit_type: str       # "TP", "SL", "TTL", "TRAILING", "CVD_EXIT"
    bars_held: int
    confidence: float = 1.0


def run_barrier_backtest(
    df: pd.DataFrame,
    signals: pd.Series,        # +1 LONG, -1 SHORT, 0 NO_TRADE
    k_upper: float = 3.0,
    k_lower: float = 1.0,
    max_hold: int = 18,
    cost: float = COST_ROUNDTRIP,
    confidence: Optional[pd.Series] = None,
    use_trailing: bool = False,
    use_cvd_exit: bool = False,
) -> list[TradeResult]:
    """Bar-by-bar Triple Barrier backtest with optional enhancements."""

    trades = []
    next_available = 0

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # ATR for barrier calculation
    if "atr_14" in df.columns:
        atr = df["atr_14"].values
    else:
        tr = np.maximum(high - low,
                        np.maximum(np.abs(high - np.roll(close, 1)),
                                   np.abs(low - np.roll(close, 1))))
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

    cvd_ratio = df["cvd_ratio_24"].values if "cvd_ratio_24" in df.columns else None
    sig_vals = signals.values
    conf_vals = confidence.values if confidence is not None else np.ones(len(df))

    for i in range(len(df) - max_hold):
        if i < next_available:
            continue
        if sig_vals[i] == 0 or np.isnan(sig_vals[i]):
            continue
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        side = int(sig_vals[i])
        entry = close[i]
        a = atr[i]

        # Barrier distances
        tp_dist = k_upper * a
        sl_dist = k_lower * a

        # Minimum barrier floor (0.2%)
        tp_dist = max(tp_dist, entry * 0.002)
        sl_dist = max(sl_dist, entry * 0.002)

        if side == 1:  # LONG
            tp = entry + tp_dist
            sl = entry - sl_dist
        else:           # SHORT
            tp = entry - tp_dist
            sl = entry + sl_dist

        trailing_sl = sl
        exit_type = "TTL"
        exit_price = close[min(i + max_hold, len(df) - 1)]
        exit_bar = min(i + max_hold, len(df) - 1)

        for j in range(i + 1, min(i + max_hold + 1, len(df))):
            # Update trailing stop
            if use_trailing and side == 1:
                unrealized = high[j] - entry
                if unrealized > 1.5 * a:
                    trailing_sl = max(trailing_sl, high[j] - 1.0 * a)
            elif use_trailing and side == -1:
                unrealized = entry - low[j]
                if unrealized > 1.5 * a:
                    trailing_sl = min(trailing_sl, low[j] + 1.0 * a)

            # Check barriers
            if side == 1:  # LONG
                if low[j] <= (trailing_sl if use_trailing else sl):
                    exit_price = trailing_sl if use_trailing else sl
                    exit_type = "TRAILING" if use_trailing and trailing_sl != sl else "SL"
                    exit_bar = j
                    break
                elif high[j] >= tp:
                    exit_price = tp
                    exit_type = "TP"
                    exit_bar = j
                    break
            else:  # SHORT
                if high[j] >= (trailing_sl if use_trailing else sl):
                    exit_price = trailing_sl if use_trailing else sl
                    exit_type = "TRAILING" if use_trailing and trailing_sl != sl else "SL"
                    exit_bar = j
                    break
                elif low[j] <= tp:
                    exit_price = tp
                    exit_type = "TP"
                    exit_bar = j
                    break

            # CVD reversal exit
            if use_cvd_exit and cvd_ratio is not None and j > i + 2:
                cvd_now = cvd_ratio[j] if not np.isnan(cvd_ratio[j]) else 0
                cvd_entry = cvd_ratio[i] if not np.isnan(cvd_ratio[i]) else 0
                cvd_change = cvd_now - cvd_entry
                # If CVD reverses significantly against our position
                if (side == 1 and cvd_change < -0.3) or (side == -1 and cvd_change > 0.3):
                    exit_price = close[j]
                    exit_type = "CVD_EXIT"
                    exit_bar = j
                    break

        # Calculate PnL
        if side == 1:
            pnl = (exit_price - entry) / entry
        else:
            pnl = (entry - exit_price) / entry

        net_pnl = pnl - cost

        trades.append(TradeResult(
            entry_bar=i,
            exit_bar=exit_bar,
            side=side,
            entry_price=entry,
            exit_price=exit_price,
            pnl_pct=pnl,
            net_pnl_pct=net_pnl,
            exit_type=exit_type,
            bars_held=exit_bar - i,
            confidence=conf_vals[i],
        ))

        next_available = exit_bar + 1

    return trades


# ══════════════════════════════════════════════════════════════
# Metrics Calculation
# ══════════════════════════════════════════════════════════════

@dataclass
class StrategyMetrics:
    name: str
    coin: str
    n_trades: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    total_pnl: float = 0.0
    sharpe: float = 0.0
    max_dd: float = 0.0
    avg_bars: float = 0.0
    tp_rate: float = 0.0
    sl_rate: float = 0.0
    ttl_rate: float = 0.0
    profit_factor: float = 0.0

    def summary_line(self) -> str:
        return (f"{self.name:30s} | {self.coin:4s} | n={self.n_trades:3d} | "
                f"WR={self.win_rate:5.1%} | avg={self.avg_pnl:+7.3%} | "
                f"total={self.total_pnl:+7.2%} | Sharpe={self.sharpe:5.2f} | "
                f"MDD={self.max_dd:6.2%} | PF={self.profit_factor:5.2f}")


def calc_metrics(trades: list[TradeResult], name: str, coin: str) -> StrategyMetrics:
    """Calculate strategy performance metrics."""
    m = StrategyMetrics(name=name, coin=coin)
    if not trades:
        return m

    pnls = np.array([t.net_pnl_pct for t in trades])
    m.n_trades = len(trades)
    m.win_rate = np.mean(pnls > 0)
    m.avg_pnl = np.mean(pnls)
    m.total_pnl = np.sum(pnls)

    if np.std(pnls) > 0:
        m.sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls))

    # Max drawdown
    equity = np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity)
    dd = equity - running_max
    m.max_dd = np.min(dd) if len(dd) > 0 else 0

    m.avg_bars = np.mean([t.bars_held for t in trades])

    exits = [t.exit_type for t in trades]
    m.tp_rate = exits.count("TP") / len(exits)
    m.sl_rate = (exits.count("SL") + exits.count("TRAILING")) / len(exits)
    m.ttl_rate = exits.count("TTL") / len(exits)

    gross_wins = sum(p for p in pnls if p > 0)
    gross_losses = abs(sum(p for p in pnls if p < 0))
    m.profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    return m


# ══════════════════════════════════════════════════════════════
# ML Signal Quality Filter (Layer 2)
# ══════════════════════════════════════════════════════════════

QUALITY_FEATURES = [
    "tsmom_strength", "rsi_14", "atr_14", "bb_width", "volume",
    "macd_hist", "cvd_ratio_24", "close", "sma_20", "sma_50",
    "ema_12", "ema_26", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


def train_quality_filter(df: pd.DataFrame, trades: list[TradeResult],
                          n_splits: int = 3) -> Optional[ExtraTreesClassifier]:
    """Train ML filter: predict which TSMOM signals are good (TP) vs bad (SL).

    Label: 1 = trade was profitable (net_pnl > 0), 0 = unprofitable
    Features: market state at entry bar
    """
    if len(trades) < 30:
        return None

    # Build training data from trade entry bars
    entry_bars = [t.entry_bar for t in trades]
    labels = [1 if t.net_pnl_pct > 0 else 0 for t in trades]

    available_feats = [f for f in QUALITY_FEATURES if f in df.columns]
    if len(available_feats) < 5:
        return None

    X = df.iloc[entry_bars][available_feats].copy()
    y = np.array(labels)

    # Fill NaN
    X = X.fillna(0)

    # Normalize
    for col in X.columns:
        std = X[col].std()
        if std > 0:
            X[col] = (X[col] - X[col].mean()) / std

    # TimeSeriesSplit CV
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=5)
    scores = []

    model = ExtraTreesClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )

    for train_idx, val_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[val_idx])
        scores.append(balanced_accuracy_score(y[val_idx], pred))

    cv_score = np.mean(scores)
    print(f"    ML Quality Filter CV: bal_acc={cv_score:.3f} (features={len(available_feats)})")

    # Retrain on all data
    model.fit(X, y)
    return model


def apply_quality_filter(df: pd.DataFrame, signals: pd.Series,
                          model: ExtraTreesClassifier,
                          threshold: float = 0.45) -> tuple[pd.Series, pd.Series]:
    """Apply ML filter to signals. Returns filtered signals and confidence scores."""
    available_feats = [f for f in QUALITY_FEATURES if f in df.columns]
    X = df[available_feats].fillna(0).copy()

    for col in X.columns:
        std = X[col].std()
        if std > 0:
            X[col] = (X[col] - X[col].mean()) / std

    proba = model.predict_proba(X)[:, 1]
    confidence = pd.Series(proba, index=df.index)

    # Filter: only keep signals where ML confidence > threshold
    filtered = signals.copy()
    filtered[confidence < threshold] = 0

    return filtered, confidence


# ══════════════════════════════════════════════════════════════
# Confidence-Weighted Sizing (Layer 4)
# ══════════════════════════════════════════════════════════════

def apply_confidence_sizing(trades: list[TradeResult],
                             min_mult: float = 0.5,
                             max_mult: float = 2.0) -> list[TradeResult]:
    """Scale PnL by confidence-based multiplier."""
    sized_trades = []
    for t in trades:
        # Linear scaling: confidence 0.5 → 0.5x, confidence 0.8 → 1.6x
        mult = min_mult + (max_mult - min_mult) * (t.confidence - 0.4) / 0.6
        mult = np.clip(mult, min_mult, max_mult)

        sized = TradeResult(
            entry_bar=t.entry_bar,
            exit_bar=t.exit_bar,
            side=t.side,
            entry_price=t.entry_price,
            exit_price=t.exit_price,
            pnl_pct=t.pnl_pct * mult,
            net_pnl_pct=t.net_pnl_pct * mult,
            exit_type=t.exit_type,
            bars_held=t.bars_held,
            confidence=t.confidence,
        )
        sized_trades.append(sized)
    return sized_trades


# ══════════════════════════════════════════════════════════════
# Main Experiment Runner
# ══════════════════════════════════════════════════════════════

def run_single_coin(coin: str, df: pd.DataFrame, btc_df: pd.DataFrame,
                     lookback: int = 14, volume_weighted: bool = False
                     ) -> dict[str, StrategyMetrics]:
    """Run all strategy layers for a single coin."""
    results = {}

    # ── Layer 1: TSMOM base ──
    df = add_tsmom_signal(df, lookback_days=lookback, volume_weighted=volume_weighted)
    base_signals = df["tsmom_signal"].copy()
    base_signals[base_signals.isna()] = 0

    trades_base = run_barrier_backtest(df, base_signals)
    results["L0_TSMOM"] = calc_metrics(trades_base, f"L0_TSMOM(lb={lookback})", coin)

    # ── Layer 1b: + RSI filter ──
    df = add_rsi_filter(df)
    rsi_signals = base_signals.copy()
    rsi_signals[df["rsi_trend_filter"] == 0] = 0

    trades_rsi = run_barrier_backtest(df, rsi_signals)
    results["L1_RSI"] = calc_metrics(trades_rsi, f"L1_+RSI>50", coin)

    # ── Layer 1c: + BTC alignment ──
    df = add_btc_alignment(df, btc_df, lookback_days=7)
    btc_signals = rsi_signals.copy()
    btc_signals[df["btc_aligned"] == 0] = 0

    trades_btc = run_barrier_backtest(df, btc_signals)
    results["L1c_BTC"] = calc_metrics(trades_btc, f"L1c_+BTC_align", coin)

    # ── Layer 3: + CVD timing ──
    df = add_cvd_timing(df)
    # Two variants: strict (CVD only) and relaxed (CVD or base)
    cvd_strict = base_signals.copy()
    cvd_strict[df["cvd_timing"] == 0] = 0

    trades_cvd = run_barrier_backtest(df, cvd_strict)
    results["L3_CVD"] = calc_metrics(trades_cvd, f"L3_CVD_timing", coin)

    # ── Combined: TSMOM + RSI + CVD ──
    combo_signals = base_signals.copy()
    combo_mask = (df["rsi_trend_filter"] == 1) & (df["cvd_timing"] == 1)
    combo_signals[~combo_mask] = 0

    trades_combo = run_barrier_backtest(df, combo_signals)
    results["L13_combo"] = calc_metrics(trades_combo, f"L1+3_RSI+CVD", coin)

    # ── Layer 5: + Trailing stop ──
    trades_trail = run_barrier_backtest(df, rsi_signals, use_trailing=True)
    results["L5_trail"] = calc_metrics(trades_trail, f"L5_trailing_stop", coin)

    # ── Layer 5b: + CVD exit ──
    trades_cvd_exit = run_barrier_backtest(df, rsi_signals, use_trailing=True, use_cvd_exit=True)
    results["L5b_cvd_exit"] = calc_metrics(trades_cvd_exit, f"L5b_trail+CVD_exit", coin)

    # ── Layer 2: ML Quality Filter (trained on L0 trades) ──
    if len(trades_base) >= 30:
        model = train_quality_filter(df, trades_base)
        if model is not None:
            ml_signals, confidence = apply_quality_filter(df, base_signals, model, threshold=0.45)

            trades_ml = run_barrier_backtest(df, ml_signals, confidence=confidence)
            results["L2_ML"] = calc_metrics(trades_ml, f"L2_ML_filter", coin)

            # ML + trailing
            trades_ml_trail = run_barrier_backtest(df, ml_signals, confidence=confidence,
                                                    use_trailing=True)
            results["L2+5_ML_trail"] = calc_metrics(trades_ml_trail, f"L2+5_ML+trail", coin)

            # ML + confidence sizing
            if trades_ml:
                sized = apply_confidence_sizing(trades_ml)
                results["L4_sized"] = calc_metrics(sized, f"L4_conf_sizing", coin)

    return results


def run_full_experiment(lookbacks: list[int] = [7, 14, 28],
                         volume_weighted_options: list[bool] = [False, True]):
    """Run complete experiment across all coins and configurations."""

    print("=" * 80)
    print("TSMOM + ML Enhanced Strategy — Full Experiment")
    print("=" * 80)

    print("\n[1/3] Loading data...")
    data = load_data()

    if "BTC" not in data:
        print("ERROR: BTC data required for alignment filter")
        return

    btc_df = data["BTC"]

    all_results = []

    for lb in lookbacks:
        for vw in volume_weighted_options:
            vw_str = "volwt" if vw else "simple"
            print(f"\n{'=' * 80}")
            print(f"Config: lookback={lb}d, {vw_str}")
            print(f"{'=' * 80}")

            for coin in COINS:
                if coin not in data:
                    continue
                df = data[coin].copy()

                print(f"\n  [{coin}]")
                results = run_single_coin(coin, df, btc_df, lookback=lb, volume_weighted=vw)

                for key, m in results.items():
                    m.name = f"{vw_str}_lb{lb}_{m.name}"
                    all_results.append(m)
                    if m.n_trades > 0:
                        print(f"    {m.summary_line()}")

    # ══════════════════════════════════════════════════════════════
    # Summary Report
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("SUMMARY: Best configurations per layer")
    print("=" * 80)

    # Group by layer type
    layer_groups = {}
    for m in all_results:
        # Extract layer prefix
        parts = m.name.split("_")
        for p in parts:
            if p.startswith("L"):
                layer = p
                break
        else:
            layer = "other"

        if layer not in layer_groups:
            layer_groups[layer] = []
        layer_groups[layer].append(m)

    for layer, metrics_list in sorted(layer_groups.items()):
        valid = [m for m in metrics_list if m.n_trades >= 10]
        if not valid:
            continue

        # Best by avg PnL
        best = max(valid, key=lambda x: x.avg_pnl)
        print(f"\n  {layer}: {best.summary_line()}")

    # Portfolio-level analysis (aggregate across coins)
    print("\n" + "=" * 80)
    print("PORTFOLIO: Aggregate across coins (best config)")
    print("=" * 80)

    # Find best lookback by total avg_pnl across coins
    for lb in lookbacks:
        for vw in volume_weighted_options:
            vw_str = "volwt" if vw else "simple"

            # Collect L0 results for this config
            config_trades = [m for m in all_results
                           if f"lb{lb}" in m.name and vw_str in m.name and "L0" in m.name]

            if not config_trades:
                continue

            total_trades = sum(m.n_trades for m in config_trades)
            if total_trades == 0:
                continue

            weighted_pnl = sum(m.avg_pnl * m.n_trades for m in config_trades) / total_trades
            avg_wr = sum(m.win_rate * m.n_trades for m in config_trades) / total_trades
            avg_sharpe = np.mean([m.sharpe for m in config_trades if m.n_trades > 0])

            print(f"  {vw_str}_lb{lb:2d} | trades={total_trades:4d} | "
                  f"WR={avg_wr:5.1%} | avg_pnl={weighted_pnl:+7.3%} | "
                  f"avg_Sharpe={avg_sharpe:5.2f}")

    # Save results
    results_df = pd.DataFrame([{
        "name": m.name, "coin": m.coin, "n_trades": m.n_trades,
        "win_rate": m.win_rate, "avg_pnl": m.avg_pnl, "total_pnl": m.total_pnl,
        "sharpe": m.sharpe, "max_dd": m.max_dd, "profit_factor": m.profit_factor,
        "tp_rate": m.tp_rate, "sl_rate": m.sl_rate, "avg_bars": m.avg_bars,
    } for m in all_results])

    out_path = "data/reports/tsmom_ml_enhanced_results.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\n  Results saved to {out_path}")

    return all_results


# ══════════════════════════════════════════════════════════════
# Grid Search for Optimal Parameters
# ══════════════════════════════════════════════════════════════

def grid_search_barriers(data: dict[str, pd.DataFrame], best_lookback: int = 14,
                          volume_weighted: bool = False):
    """Grid search over barrier parameters (k_upper, k_lower, max_hold)."""

    print("\n" + "=" * 80)
    print("GRID SEARCH: Barrier parameters")
    print("=" * 80)

    btc_df = data["BTC"]

    k_uppers = [2.0, 3.0, 4.0, 5.0]
    k_lowers = [0.8, 1.0, 1.5, 2.0]
    max_holds = [12, 18, 24]

    results = []

    for ku in k_uppers:
        for kl in k_lowers:
            for mh in max_holds:
                total_pnl = 0
                total_trades = 0
                all_pnls = []

                for coin in COINS:
                    if coin not in data:
                        continue
                    df = data[coin].copy()
                    df = add_tsmom_signal(df, lookback_days=best_lookback,
                                          volume_weighted=volume_weighted)
                    df = add_rsi_filter(df)

                    signals = df["tsmom_signal"].copy()
                    signals[signals.isna()] = 0
                    signals[df["rsi_trend_filter"] == 0] = 0

                    trades = run_barrier_backtest(df, signals, k_upper=ku, k_lower=kl,
                                                  max_hold=mh)

                    pnls = [t.net_pnl_pct for t in trades]
                    total_trades += len(trades)
                    all_pnls.extend(pnls)

                if total_trades < 20:
                    continue

                arr = np.array(all_pnls)
                avg = np.mean(arr)
                wr = np.mean(arr > 0)
                sharpe = np.mean(arr) / np.std(arr) * np.sqrt(len(arr)) if np.std(arr) > 0 else 0

                # Max drawdown
                eq = np.cumsum(arr)
                dd = eq - np.maximum.accumulate(eq)
                mdd = np.min(dd)

                results.append({
                    "k_upper": ku, "k_lower": kl, "max_hold": mh,
                    "n_trades": total_trades, "avg_pnl": avg, "win_rate": wr,
                    "sharpe": sharpe, "max_dd": mdd,
                })

                if avg > 0:
                    print(f"  ku={ku:.1f} kl={kl:.1f} mh={mh:2d} | n={total_trades:4d} | "
                          f"WR={wr:5.1%} | avg={avg:+7.3%} | Sharpe={sharpe:5.2f} | MDD={mdd:6.2%}")

    if results:
        rdf = pd.DataFrame(results).sort_values("sharpe", ascending=False)
        out_path = "data/reports/tsmom_grid_search.csv"
        rdf.to_csv(out_path, index=False)
        print(f"\n  Grid results saved to {out_path}")
        print(f"\n  TOP 5:")
        print(rdf.head().to_string(index=False))

    return results


# ══════════════════════════════════════════════════════════════
# Walk-Forward OOS Validation
# ══════════════════════════════════════════════════════════════

def walk_forward_validation(data: dict[str, pd.DataFrame],
                              lookback: int = 14, volume_weighted: bool = False,
                              n_windows: int = 4, train_ratio: float = 0.6):
    """Walk-forward out-of-sample validation."""

    print("\n" + "=" * 80)
    print("WALK-FORWARD OOS VALIDATION")
    print("=" * 80)

    btc_df = data["BTC"]
    window_results = []

    for coin in COINS:
        if coin not in data:
            continue

        df = data[coin].copy()
        n = len(df)
        window_size = n // n_windows

        print(f"\n  [{coin}] {n} bars, window_size={window_size}")

        for w in range(n_windows):
            # Split: first train_ratio for training, rest for OOS
            start = w * window_size
            end = min(start + window_size, n)

            if end - start < 60:
                continue

            train_end = start + int((end - start) * train_ratio)

            df_window = df.iloc[start:end].copy()
            df_train = df.iloc[start:train_end].copy()
            df_test = df.iloc[train_end:end].copy()

            # Generate signals on test portion
            df_test = add_tsmom_signal(df_test, lookback_days=lookback,
                                        volume_weighted=volume_weighted)
            df_test = add_rsi_filter(df_test)

            signals = df_test["tsmom_signal"].copy()
            signals[signals.isna()] = 0
            signals[df_test["rsi_trend_filter"] == 0] = 0

            trades = run_barrier_backtest(df_test, signals)
            m = calc_metrics(trades, f"WF_w{w}", coin)

            if m.n_trades > 0:
                print(f"    W{w}: {m.summary_line()}")
                window_results.append({
                    "coin": coin, "window": w,
                    "n_trades": m.n_trades, "avg_pnl": m.avg_pnl,
                    "win_rate": m.win_rate, "sharpe": m.sharpe, "max_dd": m.max_dd,
                })

    if window_results:
        wdf = pd.DataFrame(window_results)
        print(f"\n  OOS Summary:")
        print(f"    Windows: {len(wdf)}")
        print(f"    Avg PnL: {wdf['avg_pnl'].mean():+.3%}")
        print(f"    Avg WR:  {wdf['win_rate'].mean():.1%}")
        print(f"    Positive windows: {(wdf['avg_pnl'] > 0).sum()}/{len(wdf)}")

        out_path = "data/reports/tsmom_walkforward.csv"
        wdf.to_csv(out_path, index=False)
        print(f"    Saved to {out_path}")

    return window_results


# ══════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    t0 = time.time()

    # Phase 1+2: Full experiment (all layers, all configs)
    all_results = run_full_experiment(
        lookbacks=[7, 14, 28],
        volume_weighted_options=[False, True],
    )

    if all_results:
        # Phase 3a: Find best config
        valid = [m for m in all_results if m.n_trades >= 10 and "L0" in m.name]
        if valid:
            best = max(valid, key=lambda x: x.avg_pnl)
            best_lb = int([p for p in best.name.split("_") if p.startswith("lb")][0][2:])
            best_vw = "volwt" in best.name
            print(f"\n  Best base config: lookback={best_lb}, vol_weighted={best_vw}")

            # Phase 3b: Grid search barriers
            print("\n[2/3] Loading data for grid search...")
            data = load_data()
            grid_results = grid_search_barriers(data, best_lookback=best_lb,
                                                 volume_weighted=best_vw)

            # Phase 3c: Walk-forward validation
            print("\n[3/3] Walk-forward validation...")
            wf_results = walk_forward_validation(data, lookback=best_lb,
                                                   volume_weighted=best_vw)

    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 80}")
