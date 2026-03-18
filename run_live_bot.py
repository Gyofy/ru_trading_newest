#!/usr/bin/env python3
"""Autonomous Live Trading Bot -- CLAUDE_CRYPTO_AGENT v4.1.

Connects to Binance USDT-M Futures via ccxt, fetches 4h OHLCV,
computes features, runs ExtraTrees + coin-specific combos,
and manages positions with Triple Barrier (TP/SL/TTL).

Usage:
    python run_live_bot.py --mode paper          # default, testnet
    python run_live_bot.py --mode live           # real money
    python run_live_bot.py --mode paper --equity 10000  # custom equity

Environment Variables:
    BINANCE_API_KEY     -- Binance API key
    BINANCE_API_SECRET  -- Binance API secret
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

# ── Project root setup ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.execution.exchange_adapter import ExchangeAdapter, SYMBOL_MAP
from src.execution.order_ledger import OrderLedger
from src.execution.state_machine import PositionStateMachine, State
from src.execution.risk_engine import RiskEngine, RiskConfig
from src.execution.cost_model import CostModel, FeeSchedule, FundingConfig, MissFillConfig
from src.data.crawlers.crypto_ohlcv import (
    add_technical_indicators,
    _add_decomposition,
    _add_cross_asset_correlation,
)
from src.data.crawlers.signal_features import add_signal_features
from src.data.crawlers.microstructure_rollup import add_microstructure_rollup
from src.utils.feature_policy import is_excluded_feature

# ── Constants ─────────────────────────────────────────────────────
COINS = ["DOT", "ADA", "XRP", "SOL", "LINK"]
BAR_MINUTES = 240           # 4h bars
CYCLE_SECONDS = 2 * 3600    # 2 hours
MAX_HORIZON = 18            # max TTL bars
TRAIN_LOOKBACK_DAYS = 365   # 1 year of 4h data for training
HEARTBEAT_INTERVAL = 60     # seconds

LOG_DIR = PROJECT_ROOT / "data" / "reports" / "live_trading"
HEARTBEAT_FILE = LOG_DIR / "heartbeat.json"
JSONL_LOG = LOG_DIR / "events.jsonl"
MODEL_DIR = LOG_DIR / "models"
STATE_DB = LOG_DIR / "live_ledger.db"

# ── Logging Setup ─────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("live_bot")
logger.setLevel(logging.INFO)

_file_handler = logging.FileHandler(LOG_DIR / "live_bot.log", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
))
logger.addHandler(_file_handler)
logger.addHandler(_stream_handler)


# ══════════════════════════════════════════════════════════════════
#  CONFIG LOADER
# ══════════════════════════════════════════════════════════════════

def load_frozen_config() -> dict:
    """Load frozen_params_v4_1.yaml."""
    cfg_path = PROJECT_ROOT / "config" / "frozen_params_v4_1.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════════
#  JSONL EVENT LOGGER
# ══════════════════════════════════════════════════════════════════

def log_event(event_type: str, data: dict) -> None:
    """Append a structured event to JSONL log."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **data,
    }
    try:
        with open(JSONL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        logger.error(f"Failed to write JSONL: {e}")


def write_heartbeat(status: str, extra: dict | None = None) -> None:
    """Write heartbeat file for external monitoring."""
    hb = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "pid": os.getpid(),
        **(extra or {}),
    }
    try:
        HEARTBEAT_FILE.write_text(json.dumps(hb, indent=2), encoding="utf-8")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════
#  REGIME FILTER
# ══════════════════════════════════════════════════════════════════

def detect_regime(df: pd.DataFrame, lookback: int = 24) -> str:
    """Detect market regime from recent bars.

    Returns one of: TREND_UP, TREND_DOWN, RANGE_HIGH, RANGE_LOW

    Logic:
        - ADX > 25 + positive DI diff -> TREND_UP
        - ADX > 25 + negative DI diff -> TREND_DOWN
        - ADX <= 25 + ATR/close > median -> RANGE_HIGH
        - ADX <= 25 + ATR/close <= median -> RANGE_LOW
    """
    if len(df) < lookback:
        return "UNKNOWN"

    recent = df.iloc[-lookback:]
    close = recent["close"]

    # ADX calculation
    if "adx_14" in df.columns:
        adx = df["adx_14"].iloc[-1]
    else:
        adx = 20.0  # fallback

    # DI diff
    if "di_diff" in df.columns:
        di_diff = df["di_diff"].iloc[-1]
    elif "plus_di_14" in df.columns and "minus_di_14" in df.columns:
        di_diff = df["plus_di_14"].iloc[-1] - df["minus_di_14"].iloc[-1]
    else:
        di_diff = 0.0

    # Volatility relative
    if "atr_14" in df.columns:
        atr_pct = df["atr_14"].iloc[-1] / (close.iloc[-1] + 1e-10)
        median_atr_pct = (df["atr_14"].iloc[-96:] / (df["close"].iloc[-96:] + 1e-10)).median()
    else:
        atr_pct = close.pct_change().abs().rolling(14).mean().iloc[-1]
        median_atr_pct = atr_pct

    if adx > 25:
        return "TREND_UP" if di_diff > 0 else "TREND_DOWN"
    else:
        return "RANGE_HIGH" if atr_pct > median_atr_pct else "RANGE_LOW"


# ══════════════════════════════════════════════════════════════════
#  DATA FETCHING (ccxt)
# ══════════════════════════════════════════════════════════════════

async def fetch_ohlcv_ccxt(
    exchange,
    symbol: str,
    timeframe: str = "4h",
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch OHLCV from Binance via ccxt async."""
    ccxt_sym = SYMBOL_MAP.get(symbol, f"{symbol}/USDT:USDT")
    max_retries = 3

    for attempt in range(max_retries):
        try:
            ohlcv = await exchange._exchange.fetch_ohlcv(
                ccxt_sym, timeframe=timeframe, limit=limit,
            )
            if not ohlcv:
                raise ValueError(f"Empty OHLCV for {symbol}")

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            return df

        except Exception as e:
            logger.warning(f"[OHLCV] {symbol} attempt {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)

    logger.error(f"[OHLCV] {symbol} all retries failed")
    return pd.DataFrame()


async def fetch_all_ohlcv(
    exchange: ExchangeAdapter,
    coins: list[str],
    limit: int = 500,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for all coins concurrently."""
    tasks = {
        coin: fetch_ohlcv_ccxt(exchange, coin, "4h", limit)
        for coin in coins
    }

    results = {}
    for coin, coro in tasks.items():
        try:
            df = await coro
            if not df.empty:
                results[coin] = df
                logger.info(f"[OHLCV] {coin}: {len(df)} bars, latest ${df['close'].iloc[-1]:,.4f}")
            else:
                logger.warning(f"[OHLCV] {coin}: empty")
        except Exception as e:
            logger.error(f"[OHLCV] {coin}: {e}")

    return results


# ══════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

def compute_features(
    raw_data: dict[str, pd.DataFrame],
    excluded_keywords: list[str],
) -> dict[str, pd.DataFrame]:
    """Compute all features for each coin."""
    featured = {}

    for coin, df in raw_data.items():
        if len(df) < 100:
            logger.warning(f"[Features] {coin}: too few bars ({len(df)}), skipping")
            continue

        try:
            df = df.copy()

            # Technical indicators
            df = add_technical_indicators(df)

            # Decomposition with 4h period (6 bars per day)
            df = _add_decomposition(df, period=42)

            # Signal features (wavelet, FFT, hilbert, entropy, etc.)
            df = add_signal_features(df, verbose=False)

            # Microstructure rollup (CVD, OFI, VPIN, Roll, Amihud)
            df = add_microstructure_rollup(df, verbose=False)

            # Clean NaN/inf
            df.ffill(inplace=True)
            df.bfill(inplace=True)
            df.replace([np.inf, -np.inf], 0, inplace=True)

            # Drop excluded features
            cols_to_drop = [c for c in df.columns if is_excluded_feature(c)]
            if cols_to_drop:
                df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

            featured[coin] = df
            logger.info(f"[Features] {coin}: {len(df)} bars, {len(df.columns)} features")

        except Exception as e:
            logger.error(f"[Features] {coin}: {e}\n{traceback.format_exc()}")

    # Cross-asset correlation
    if len(featured) >= 2:
        # Need BTC and ETH for correlation -- fetch separately if needed
        for coin in list(featured.keys()):
            try:
                featured[coin] = _add_cross_asset_correlation(featured, coin, window=20)
            except Exception:
                pass

    return featured


# ══════════════════════════════════════════════════════════════════
#  LABELING (Fee-Aware Triple Barrier)
# ══════════════════════════════════════════════════════════════════

def create_labels(
    df: pd.DataFrame,
    k_upper: float = 3.0,
    k_lower: float = 0.6,
    max_horizon: int = 18,
    min_barrier_pct: float = 0.002,
) -> pd.Series:
    """Fee-aware triple barrier labeling.

    UP=1, DOWN=0. HOLD samples are dropped.
    Returns labels aligned to df.index.
    """
    close = df["close"].values
    atr = df["atr_14"].values if "atr_14" in df.columns else np.full(len(df), 0.01)
    n = len(df)
    labels = np.full(n, np.nan)

    for i in range(n - max_horizon):
        entry = close[i]
        a = atr[i]
        if np.isnan(a) or a < 1e-10:
            a = entry * 0.01

        upper_dist = max(k_upper * a, entry * min_barrier_pct)
        lower_dist = max(k_lower * a, entry * min_barrier_pct)
        tp = entry + upper_dist
        sl = entry - lower_dist

        for j in range(1, max_horizon + 1):
            idx = i + j
            if idx >= n:
                break
            h = close[idx]
            if h >= tp:
                labels[i] = 1  # UP (BUY wins)
                break
            elif h <= sl:
                labels[i] = 0  # DOWN (SELL wins)
                break

        # If TTL expires without hitting barrier, check net movement
        if np.isnan(labels[i]):
            end_price = close[min(i + max_horizon, n - 1)]
            ret = (end_price - entry) / entry
            fee_threshold = 0.002  # 0.2%
            if ret > fee_threshold:
                labels[i] = 1
            elif ret < -fee_threshold:
                labels[i] = 0
            # else: remains NaN (HOLD, dropped)

    return pd.Series(labels, index=df.index)


# ══════════════════════════════════════════════════════════════════
#  MODEL TRAINING & PREDICTION
# ══════════════════════════════════════════════════════════════════

@dataclass
class TrainedModel:
    """Holds a trained model and its metadata."""
    coin: str
    model: object
    feature_columns: list[str]
    train_date: str
    train_samples: int
    train_score: float


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Extract numeric feature columns, excluding OHLCV and labels."""
    exclude = {"open", "high", "low", "close", "volume", "label"}
    cols = []
    for c in df.columns:
        if c.lower() in exclude:
            continue
        if df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int):
            cols.append(c)
    return cols


def train_model_for_coin(
    df: pd.DataFrame,
    coin: str,
    config: dict,
    coin_config: dict,
) -> TrainedModel | None:
    """Train ExtraTrees + optional secondary model for a coin."""
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import balanced_accuracy_score

    # Create labels
    common = config["common"]
    labels = create_labels(
        df,
        k_upper=common["k_upper"],
        k_lower=common["k_lower"],
        max_horizon=common["max_horizon"],
        min_barrier_pct=common["min_barrier_pct"],
    )

    # Merge labels
    df = df.copy()
    df["label"] = labels
    df.dropna(subset=["label"], inplace=True)

    if len(df) < 200:
        logger.warning(f"[Train] {coin}: too few labeled samples ({len(df)})")
        return None

    feature_cols = get_feature_columns(df)

    # Limit features per coin config
    max_feat = coin_config.get("max_features", 80)
    if len(feature_cols) > max_feat:
        # Use MI or variance-based selection
        from sklearn.feature_selection import mutual_info_classif
        X_sel = df[feature_cols].values
        y_sel = df["label"].values.astype(int)

        # Quick MI computation
        try:
            mi = mutual_info_classif(X_sel, y_sel, random_state=42, n_neighbors=5)
            mi_idx = np.argsort(mi)[::-1][:max_feat]
            feature_cols = [feature_cols[i] for i in mi_idx]
        except Exception:
            feature_cols = feature_cols[:max_feat]

    X = df[feature_cols].values
    y = df["label"].values.astype(int)

    # Train ExtraTrees (primary)
    et = ExtraTreesClassifier(
        n_estimators=coin_config.get("n_estimators", 300),
        max_depth=coin_config.get("max_depth_tree", 8),
        min_samples_leaf=coin_config.get("min_child_samples", 10),
        class_weight="balanced",
        n_jobs=6,
        random_state=42,
    )

    # TimeSeriesSplit validation
    tscv = TimeSeriesSplit(n_splits=3, gap=12)
    val_scores = []
    for train_idx, val_idx in tscv.split(X):
        et_cv = ExtraTreesClassifier(
            n_estimators=coin_config.get("n_estimators", 300),
            max_depth=coin_config.get("max_depth_tree", 8),
            min_samples_leaf=coin_config.get("min_child_samples", 10),
            class_weight="balanced",
            n_jobs=6,
            random_state=42,
        )
        et_cv.fit(X[train_idx], y[train_idx])
        preds = et_cv.predict(X[val_idx])
        ba = balanced_accuracy_score(y[val_idx], preds)
        val_scores.append(ba)

    avg_ba = np.mean(val_scores)
    logger.info(f"[Train] {coin}: CV balanced_accuracy = {avg_ba:.4f} (3 folds)")

    # Fit on full data
    et.fit(X, y)

    trained = TrainedModel(
        coin=coin,
        model=et,
        feature_columns=feature_cols,
        train_date=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        train_samples=len(X),
        train_score=round(avg_ba, 4),
    )

    log_event("model_trained", {
        "coin": coin,
        "samples": len(X),
        "features": len(feature_cols),
        "cv_balanced_accuracy": round(avg_ba, 4),
        "combo": coin_config.get("model_combo", "et"),
    })

    return trained


def predict_signal(
    model: TrainedModel,
    df: pd.DataFrame,
) -> tuple[str, float]:
    """Generate prediction from latest bar.

    Returns (side, probability).
    side: "BUY", "SELL", or "HOLD"
    """
    feature_cols = model.feature_columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning(f"[Predict] {model.coin}: {len(missing)} missing features")
        # Add missing as zeros
        for c in missing:
            df[c] = 0.0

    X = df[feature_cols].iloc[[-1]].values

    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    proba = model.model.predict_proba(X)[0]

    # Binary: class 0 = DOWN, class 1 = UP
    p_up = proba[1] if len(proba) > 1 else proba[0]
    p_down = 1 - p_up

    if p_up > 0.5:
        return "BUY", float(p_up)
    elif p_down > 0.5:
        return "SELL", float(p_down)
    else:
        return "HOLD", 0.5


# ══════════════════════════════════════════════════════════════════
#  MICROSTRUCTURE SIZING ADJUSTMENTS
# ══════════════════════════════════════════════════════════════════

def apply_microstructure_sizing(
    base_qty: float,
    df: pd.DataFrame,
    side: str,
    micro_config: dict,
) -> float:
    """Apply CVD and OFI-based sizing multipliers from v4.1 config."""
    qty = base_qty

    # CVD filter
    if micro_config.get("cvd_filter_enabled", False):
        cvd_cols = [c for c in df.columns if c.startswith("cvd_ratio_")]
        if cvd_cols:
            latest_cvd = df[cvd_cols[0]].iloc[-1]
            q25 = df[cvd_cols[0]].quantile(0.25)
            q75 = df[cvd_cols[0]].quantile(0.75)

            if side == "BUY":
                if latest_cvd > q75:
                    qty *= micro_config.get("cvd_long_mult_high", 0.6)
                elif latest_cvd < q25:
                    qty *= micro_config.get("cvd_long_mult_low", 1.2)
            else:  # SELL
                if latest_cvd < q25:
                    qty *= micro_config.get("cvd_long_mult_high", 0.6)
                elif latest_cvd > q75:
                    qty *= micro_config.get("cvd_long_mult_low", 1.2)

    # OFI timing
    if micro_config.get("ofi_timing_enabled", False):
        ofi_cols = [c for c in df.columns if c.startswith("ofi_sum_")]
        if ofi_cols:
            latest_ofi = df[ofi_cols[0]].iloc[-1]
            q67 = df[ofi_cols[0]].quantile(0.67)
            if (side == "BUY" and latest_ofi > q67) or \
               (side == "SELL" and latest_ofi < -q67):
                qty *= micro_config.get("ofi_timing_mult", 1.1)

    return qty


# ══════════════════════════════════════════════════════════════════
#  POSITION MANAGER
# ══════════════════════════════════════════════════════════════════

@dataclass
class OpenPosition:
    """Track an open position in memory."""
    coin: str
    side: str
    entry_price: float
    qty: float
    sl_price: float
    tp_price: float
    entry_time: datetime
    ttl_bars: int
    bars_held: int = 0
    entry_order_id: str = ""
    sl_order_id: str = ""
    tp_order_id: str = ""


class PositionManager:
    """Manages non-overlapping positions for all coins."""

    def __init__(self):
        self.positions: dict[str, OpenPosition] = {}

    def has_position(self, coin: str) -> bool:
        return coin in self.positions

    def add_position(self, pos: OpenPosition) -> None:
        self.positions[pos.coin] = pos

    def remove_position(self, coin: str) -> OpenPosition | None:
        return self.positions.pop(coin, None)

    def get_position(self, coin: str) -> OpenPosition | None:
        return self.positions.get(coin)

    def all_positions(self) -> list[OpenPosition]:
        return list(self.positions.values())

    def increment_bars(self) -> list[str]:
        """Increment bars_held for all positions. Return coins that hit TTL."""
        expired = []
        for coin, pos in self.positions.items():
            pos.bars_held += 1
            if pos.bars_held >= pos.ttl_bars:
                expired.append(coin)
        return expired


# ══════════════════════════════════════════════════════════════════
#  KILL SWITCH / DRAWDOWN TRACKER
# ══════════════════════════════════════════════════════════════════

class DrawdownTracker:
    """Track daily and weekly P&L for kill switch logic."""

    def __init__(
        self,
        daily_limit_pct: float = 0.02,
        weekly_limit_pct: float = 0.05,
    ):
        self.daily_limit = daily_limit_pct
        self.weekly_limit = weekly_limit_pct
        self.daily_start_equity: float = 0.0
        self.weekly_start_equity: float = 0.0
        self.daily_pnl: float = 0.0
        self.weekly_pnl: float = 0.0
        self.killed: bool = False
        self.kill_reason: str = ""
        self._last_daily_reset: date | None = None
        self._last_weekly_reset: date | None = None

    def set_initial_equity(self, equity: float) -> None:
        self.daily_start_equity = equity
        self.weekly_start_equity = equity

    def record_trade_pnl(self, pnl_usdt: float) -> None:
        self.daily_pnl += pnl_usdt
        self.weekly_pnl += pnl_usdt

    def check_limits(self, current_equity: float) -> tuple[bool, str]:
        """Returns (ok, reason). ok=False means kill switch should activate."""
        if self.daily_start_equity <= 0:
            return True, ""

        daily_dd = -self.daily_pnl / self.daily_start_equity
        if daily_dd >= self.daily_limit:
            self.killed = True
            self.kill_reason = f"Daily drawdown {daily_dd:.2%} >= {self.daily_limit:.0%}"
            return False, self.kill_reason

        weekly_dd = -self.weekly_pnl / self.weekly_start_equity
        if weekly_dd >= self.weekly_limit:
            self.killed = True
            self.kill_reason = f"Weekly drawdown {weekly_dd:.2%} >= {self.weekly_limit:.0%}"
            return False, self.kill_reason

        return True, ""

    def maybe_reset(self, equity: float) -> None:
        """Reset daily/weekly counters at appropriate times."""
        today = datetime.now(timezone.utc).date()

        if self._last_daily_reset != today:
            self.daily_pnl = 0.0
            self.daily_start_equity = equity
            self._last_daily_reset = today
            if self.killed and "Daily" in self.kill_reason:
                self.killed = False
                self.kill_reason = ""
                logger.info("[KillSwitch] Daily reset -- lifted")

        # Weekly reset on Monday
        if today.weekday() == 0 and self._last_weekly_reset != today:
            self.weekly_pnl = 0.0
            self.weekly_start_equity = equity
            self._last_weekly_reset = today
            if self.killed and "Weekly" in self.kill_reason:
                self.killed = False
                self.kill_reason = ""
                logger.info("[KillSwitch] Weekly reset -- lifted")


# ══════════════════════════════════════════════════════════════════
#  MAIN BOT CLASS
# ══════════════════════════════════════════════════════════════════

class LiveTradingBot:
    """Autonomous trading bot orchestrator."""

    def __init__(
        self,
        mode: str = "paper",
        initial_equity: float = 10000.0,
    ):
        self.mode = mode
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.running = False
        self._shutdown_event = asyncio.Event()

        # Load config
        self.config = load_frozen_config()
        self.common = self.config["common"]
        self.coin_configs = self.config["coins"]
        self.micro_config = self.config.get("microstructure", {})
        self.blocked_regimes = self.config.get("blocked_regimes", ["RANGE_LOW"])
        self.excluded_keywords = self.config.get("excluded_feature_keywords", [])

        # Cost model from config
        cm_cfg = self.config.get("cost_model", {})
        self.cost_model = CostModel(
            fee_schedule=FeeSchedule(
                maker_fee=cm_cfg.get("maker_fee", 0.0002),
                taker_fee=cm_cfg.get("taker_fee", 0.00055),
                slippage_entry=cm_cfg.get("slippage_entry", 0.0003),
                slippage_exit_limit=cm_cfg.get("slippage_exit_limit", 0.0001),
                slippage_exit_market=cm_cfg.get("slippage_exit_market", 0.0005),
            ),
            funding_config=FundingConfig(
                interval_hours=cm_cfg.get("funding_interval_hours", 8.0),
                default_rate=cm_cfg.get("funding_default_rate", 0.0001),
            ),
            miss_fill_config=MissFillConfig(
                reject_prob=cm_cfg.get("miss_fill_reject_prob", 0.15),
                missed_ev_pct=cm_cfg.get("miss_fill_missed_ev", 0.0015),
            ),
        )

        # Risk engine
        self.risk_engine = RiskEngine(RiskConfig(
            risk_frac=self.common["risk_frac"],
            daily_drawdown_pct=0.02,
            weekly_drawdown_pct=0.05,
            leverage=1.0,
        ))

        # Components
        self.exchange: ExchangeAdapter | None = None
        self.ledger = OrderLedger(db_path=STATE_DB)
        self.pos_manager = PositionManager()
        self.dd_tracker = DrawdownTracker()
        self.models: dict[str, TrainedModel] = {}
        self.last_train_date: date | None = None
        self.raw_data: dict[str, pd.DataFrame] = {}
        self.featured_data: dict[str, pd.DataFrame] = {}

        # Cycle counter
        self.cycle_count = 0

    async def initialize(self) -> None:
        """Initialize exchange connection and load models."""
        api_key = os.environ.get("BINANCE_API_KEY", "")
        api_secret = os.environ.get("BINANCE_API_SECRET", "")

        if not api_key or not api_secret:
            logger.error("BINANCE_API_KEY and BINANCE_API_SECRET must be set")
            raise ValueError("Missing Binance API credentials")

        exchange_mode = "sandbox" if self.mode == "paper" else "live"
        self.exchange = ExchangeAdapter(
            mode=exchange_mode,
            api_key=api_key,
            secret=api_secret,
        )
        await self.exchange.initialize()

        # Fetch initial balance
        try:
            balance = await self.exchange.fetch_balance()
            if balance["total"] > 0:
                self.equity = balance["total"]
                self.initial_equity = self.equity
            logger.info(f"[Init] Balance: {self.equity:.2f} USDT")
        except Exception as e:
            logger.warning(f"[Init] Could not fetch balance: {e}, using {self.equity:.2f}")

        self.dd_tracker.set_initial_equity(self.equity)
        self.risk_engine.set_initial_equity(self.equity)

        log_event("bot_started", {
            "mode": self.mode,
            "equity": self.equity,
            "coins": COINS,
            "config_version": self.config.get("version", "unknown"),
        })

        logger.info(f"{'='*60}")
        logger.info(f"  LIVE TRADING BOT STARTED")
        logger.info(f"  Mode:    {self.mode}")
        logger.info(f"  Equity:  {self.equity:.2f} USDT")
        logger.info(f"  Coins:   {', '.join(COINS)}")
        logger.info(f"  Config:  v{self.config.get('version', '?')}")
        logger.info(f"  Cycle:   every {CYCLE_SECONDS//3600}h")
        logger.info(f"{'='*60}")

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("[Shutdown] Shutting down...")
        self.running = False
        self._shutdown_event.set()

        # Close all positions in paper mode
        if self.mode == "paper":
            for coin in list(self.pos_manager.positions.keys()):
                pos = self.pos_manager.get_position(coin)
                if pos:
                    logger.info(f"[Shutdown] Closing {coin} position")
                    await self._close_position(coin, "SHUTDOWN")

        if self.exchange:
            await self.exchange.close()

        self.ledger.close()
        write_heartbeat("stopped")
        log_event("bot_stopped", {"reason": "shutdown"})
        logger.info("[Shutdown] Complete")

    # ── Main Loop ─────────────────────────────────────────────

    async def run(self) -> None:
        """Main bot loop. Runs every 2 hours."""
        self.running = True
        last_heartbeat = 0.0

        while self.running:
            try:
                cycle_start = time.time()
                self.cycle_count += 1

                logger.info(f"\n{'='*60}")
                logger.info(f"  CYCLE #{self.cycle_count} @ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
                logger.info(f"{'='*60}")

                # Reset daily/weekly limits
                self.dd_tracker.maybe_reset(self.equity)

                # Check kill switch
                if self.dd_tracker.killed:
                    logger.warning(f"[KillSwitch] ACTIVE: {self.dd_tracker.kill_reason}")
                    log_event("kill_switch_active", {"reason": self.dd_tracker.kill_reason})
                    # Close all positions
                    for coin in list(self.pos_manager.positions.keys()):
                        await self._close_position(coin, "KILL_SWITCH")
                    await self._sleep_until_next_cycle(cycle_start)
                    continue

                # 1. Fetch data
                await self._step_fetch_data()

                # 2. Train/retrain models if needed
                await self._step_train_models()

                # 3. Check exits on open positions
                await self._step_check_exits()

                # 4. Generate signals and enter new positions
                await self._step_generate_signals()

                # 5. Update equity
                await self._step_update_equity()

                # Log cycle summary
                cycle_elapsed = time.time() - cycle_start
                log_event("cycle_complete", {
                    "cycle": self.cycle_count,
                    "elapsed_s": round(cycle_elapsed, 1),
                    "equity": round(self.equity, 2),
                    "open_positions": len(self.pos_manager.positions),
                    "positions": list(self.pos_manager.positions.keys()),
                })

                logger.info(
                    f"[Cycle] #{self.cycle_count} complete in {cycle_elapsed:.1f}s | "
                    f"Equity: {self.equity:.2f} | "
                    f"Open: {list(self.pos_manager.positions.keys())}"
                )

                # Wait for next cycle
                await self._sleep_until_next_cycle(cycle_start)

            except asyncio.CancelledError:
                logger.info("[Loop] Cancelled")
                break
            except Exception as e:
                logger.error(f"[Loop] Cycle error: {e}\n{traceback.format_exc()}")
                log_event("cycle_error", {"error": str(e), "cycle": self.cycle_count})
                await self._sleep_until_next_cycle(time.time())

    async def _sleep_until_next_cycle(self, cycle_start: float) -> None:
        """Sleep until next 2h cycle, writing heartbeats periodically."""
        elapsed = time.time() - cycle_start
        remaining = max(10, CYCLE_SECONDS - elapsed)
        end_time = time.time() + remaining

        while time.time() < end_time and self.running:
            write_heartbeat("running", {
                "cycle": self.cycle_count,
                "equity": round(self.equity, 2),
                "open_positions": len(self.pos_manager.positions),
                "next_cycle_s": int(end_time - time.time()),
            })
            sleep_chunk = min(HEARTBEAT_INTERVAL, end_time - time.time())
            if sleep_chunk <= 0:
                break
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=sleep_chunk,
                )
                break  # shutdown requested
            except asyncio.TimeoutError:
                pass

    # ── Step 1: Fetch Data ────────────────────────────────────

    async def _step_fetch_data(self) -> None:
        """Fetch 4h OHLCV for all coins."""
        logger.info("[Step 1] Fetching OHLCV data...")

        # Also fetch BTC and ETH for cross-correlation if not in COINS
        fetch_coins = list(set(COINS + ["BTC", "ETH"]))

        self.raw_data = await fetch_all_ohlcv(
            self.exchange, fetch_coins, limit=500,
        )

        if not self.raw_data:
            logger.error("[Step 1] No data fetched!")
            return

        # Compute features
        self.featured_data = compute_features(
            self.raw_data, self.excluded_keywords,
        )

        logger.info(f"[Step 1] Data ready: {list(self.featured_data.keys())}")

    # ── Step 2: Train Models ──────────────────────────────────

    async def _step_train_models(self) -> None:
        """Train or retrain models. Retrain daily."""
        today = datetime.now(timezone.utc).date()

        need_retrain = (
            not self.models
            or self.last_train_date != today
        )

        if not need_retrain:
            logger.info("[Step 2] Models up to date, skipping retrain")
            return

        logger.info("[Step 2] Training models...")

        for coin in COINS:
            if coin not in self.featured_data:
                logger.warning(f"[Train] {coin}: no data, skipping")
                continue

            df = self.featured_data[coin]
            coin_cfg = self.coin_configs.get(coin, {})

            try:
                model = train_model_for_coin(df, coin, self.config, coin_cfg)
                if model:
                    self.models[coin] = model
                    logger.info(
                        f"[Train] {coin}: OK (score={model.train_score:.4f}, "
                        f"samples={model.train_samples})"
                    )
                else:
                    logger.warning(f"[Train] {coin}: training failed")
            except Exception as e:
                logger.error(f"[Train] {coin}: {e}\n{traceback.format_exc()}")

        self.last_train_date = today
        logger.info(f"[Step 2] Models trained: {list(self.models.keys())}")

    # ── Step 3: Check Exits ───────────────────────────────────

    async def _step_check_exits(self) -> None:
        """Check SL/TP/TTL for open positions."""
        logger.info("[Step 3] Checking exits...")

        # Increment bar count for all positions
        expired_coins = self.pos_manager.increment_bars()

        for coin in list(self.pos_manager.positions.keys()):
            pos = self.pos_manager.get_position(coin)
            if not pos:
                continue

            # Get current price
            try:
                ticker = await self.exchange.fetch_ticker(coin)
                current_price = ticker["last"]
            except Exception as e:
                logger.error(f"[Exit] {coin}: failed to get price: {e}")
                continue

            # Check SL
            if pos.side == "BUY" and current_price <= pos.sl_price:
                logger.info(f"[Exit] {coin} SL HIT: {current_price:.4f} <= {pos.sl_price:.4f}")
                await self._close_position(coin, "SL_HIT", current_price)
                continue

            if pos.side == "SELL" and current_price >= pos.sl_price:
                logger.info(f"[Exit] {coin} SL HIT: {current_price:.4f} >= {pos.sl_price:.4f}")
                await self._close_position(coin, "SL_HIT", current_price)
                continue

            # Check TP
            if pos.side == "BUY" and current_price >= pos.tp_price:
                logger.info(f"[Exit] {coin} TP HIT: {current_price:.4f} >= {pos.tp_price:.4f}")
                await self._close_position(coin, "TP_HIT", current_price)
                continue

            if pos.side == "SELL" and current_price <= pos.tp_price:
                logger.info(f"[Exit] {coin} TP HIT: {current_price:.4f} <= {pos.tp_price:.4f}")
                await self._close_position(coin, "TP_HIT", current_price)
                continue

            # Check TTL
            if coin in expired_coins:
                logger.info(f"[Exit] {coin} TTL EXPIRED after {pos.bars_held} bars")
                await self._close_position(coin, "TIME_STOP", current_price)
                continue

            # Log position status
            if pos.side == "BUY":
                unrealized_pct = (current_price - pos.entry_price) / pos.entry_price
            else:
                unrealized_pct = (pos.entry_price - current_price) / pos.entry_price

            logger.info(
                f"[Position] {coin} {pos.side} | entry={pos.entry_price:.4f} | "
                f"current={current_price:.4f} | pnl={unrealized_pct:.2%} | "
                f"bars={pos.bars_held}/{pos.ttl_bars}"
            )

    async def _close_position(
        self, coin: str, reason: str, exit_price: float = 0.0,
    ) -> None:
        """Close a position and record P&L."""
        pos = self.pos_manager.get_position(coin)
        if not pos:
            return

        # Get exit price if not provided
        if exit_price <= 0:
            try:
                ticker = await self.exchange.fetch_ticker(coin)
                exit_price = ticker["last"]
            except Exception:
                exit_price = pos.entry_price  # fallback

        # Calculate P&L
        if pos.side == "BUY":
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        pnl_usdt = pnl_pct * pos.qty * pos.entry_price

        # Record
        self.dd_tracker.record_trade_pnl(pnl_usdt)
        self.ledger.record_pnl(coin, pnl_usdt, 0.0)

        # Place market close order (live mode)
        if self.mode == "live":
            exit_side = "SELL" if pos.side == "BUY" else "BUY"
            order_id = ExchangeAdapter.make_order_id(coin, exit_side, prefix="exit")
            try:
                result = await self.exchange.market_close(
                    coin, exit_side, pos.qty, order_id,
                )
                if result.get("success"):
                    exit_price = result.get("fill_price", exit_price)
                    logger.info(f"[Close] {coin} market close OK: {exit_price:.4f}")
                else:
                    logger.error(f"[Close] {coin} market close FAILED: {result}")
            except Exception as e:
                logger.error(f"[Close] {coin} market close error: {e}")

            # Cancel protective orders
            if pos.sl_order_id:
                try:
                    await self.exchange.cancel_order(coin, order_link_id=pos.sl_order_id)
                except Exception:
                    pass
            if pos.tp_order_id:
                try:
                    await self.exchange.cancel_order(coin, order_link_id=pos.tp_order_id)
                except Exception:
                    pass

        # Remove from manager
        self.pos_manager.remove_position(coin)

        log_event("position_closed", {
            "coin": coin,
            "reason": reason,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "pnl_pct": round(pnl_pct, 6),
            "pnl_usdt": round(pnl_usdt, 4),
            "bars_held": pos.bars_held,
        })

        logger.info(
            f"[Close] {coin} {pos.side} closed ({reason}) | "
            f"entry={pos.entry_price:.4f} exit={exit_price:.4f} | "
            f"pnl={pnl_pct:.2%} ({pnl_usdt:+.2f} USDT)"
        )

        # Check kill switch after recording P&L
        ok, reason_msg = self.dd_tracker.check_limits(self.equity)
        if not ok:
            logger.warning(f"[KillSwitch] TRIGGERED: {reason_msg}")
            log_event("kill_switch_triggered", {"reason": reason_msg})

    # ── Step 4: Generate Signals ──────────────────────────────

    async def _step_generate_signals(self) -> None:
        """Generate signals and enter new positions."""
        logger.info("[Step 4] Generating signals...")

        if self.dd_tracker.killed:
            logger.warning("[Step 4] Kill switch active, no new entries")
            return

        for coin in COINS:
            try:
                await self._process_coin_signal(coin)
            except Exception as e:
                logger.error(f"[Signal] {coin}: {e}\n{traceback.format_exc()}")

    async def _process_coin_signal(self, coin: str) -> None:
        """Process signal for a single coin."""
        # Skip if already has position (non-overlapping)
        if self.pos_manager.has_position(coin):
            logger.debug(f"[Signal] {coin}: already has position, skipping")
            return

        # Skip if no model
        if coin not in self.models:
            logger.debug(f"[Signal] {coin}: no model, skipping")
            return

        # Skip if no data
        if coin not in self.featured_data:
            logger.debug(f"[Signal] {coin}: no data, skipping")
            return

        df = self.featured_data[coin]

        # Regime filter
        regime = detect_regime(df)
        coin_cfg = self.coin_configs.get(coin, {})
        coin_blocked = coin_cfg.get("blocked_regimes_override", self.blocked_regimes)

        if regime in coin_blocked:
            logger.info(f"[Signal] {coin}: regime={regime} BLOCKED")
            log_event("regime_blocked", {"coin": coin, "regime": regime})
            return

        # Predict
        model = self.models[coin]
        side, probability = predict_signal(model, df)

        if side == "HOLD":
            logger.info(f"[Signal] {coin}: HOLD (p={probability:.3f})")
            return

        # Stage 1 threshold check
        s1_thresh = coin_cfg.get("stage1_threshold", 0.50)
        if probability < s1_thresh:
            logger.info(f"[Signal] {coin}: {side} p={probability:.3f} < threshold {s1_thresh}")
            return

        logger.info(f"[Signal] {coin}: {side} p={probability:.3f} (regime={regime})")

        # Get current price
        try:
            ticker = await self.exchange.fetch_ticker(coin)
            entry_price = ticker["bid"] if side == "BUY" else ticker["ask"]
            spread_bps = ticker["spread_bps"]
        except Exception as e:
            logger.error(f"[Signal] {coin}: failed to get ticker: {e}")
            return

        # Compute barriers
        atr = df["atr_14"].iloc[-1] if "atr_14" in df.columns else entry_price * 0.01
        if np.isnan(atr) or atr < 1e-10:
            atr = entry_price * 0.01

        sl_price, tp_price = RiskEngine.compute_barriers(
            entry_price=entry_price,
            atr=atr,
            side=side,
            k_upper=self.common["k_upper"],
            k_lower=self.common["k_lower"],
            min_barrier_pct=self.common["min_barrier_pct"],
        )

        # Risk engine pre-trade gate
        try:
            funding_rate = await self.exchange.fetch_funding_rate(coin)
        except Exception:
            funding_rate = 0.0

        pre_check = self.risk_engine.pre_trade_gate(
            symbol=coin,
            side=side,
            entry_price=entry_price,
            sl_price=sl_price,
            equity_usdt=self.equity,
            p_trade=probability,
            atr=atr,
            funding_rate=funding_rate,
            spread_bps=spread_bps,
        )

        if not pre_check.approved:
            logger.info(f"[Signal] {coin}: REJECTED by risk engine: {pre_check.reason}")
            log_event("signal_rejected", {
                "coin": coin, "side": side, "reason": pre_check.reason,
                "p": round(probability, 4),
            })
            return

        qty = pre_check.sizing.qty

        # Apply microstructure sizing adjustments
        qty = apply_microstructure_sizing(qty, df, side, self.micro_config)

        # Round to exchange precision
        qty = self.exchange.round_qty(coin, qty)
        entry_price = self.exchange.round_price(coin, entry_price)
        sl_price = self.exchange.round_price(coin, sl_price)
        tp_price = self.exchange.round_price(coin, tp_price)

        # Check minimum quantity
        min_qty = self.exchange.get_min_qty(coin)
        if qty < min_qty:
            logger.info(f"[Signal] {coin}: qty {qty} < min {min_qty}")
            return

        logger.info(
            f"[Entry] {coin} {side} | price={entry_price:.4f} | "
            f"qty={qty} | SL={sl_price:.4f} | TP={tp_price:.4f} | "
            f"p={probability:.3f} | regime={regime}"
        )

        # Place entry order
        order_id = ExchangeAdapter.make_order_id(coin, side, prefix="lb")

        if self.mode == "live":
            await self._place_live_entry(
                coin, side, qty, entry_price, sl_price, tp_price,
                order_id, probability,
            )
        else:
            # Paper mode: simulate fill at bid/ask
            await self._place_paper_entry(
                coin, side, qty, entry_price, sl_price, tp_price,
                order_id, probability,
            )

    async def _place_paper_entry(
        self,
        coin: str, side: str, qty: float,
        entry_price: float, sl_price: float, tp_price: float,
        order_id: str, probability: float,
    ) -> None:
        """Paper mode: simulate immediate fill."""
        pos = OpenPosition(
            coin=coin,
            side=side,
            entry_price=entry_price,
            qty=qty,
            sl_price=sl_price,
            tp_price=tp_price,
            entry_time=datetime.now(timezone.utc),
            ttl_bars=self.common["max_horizon"],
            entry_order_id=order_id,
        )
        self.pos_manager.add_position(pos)

        # Record in ledger
        self.ledger.insert_order(
            order_link_id=order_id,
            symbol=coin,
            side=side,
            order_type="LIMIT",
            qty=qty,
            price=entry_price,
            purpose="entry",
            metadata={"probability": probability, "mode": "paper"},
        )
        self.ledger.update_order_status(order_id, "FILLED")
        self.ledger.insert_fill(order_id, entry_price, qty)
        self.ledger.log_transition(
            coin, "IDLE", "FILLED", "paper_fill", order_id,
        )

        log_event("position_opened", {
            "coin": coin,
            "side": side,
            "entry_price": entry_price,
            "qty": qty,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "probability": round(probability, 4),
            "mode": "paper",
        })

        logger.info(f"[Paper] {coin} {side} FILLED @ {entry_price:.4f} (qty={qty})")

    async def _place_live_entry(
        self,
        coin: str, side: str, qty: float,
        entry_price: float, sl_price: float, tp_price: float,
        order_id: str, probability: float,
    ) -> None:
        """Live mode: Post-Only entry + protective orders."""
        # Record pending order
        self.ledger.insert_order(
            order_link_id=order_id,
            symbol=coin,
            side=side,
            order_type="LIMIT",
            qty=qty,
            price=entry_price,
            purpose="entry",
            metadata={"probability": probability, "mode": "live"},
        )

        # Place Post-Only entry
        result = await self.exchange.place_post_only_entry(
            coin, side, qty, entry_price, order_id,
        )

        if not result.get("success"):
            logger.warning(f"[Entry] {coin} Post-Only FAILED: {result.get('error')}")
            self.ledger.update_order_status(order_id, "REJECTED")
            log_event("entry_rejected", {
                "coin": coin, "error": result.get("error"),
            })
            return

        # Wait for fill (20s timeout)
        fill = await self.exchange.wait_fill_or_cancel(
            coin, order_id, ttl_sec=20.0,
        )

        if not fill:
            logger.info(f"[Entry] {coin} Post-Only TIMEOUT, cancelled")
            self.ledger.update_order_status(order_id, "CANCELLED")
            log_event("entry_timeout", {"coin": coin, "order_id": order_id})
            return

        # Entry filled
        fill_price = fill["fill_price"]
        fill_qty = fill["fill_qty"]
        fee = fill.get("fee", 0)

        self.ledger.update_order_status(order_id, "FILLED",
                                        result.get("exchange_order_id"))
        self.ledger.insert_fill(order_id, fill_price, fill_qty, fee)

        # Recalculate barriers with actual fill price
        atr_val = self.featured_data[coin]["atr_14"].iloc[-1]
        sl_price_new, tp_price_new = RiskEngine.compute_barriers(
            fill_price, atr_val, side,
            self.common["k_upper"], self.common["k_lower"],
            self.common["min_barrier_pct"],
        )
        sl_price = self.exchange.round_price(coin, sl_price_new)
        tp_price = self.exchange.round_price(coin, tp_price_new)

        # Place protective SL
        exit_side = "SELL" if side == "BUY" else "BUY"
        sl_order_id = order_id + "-sl"
        sl_result = await self.exchange.place_protective_stop(
            coin, exit_side, fill_qty, sl_price, sl_order_id, order_id,
        )

        # Place TP
        tp_order_id = order_id + "-tp"
        tp_result = await self.exchange.place_take_profit(
            coin, exit_side, fill_qty, tp_price, tp_order_id, order_id,
        )

        # Record protective orders
        self.ledger.insert_order(
            sl_order_id, coin, exit_side, "STOP_MARKET",
            fill_qty, stop_trigger=sl_price,
            purpose="stop_loss", parent_id=order_id,
        )
        self.ledger.insert_order(
            tp_order_id, coin, exit_side, "TAKE_PROFIT_MARKET",
            fill_qty, stop_trigger=tp_price,
            purpose="take_profit", parent_id=order_id,
        )

        # Create position
        pos = OpenPosition(
            coin=coin,
            side=side,
            entry_price=fill_price,
            qty=fill_qty,
            sl_price=sl_price,
            tp_price=tp_price,
            entry_time=datetime.now(timezone.utc),
            ttl_bars=self.common["max_horizon"],
            entry_order_id=order_id,
            sl_order_id=sl_order_id,
            tp_order_id=tp_order_id,
        )
        self.pos_manager.add_position(pos)

        self.ledger.log_transition(
            coin, "IDLE", "PROTECTED", "entry_filled", order_id,
        )

        log_event("position_opened", {
            "coin": coin,
            "side": side,
            "entry_price": fill_price,
            "qty": fill_qty,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "probability": round(probability, 4),
            "fee": fee,
            "mode": "live",
        })

        logger.info(
            f"[Live] {coin} {side} FILLED @ {fill_price:.4f} | "
            f"SL={sl_price:.4f} TP={tp_price:.4f}"
        )

    # ── Step 5: Update Equity ─────────────────────────────────

    async def _step_update_equity(self) -> None:
        """Update equity from exchange or paper calculation."""
        if self.mode == "live":
            try:
                balance = await self.exchange.fetch_balance()
                self.equity = balance["total"]
            except Exception as e:
                logger.warning(f"[Equity] Failed to fetch: {e}")
        else:
            # Paper mode: initial + realized P&L
            daily = self.ledger.get_daily_total_pnl()
            self.equity = self.initial_equity + daily

        logger.info(f"[Equity] Current: {self.equity:.2f} USDT")


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLAUDE_CRYPTO_AGENT Live Trading Bot v4.1",
    )
    parser.add_argument(
        "--mode", choices=["paper", "live"], default="paper",
        help="Trading mode: paper (testnet, default) or live",
    )
    parser.add_argument(
        "--equity", type=float, default=10000.0,
        help="Initial equity in USDT (paper mode fallback)",
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    bot = LiveTradingBot(mode=args.mode, initial_equity=args.equity)

    # Signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    shutdown_triggered = False

    def handle_signal(signum, frame):
        nonlocal shutdown_triggered
        if not shutdown_triggered:
            shutdown_triggered = True
            logger.info(f"\n[Signal] Received {signal.Signals(signum).name}, shutting down...")
            asyncio.ensure_future(bot.shutdown())

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        await bot.initialize()
        await bot.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"[Fatal] {e}\n{traceback.format_exc()}")
        log_event("fatal_error", {"error": str(e)})
    finally:
        if bot.running:
            await bot.shutdown()


def main():
    args = parse_args()

    # Safety check for live mode
    if args.mode == "live":
        print("\n" + "!" * 60)
        print("  WARNING: LIVE MODE -- REAL MONEY AT RISK")
        print("!" * 60)
        confirm = input("Type 'YES I UNDERSTAND' to continue: ")
        if confirm.strip() != "YES I UNDERSTAND":
            print("Aborted.")
            sys.exit(1)

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
