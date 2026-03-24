#!/usr/bin/env python3
"""Autonomous Live Trading Bot v4.3 -- Clean Architecture.

Changes from v4.1 run_live_bot.py:
  1. 2-Stage Binary: S1(Trade/NoTrade) → S2(Long/Short) aligned with backtest
  2. Multi-Model Combos: et+cb, et+xgb, et+tabm per Mega Search v2
  3. 30s SL/TP monitoring (was 2h)
  4. Model artifact save/load (no retrain on restart)
  5. Position persistence (crash recovery)
  6. Cumulative equity tracking (was daily-only reset)
  7. Unified labeling via masking_loop.create_labels_triple_barrier

Usage:
    python run_live_bot_v2.py --mode paper
    python run_live_bot_v2.py --mode live --equity 10000
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
import urllib.request
from datetime import datetime, timezone, date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── Project root ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.execution.exchange_adapter import ExchangeAdapter, SYMBOL_MAP
from src.execution.order_ledger import OrderLedger
from src.execution.risk_engine import RiskEngine, RiskConfig
from src.execution.cost_model import CostModel, FeeSchedule, FundingConfig, MissFillConfig
from src.execution.live_predictor import (
    train_combo, predict_2stage, save_combo, load_combo, TrainedCombo,
)
from src.execution.sl_tp_monitor import SlTpMonitor
from src.execution.position_store import PositionManager, OpenPosition
from src.data.crawlers.crypto_ohlcv import (
    add_technical_indicators, _add_decomposition, _add_cross_asset_correlation,
)
from src.data.crawlers.signal_features import add_signal_features
from src.data.crawlers.microstructure_rollup import add_microstructure_rollup
from src.data.crawlers.binance_public_features import add_binance_public_features
from src.utils.feature_policy import is_excluded_feature
from src.rl.state_builder import build_rl_state
from src.rl.signal_logger import SignalLogger, SIZING_MAP
from src.rl.rl_gate import RLGate

# ── Constants ──────────────────────────────────────────────
COINS = ["SOL"]  # SOL 단일 운영 (2026-03-19)
CYCLE_SECONDS = 2 * 3600  # default; overridden by config bar_minutes
HEARTBEAT_INTERVAL = 60

LOG_DIR = PROJECT_ROOT / "data" / "reports" / "live_trading_v2"
JSONL_LOG = LOG_DIR / "events.jsonl"
STATE_DB = LOG_DIR / "ledger.db"
POSITIONS_FILE = LOG_DIR / "open_positions.json"
EQUITY_FILE = LOG_DIR / "equity_state.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────
logger = logging.getLogger("live_bot")
logger.setLevel(logging.INFO)
_fh = logging.FileHandler(LOG_DIR / "bot.log", encoding="utf-8")
_fh.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
_sh = logging.StreamHandler()
_sh.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S",
))
logger.addHandler(_fh)
logger.addHandler(_sh)


# ══════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════

_DISCORD_VERSION = "CLAUDE_CRYPTO_AGENT v4.3"
_discord_state = {"total_trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}


def _discord_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _safe_oid(base: str, suffix: str, maxlen: int = 36) -> str:
    """Binance clientOrderId 최대 36자 보장."""
    return (base + suffix)[:maxlen]


def send_discord(msg: str) -> None:
    """단순 텍스트 메시지 전송 (경보/에러용)."""
    _discord_post({"content": msg})


def send_discord_embed(embeds: list) -> None:
    """Embed 리포트 전송."""
    _discord_post({"embeds": embeds})


def _discord_post(payload: dict) -> None:
    url = os.getenv("DISCORD_WEBHOOK_URL", "")
    if not url:
        return
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={
            "Content-Type": "application/json",
            "User-Agent": "DiscordBot (trading-bot, 1.0)",
        })
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


def discord_report_entry(coin: str, side: str, entry: float, qty: float,
                          sl: float, tp: float, p_trade: float, p_dir: float) -> None:
    notional = entry * qty
    sl_pct = abs(entry - sl) / entry * 100 if entry else 0
    tp_pct = abs(tp - entry) / entry * 100 if (entry and tp) else 0
    rr = tp_pct / sl_pct if sl_pct > 0 else 0
    side_label = "🟢 LONG" if side == "BUY" else "🔴 SHORT"
    color = 0x2ecc71 if side == "BUY" else 0xe74c3c
    sl_sign = "-" if side == "BUY" else "+"
    tp_sign = "+" if side == "BUY" else "-"
    send_discord_embed([{
        "title": f"📋 진입 리포트 — {coin} {side_label}",
        "color": color,
        "fields": [
            {"name": "진입가", "value": f"${entry:.4f}",                              "inline": True},
            {"name": "수량",   "value": f"{qty} ({notional:.2f} USDT)",               "inline": True},
            {"name": "모드",   "value": "LIVE",                                        "inline": True},
            {"name": "SL",     "value": f"${sl:.4f} ({sl_sign}{sl_pct:.2f}%)",        "inline": True},
            {"name": "TP",     "value": f"${tp:.4f} ({tp_sign}{tp_pct:.2f}%)",        "inline": True},
            {"name": "R:R",    "value": f"1 : {rr:.2f}",                               "inline": True},
            {"name": "S1 확률 (거래여부)", "value": f"{p_trade*100:.1f}%",             "inline": True},
            {"name": "S2 확률 (방향)",     "value": f"{p_dir*100:.1f}%",               "inline": True},
        ],
        "footer": {"text": f"{_DISCORD_VERSION} | {_discord_now()}"},
    }])


def discord_report_close(coin: str, side: str, reason: str,
                          entry: float, exit_p: float,
                          pnl_usdt: float, pnl_pct: float, bars: int,
                          bar_minutes: int = 15) -> None:
    _discord_state["total_trades"] += 1
    _discord_state["total_pnl"] += pnl_usdt
    if pnl_usdt >= 0:
        _discord_state["wins"] += 1
    else:
        _discord_state["losses"] += 1
    t = _discord_state["total_trades"]
    w = _discord_state["wins"]
    l = _discord_state["losses"]
    win_rate = w / t * 100 if t else 0

    reason_map = {
        "TP_HIT":   "🎯 TP_HIT",  "TP1_HIT": "🎯 TP1_HIT",
        "TP2_HIT":  "🎯 TP2_HIT", "TP3_HIT": "🎯 TP3_HIT",
        "SL_HIT":   "🛑 SL_HIT",
        "TIME_STOP": "⏱ TTL 만료", "TTL": "⏱ TTL 만료",
        "SHUTDOWN":  "🔌 종료",    "KILL_SWITCH": "⛔ 킬스위치",
        "SL/TP(거래소)": "🔔 SL/TP(거래소)",
    }
    reason_label = reason_map.get(reason, reason_map.get(reason.upper(), f"❓ {reason}"))
    result_emoji = "✅ 수익" if pnl_usdt >= 0 else "❌ 손실"
    color = 0x2ecc71 if pnl_usdt >= 0 else 0xe74c3c
    side_label = "LONG" if side == "BUY" else "SHORT"
    hold_h = round(bars * bar_minutes / 60, 1)

    send_discord_embed([{
        "title": f"🗒️ 청산 리포트 — {coin} {side_label}",
        "color": color,
        "fields": [
            {"name": "청산 사유",  "value": reason_label,                                    "inline": True},
            {"name": "결과",       "value": result_emoji,                                    "inline": True},
            {"name": "보유기간",   "value": f"{bars}봉 ({hold_h}h)",                         "inline": True},
            {"name": "진입가",     "value": f"${entry:.4f}",                                 "inline": True},
            {"name": "청산가",     "value": f"${exit_p:.4f}",                                "inline": True},
            {"name": "PnL",        "value": f"{pnl_usdt:+.4f} USDT ({pnl_pct*100:+.2f}%)", "inline": True},
            {"name": "누적 거래",  "value": f"{t}건",                                        "inline": True},
            {"name": "승률",       "value": f"{win_rate:.0f}% ({w}W/{l}L)",                  "inline": True},
            {"name": "누적 PnL",   "value": f"{_discord_state['total_pnl']:+.4f} USDT",     "inline": True},
        ],
        "footer": {"text": f"{_DISCORD_VERSION} | {_discord_now()}"},
    }])


def log_event(event_type: str, data: dict) -> None:
    record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event_type, **data}
    try:
        with open(JSONL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


def write_heartbeat(path: Path, status: str, extra: dict = None) -> None:
    hb = {"ts": datetime.now(timezone.utc).isoformat(), "status": status,
           "pid": os.getpid(), **(extra or {})}
    try:
        path.write_text(json.dumps(hb, indent=2), encoding="utf-8")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════
#  REGIME DETECTION
# ══════════════════════════════════════════════════════════

def detect_regime(df, lookback: int = 24) -> str:
    if len(df) < lookback:
        return "UNKNOWN"
    has_adx = "adx_14" in df.columns
    adx = df["adx_14"].iloc[-1] if has_adx else 20.0
    if "di_diff" in df.columns:
        di_diff = df["di_diff"].iloc[-1]
    elif "plus_di_14" in df.columns and "minus_di_14" in df.columns:
        di_diff = df["plus_di_14"].iloc[-1] - df["minus_di_14"].iloc[-1]
    else:
        di_diff = 0.0
    if not has_adx:
        logger.warning("[Regime] adx_14 missing — regime UNKNOWN (signal_features may have failed)")
        return "UNKNOWN"
    if "atr_14" in df.columns:
        atr_pct = df["atr_14"].iloc[-1] / (df["close"].iloc[-1] + 1e-10)
        median_pct = (df["atr_14"].iloc[-96:] / (df["close"].iloc[-96:] + 1e-10)).median()
    else:
        atr_pct = df["close"].pct_change().abs().rolling(14).mean().iloc[-1]
        median_pct = atr_pct
    if adx > 25:
        return "TREND_UP" if di_diff > 0 else "TREND_DOWN"
    return "RANGE_HIGH" if atr_pct > median_pct else "RANGE_LOW"


# ══════════════════════════════════════════════════════════
#  DATA FETCHING
# ══════════════════════════════════════════════════════════

async def fetch_ohlcv(exchange, symbol: str, limit: int = 500, timeframe: str = "4h"):
    ccxt_sym = SYMBOL_MAP.get(symbol, f"{symbol}/USDT:USDT")
    for attempt in range(3):
        try:
            ohlcv = await exchange._exchange.fetch_ohlcv(ccxt_sym, timeframe, limit=limit)
            if not ohlcv:
                raise ValueError(f"Empty OHLCV for {symbol}")
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)
            return df
        except Exception as e:
            logger.warning(f"[OHLCV] {symbol} attempt {attempt+1}: {e}")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    return None


_FETCH_SEM = asyncio.Semaphore(3)  # max 3 concurrent ccxt requests


async def fetch_all_ohlcv(exchange, coins: list, timeframe: str = "4h", n_bars: int = 500) -> dict:
    """Fetch OHLCV with rate-limited concurrency (max 3 parallel)."""
    fetch_list = list(set(coins + ["BTC", "ETH"]))

    async def _limited(coin):
        async with _FETCH_SEM:
            return await fetch_ohlcv(exchange, coin, limit=n_bars, timeframe=timeframe)

    dfs = await asyncio.gather(*[_limited(c) for c in fetch_list], return_exceptions=True)

    results = {}
    for coin, df in zip(fetch_list, dfs):
        if isinstance(df, Exception):
            logger.error(f"[OHLCV] {coin}: {df}")
        elif df is not None and not df.empty:
            results[coin] = df
            logger.info(f"[OHLCV] {coin}: {len(df)} bars, ${df['close'].iloc[-1]:,.4f}")
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
            df = add_binance_public_features(df, coin)
            df.ffill(inplace=True)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)  # no bfill — prevents future data leakage
            cols_to_drop = [c for c in df.columns if is_excluded_feature(c)]
            if cols_to_drop:
                df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
            featured[coin] = df
        except Exception as e:
            logger.error(f"[Features] {coin}: {e}")
    # Cross-asset correlation (needs BTC/ETH as reference)
    if len(featured) >= 2:
        for coin in COINS:
            if coin in featured:
                try:
                    featured[coin] = _add_cross_asset_correlation(featured, coin, window=20)
                except Exception:
                    pass
    return featured


# DrawdownTracker removed — functionality merged into RiskEngine.
# Use risk_engine.record_pnl(), risk_engine.maybe_reset(), risk_engine.check_drawdown().


# ══════════════════════════════════════════════════════════
#  EQUITY STATE (cumulative, persisted)
# ══════════════════════════════════════════════════════════

class EquityTracker:
    def __init__(self, initial: float, path: Path):
        self.initial = initial
        self.current = initial
        self.cumulative_pnl = 0.0
        self._path = path
        self._load()

    def record_pnl(self, pnl: float):
        self.cumulative_pnl += pnl
        self.current = self.initial + self.cumulative_pnl
        self._save()

    def sync_from_exchange(self, balance: float):
        self.current = balance
        self.cumulative_pnl = balance - self.initial
        self._save()

    def _save(self):
        try:
            self._path.write_text(json.dumps({
                "initial": self.initial,
                "current": round(self.current, 4),
                "cumulative_pnl": round(self.cumulative_pnl, 4),
                "updated": datetime.now(timezone.utc).isoformat(),
            }, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load(self):
        if not self._path.exists():
            return
        try:
            d = json.loads(self._path.read_text(encoding="utf-8"))
            self.initial = d.get("initial", self.initial)
            self.current = d.get("current", self.initial)
            self.cumulative_pnl = d.get("cumulative_pnl", 0.0)
            logger.info(f"[Equity] Recovered: {self.current:.2f} USDT (cum PnL: {self.cumulative_pnl:+.2f})")
        except Exception:
            pass


# ══════════════════════════════════════════════════════════
#  CONFIDENCE-TIERED RISK + DD BRAKE
# ══════════════════════════════════════════════════════════

def compute_risk_frac(
    confidence: float,
    dd_pct: float,
    tier_high: float = 0.150,
    tier_mid: float = 0.100,
    tier_low: float = 0.050,
    dd_brake_threshold: float = 0.015,
) -> float:
    """Dynamic risk_frac: bigger bets on strong signals, brake on drawdown.

    Tiers (by confidence = p_trade × p_direction):
      > 0.65  → 15% of equity
      0.50-0.65 → 10%
      < 0.50  → 5%

    DD brake: if daily DD > 1.5%, halve everything.
    """
    if confidence > 0.65:
        base = tier_high
    elif confidence > 0.50:
        base = tier_mid
    else:
        base = tier_low

    if dd_pct > dd_brake_threshold:
        base *= 0.5

    return base


# ══════════════════════════════════════════════════════════
#  MICROSTRUCTURE SIZING
# ══════════════════════════════════════════════════════════

def apply_micro_sizing(qty: float, df, side: str, cfg: dict) -> float:
    if cfg.get("cvd_filter_enabled"):
        cvd_cols = [c for c in df.columns if c.startswith("cvd_ratio_")]
        if cvd_cols:
            val = df[cvd_cols[0]].iloc[-1]
            q25 = df[cvd_cols[0]].quantile(0.25)
            q75 = df[cvd_cols[0]].quantile(0.75)
            if side == "BUY":
                if val > q75:   qty *= cfg.get("cvd_long_mult_high", 1.2)   # 매수압력 강함 → 증가
                elif val < q25: qty *= cfg.get("cvd_long_mult_low",  0.6)   # 매도압력 강함 → 감소
            else:
                if val < q25:   qty *= cfg.get("cvd_long_mult_high", 1.2)   # 매도압력 강함 → 증가
                elif val > q75: qty *= cfg.get("cvd_long_mult_low",  0.6)   # 매수압력 강함 → 감소
    if cfg.get("ofi_timing_enabled"):
        ofi_cols = [c for c in df.columns if c.startswith("ofi_sum_")]
        if ofi_cols:
            val = df[ofi_cols[0]].iloc[-1]
            q67 = df[ofi_cols[0]].quantile(0.67)
            if (side == "BUY" and val > q67) or (side == "SELL" and val < -q67):
                qty *= cfg.get("ofi_timing_mult", 1.1)
    return qty


# ══════════════════════════════════════════════════════════
#  MAIN BOT
# ══════════════════════════════════════════════════════════

class LiveTradingBot:

    def __init__(self, mode: str = "paper", initial_equity: float = 10000.0, config_path: str = None):
        self.mode = mode
        self.running = False
        self._shutdown_event = asyncio.Event()

        # Config
        cfg_path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "frozen_params_v4_3_1m.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.common = self.config["common"]
        self.coin_cfgs = self.config["coins"]
        # Cycle: 1 bar 마다 실행 (bar_minutes 기반)
        bar_min = self.common.get("bar_minutes", 240)
        self._cycle_seconds = max(60, bar_min * 60)
        # OHLCV timeframe
        self._timeframe = self.config.get("data_fetch", {}).get("timeframe", f"{bar_min}m" if bar_min < 60 else f"{bar_min//60}h")
        self._n_bars = self.config.get("data_fetch", {}).get("n_bars", 500)
        self.micro_cfg = self.config.get("microstructure", {})
        self.monitor_cfg = self.config.get("monitoring", {})
        self.blocked_regimes = self.config.get("blocked_regimes", ["RANGE_LOW"])
        # Always block UNKNOWN — regime detection failed, do not trade
        if "UNKNOWN" not in self.blocked_regimes:
            self.blocked_regimes = list(self.blocked_regimes) + ["UNKNOWN"]

        # Cost model
        cm = self.config.get("cost_model", {})
        self.cost_model = CostModel(
            fee_schedule=FeeSchedule(
                maker_fee=cm.get("maker_fee", 0.0002),
                taker_fee=cm.get("taker_fee", 0.00055),
                slippage_entry=cm.get("slippage_entry", 0.0003),
                slippage_exit_limit=cm.get("slippage_exit_limit", 0.0001),
                slippage_exit_market=cm.get("slippage_exit_market", 0.0005),
            ),
            funding_config=FundingConfig(
                interval_hours=cm.get("funding_interval_hours", 8.0),
                default_rate=cm.get("funding_default_rate", 0.0001),
            ),
            miss_fill_config=MissFillConfig(
                reject_prob=cm.get("miss_fill_reject_prob", 0.15),
                missed_ev_pct=cm.get("miss_fill_missed_ev", 0.0015),
            ),
        )

        # Risk engine
        self.risk_engine = RiskEngine(RiskConfig(
            risk_frac=self.common["risk_frac"],
            daily_drawdown_pct=0.10,   # -10% 일일 손실 한도
            weekly_drawdown_pct=1.0,   # 비활성화
            leverage=4.0,
            min_stop_distance_pct=self.common.get("min_barrier_pct", 0.003) * 0.9,
            max_notional_usdt=initial_equity * 0.10,  # 자본의 10%
        ))

        # Components
        self.exchange: ExchangeAdapter | None = None
        self.ledger = OrderLedger(db_path=STATE_DB)
        self.pos_manager = PositionManager(persist_path=POSITIONS_FILE)
        self.equity = EquityTracker(initial_equity, EQUITY_FILE)
        self.models: dict[str, TrainedCombo] = {}
        self.monitor: SlTpMonitor | None = None
        self.raw_data: dict = {}
        self.featured_data: dict = {}
        self.last_train_date: date | None = None
        self.cycle_count = 0

        # RL meta-layer (v4.3)
        rl_cfg = self.config.get("rl", {})
        self.signal_logger = SignalLogger(LOG_DIR / "signal_log.jsonl")
        self.rl_gate = RLGate(
            warmup=rl_cfg.get("warmup_signals", 200),
            shadow_mode=rl_cfg.get("shadow_mode", True),
            model_path=PROJECT_ROOT / rl_cfg["model_path"] if rl_cfg.get("model_path") else None,
        )
        self.max_new_per_cycle = rl_cfg.get("max_new_per_cycle", 3)
        self._last_funding_cache: dict[str, float] = {}
        self._closing: set[str] = set()  # prevents double-close race condition

    # ── Lifecycle ──────────────────────────────────────────

    async def initialize(self):
        api_key = os.environ.get("BINANCE_API_KEY", "")
        api_secret = os.environ.get("BINANCE_API_SECRET", "")
        if not api_key or not api_secret:
            raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET required")

        ex_mode = "sandbox" if self.mode == "paper" else "live"
        self.exchange = ExchangeAdapter(mode=ex_mode, api_key=api_key, secret=api_secret)
        await self.exchange.initialize()

        # Sync equity from exchange
        try:
            bal = await self.exchange.fetch_balance()
            if bal["total"] > 0:
                self.equity.sync_from_exchange(bal["total"])
        except Exception as e:
            logger.warning(f"[Init] Balance fetch failed: {e}")

        self.risk_engine.set_initial_equity(self.equity.current)

        # Try loading saved models
        for coin in COINS:
            combo = load_combo(coin)
            if combo:
                self.models[coin] = combo
                logger.info(f"[Init] Loaded {coin} model (trained {combo.train_date})")

        # Sync positions with exchange (live mode only)
        if self.mode == "live":
            await self._sync_exchange_positions()

        # Start SL/TP monitor — 1m 모드는 빠른 폴링 필요
        poll_s = self.monitor_cfg.get("sl_tp_poll_seconds", 30)
        if self.common.get("bar_minutes", 240) <= 5:
            poll_s = min(poll_s, 10)  # 1m/5m bar → 10s polling
        self.monitor = SlTpMonitor(
            exchange=self.exchange,
            pos_manager=self.pos_manager,
            close_callback=self._close_position,
            partial_tp_callback=None,  # 분할 익절 비활성화
            poll_seconds=poll_s,
            bar_minutes=self.common["bar_minutes"],
        )
        self.monitor.start()

        log_event("bot_started", {
            "mode": self.mode, "equity": self.equity.current,
            "coins": COINS, "version": self.config.get("version"),
            "recovered_positions": self.pos_manager.count(),
            "loaded_models": list(self.models.keys()),
        })

        logger.info(f"{'='*60}")
        logger.info(f"  LIVE BOT v4.3 STARTED")
        logger.info(f"  Mode:      {self.mode}")
        logger.info(f"  Equity:    {self.equity.current:.2f} USDT")
        logger.info(f"  Coins:     {', '.join(COINS)}")
        logger.info(f"  Models:    {list(self.models.keys()) or 'will train'}")
        logger.info(f"  Positions: {[p.coin for p in self.pos_manager.all_positions()]}")
        logger.info(f"  SL/TP:     every {poll_s}s")
        logger.info(f"{'='*60}")
        if self.mode == "live":
            coins_str = " · ".join(COINS)
            send_discord_embed([{
                "title": "🚀 봇 시작 — LIVE 모드",
                "color": 0x3498db,
                "fields": [
                    {"name": "잔고",    "value": f"{self.equity.current:.2f} USDT", "inline": True},
                    {"name": "모드",    "value": "LIVE",                             "inline": True},
                    {"name": "버전",    "value": _DISCORD_VERSION,                   "inline": True},
                    {"name": "거래 코인", "value": coins_str,                        "inline": False},
                ],
                "footer": {"text": f"{_DISCORD_VERSION} | {_discord_now()}"},
            }])

    async def _sync_exchange_positions(self):
        """봇 재시작 시 거래소 실제 포지션과 position_store 동기화.

        1. 거래소에 있으나 store에 없는 포지션 → 추가 (미등록 포지션 복구)
        2. store에 있으나 거래소에 없는 포지션 → 제거 (이미 청산됨)
        3. 포지션 없는 코인의 잔여 주문 → 취소
        """
        logger.info("[Sync] 거래소 포지션 동기화 시작...")
        def _pos_qty(p: dict) -> float:
            for key in ("contracts", "positionAmt", "size"):
                v = p.get(key)
                if v is not None:
                    try:
                        return abs(float(v))
                    except (TypeError, ValueError):
                        pass
            return 0.0

        try:
            ex_positions = await self.exchange._exchange.fetch_positions()
            ex_open = {
                p["symbol"].split("/")[0]: p
                for p in ex_positions
                if _pos_qty(p) > 0 and p["symbol"].split("/")[0] in COINS
            }
        except Exception as e:
            logger.warning(f"[Sync] 거래소 포지션 조회 실패: {e}")
            return

        store_coins = set(self.pos_manager.positions.keys())
        ex_coins    = set(ex_open.keys())

        # ── 거래소에 없는데 store에 있는 포지션 제거 (SL/TP 거래소 체결) ──
        for coin in store_coins - ex_coins:
            pos = self.pos_manager.get_position(coin)
            if pos:
                # 현재가 조회해서 PnL 추정
                try:
                    ticker = await self.exchange.fetch_ticker(coin)
                    exit_price = ticker["last"]
                except Exception:
                    exit_price = pos.entry_price
                if pos.side == "BUY":
                    pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
                else:
                    pnl_pct = (pos.entry_price - exit_price) / pos.entry_price
                pnl_usdt = pnl_pct * pos.current_qty * pos.entry_price
                self.risk_engine.record_pnl(pnl_usdt)
                self.equity.record_pnl(pnl_usdt)
                self.ledger.record_pnl(coin, pnl_usdt, 0.0)
                log_event("position_closed", {
                    "coin": coin, "reason": "EXCHANGE_CLOSED", "side": pos.side,
                    "entry": pos.entry_price, "exit": exit_price,
                    "pnl_pct": round(pnl_pct, 6), "pnl_usdt": round(pnl_usdt, 4),
                    "bars": pos.bars_held,
                })
                logger.warning(
                    f"[Sync] {coin}: 거래소에서 자동 청산됨 (SL/TP 체결 추정) "
                    f"pnl={pnl_pct:.2%} ({pnl_usdt:+.2f} USDT)"
                )
                discord_report_close(coin, pos.side, "SL/TP(거래소)", pos.entry_price,
                                     exit_price, pnl_usdt, pnl_pct, pos.bars_held,
                                     bar_minutes=self.common.get("bar_minutes", 15))
            self.pos_manager.remove_position(coin)

        # ── 거래소에 있는데 store에 없는 포지션 복구 ──────
        for coin in ex_coins - store_coins:
            p = ex_open[coin]
            try:
                entry = float(p.get("entryPrice", 0))
                qty   = _pos_qty(p)
            except (TypeError, ValueError) as e:
                logger.error(f"[Sync] {coin}: invalid position data {e} — skip")
                continue
            if entry <= 0 or qty <= 0:
                logger.warning(f"[Sync] {coin}: zero entry/qty, skip")
                continue
            side   = "BUY" if p.get("side", "") == "long" else "SELL"

            # SL/TP 재계산: ATR 없으므로 entry 기준 기본 비율 사용
            sl_pct = self.common["k_lower"] * 0.01   # k_lower × 1%
            tp_pct = self.common["k_upper"] * 0.01   # k_upper × 1%
            if side == "BUY":
                sl = round(entry * (1 - sl_pct), 6)
                tp = round(entry * (1 + tp_pct), 6)
            else:
                sl = round(entry * (1 + sl_pct), 6)
                tp = round(entry * (1 - tp_pct), 6)

            pos = OpenPosition(
                coin=coin, side=side, entry_price=entry, qty=qty,
                sl_price=sl, tp_price=tp,
                entry_time=datetime.now(timezone.utc).isoformat(),
                ttl_bars=self.common["max_horizon"],
            )
            self.pos_manager.add_position(pos)
            logger.warning(
                f"[Sync] {coin} 미등록 포지션 복구: {side} {qty} @ {entry} "
                f"| SL={sl:.4f} TP={tp:.4f}"
            )

            # Live 모드: 거래소에 SL/TP 주문 재등록 (closePosition=True, Position 탭 표시)
            if self.mode == "live":
                exit_side = "SELL" if side == "BUY" else "BUY"
                sl_oid = ExchangeAdapter.make_order_id(coin, exit_side, prefix="rec-sl")
                tp_oid = ExchangeAdapter.make_order_id(coin, exit_side, prefix="rec-tp")
                sl_result = await self.exchange.place_protective_stop(
                    coin, exit_side, qty, sl, sl_oid
                )
                if sl_result.get("success"):
                    pos.sl_order_id = sl_oid
                    self.pos_manager.add_position(pos)  # order_id 업데이트 저장
                    logger.info(f"[Sync] {coin} SL 재등록 완료: {sl:.4f}")
                else:
                    logger.error(f"[Sync] {coin} SL 재등록 실패: {sl_result.get('error')}")
                tp_result = await self.exchange.place_take_profit(
                    coin, exit_side, qty, tp, tp_oid
                )
                if tp_result.get("success"):
                    pos.tp_order_id = tp_oid
                    self.pos_manager.add_position(pos)  # order_id 업데이트 저장
                    logger.info(f"[Sync] {coin} TP 재등록 완료: {tp:.4f}")
                else:
                    logger.warning(f"[Sync] {coin} TP 재등록 실패: {tp_result.get('error')}")

            log_event("position_recovered", {
                "coin": coin, "side": side, "entry": entry, "qty": qty,
                "sl": sl, "tp": tp,
            })

        # ── 포지션 없는 코인의 잔여 주문 취소 ─────────────
        active_coins = set(self.pos_manager.positions.keys())
        for coin in COINS:
            if coin in active_coins:
                continue
            ccxt_sym = f"{coin}/USDT:USDT"
            try:
                await self.exchange._exchange.cancel_all_orders(ccxt_sym)
                logger.info(f"[Sync] {coin}: 잔여 주문 취소 완료")
            except Exception:
                pass

        logger.info(
            f"[Sync] 완료 — 복구: {len(ex_coins - store_coins)}개 "
            f"| 제거: {len(store_coins - ex_coins)}개 "
            f"| 현재 포지션: {list(self.pos_manager.positions.keys())}"
        )

    async def shutdown(self):
        logger.info("[Shutdown] Starting...")
        self.running = False
        self._shutdown_event.set()
        if self.monitor:
            await self.monitor.stop()
        if self.mode == "paper":
            for coin in list(self.pos_manager.positions.keys()):
                await self._close_position(coin, "SHUTDOWN")
        if self.exchange:
            await self.exchange.close()
        self.ledger.close()
        write_heartbeat(LOG_DIR / "heartbeat.json", "stopped")
        log_event("bot_stopped", {"reason": "shutdown"})
        logger.info("[Shutdown] Complete")

    # ── Main Loop ──────────────────────────────────────────

    async def run(self):
        self.running = True
        while self.running:
            try:
                t0 = time.time()
                self.cycle_count += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"  CYCLE #{self.cycle_count} @ {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
                logger.info(f"{'='*60}")

                self.risk_engine.maybe_reset(self.equity.current)
                # Gate 5/6 제거 보완: 루프 진입 전 일일/주간 DD 체크
                _dd_ok, _dd_msg = self.risk_engine.check_drawdown()
                if not _dd_ok:
                    logger.warning(f"[KillSwitch] {_dd_msg}")
                if self.risk_engine.is_killed:
                    logger.warning(f"[KillSwitch] {self.risk_engine.kill_reason}")
                    for coin in list(self.pos_manager.positions.keys()):
                        await self._close_position(coin, "KILL_SWITCH")
                    await self._wait_next_cycle(t0)
                    continue

                # Step 1: Data
                await self._fetch_data()

                # Step 2: Train (daily or on cold start)
                await self._train_models()

                # Step 3: Increment TTL bars for monitor
                if self.monitor:
                    self.monitor.increment_all_bars()

                # Step 4: Generate signals, enter positions
                await self._generate_signals()

                # Step 5: Update equity
                await self._update_equity()

                elapsed = time.time() - t0
                log_event("cycle_complete", {
                    "cycle": self.cycle_count, "elapsed_s": round(elapsed, 1),
                    "equity": round(self.equity.current, 2),
                    "positions": [p.coin for p in self.pos_manager.all_positions()],
                })
                logger.info(
                    f"[Cycle] #{self.cycle_count} done ({elapsed:.1f}s) | "
                    f"Equity: {self.equity.current:.2f} | "
                    f"Open: {[p.coin for p in self.pos_manager.all_positions()]}"
                )
                self._export_trading_result()
                await self._wait_next_cycle(t0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Loop] {e}\n{traceback.format_exc()}")
                log_event("cycle_error", {"error": str(e)})
                await self._wait_next_cycle(time.time())

    async def _wait_next_cycle(self, cycle_start: float):
        remaining = max(10, self._cycle_seconds - (time.time() - cycle_start))
        end_time = time.time() + remaining
        while time.time() < end_time and self.running:
            write_heartbeat(LOG_DIR / "heartbeat.json", "running", {
                "cycle": self.cycle_count,
                "equity": round(self.equity.current, 2),
                "positions": self.pos_manager.count(),
                "next_s": int(end_time - time.time()),
            })
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=min(HEARTBEAT_INTERVAL, end_time - time.time()),
                )
                break
            except asyncio.TimeoutError:
                pass

    # ── Step 1: Fetch Data ─────────────────────────────────

    async def _fetch_data(self):
        logger.info("[Step 1] Fetching OHLCV...")
        self.raw_data = await fetch_all_ohlcv(self.exchange, COINS, timeframe=self._timeframe, n_bars=self._n_bars)
        if not self.raw_data:
            logger.error("[Step 1] No data!")
            return
        self.featured_data = compute_features(self.raw_data)
        logger.info(f"[Step 1] Ready: {list(self.featured_data.keys())}")

    # ── Step 2: Train Models ───────────────────────────────

    async def _train_models(self):
        today = datetime.now(timezone.utc).date()
        if self.models and self.last_train_date == today:
            logger.info("[Step 2] Models up to date")
            return

        logger.info("[Step 2] Training 2-stage combos...")
        for coin in COINS:
            if coin not in self.featured_data:
                continue
            try:
                combo = train_combo(
                    self.featured_data[coin], coin,
                    self.common, self.coin_cfgs.get(coin, {}),
                )
                if combo:
                    self.models[coin] = combo
                    save_combo(combo)
                    log_event("model_trained", {
                        "coin": coin, "combo": self.coin_cfgs.get(coin, {}).get("model_combo"),
                        "cv_score": combo.cv_score, "samples": combo.train_samples,
                    })
            except Exception as e:
                logger.error(f"[Train] {coin}: {e}\n{traceback.format_exc()}")

        self.last_train_date = today
        logger.info(f"[Step 2] Trained: {list(self.models.keys())}")

    # ── Step 3: Signals (candidate ranking) ──────────────

    async def _generate_signals(self):
        logger.info("[Step 3] Generating signals...")
        if self.risk_engine.is_killed:
            return

        # Collect candidates from all coins
        candidates = []
        for coin in COINS:
            try:
                cand = await self._evaluate_coin(coin)
                if cand:
                    candidates.append(cand)
            except Exception as e:
                logger.error(f"[Signal] {coin}: {e}\n{traceback.format_exc()}")

        if not candidates:
            return

        # RL ranking: sort by rl_score descending, REJECT filtered
        accepted = [c for c in candidates if c["rl_effective_action"] > 0]
        accepted.sort(key=lambda c: c["rl_score"], reverse=True)

        # Portfolio limit: max new entries this cycle
        slots = max(0, self.max_new_per_cycle - self.pos_manager.count())
        taken = accepted[:slots]

        for c in taken:
            await self._execute_entry(c)

        # Log ALL candidates (accept + reject) for RL learning
        for c in candidates:
            self.signal_logger.log(
                coin=c["coin"], side=c["pred"].side, regime=c["regime"],
                state=c["state"].tolist(),
                p_trade=c["pred"].p_trade, p_direction=c["pred"].p_direction,
                action=c["rl_action"], rl_score=c["rl_score"],
                entry_price=c["entry_price"], sl_price=c["sl_price"], tp_price=c["tp_price"],
                risk_gate_passed=c["risk_ok"],
                executed=c["coin"] in [t["coin"] for t in taken],
            )

        if taken:
            logger.info(f"[Step 3] Entered: {[t['coin'] for t in taken]} "
                        f"(from {len(candidates)} candidates, {len(accepted)} accepted)")

    # ── Signal Evaluation (decomposed) ────────────────────

    def _check_regime_and_predict(self, coin: str, df, coin_cfg: dict):
        """Check regime filter + CV gate + 2-stage prediction.

        Returns (regime, pred) or (None, None) if blocked.
        """
        regime = detect_regime(df)
        blocked = coin_cfg.get("blocked_regimes_override", self.blocked_regimes)
        if regime in blocked:
            return None, None

        combo = self.models[coin]
        min_cv = self.common.get("min_cv_score", 0.0)
        if min_cv > 0 and combo.cv_score < min_cv:
            logger.info(f"[Signal] {coin}: CV {combo.cv_score:.3f} < min {min_cv:.2f} — 차단")
            return None, None

        s1_thresh = coin_cfg.get("stage1_threshold", 0.50)
        pred = predict_2stage(combo, df, s1_thresh)
        if pred.side == "HOLD":
            return None, None

        # Deadzone: p_long이 0.5 근처(±0.10)이면 방향 확신 부족 → HOLD
        # 효과: WR +4.2%p, MDD -5.6%p (백테스트 검증)
        deadzone = coin_cfg.get("direction_deadzone", self.common.get("direction_deadzone", 0.10))
        if deadzone > 0:
            actual_p_long = pred.p_direction if pred.side == "BUY" else (1 - pred.p_direction)
            if abs(actual_p_long - 0.5) < deadzone:
                logger.info(f"[Signal] {coin}: {pred.side} in deadzone (p_long={actual_p_long:.3f}) — HOLD")
                return None, None

        # Momentum filter: 방향과 최근 가격 모멘텀이 일치해야 진입
        # BUY인데 최근 3 bars 하락 중이면 진입하지 않음 (역추세 진입 방지)
        if len(df) >= 4 and "close" in df.columns:
            recent_ret = (df["close"].iloc[-1] / df["close"].iloc[-4] - 1)
            if pred.side == "BUY" and recent_ret < -0.01:
                logger.info(f"[Signal] {coin}: BUY blocked — price dropping {recent_ret:.2%} in 3 bars")
                return None, None
            if pred.side == "SELL" and recent_ret > 0.01:
                logger.info(f"[Signal] {coin}: SELL blocked — price rising {recent_ret:.2%} in 3 bars")
                return None, None

        return regime, pred

    def _build_rl_decision(self, coin: str, df, pred, regime: str):
        """Build RL state, run gate, return (state, rl_action, rl_effective, rl_score, sizing_mult)."""
        coin_history = self._get_coin_history(coin)
        s1_thresh = self.coin_cfgs.get(coin, {}).get("stage1_threshold", 0.50)
        dd_denom = max(
            self.risk_engine._daily_start_equity * self.risk_engine.config.daily_drawdown_pct,
            1e-10,
        )
        state = build_rl_state(
            df=df, pred_side=pred.side,
            p_trade=pred.p_trade, p_direction=pred.p_direction,
            s1_threshold=s1_thresh, coin=coin,
            equity=self.equity.current,
            daily_pnl=self.risk_engine.daily_pnl,
            weekly_pnl=self.risk_engine.weekly_pnl,
            dd_ratio=max(0.0, -self.risk_engine.daily_pnl) / dd_denom,
            open_count=self.pos_manager.count(),
            coin_win_rate_5=coin_history["win_rate_5"],
            coin_avg_pnl_5=coin_history["avg_pnl_5"],
            coin_streak=coin_history["streak"],
            bars_since_last=coin_history["bars_since_last"],
            max_horizon=self.common["max_horizon"],
            last_funding=self._last_funding_cache.get(coin, 0.0),
            btc_df=self.featured_data.get("BTC"),
        )
        rl_action, rl_score = self.rl_gate.decide(state)
        rl_effective = self.rl_gate.effective_action(rl_action)
        sizing_mult = SIZING_MAP.get(rl_effective, 1.0)
        return state, rl_action, rl_effective, rl_score, sizing_mult

    def _compute_atr_and_barriers(self, coin: str, df, entry_price: float, side: str):
        """Read ATR from features and compute SL/TP barriers."""
        atr_raw = df["atr_14"].iloc[-1] if "atr_14" in df.columns else None
        try:
            atr = float(atr_raw)
        except (TypeError, ValueError):
            atr = float("nan")
        if np.isnan(atr) or atr < 1e-10:
            atr = entry_price * 0.01
        sl_price, tp_price = RiskEngine.compute_barriers(
            entry_price, atr, side,
            self.common["k_upper"], self.common["k_lower"], self.common["min_barrier_pct"],
        )
        return atr, sl_price, tp_price

    async def _fetch_market_inputs(self, coin: str, side: str, df):
        """Fetch ticker + funding rate. Returns dict or None on failure."""
        try:
            ticker = await self.exchange.fetch_ticker(coin)
            bid = ticker.get("bid")
            ask = ticker.get("ask")
            last = ticker.get("last")
            entry_price = (bid if side == "BUY" else ask) or last
            if not entry_price:
                logger.warning(f"[Signal] {coin}: bid/ask/last all None, skip")
                return None
            spread_bps = ticker["spread_bps"]
        except Exception as e:
            logger.error(f"[Signal] {coin}: ticker failed: {e}")
            return None

        try:
            funding = await self.exchange.fetch_funding_rate(coin)
            self._last_funding_cache[coin] = funding
        except Exception:
            funding = self._last_funding_cache.get(coin, 0.0)

        atr, sl_price, tp_price = self._compute_atr_and_barriers(coin, df, entry_price, side)
        return {
            "entry_price": entry_price, "sl_price": sl_price, "tp_price": tp_price,
            "atr": atr, "spread_bps": spread_bps, "funding": funding,
        }

    def _run_risk_gate(self, coin: str, pred, market: dict):
        """Run pre-trade risk gate with confidence-tiered dynamic risk_frac."""
        dd_pct = max(0.0, -self.risk_engine.daily_pnl) / (
            self.risk_engine._daily_start_equity + 1e-10
        )
        sizing_cfg = self.config.get("sizing_tiers", {})
        dynamic_risk_frac = compute_risk_frac(
            confidence=pred.confidence, dd_pct=dd_pct,
            tier_high=sizing_cfg.get("tier_high", 0.015),
            tier_mid=sizing_cfg.get("tier_mid", 0.010),
            tier_low=sizing_cfg.get("tier_low", 0.005),
            dd_brake_threshold=sizing_cfg.get("dd_brake_threshold", 0.015),
        )
        orig = self.risk_engine.config.risk_frac
        self.risk_engine.config.risk_frac = dynamic_risk_frac
        try:
            return self.risk_engine.pre_trade_gate(
                symbol=coin, side=pred.side,
                entry_price=market["entry_price"], sl_price=market["sl_price"],
                equity_usdt=self.equity.current, p_trade=pred.p_trade,
                atr=market["atr"], funding_rate=market["funding"],
                spread_bps=market["spread_bps"],
            )
        finally:
            self.risk_engine.config.risk_frac = orig

    async def _evaluate_coin(self, coin: str) -> dict | None:
        """Orchestrate per-coin evaluation: predict → RL gate → risk check."""
        if self.pos_manager.has_position(coin):
            return None
        if coin not in self.models or coin not in self.featured_data:
            return None

        df = self.featured_data[coin]
        coin_cfg = self.coin_cfgs.get(coin, {})

        # 1. Regime filter + CV check + prediction
        regime, pred = self._check_regime_and_predict(coin, df, coin_cfg)
        if pred is None:
            return None

        # 2. RL gate
        state, rl_action, rl_effective, rl_score, sizing_mult = self._build_rl_decision(
            coin, df, pred, regime
        )
        logger.info(
            f"[Signal] {coin}: {pred.side} p_trade={pred.p_trade:.3f} "
            f"conf={pred.confidence:.3f} rl={rl_action}({self.rl_gate.status}) "
            f"score={rl_score:.3f} regime={regime}"
        )

        # 3. RL REJECT — return counterfactual for logging (no live market fetch)
        if rl_effective == 0:
            close_price = df["close"].iloc[-1]
            atr, sl, tp = self._compute_atr_and_barriers(coin, df, close_price, pred.side)
            return {
                "coin": coin, "pred": pred, "regime": regime, "state": state,
                "rl_action": rl_action, "rl_effective_action": 0,
                "rl_score": rl_score, "sizing_mult": 0.0,
                "entry_price": close_price, "sl_price": sl, "tp_price": tp,
                "risk_ok": False, "check": None, "atr": atr, "df": df,
            }

        # 4. Live market fetch + barriers
        market = await self._fetch_market_inputs(coin, pred.side, df)
        if market is None:
            return None

        # 5. Risk gate
        check = self._run_risk_gate(coin, pred, market)
        base = {
            "coin": coin, "pred": pred, "regime": regime, "state": state,
            "rl_action": rl_action, "rl_effective_action": rl_effective,
            "rl_score": rl_score, "sizing_mult": sizing_mult, "df": df,
            **market,
        }
        if not check.approved:
            logger.info(f"[Signal] {coin}: risk REJECTED ({check.reason})")
            return {**base, "risk_ok": False, "check": check}
        return {**base, "risk_ok": True, "check": check}

    async def _execute_entry(self, c: dict):
        """Execute a ranked candidate entry."""
        coin = c["coin"]
        pred = c["pred"]
        df = c["df"]

        if not c["check"].sizing:
            logger.warning(f"[Entry] {coin} sizing is None, skip")
            return
        qty = c["check"].sizing.qty
        qty *= c["sizing_mult"]  # RL sizing adjustment
        qty = apply_micro_sizing(qty, df, pred.side, self.micro_cfg)
        qty = self.exchange.round_qty(coin, qty)
        raw_entry   = c["entry_price"]
        # Post-Only maker offset: BUY 1 tick below bid, SELL 1 tick above ask
        tick = self.exchange.get_tick_size(coin)
        if pred.side == "BUY":
            raw_entry -= tick
        else:
            raw_entry += tick
        entry_price = self.exchange.round_price(coin, raw_entry)
        sl_price    = self.exchange.round_price(coin, c["sl_price"])
        tp_price    = self.exchange.round_price(coin, c["tp_price"])

        # 단일 TP 사용 (분할 익절 제거 — 수수료가 2.35배 더 드는 구조)
        # tp1/tp2 = 0.0 → partial_tp_enabled=False → 모니터가 단일 TP 체크
        tp1_price = 0.0
        tp2_price = 0.0
        tp3_price = tp_price
        atr_val   = c.get("atr", 0.0)

        if qty < self.exchange.get_min_qty(coin):
            return

        logger.info(
            f"[Entry] {coin} {pred.side} | price={entry_price} qty={qty} "
            f"SL={sl_price} TP={tp_price} rl_mult={c['sizing_mult']:.2f}"
        )

        order_id = ExchangeAdapter.make_order_id(coin, pred.side, prefix="v43")

        if self.mode == "live":
            await self._live_entry(
                coin, pred, qty, entry_price, sl_price, tp_price, order_id,
                tp1_price, tp2_price, tp3_price,
                atr_val=c.get("atr", 0.0),
            )
        else:
            await self._paper_entry(
                coin, pred, qty, entry_price, sl_price, tp_price, order_id,
                tp1_price, tp2_price, tp3_price,
            )

    def _get_coin_history(self, coin: str) -> dict:
        """Query recent trade history for a coin from ledger."""
        default = {"win_rate_5": 0.5, "avg_pnl_5": 0.0, "streak": 0,
                    "bars_since_last": self.common["max_horizon"]}
        try:
            transitions = self.ledger.get_transitions(coin, limit=10)
            exits = [t for t in transitions
                     if t.get("to_state") in ("TP_HIT", "SL_HIT", "TIME_STOP")]
            if not exits:
                return default

            recent = exits[:5]
            wins = sum(1 for t in recent if t.get("to_state") == "TP_HIT")
            win_rate = wins / len(recent)

            # Streak (from most recent)
            streak = 0
            for t in exits:
                is_win = t.get("to_state") == "TP_HIT"
                if streak == 0:
                    streak = 1 if is_win else -1
                elif is_win and streak > 0:
                    streak += 1
                elif not is_win and streak < 0:
                    streak -= 1
                else:
                    break

            # Bars since last
            bars_since = self.common["max_horizon"]
            last_ts = exits[0].get("ts", "") if exits else ""
            if last_ts:
                try:
                    last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                    hours = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
                    bars_since = int(hours / (self.common["bar_minutes"] / 60))
                except Exception:
                    pass

            return {
                "win_rate_5": win_rate,
                "avg_pnl_5": 0.0,  # PnL% not stored in transitions; use 0 until signal_log available
                "streak": streak,
                "bars_since_last": bars_since,
            }
        except Exception:
            return default

    async def _paper_entry(self, coin, pred, qty, entry, sl, tp, oid,
                            tp1=0.0, tp2=0.0, tp3=0.0):
        pos = OpenPosition(
            coin=coin, side=pred.side, entry_price=entry, qty=qty,
            sl_price=sl, tp_price=tp,
            entry_time=datetime.now(timezone.utc).isoformat(),
            ttl_bars=self.common["max_horizon"], entry_order_id=oid,
            tp1_price=tp1, tp2_price=tp2, tp3_price=tp3,
        )
        self.pos_manager.add_position(pos)
        self.ledger.insert_order(oid, coin, pred.side, "LIMIT", qty, price=entry,
                                 purpose="entry", metadata={"p_trade": pred.p_trade, "mode": "paper"})
        self.ledger.update_order_status(oid, "FILLED")
        self.ledger.insert_fill(oid, entry, qty)
        log_event("position_opened", {
            "coin": coin, "side": pred.side, "entry": entry, "qty": qty,
            "sl": sl, "tp": tp, "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "p_trade": round(pred.p_trade, 4),
            "p_dir": round(pred.p_direction, 4), "mode": "paper",
        })
        logger.info(f"[Paper] {coin} {pred.side} FILLED @ {entry} | TP1={tp1} TP2={tp2} TP3={tp3}")

    async def _live_entry(self, coin, pred, qty, entry, sl, tp, oid,
                           tp1=0.0, tp2=0.0, tp3=0.0, atr_val=0.0):
        self.ledger.insert_order(oid, coin, pred.side, "LIMIT", qty, price=entry,
                                 purpose="entry", metadata={"p_trade": pred.p_trade, "mode": "live"})
        result = await self.exchange.place_post_only_entry(coin, pred.side, qty, entry, oid)
        if not result.get("success"):
            logger.warning(f"[Entry] {coin} order FAILED: {result.get('error')}")
            self.ledger.update_order_status(oid, "REJECTED")
            return

        fill = await self.exchange.wait_fill_or_cancel(coin, oid, ttl_sec=20.0)
        if not fill:
            # WaitFill None 반환 시 거래소 포지션 직접 확인 (미등록 포지션 방지)
            try:
                ex_pos = await self.exchange.fetch_position(coin)
                if ex_pos and float(ex_pos.get("qty", 0)) > 0:
                    logger.error(
                        f"[Entry] {coin} WaitFill None이지만 거래소 포지션 존재 "
                        f"qty={ex_pos['qty']} — 즉시 시장가 청산"
                    )
                    exit_side = "SELL" if pred.side == "BUY" else "BUY"
                    await self.exchange.market_close(
                        coin, exit_side, float(ex_pos["qty"]),
                        _safe_oid(oid, "-unreg"),
                    )
                    send_discord_embed([{"title": f"⚠️ 미등록 포지션 강제 청산 — {coin}",
                        "color": 0xe67e22, "description": "체결 확인 실패 → 거래소에 포지션 존재 → 즉시 시장가 청산",
                        "footer": {"text": f"{_DISCORD_VERSION} | {_discord_now()}"}}])
            except Exception as ce:
                logger.error(f"[Entry] {coin} 미등록 포지션 확인 실패: {ce}")
            logger.info(f"[Entry] {coin} TIMEOUT, cancelled")
            self.ledger.update_order_status(oid, "CANCELLED")
            return

        fill_price = fill["fill_price"]
        fill_qty   = fill["fill_qty"]

        # fill_price 검증: 0이거나 의도 진입가 대비 5% 이상 괴리 시 거래소 포지션에서 실체결가 조회
        if fill_price <= 0 or abs(fill_price - entry) / max(entry, 1e-8) > 0.05:
            try:
                ex_pos = await self.exchange.fetch_position(coin)
                if ex_pos and ex_pos.get("entry_price", 0) > 0:
                    corrected = ex_pos["entry_price"]
                    logger.warning(
                        f"[Entry] {coin} fill_price 보정: {fill_price} → {corrected} "
                        f"(의도진입가={entry}, 거래소 entryPrice 기준)"
                    )
                    fill_price = corrected
            except Exception as ep:
                logger.warning(f"[Entry] {coin} fill_price 보정 실패: {ep}, fallback={entry}")
                fill_price = entry

        self.ledger.update_order_status(oid, "FILLED", result.get("exchange_order_id"))
        self.ledger.insert_fill(oid, fill_price, fill_qty, fill.get("fee", 0))

        # 단일 TP 사용 — tp1/tp2 = 0 으로 partial_tp_enabled=False
        tp3 = tp

        exit_side = "SELL" if pred.side == "BUY" else "BUY"
        sl_oid = oid + "-sl"
        sl_result = await self.exchange.place_protective_stop(coin, exit_side, fill_qty, sl, sl_oid, oid)
        if not sl_result.get("success"):
            logger.error(f"[Entry] {coin} SL 주문 실패: {sl_result.get('error')} — 즉시 시장가 청산")
            send_discord_embed([{"title": f"🚨 SL 주문 실패 — {coin}",
                "color": 0xe74c3c, "description": "SL 거래소 등록 실패 → 즉시 시장가 청산 실행",
                "footer": {"text": f"{_DISCORD_VERSION} | {_discord_now()}"}}])
            await self.exchange.market_close(coin, exit_side, fill_qty, _safe_oid(oid, "-slf"))
            self.ledger.update_order_status(oid, "SL_FAIL_CLOSED")
            return

        # TP3를 거래소에 TAKE_PROFIT_MARKET 주문으로 즉시 등록 (30s 모니터 의존 제거)
        tp_oid = oid + "-tp"
        tp_result = await self.exchange.place_take_profit(coin, exit_side, fill_qty, tp3, tp_oid, oid)
        if not tp_result.get("success"):
            logger.warning(f"[Entry] {coin} TP 주문 실패 (모니터 폴백): {tp_result.get('error')}")
            tp_oid = ""

        pos = OpenPosition(
            coin=coin, side=pred.side, entry_price=fill_price, qty=fill_qty,
            sl_price=sl, tp_price=tp3,
            entry_time=datetime.now(timezone.utc).isoformat(),
            ttl_bars=self.common["max_horizon"],
            entry_order_id=oid, sl_order_id=sl_oid, tp_order_id=tp_oid,
        )
        self.pos_manager.add_position(pos)
        log_event("position_opened", {
            "coin": coin, "side": pred.side, "entry": fill_price, "qty": fill_qty,
            "sl": sl, "tp": tp3, "mode": "live",
            "p_trade": round(pred.p_trade, 4), "p_dir": round(pred.p_direction, 4),
        })
        logger.info(
            f"[Live] {coin} {pred.side} FILLED @ {fill_price} "
            f"| SL={sl} TP={tp3}"
        )
        discord_report_entry(coin, pred.side, fill_price, fill_qty, sl, tp3,
                             pred.p_trade, pred.p_direction)

    # ── Partial Close (분할 익절) ───────────────────────────

    def _export_trading_result(self) -> None:
        """사이클마다 trading_result/ 폴더에 거래 기록 CSV 갱신."""
        import csv as _csv, shutil as _shutil
        out = PROJECT_ROOT / "trading_result"
        out.mkdir(exist_ok=True)
        try:
            conn = self.ledger._conn
            for table, fname in [("orders", "orders.csv"), ("fills", "fills.csv"), ("daily_pnl", "daily_pnl.csv")]:
                rows = conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                if rows:
                    keys = [d[0] for d in conn.execute(f"SELECT * FROM {table} LIMIT 0").description]
                    with open(out / fname, "w", newline="", encoding="utf-8") as f:
                        w = _csv.writer(f)
                        w.writerow(keys)
                        w.writerows(rows)
            _shutil.copy(EQUITY_FILE, out / "equity_state.json")
            _shutil.copy(JSONL_LOG, out / "events.jsonl")
        except Exception as e:
            logger.debug(f"[Export] trading_result 갱신 실패: {e}")

    async def _partial_close_position(self, coin: str, stage: int, exit_price: float):
        """TP1/TP2 분할 익절: 일부 청산 + SL 이동."""
        pos = self.pos_manager.get_position(coin)
        if not pos:
            return
        if stage == 1 and pos.tp1_hit:
            return
        if stage == 2 and pos.tp2_hit:
            return

        # 청산 수량 결정
        original_qty = pos.qty
        if stage in (1, 2):
            partial_qty = self.exchange.round_qty(coin, original_qty * 0.33)
        else:
            partial_qty = pos.current_qty

        partial_qty = max(partial_qty, self.exchange.get_min_qty(coin))
        if partial_qty > pos.current_qty:
            partial_qty = pos.current_qty

        pnl_pct  = (exit_price - pos.entry_price) / pos.entry_price
        if pos.side == "SELL":
            pnl_pct = -pnl_pct
        pnl_usdt = pnl_pct * partial_qty * pos.entry_price

        logger.info(
            f"[TP{stage}] {coin} {pos.side} 분할 청산 {partial_qty}/{pos.current_qty} @ {exit_price:.4f} "
            f"| pnl={pnl_usdt:+.4f} USDT"
        )

        # 거래소 시장가 부분 청산 (live mode)
        if self.mode == "live":
            exit_side = "SELL" if pos.side == "BUY" else "BUY"
            close_oid = ExchangeAdapter.make_order_id(coin, exit_side, prefix=f"tp{stage}")
            try:
                await self.exchange.market_close(coin, exit_side, partial_qty, close_oid)
            except Exception as e:
                logger.error(f"[TP{stage}] {coin} 부분 청산 실패: {e}")
                return

        # 포지션 상태 업데이트
        pos.remaining_qty = round(pos.current_qty - partial_qty, 8)
        if stage == 1:
            pos.tp1_hit = True
            new_sl = pos.entry_price  # BEP로 이동
            logger.info(f"[TP1] {coin} SL → BEP {new_sl:.4f}")
        else:  # stage == 2
            pos.tp2_hit = True
            new_sl = pos.tp1_price   # TP1 가격으로 이동
            logger.info(f"[TP2] {coin} SL → TP1 {new_sl:.4f}")

        # 거래소 SL 주문 갱신 (cancel + replace)
        if self.mode == "live" and pos.sl_order_id:
            exit_side = "SELL" if pos.side == "BUY" else "BUY"
            # SL 주문 갱신 (cancel + replace at new SL, reduced qty)
            try:
                await self.exchange.cancel_order(coin, order_link_id=pos.sl_order_id)
            except Exception:
                pass
            new_sl_oid = pos.sl_order_id + f"-s{stage}"
            await self.exchange.place_protective_stop(
                coin, exit_side, pos.remaining_qty,
                self.exchange.round_price(coin, new_sl), new_sl_oid, pos.entry_order_id,
            )
            pos.sl_order_id = new_sl_oid
            # TP 주문 갱신 (cancel + replace at TP3, reduced qty)
            if pos.tp_order_id:
                try:
                    await self.exchange.cancel_order(coin, order_link_id=pos.tp_order_id)
                except Exception:
                    pass
                new_tp_oid = pos.entry_order_id + f"-tp-s{stage}"
                tp_r = await self.exchange.place_take_profit(
                    coin, exit_side, pos.remaining_qty,
                    self.exchange.round_price(coin, pos.tp3_price), new_tp_oid, pos.entry_order_id,
                )
                pos.tp_order_id = new_tp_oid if tp_r.get("success") else ""

        pos.sl_price = new_sl
        self.pos_manager._save()

        self.risk_engine.record_pnl(pnl_usdt)
        self.equity.record_pnl(pnl_usdt)

        log_event("partial_tp", {
            "coin": coin, "stage": stage, "side": pos.side,
            "entry": pos.entry_price, "exit": exit_price,
            "partial_qty": partial_qty, "remaining_qty": pos.remaining_qty,
            "pnl_pct": round(pnl_pct, 6), "pnl_usdt": round(pnl_usdt, 4),
            "new_sl": new_sl,
        })

    # ── Close Position ─────────────────────────────────────

    async def _close_position(self, coin: str, reason: str, exit_price: float = 0.0):
        # Prevent double-close from monitor + main loop race
        if coin in self._closing:
            return
        self._closing.add(coin)
        try:
            await self._close_position_inner(coin, reason, exit_price)
        finally:
            self._closing.discard(coin)

    async def _close_position_inner(self, coin: str, reason: str, exit_price: float):
        pos = self.pos_manager.get_position(coin)
        if not pos:
            return
        if exit_price <= 0:
            try:
                ticker = await self.exchange.fetch_ticker(coin)
                exit_price = ticker["last"]
            except Exception:
                # Use last tracked price from SlTpMonitor, not entry (Issue 1.4)
                # Conservative fallback: use mid of tracked range
                exit_price = (pos.price_high + pos.price_low) / 2
                if exit_price <= 0:
                    exit_price = pos.entry_price
                logger.warning(f"[Close] {coin}: using fallback price {exit_price:.4f}")

        if pos.side == "BUY":
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price
        pnl_usdt = pnl_pct * pos.current_qty * pos.entry_price

        self.risk_engine.record_pnl(pnl_usdt)
        self.equity.record_pnl(pnl_usdt)
        self.ledger.record_pnl(coin, pnl_usdt, 0.0)

        if self.mode == "live":
            exit_side = "SELL" if pos.side == "BUY" else "BUY"
            # Cancel SL/TP BEFORE market close to prevent exchange double-fill
            for oid in [pos.sl_order_id, pos.tp_order_id]:
                if oid:
                    try:
                        await self.exchange.cancel_order(coin, order_link_id=oid)
                    except Exception:
                        pass
            try:
                close_oid = ExchangeAdapter.make_order_id(coin, exit_side, prefix="exit")
                await self.exchange.market_close(coin, exit_side, pos.current_qty, close_oid)
            except Exception as e:
                logger.error(f"[Close] {coin}: {e}")

        self.pos_manager.remove_position(coin)

        # RL signal logger: backfill result with MFE/MAE
        self.signal_logger.update_result(
            coin, pnl_pct, reason, pos.bars_held,
            mfe_pct=pos.mfe_pct, mae_pct=pos.mae_pct,
        )

        log_event("position_closed", {
            "coin": coin, "reason": reason, "side": pos.side,
            "entry": pos.entry_price, "exit": exit_price,
            "pnl_pct": round(pnl_pct, 6), "pnl_usdt": round(pnl_usdt, 4),
            "bars": pos.bars_held,
        })
        logger.info(
            f"[Close] {coin} {pos.side} ({reason}) | "
            f"entry={pos.entry_price:.4f} exit={exit_price:.4f} | "
            f"pnl={pnl_pct:.2%} ({pnl_usdt:+.2f} USDT)"
        )
        discord_report_close(coin, pos.side, reason, pos.entry_price, exit_price,
                             pnl_usdt, pnl_pct, pos.bars_held,
                             bar_minutes=self.common.get("bar_minutes", 15))
        ok, msg = self.risk_engine.check_drawdown()
        if not ok:
            logger.warning(f"[KillSwitch] {msg}")

    # ── Step 5: Equity ─────────────────────────────────────

    async def _update_equity(self):
        if self.mode == "live":
            try:
                bal = await self.exchange.fetch_balance()
                self.equity.sync_from_exchange(bal["total"])
            except Exception:
                pass
        logger.info(f"[Equity] {self.equity.current:.2f} USDT (cum PnL: {self.equity.cumulative_pnl:+.2f})")


# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="CLAUDE_CRYPTO_AGENT Live Bot v4.3")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--equity", type=float, default=10000.0)
    parser.add_argument("--config", type=str, default=None, help="config yaml path (relative to project root)")
    parser.add_argument("--yes", action="store_true", help="skip live mode confirmation prompt")
    return parser.parse_args()


async def async_main(args):
    bot = LiveTradingBot(mode=args.mode, initial_equity=args.equity, config_path=args.config)
    shutdown_triggered = False

    def handle_sig(signum, frame):
        nonlocal shutdown_triggered
        if not shutdown_triggered:
            shutdown_triggered = True
            logger.info(f"\n[Signal] {signal.Signals(signum).name}, shutting down...")
            asyncio.ensure_future(bot.shutdown())

    signal.signal(signal.SIGINT, handle_sig)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_sig)

    try:
        await bot.initialize()
        await bot.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"[Fatal] {e}\n{traceback.format_exc()}")
    finally:
        if bot.running:
            await bot.shutdown()


def main():
    args = parse_args()
    if args.mode == "live" and not args.yes:
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
