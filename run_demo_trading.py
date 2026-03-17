"""Demo Trading — Binance Futures Testnet 실주문.

run_paper_v3_4.py의 학습·신호 로직 100% 재사용.
시뮬레이션 대신 Binance Testnet에 실제 주문 집행.

Usage:
    BINANCE_API_KEY=xxx BINANCE_API_SECRET=yyy python run_demo_trading.py

Coins : DOT, ADA
Mode  : sandbox (testnet.binancefuture.com)
Config: config/frozen_params_v3_4.yaml
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

# .env 파일 자동 로드 (python-dotenv 있으면 사용, 없으면 직접 파싱)
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        for _line in _env_file.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ── 학습 파이프라인 (run_paper_v3_4와 동일) ───────────────────
from src.data.crawlers.crypto_ohlcv import fetch_all_top10
from src.data.crawlers.macro_commodity_crawler import crawl_all_macro_data
from src.models.masking_loop import create_labels_triple_barrier, LABEL_MAP, HORIZONS
from src.models.enhanced_ensemble import EnhancedEnsemble
from src.execution.exchange_adapter import ExchangeAdapter
from src.execution.cost_model import CostModel, FeeSchedule, FundingConfig, MissFillConfig
from sklearn.feature_selection import mutual_info_classif

# ── 설정 ──────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent
YAML_PATH     = PROJECT_ROOT / "config" / "frozen_params_v3_4.yaml"
REPORT_DIR    = PROJECT_ROOT / "data" / "reports" / "demo_trading"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE      = REPORT_DIR / "demo_log.jsonl"

with open(YAML_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

COMMON        = CFG["common"]
COIN_PARAMS   = CFG["coins"]
COST_CFG      = CFG["cost_model"]
EXCLUDED_KW   = CFG["excluded_feature_keywords"]
GLOBAL_BLOCKED = CFG["blocked_regimes"]
ACTIVE_COINS  = list(COIN_PARAMS.keys())   # DOT, ADA

BM            = COMMON["bar_minutes"]       # 240 min
MAX_HORIZON   = COMMON["max_horizon"]       # 18 bars
RISK_FRAC     = COMMON["risk_frac"]         # 0.005
MIN_NOTIONAL  = 5.0                         # Binance 최소 $5
N_JOBS        = 4
POLL_SEC      = 60                          # 폴링 주기 (초)
RETRAIN_HOURS = 24
MONITOR_SEC   = 600                         # 10분 모니터링 주기

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(REPORT_DIR / "demo_trading.log", encoding="utf-8", mode="a"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("demo")
monitor_log = logging.getLogger("monitor")
_mh = logging.FileHandler(REPORT_DIR / "monitor.log", encoding="utf-8", mode="a")
_mh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
monitor_log.addHandler(_mh)
monitor_log.setLevel(logging.INFO)


# ── 헬퍼 ──────────────────────────────────────────────────────

def coin_k(coin, key):
    override = f"{key}_override"
    if override in COIN_PARAMS.get(coin, {}):
        return COIN_PARAMS[coin][override]
    return COMMON[key]

def coin_blocked_regimes(coin):
    return COIN_PARAMS.get(coin, {}).get("blocked_regimes_override", GLOBAL_BLOCKED)

def is_excluded(col):
    return any(kw in col.lower() for kw in EXCLUDED_KW)

def get_regime(df: pd.DataFrame, i: int) -> str:
    if i < 42:
        return "UNKNOWN"
    close = df["close"].values[max(0, i - 41):i + 1]
    ema_f = pd.Series(close).ewm(span=10).mean().iloc[-1]
    ema_s = pd.Series(close).ewm(span=30).mean().iloc[-1]
    ret_std = np.std(np.diff(close) / close[:-1]) if len(close) > 1 else 0
    if ema_f > ema_s * 1.005:
        return "TREND_UP"
    elif ema_f < ema_s * 0.995:
        return "TREND_DOWN"
    return "RANGE_HIGH" if ret_std > 0.02 else "RANGE_LOW"

def write_log(event: dict):
    event["ts"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")
    log.info(json.dumps(event, default=str))


# ── 데이터 ────────────────────────────────────────────────────

def fetch_data() -> dict[str, pd.DataFrame]:
    log.info("데이터 수집 중...")
    ohlcv = fetch_all_top10("365d", "1h")
    first = list(ohlcv.values())[0]
    macro = crawl_all_macro_data(first.index)
    ma = macro.get("aligned", pd.DataFrame())
    data = {}
    for coin in ACTIVE_COINS:
        if coin not in ohlcv:
            continue
        df = ohlcv[coin].copy()
        if len(ma) > 0:
            for col in ma.columns:
                df[col] = ma[col].reindex(df.index).ffill().bfill().fillna(0)
        data[coin] = df
    return data


def refresh_coin_live(coin: str, train_df: pd.DataFrame) -> pd.DataFrame:
    """바 종가 후 해당 코인만 최신 60일 데이터로 갱신 (빠른 피처 재계산).

    학습 파이프라인과 동일한 add_technical_indicators + add_signal_features 적용.
    실패 시 train_df 그대로 반환 (graceful degradation).
    """
    try:
        from src.data.crawlers.crypto_ohlcv import (
            fetch_ohlcv, resample_to_4h, add_technical_indicators,
            _add_decomposition, TOP10_YAHOO,
        )
        from src.data.crawlers.signal_features import add_signal_features

        yahoo_sym = TOP10_YAHOO.get(coin, f"{coin}-USD")
        df = fetch_ohlcv(coin, yahoo_sym, "60d", "1h")
        if df.empty:
            return train_df
        df = resample_to_4h(df)
        df = add_technical_indicators(df)
        df = _add_decomposition(df, period=42)
        df = add_signal_features(df, verbose=False)

        # 학습 df에 있는 매크로 컬럼 마지막값 채우기
        macro_cols = [c for c in train_df.columns if c not in df.columns]
        for c in macro_cols:
            df[c] = train_df[c].iloc[-1]

        log.info(f"[{coin}] 실시간 갱신 완료: {len(df)}bars, {len(df.columns)}cols, "
                 f"latest=${df['close'].iloc[-1]:,.4f}")
        return df
    except Exception as e:
        log.warning(f"[{coin}] 실시간 갱신 실패, 학습 데이터 사용: {e}")
        return train_df


# ── 학습 ──────────────────────────────────────────────────────

def train_models(data: dict) -> dict:
    """frozen_params_v3_4 기반 2-Stage Ensemble 학습."""
    models = {}
    for coin in ACTIVE_COINS:
        if coin not in data:
            continue
        params = COIN_PARAMS[coin]
        df = data[coin]
        h = HORIZONS[-1]
        h_label = f"label_{h * BM}min"
        ku = coin_k(coin, "k_upper")
        kl = coin_k(coin, "k_lower")

        labeled = create_labels_triple_barrier(
            df.copy(), h, k_upper_override=ku, k_lower_override=kl, verbose=False)
        if h_label not in labeled.columns:
            log.warning(f"[{coin}] label 생성 실패")
            continue

        exclude = {"label", "future_return", "open", "high", "low", "close", "volume"}
        for hh in HORIZONS:
            exclude.update([f"label_{hh*BM}min", f"return_{hh*BM}min"])
        leak_kw = ["future", "target", "label_", "return_", "fwd_", "forward_"]

        fcols = [c for c in labeled.columns
                 if c not in exclude
                 and not any(kw in c.lower() for kw in leak_kw)
                 and not is_excluded(c)
                 and labeled[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]

        mf = params["max_features"]
        clean = labeled.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        X = clean[fcols].fillna(0).values
        y = clean[h_label].fillna(1).values.astype(int)

        if len(fcols) > mf:
            nmi = min(2000, len(X))
            mi = mutual_info_classif(X[:nmi], y[:nmi], discrete_features=False,
                                     random_state=42, n_neighbors=5)
            top = np.argsort(mi)[-mf:]
            fcols = [fcols[i] for i in sorted(top)]
            X = clean[fcols].fillna(0).values

        y_s1 = (y != LABEL_MAP["HOLD"]).astype(int)
        if len(np.unique(y_s1)) < 2:
            continue
        s1c = np.bincount(y_s1, minlength=2)
        s1w = np.where(s1c > 0, len(y_s1) / (2 * s1c + 1e-10), 1.0)[y_s1]
        s1 = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
        s1.fit(X, y_s1, sample_weight=s1w)

        s2 = None
        tm = y != LABEL_MAP["HOLD"]
        if tm.sum() >= 30:
            y_s2 = (y[tm] == LABEL_MAP["UP"]).astype(int)
            if len(np.unique(y_s2)) >= 2:
                s2c = np.bincount(y_s2, minlength=2)
                s2w = np.where(s2c > 0, len(y_s2) / (2 * s2c + 1e-10), 1.0)[y_s2]
                s2 = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
                s2.fit(X[tm], y_s2, sample_weight=s2w)

        models[coin] = (s1, s2, fcols)
        log.info(f"[{coin}] 학습 완료: {len(X)}샘플 / {len(fcols)}피처")

    return models


# ── 신호 생성 ─────────────────────────────────────────────────

def generate_signal(coin: str, df: pd.DataFrame, models: dict) -> dict | None:
    """bar 종가에서 신호 생성. None = 신호 없음."""
    if coin not in models:
        return None
    params = COIN_PARAMS[coin]
    s1m, s2m, fcols = models[coin]

    idx = len(df) - 1
    regime = get_regime(df, idx)

    # Stage 1: Trade/NoTrade
    xf = df[fcols].iloc[-1:].replace([np.inf, -np.inf], 0).fillna(0).values
    s1p = s1m.predict_proba(xf)
    s1_prob = float(s1p[0, 1])

    if s1_prob < params["stage1_threshold"]:
        return None

    # Regime block
    if regime in coin_blocked_regimes(coin):
        log.info(f"[{coin}] 레짐 차단: {regime} (s1={s1_prob:.3f})")
        return None

    # Stage 2: Long/Short
    s2_prob = 0.5
    side = "SELL"
    if s2m is not None:
        s2p = s2m.predict_proba(xf)
        s2_prob = float(s2p[0, 1])
        side = "BUY" if np.argmax(s2p[0]) == 1 else "SELL"

    bar = df.iloc[-1]
    entry = float(bar["close"])
    ku = coin_k(coin, "k_upper")
    kl = coin_k(coin, "k_lower")
    atr = float(bar.get("atr_14", entry * 0.01))
    if np.isnan(atr) or atr <= 0:
        atr = entry * 0.01

    ud = max(ku * atr, COMMON["min_barrier_pct"] * entry)
    ld = max(kl * atr, COMMON["min_barrier_pct"] * entry)
    tp = entry + ud if side == "BUY" else entry - ud
    sl = entry - ld if side == "BUY" else entry + ld

    return {
        "coin": coin, "side": side, "entry": entry,
        "tp": tp, "sl": sl, "atr": atr, "kl": kl,
        "regime": regime, "s1_prob": s1_prob, "s2_prob": s2_prob,
    }


# ── 실행 엔진 ─────────────────────────────────────────────────

class DemoEngine:
    """Binance Testnet 실주문 트래커."""

    def __init__(self, adapter: ExchangeAdapter, equity_usdt: float):
        self.adapter = adapter
        self.equity  = equity_usdt
        self.positions: dict[str, dict] = {}   # coin → position info
        self.trades: list[dict] = []

    async def enter(self, sig: dict) -> bool:
        coin  = sig["coin"]
        side  = sig["side"]
        entry = sig["entry"]
        sl    = sig["sl"]
        tp    = sig["tp"]
        kl    = sig["kl"]
        atr   = sig["atr"]

        # 수량 계산: risk_frac / (kl * atr / entry)
        stop_dist = max(kl * atr, COMMON["min_barrier_pct"] * entry)
        risk_usdt = self.equity * RISK_FRAC
        qty_raw   = risk_usdt / stop_dist
        # 최소 노셔널 $5 보장
        qty_raw   = max(qty_raw, MIN_NOTIONAL / entry * 1.1)
        qty       = self.adapter.round_qty(coin, qty_raw)
        sl        = self.adapter.round_price(coin, sl)
        tp        = self.adapter.round_price(coin, tp)

        # 시장가 진입
        entry_id = ExchangeAdapter.make_order_id(coin, side, prefix="demo")
        try:
            order = await self.adapter._exchange.create_order(
                symbol=self.adapter._ccxt_symbol(coin),
                type="market",
                side=side.lower(),
                amount=qty,
                params={"newClientOrderId": entry_id},
            )
            fill_price = order.get("average") or order.get("price") or entry
        except Exception as e:
            log.error(f"[{coin}] 진입 실패: {e}")
            write_log({"event": "ENTRY_FAIL", "coin": coin, "error": str(e)})
            return False

        # SL 등록
        sl_id  = ExchangeAdapter.make_order_id(coin, "SELL" if side == "BUY" else "BUY",
                                               seq=2, prefix="demo")
        sl_res = await self.adapter.place_protective_stop(
            symbol=coin, side="SELL" if side == "BUY" else "BUY",
            qty=qty, stop_price=sl, order_link_id=sl_id,
        )

        # TP 등록
        tp_id  = ExchangeAdapter.make_order_id(coin, "SELL" if side == "BUY" else "BUY",
                                               seq=3, prefix="demo")
        tp_res = await self.adapter.place_take_profit(
            symbol=coin, side="SELL" if side == "BUY" else "BUY",
            qty=qty, tp_price=tp, order_link_id=tp_id,
        )

        self.positions[coin] = {
            "side": side, "entry": fill_price, "qty": qty,
            "tp": tp, "sl": sl, "atr": atr,
            "sl_id": sl_id, "sl_exch": sl_res.get("exchange_order_id", ""),
            "tp_id": tp_id, "tp_exch": tp_res.get("exchange_order_id", ""),
            "bars": 0,
            "regime": sig["regime"],
            "s1_prob": sig["s1_prob"],
            "s2_prob": sig["s2_prob"],
            "open_time": datetime.now(timezone.utc).isoformat(),
        }

        write_log({
            "event": "OPEN", "coin": coin, "side": side,
            "entry": round(fill_price, 6), "qty": qty,
            "tp": round(tp, 6), "sl": round(sl, 6),
            "regime": sig["regime"],
            "s1": round(sig["s1_prob"], 3), "s2": round(sig["s2_prob"], 3),
        })
        log.info(f"[{coin}] OPEN {side} @{fill_price:.4f} SL={sl:.4f} TP={tp:.4f} qty={qty}")
        return True

    async def check_exits(self, coin: str, df: pd.DataFrame):
        """TTL 체크. SL/TP는 거래소 주문이 처리."""
        if coin not in self.positions:
            return
        pos = self.positions[coin]
        pos["bars"] += 1

        # 거래소 포지션 실제 확인
        exch_pos = await self.adapter.fetch_position(coin)

        # 포지션이 이미 청산됐으면 (SL/TP 체결) 로그 처리
        if exch_pos is None:
            close_price = float(df["close"].iloc[-1])
            await self._record_close(coin, "sl_or_tp", close_price)
            return

        # TTL 초과 시 강제 청산
        if pos["bars"] >= MAX_HORIZON:
            log.info(f"[{coin}] TTL 도달 ({MAX_HORIZON}bars) → 강제 청산")
            await self.close(coin, "time_stop", float(df["close"].iloc[-1]))

    async def close(self, coin: str, reason: str, price: float):
        if coin not in self.positions:
            return
        pos = self.positions[coin]

        # SL/TP 취소
        close_side = "SELL" if pos["side"] == "BUY" else "BUY"
        for oid, exch_id in [(pos["sl_id"], pos["sl_exch"]),
                              (pos["tp_id"], pos["tp_exch"])]:
            await self.adapter.cancel_order(coin,
                                            order_link_id=oid,
                                            exchange_order_id=exch_id)

        # 시장가 청산
        close_id = ExchangeAdapter.make_order_id(coin, close_side, seq=9, prefix="demo")
        res = await self.adapter.market_close(
            symbol=coin, side=close_side,
            qty=pos["qty"], order_link_id=close_id,
        )
        fill = res.get("fill_price") or price
        await self._record_close(coin, reason, fill)

    async def _record_close(self, coin: str, reason: str, exit_price: float):
        pos = self.positions.pop(coin, None)
        if pos is None:
            return

        if pos["side"] == "BUY":
            gpnl = (exit_price - pos["entry"]) / pos["entry"]
        else:
            gpnl = (pos["entry"] - exit_price) / pos["entry"]

        sd   = max(pos["atr"] * coin_k(coin, "k_lower") / pos["entry"], 0.003)
        nr   = RISK_FRAC / sd
        net  = gpnl * nr * self.equity   # approximate USDT P&L

        self.equity += net
        trade = {
            "coin": coin, "side": pos["side"],
            "entry": round(pos["entry"], 6), "exit": round(exit_price, 6),
            "reason": reason, "bars": pos["bars"],
            "net_usdt": round(net, 4),
            "regime": pos["regime"],
            "open_time": pos["open_time"],
            "close_time": datetime.now(timezone.utc).isoformat(),
        }
        self.trades.append(trade)
        write_log({"event": "CLOSE", **trade})
        log.info(f"[{coin}] CLOSE {reason} @{exit_price:.4f} net={net:+.4f} USDT")

    def summary(self) -> dict:
        n = len(self.trades)
        if n == 0:
            return {"trades": 0, "equity": self.equity}
        pnls = [t["net_usdt"] for t in self.trades]
        wins = sum(1 for p in pnls if p > 0)
        return {
            "trades": n,
            "win_rate": round(wins / n, 3),
            "equity": round(self.equity, 2),
            "total_pnl_usdt": round(sum(pnls), 4),
            "avg_pnl_usdt": round(np.mean(pnls), 4),
        }


# ── 4h 바 감지 ────────────────────────────────────────────────

BAR_HOURS = [0, 4, 8, 12, 16, 20]

def current_bar_close(now: datetime) -> datetime:
    """현재 시각 기준 가장 최근 확정 4h 바 종가 시각 (UTC, +5분 여유)."""
    for h in reversed(BAR_HOURS):
        if now.hour > h or (now.hour == h and now.minute >= 5):
            return now.replace(hour=h, minute=0, second=0, microsecond=0,
                                tzinfo=timezone.utc)
    return (now - timedelta(days=1)).replace(
        hour=20, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)


def _next_bar_str(now: datetime) -> str:
    """다음 4h 바 종가까지 남은 시간 문자열."""
    for h in BAR_HOURS:
        candidate = now.replace(hour=h, minute=5, second=0, microsecond=0)
        if candidate > now:
            diff = candidate - now
            m = int(diff.total_seconds() // 60)
            return f"{h:02d}:05 UTC (약 {m}분 후)"
    # 자정 넘어
    candidate = (now + timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0)
    diff = candidate - now
    m = int(diff.total_seconds() // 60)
    return f"00:05 UTC (약 {m}분 후)"


# ── 메인 루프 ─────────────────────────────────────────────────

async def main():
    api_key    = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")

    if not api_key:
        log.error("BINANCE_API_KEY 환경변수가 없습니다.")
        sys.exit(1)

    log.info("=" * 60)
    log.info("  DEMO TRADING — Binance Futures Testnet")
    log.info(f"  Coins : {ACTIVE_COINS}")
    log.info(f"  Config: {YAML_PATH}")
    log.info(f"  Log   : {LOG_FILE}")
    log.info("=" * 60)

    async with ExchangeAdapter(mode="sandbox",
                               api_key=api_key,
                               secret=api_secret) as adapter:
        # 초기 잔고
        bal = await adapter.fetch_balance()
        equity = bal["free"]
        log.info(f"  초기 잔고: {equity:.2f} USDT")

        engine     = DemoEngine(adapter, equity)
        models     = {}
        last_train = None
        last_bar   : dict[str, datetime] = {}

        last_monitor  = datetime.min.replace(tzinfo=timezone.utc)
        live_data: dict[str, pd.DataFrame] = {}  # 실시간 갱신된 데이터

        while True:
            now = datetime.now(timezone.utc)

            # ── 모델 학습 (시작 시 + 24h마다) ──────────────────
            if last_train is None or (now - last_train) > timedelta(hours=RETRAIN_HOURS):
                log.info("[TRAIN] 모델 학습 시작...")
                try:
                    data = await asyncio.get_event_loop().run_in_executor(
                        None, fetch_data)
                    models = await asyncio.get_event_loop().run_in_executor(
                        None, train_models, data)
                    live_data = dict(data)  # 학습 데이터로 초기화
                    last_train = now
                    log.info(f"[TRAIN] 완료: {list(models.keys())}")
                except Exception as e:
                    log.error(f"[TRAIN] 실패: {e}")
                    await asyncio.sleep(POLL_SEC)
                    continue

            # ── 바 감지 ────────────────────────────────────────
            bar_close = current_bar_close(now)

            for coin in ACTIVE_COINS:
                if coin not in data or coin not in models:
                    continue

                # 새 바인지 확인
                if last_bar.get(coin) == bar_close:
                    continue
                last_bar[coin] = bar_close

                # ── 실시간 피처 갱신 ───────────────────────────
                log.info(f"[{coin}] 새 4h바 확정 ({bar_close.strftime('%H:%M UTC')}) — 피처 갱신 중...")
                live_df = await asyncio.get_event_loop().run_in_executor(
                    None, refresh_coin_live, coin, data[coin])
                live_data[coin] = live_df

                # 포지션 TTL 체크
                await engine.check_exits(coin, live_df)

                # 포지션 없을 때만 진입 시도
                if coin not in engine.positions:
                    sig = generate_signal(coin, live_df, models)
                    if sig:
                        log.info(
                            f"[{coin}] ★ 신호 발생: {sig['side']} "
                            f"entry={sig['entry']:.4f} "
                            f"TP={sig['tp']:.4f} SL={sig['sl']:.4f} "
                            f"s1={sig['s1_prob']:.3f} s2={sig['s2_prob']:.3f} "
                            f"regime={sig['regime']}"
                        )
                        write_log({"event": "SIGNAL", "coin": coin,
                                   "side": sig["side"], "entry": sig["entry"],
                                   "tp": sig["tp"], "sl": sig["sl"],
                                   "s1_prob": sig["s1_prob"], "s2_prob": sig["s2_prob"],
                                   "regime": sig["regime"]})
                        await engine.enter(sig)
                    else:
                        log.info(f"[{coin}] 신호 없음 (bar={bar_close.strftime('%H:%M')})")
                else:
                    pos = engine.positions[coin]
                    log.info(
                        f"[{coin}] 포지션 유지 중: {pos['side']} @{pos['entry']:.4f} "
                        f"bars={pos['bars']}/{MAX_HORIZON} "
                        f"TP={pos['tp']:.4f} SL={pos['sl']:.4f}"
                    )

            # ── 10분 모니터링 출력 ─────────────────────────────
            if (now - last_monitor).total_seconds() >= MONITOR_SEC:
                last_monitor = now
                s = engine.summary()

                # 현재 가격 수집
                prices = {}
                for coin in ACTIVE_COINS:
                    if coin in live_data and len(live_data[coin]) > 0:
                        prices[coin] = float(live_data[coin]["close"].iloc[-1])

                # 미결 포지션 상세
                pos_lines = []
                for coin, pos in engine.positions.items():
                    cur = prices.get(coin, pos["entry"])
                    if pos["side"] == "BUY":
                        upnl = (cur - pos["entry"]) / pos["entry"] * 100
                    else:
                        upnl = (pos["entry"] - cur) / pos["entry"] * 100
                    pos_lines.append(
                        f"    {coin} {pos['side']} @{pos['entry']:.4f} "
                        f"현재={cur:.4f} upnl={upnl:+.2f}% "
                        f"TP={pos['tp']:.4f} SL={pos['sl']:.4f} "
                        f"bars={pos['bars']}/{MAX_HORIZON}"
                    )

                # 최근 거래 5건
                recent = engine.trades[-5:] if engine.trades else []
                trade_lines = []
                for t in recent:
                    trade_lines.append(
                        f"    {t['coin']} {t['side']} {t['entry']:.4f}→{t['exit']:.4f} "
                        f"net={t['net_usdt']:+.4f}USDT reason={t['reason']}"
                    )

                sep = "=" * 70
                report = (
                    f"\n{sep}\n"
                    f"  [MONITOR] {now.strftime('%Y-%m-%d %H:%M UTC')}\n"
                    f"{sep}\n"
                    f"  잔고    : {s['equity']:.2f} USDT\n"
                    f"  총거래  : {s['trades']}건"
                    + (f"  승률={s.get('win_rate',0)*100:.1f}%"
                       f"  총손익={s.get('total_pnl_usdt',0):+.4f} USDT" if s['trades'] else "") + "\n"
                    f"  다음바  : "
                    + _next_bar_str(now) + "\n"
                    f"\n  ─ 오픈 포지션 ({len(engine.positions)}건) ─\n"
                    + ("\n".join(pos_lines) if pos_lines else "    없음") + "\n"
                    f"\n  ─ 최근 거래 ({len(recent)}건) ─\n"
                    + ("\n".join(trade_lines) if trade_lines else "    없음") + "\n"
                    f"{sep}"
                )
                log.info(report)
                monitor_log.info(report)
                write_log({"event": "MONITOR", **s,
                           "open_positions": len(engine.positions),
                           "prices": prices})

            await asyncio.sleep(POLL_SEC)


if __name__ == "__main__":
    asyncio.run(main())
