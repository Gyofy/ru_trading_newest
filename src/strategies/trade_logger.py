"""TradeLogger — per-coin optimization-ready trade logging pipeline.

Captures full market context at entry and exit for every trade.
Enables per-coin parameter optimization by storing:
  - Signal conditions (CVD, OFI, z-score, quantile breach)
  - Market microstructure (ATR, volatility, spread, volume, VPIN, funding)
  - Execution quality (fill price vs mid, slippage, maker/taker)
  - Position trajectory (bar-by-bar MFE/MAE snapshots)
  - Outcome (PnL, exit reason, bars held)

Output: data/reports/multi_strategy/trade_context.jsonl
Each line = 1 complete trade with full entry/exit context.

Usage:
    logger = TradeLogger(Path("data/reports/multi_strategy"))
    ctx = await logger.capture_entry_context(data_hub, coin, signal, strategy_name)
    # ... position held ...
    logger.record_close(ctx, exit_price, reason, pos)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("trade_logger")


@dataclass
class TradeContext:
    """Full context for one trade — captured at entry, completed at exit."""

    # ── Identity ──
    trade_id: str = ""
    ts_entry: str = ""
    ts_exit: str = ""
    coin: str = ""
    strategy: str = ""
    side: str = ""  # BUY or SELL

    # ── Signal conditions at entry ──
    trigger_type: str = ""       # e.g., "cvd_q99_3sigma", "cvd_extreme_buy"
    cvd_value: float = 0.0      # raw CVD delta at entry
    cvd_z_score: float = 0.0    # z-score of CVD delta
    cvd_quantile_breach: float = 0.0  # how far above Q threshold
    ofi_value: float = 0.0      # OFI at entry
    signal_strength: float = 0.0
    signal_confidence: float = 0.0

    # ── Market microstructure at entry ──
    entry_price: float = 0.0
    entry_atr: float = 0.0
    entry_atr_pct: float = 0.0  # ATR / price
    entry_volatility_24h: float = 0.0  # (high-low)/last from ticker
    entry_volume_24h_usdt: float = 0.0
    entry_spread_bps: float = 0.0
    entry_vpin: float = 0.0
    entry_funding_rate: float = 0.0
    entry_bid: float = 0.0
    entry_ask: float = 0.0

    # ── Barriers ──
    sl_price: float = 0.0
    tp_price: float = 0.0
    sl_distance_pct: float = 0.0
    tp_distance_pct: float = 0.0
    trailing_sl: bool = False
    trail_distance: float = 0.0
    risk_usdt: float = 0.0      # planned risk in USD
    rr_estimate: float = 0.0    # estimated R:R at entry

    # ── Sizing ──
    qty: float = 0.0
    notional_usdt: float = 0.0
    leverage: int = 1

    # ── Market context at exit ──
    exit_price: float = 0.0
    exit_atr: float = 0.0
    exit_spread_bps: float = 0.0
    exit_funding_rate: float = 0.0

    # ── Outcome ──
    exit_reason: str = ""        # SL_HIT, TP_HIT, TIME_STOP, TRAILING
    pnl_usdt: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0
    hold_duration_sec: float = 0.0

    # ── Trajectory (MFE/MAE) ──
    mfe_pct: float = 0.0        # max favorable excursion
    mae_pct: float = 0.0        # max adverse excursion
    price_high: float = 0.0     # highest price during hold
    price_low: float = 0.0      # lowest price during hold

    # ── Trailing stop history ──
    trail_sl_final: float = 0.0  # final SL after trailing

    # ── Execution quality ──
    fill_price: float = 0.0
    slippage_entry_bps: float = 0.0  # (fill - mid) / mid * 10000
    paper_mode: bool = True

    # ── Strategy-specific params (frozen at entry) ──
    strategy_params: dict = field(default_factory=dict)

    # ── Time & market regime features (added for strategy optimization) ──
    hour_utc: int = -1            # 0-23 UTC hour at entry
    day_of_week: int = -1         # 0=Mon, 6=Sun
    session_utc: str = ""         # "Asia" | "London" | "NY" | "Off"
    trend_5bar: float = 0.0       # (close[-1] - close[-6]) / close[-6]
    trend_20bar: float = 0.0      # (close[-1] - close[-21]) / close[-21]
    bb_position: float = 0.5      # where price is in BB (0=lower band, 1=upper)
    rsi_value: float = 50.0       # RSI-14 at entry
    volatility_regime: str = ""   # "HIGH" | "MED" | "LOW" (ATR percentile vs 100-bar window)

    # ── Macro regime at entry (BTC 1h EMA filter) ──
    btc_ema20_1h: float = 0.0     # BTC EMA-20 on 1h chart
    btc_ema50_1h: float = 0.0     # BTC EMA-50 on 1h chart
    btc_regime_1h: str = ""       # "BULL" | "BEAR" | "NEUTRAL" (ema20 vs ema50)
    btc_price_vs_ema20: float = 0.0  # (btc_price - ema20) / ema20
    trade_vs_btc_regime: str = ""  # "ALIGNED" | "COUNTER" | "NEUTRAL"

    # ── Entry quality ──
    fill_type: str = "paper"       # "maker" | "taker" | "paper"
    fee_estimate_usdt: float = 0.0  # round-trip fee estimate at entry

    # ── Trade lifecycle (populated at close) ──
    mfe_to_trail_ratio: float = 0.0  # mfe_pct / (trail_distance/entry_price) — how close trail was to MFE
    sl_tighten_count: int = 0        # how many times trailing SL moved
    time_to_mfe_bars: int = 0        # bar number when MFE was reached (0 = unknown)
    bars_between_mfe_exit: int = 0   # bars from MFE to exit — reversal speed proxy
    atr_ratio_exit_entry: float = 0.0  # exit_atr / entry_atr — volatility expansion(>1) or contraction(<1)

    # ── Actual fee & net PnL (populated at close) ──
    fee_actual_usdt: float = 0.0     # actual round-trip fee paid (entry + exit)
    pnl_net_usdt: float = 0.0        # PnL after fees
    pnl_net_pct: float = 0.0         # net PnL % after fees
    fee_drag_pct: float = 0.0        # fee as % of entry notional (how much fee eats into trade)
    breakeven_move_pct: float = 0.0  # minimum price move needed just to cover fees

    # ── Portfolio context at entry ──
    concurrent_positions: int = 0    # how many other positions were open when this was entered
    portfolio_notional_at_entry: float = 0.0  # total open notional across all strategies at entry

    # ── Timestamps for duration calc ──
    _entry_unix: float = 0.0


class TradeLogger:
    """Append-only JSONL logger for optimization-ready trade data."""

    def __init__(self, report_dir: Path):
        self._dir = report_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "trade_context.jsonl"
        self._pending: dict[str, TradeContext] = {}  # key = "strategy:coin"
        self._trade_counter = 0

    async def capture_entry_context(
        self,
        data_hub,
        coin: str,
        strategy_name: str,
        side: str,
        signal_extra: dict,
        fill_price: float,
        sl_price: float,
        tp_price: float,
        qty: float,
        notional: float,
        leverage: int,
        atr: float,
        trailing_sl: bool,
        trail_distance: float,
        risk_usdt: float,
        rr_estimate: float,
        strategy_params: dict,
        paper_mode: bool = True,
        concurrent_positions: int = 0,
        portfolio_notional: float = 0.0,
    ) -> TradeContext:
        """Capture full market context at entry time."""
        self._trade_counter += 1
        now = datetime.now(timezone.utc)
        trade_id = f"{strategy_name}_{coin}_{now.strftime('%m%d_%H%M%S')}_{self._trade_counter:04d}"

        # ── RACE CONDITION FIX: save skeleton to _pending BEFORE any async I/O ──
        # If SL hits while we're awaiting ticker/ohlcv/BTC data (can take 10-20s),
        # record_close will still find a valid context. Fields are enriched below.
        from src.strategies.base import ROUND_TRIP_FEE_RATE as _FEE_RATE
        _sl_dist_pct = abs(fill_price - sl_price) / fill_price * 100 if fill_price > 0 else 0
        _tp_dist_pct = abs(tp_price - fill_price) / fill_price * 100 if (fill_price > 0 and tp_price > 0) else 0
        _breakeven_pct = (_FEE_RATE * 100) if fill_price > 0 else 0  # round-trip fee % of notional
        _sl_dist_abs = abs(fill_price - sl_price) if fill_price > 0 else 0
        _rr_est = abs(tp_price - fill_price) / _sl_dist_abs if (_sl_dist_abs > 0 and tp_price > 0) else 0.0
        key = f"{strategy_name}:{coin}"
        _skeleton = TradeContext(
            trade_id=trade_id,
            ts_entry=now.isoformat(),
            coin=coin,
            strategy=strategy_name,
            side=side,
            trigger_type=signal_extra.get("trigger", ""),
            cvd_value=signal_extra.get("cvd_value", 0.0),
            cvd_z_score=signal_extra.get("cvd_z_score", signal_extra.get("z_score", 0.0)),
            # cvd_quantile_rank (cvd_spike) / cvd_quantile_breach (asymmetric_sniper) 둘 다 수용
            cvd_quantile_breach=signal_extra.get("cvd_quantile_rank",
                                signal_extra.get("cvd_quantile_breach", 0.0)),
            ofi_value=signal_extra.get("ofi_value", 0.0),
            signal_strength=signal_extra.get("strength", 0.0),
            signal_confidence=signal_extra.get("confidence", 0.0),
            entry_price=fill_price,
            entry_atr=atr,
            entry_atr_pct=atr / fill_price * 100 if fill_price > 0 else 0,
            sl_price=sl_price,
            tp_price=tp_price,
            sl_distance_pct=_sl_dist_pct,
            tp_distance_pct=_tp_dist_pct,
            trailing_sl=trailing_sl,
            trail_distance=trail_distance,
            risk_usdt=risk_usdt,
            rr_estimate=round(_rr_est, 3),
            qty=qty,
            notional_usdt=notional,
            leverage=leverage,
            fill_price=fill_price,
            paper_mode=paper_mode,
            strategy_params=strategy_params,
            hour_utc=now.hour,
            day_of_week=now.weekday(),
            fee_estimate_usdt=round(notional * _FEE_RATE, 4),
            breakeven_move_pct=round(_breakeven_pct, 4),
            concurrent_positions=concurrent_positions,
            portfolio_notional_at_entry=round(portfolio_notional, 2),
            _entry_unix=time.time(),
        )
        self._pending[key] = _skeleton
        logger.info(f"[TradeLog] Entry skeleton saved: {trade_id}")

        # Fetch current market data
        ticker = await data_hub.get_ticker(coin)
        bid = ticker.get("bid", 0) or 0
        ask = ticker.get("ask", 0) or 0
        mid = (bid + ask) / 2 if (bid + ask) > 0 else fill_price
        high_24h = ticker.get("high", 0) or 0
        low_24h = ticker.get("low", 0) or 0
        last = ticker.get("last", 0) or fill_price

        # Volatility
        volatility_24h = (high_24h - low_24h) / last * 100 if last > 0 else 0

        # Volume
        vol_usdt = ticker.get("quoteVolume", 0) or 0

        # VPIN
        try:
            df = await data_hub.get_ohlcv(coin, "1m", limit=30)
            vpin = data_hub.compute_vpin(df, window=24)
        except Exception:
            vpin = 0.5

        # Funding
        try:
            funding = await data_hub.get_funding_rate(coin)
            if funding is None:
                funding = 0.0
        except Exception:
            funding = 0.0

        # ── Time & market regime features ──
        hour_utc = now.hour
        day_of_week = now.weekday()
        # UTC session classification (non-overlapping boundaries)
        if 0 <= hour_utc < 7:
            session_utc = "Asia"
        elif 7 <= hour_utc < 12:
            session_utc = "London"
        elif 12 <= hour_utc < 21:
            session_utc = "NY"
        else:
            session_utc = "Off"

        # Technical features from OHLCV
        trend_5bar = 0.0
        trend_20bar = 0.0
        bb_position = 0.5
        rsi_value = 50.0
        volatility_regime = "MED"
        try:
            df_feat = await data_hub.get_ohlcv(coin, "1m", limit=120)
            if df_feat is not None and len(df_feat) >= 22:
                closes = df_feat["close"].values
                # trend
                trend_5bar = float((closes[-1] - closes[-6]) / closes[-6]) if closes[-6] > 0 else 0.0
                trend_20bar = float((closes[-1] - closes[-21]) / closes[-21]) if closes[-21] > 0 else 0.0
                # BB position: (price - lower) / (upper - lower) over 20 bars
                bb_closes = closes[-20:]
                bb_mean = float(bb_closes.mean())
                bb_std = float(bb_closes.std()) + 1e-10
                bb_upper = bb_mean + 2 * bb_std
                bb_lower = bb_mean - 2 * bb_std
                bb_range = bb_upper - bb_lower
                bb_position = float((closes[-1] - bb_lower) / bb_range) if bb_range > 1e-10 else 0.5
                bb_position = max(0.0, min(1.0, bb_position))
                # RSI-14
                if len(closes) >= 16:
                    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
                    gains = [max(d, 0) for d in deltas[-14:]]
                    losses = [max(-d, 0) for d in deltas[-14:]]
                    avg_gain = sum(gains) / 14
                    avg_loss = sum(losses) / 14
                    rs = avg_gain / avg_loss if avg_loss > 1e-10 else 100.0
                    rsi_value = float(100 - 100 / (1 + rs))
                # Volatility regime: ATR percentile vs last 100 bars
                if len(df_feat) >= 50:
                    atrs = []
                    for i in range(1, min(101, len(df_feat))):
                        h = df_feat["high"].iloc[-i]
                        l = df_feat["low"].iloc[-i]
                        c_prev = df_feat["close"].iloc[-i-1]
                        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
                        atrs.append(tr)
                    atr_pct_now = atr / fill_price if fill_price > 0 else 0
                    sorted_atrs = sorted(atrs)
                    threshold_hi = sorted_atrs[int(len(sorted_atrs) * 0.75)]
                    threshold_lo = sorted_atrs[int(len(sorted_atrs) * 0.25)]
                    if atr >= threshold_hi:
                        volatility_regime = "HIGH"
                    elif atr <= threshold_lo:
                        volatility_regime = "LOW"
                    else:
                        volatility_regime = "MED"
        except Exception as _e:
            logger.debug(f"[TradeLog] Feature compute failed for {coin}: {_e}")

        # CVD/OFI context from signal (enriched by strategies)
        cvd_value = signal_extra.get("cvd_value", 0.0)
        cvd_z = signal_extra.get("cvd_z_score", signal_extra.get("z_score", 0.0))
        ofi_value = signal_extra.get("ofi_value", 0.0)
        strength = signal_extra.get("strength", 0.0)
        trigger = signal_extra.get("trigger", "")

        # ── Macro regime: BTC 1h EMA20 vs EMA50 ──
        btc_ema20_1h = 0.0
        btc_ema50_1h = 0.0
        btc_regime_1h = "NEUTRAL"
        btc_price_vs_ema20 = 0.0
        try:
            df_btc = await data_hub.get_ohlcv("BTC", "1h", limit=60)
            if df_btc is not None and len(df_btc) >= 50:
                btc_closes = df_btc["close"].values
                btc_ema20_1h = float(sum(btc_closes[-20:]) / 20)  # simple approx
                btc_ema50_1h = float(sum(btc_closes[-50:]) / 50)
                btc_price = float(btc_closes[-1])
                btc_price_vs_ema20 = (btc_price - btc_ema20_1h) / btc_ema20_1h if btc_ema20_1h > 0 else 0.0
                if btc_ema20_1h > btc_ema50_1h * 1.001:
                    btc_regime_1h = "BULL"
                elif btc_ema20_1h < btc_ema50_1h * 0.999:
                    btc_regime_1h = "BEAR"
        except Exception:
            pass

        # Determine if entry is aligned or counter to BTC macro regime
        if btc_regime_1h == "BULL":
            trade_vs_btc_regime = "ALIGNED" if side == "BUY" else "COUNTER"
        elif btc_regime_1h == "BEAR":
            trade_vs_btc_regime = "ALIGNED" if side == "SELL" else "COUNTER"
        else:
            trade_vs_btc_regime = "NEUTRAL"

        # Fill type (paper vs maker/taker)
        fill_type = "paper" if paper_mode else ("taker" if signal_extra.get("fill_taker", False) else "maker")

        # Fee estimate
        from src.strategies.base import ROUND_TRIP_FEE_RATE
        fee_estimate_usdt = notional * ROUND_TRIP_FEE_RATE
        breakeven_move_pct = ROUND_TRIP_FEE_RATE * 100  # % of notional to cover round-trip fees

        # Slippage
        slippage_bps = (fill_price - mid) / mid * 10000 if mid > 0 else 0
        if side == "SELL":
            slippage_bps = -slippage_bps  # normalize: positive = bad

        # SL/TP distance
        sl_dist_pct = abs(fill_price - sl_price) / fill_price * 100 if fill_price > 0 else 0
        tp_dist_pct = abs(tp_price - fill_price) / fill_price * 100 if (fill_price > 0 and tp_price > 0) else 0

        # Update the skeleton with all async-fetched fields
        ctx = self._pending[key]  # skeleton saved earlier
        ctx.trigger_type = trigger
        ctx.cvd_value = cvd_value
        ctx.cvd_z_score = cvd_z
        ctx.cvd_quantile_breach = signal_extra.get("cvd_quantile_rank",
                                   signal_extra.get("cvd_quantile_breach", 0.0))
        ctx.ofi_value = ofi_value
        ctx.signal_strength = strength
        ctx.signal_confidence = signal_extra.get("confidence", 0.0)
        ctx.entry_volatility_24h = volatility_24h
        ctx.entry_volume_24h_usdt = vol_usdt
        ctx.entry_spread_bps = ticker.get("spread_bps", 0)
        ctx.entry_vpin = vpin
        ctx.entry_funding_rate = funding
        ctx.entry_bid = bid
        ctx.entry_ask = ask
        ctx.sl_distance_pct = sl_dist_pct
        ctx.tp_distance_pct = tp_dist_pct
        ctx.slippage_entry_bps = round(slippage_bps, 2)
        ctx.session_utc = session_utc
        ctx.trend_5bar = round(trend_5bar, 6)
        ctx.trend_20bar = round(trend_20bar, 6)
        ctx.bb_position = round(bb_position, 4)
        ctx.rsi_value = round(rsi_value, 2)
        ctx.volatility_regime = volatility_regime
        ctx.btc_ema20_1h = round(btc_ema20_1h, 2)
        ctx.btc_ema50_1h = round(btc_ema50_1h, 2)
        ctx.btc_regime_1h = btc_regime_1h
        ctx.btc_price_vs_ema20 = round(btc_price_vs_ema20, 6)
        ctx.trade_vs_btc_regime = trade_vs_btc_regime
        ctx.fill_type = fill_type
        ctx.fee_estimate_usdt = round(fee_estimate_usdt, 4)
        ctx.breakeven_move_pct = round(breakeven_move_pct, 4)
        # skeleton already in _pending[key], no need to re-assign
        logger.info(f"[TradeLog] Entry enriched: {trade_id}")

        return ctx

    def record_close(
        self,
        strategy: str,
        coin: str,
        exit_price: float,
        exit_reason: str,
        pos,
        exit_atr: float = 0.0,
        exit_spread_bps: float = 0.0,
        exit_funding_rate: float = 0.0,
    ) -> Optional[dict]:
        """Record exit and write complete trade context to JSONL."""
        key = f"{strategy}:{coin}"
        ctx = self._pending.pop(key, None)

        if ctx is None:
            logger.warning(f"[TradeLog] No pending context for {key}")
            return None

        now = datetime.now(timezone.utc)

        # Fill exit fields
        ctx.ts_exit = now.isoformat()
        ctx.exit_price = exit_price
        ctx.exit_reason = exit_reason
        ctx.exit_atr = exit_atr
        ctx.exit_spread_bps = exit_spread_bps
        ctx.exit_funding_rate = exit_funding_rate

        # Outcome
        if ctx.side == "BUY":
            ctx.pnl_pct = (exit_price - ctx.entry_price) / ctx.entry_price
        else:
            ctx.pnl_pct = (ctx.entry_price - exit_price) / ctx.entry_price
        ctx.pnl_usdt = ctx.pnl_pct * ctx.entry_price * ctx.qty

        # Actual fee: entry notional × rate + exit notional × rate
        from src.strategies.base import ROUND_TRIP_FEE_RATE
        entry_notional = ctx.entry_price * ctx.qty
        exit_notional = exit_price * ctx.qty
        ctx.fee_actual_usdt = round((entry_notional + exit_notional) * (ROUND_TRIP_FEE_RATE / 2), 4)
        ctx.pnl_net_usdt = round(ctx.pnl_usdt - ctx.fee_actual_usdt, 4)
        ctx.pnl_net_pct = round(ctx.pnl_net_usdt / entry_notional, 6) if entry_notional > 0 else 0.0
        ctx.fee_drag_pct = round(ctx.fee_actual_usdt / entry_notional * 100, 4) if entry_notional > 0 else 0.0

        # Position data
        if pos is not None:
            ctx.bars_held = getattr(pos, "bars_held", 0)
            ctx.mfe_pct = getattr(pos, "mfe_pct", 0.0)
            ctx.mae_pct = getattr(pos, "mae_pct", 0.0)
            ctx.price_high = getattr(pos, "price_high", 0.0)
            ctx.price_low = getattr(pos, "price_low", 0.0)
            ctx.trail_sl_final = getattr(pos, "sl_price", 0.0)
            ctx.sl_tighten_count = getattr(pos, "sl_tighten_count", 0)

            # mfe_to_trail_ratio: how close the trail was to capturing MFE
            # ratio > 1 means trail was tight enough (MFE > trail distance → should have captured)
            # ratio < 1 means MFE was within trail distance (trail didn't tighten because MFE too small)
            if ctx.trail_distance > 0 and ctx.entry_price > 0:
                trail_pct = ctx.trail_distance / ctx.entry_price
                ctx.mfe_to_trail_ratio = round(abs(ctx.mfe_pct) / trail_pct, 3) if trail_pct > 0 else 0.0

        # exit_atr 기반 변동성 변화율 (진입 대비 청산 시 변동성 확장/축소)
        if exit_atr > 0 and ctx.entry_atr > 0:
            ctx.atr_ratio_exit_entry = round(exit_atr / ctx.entry_atr, 4)

        # Duration
        ctx.hold_duration_sec = time.time() - ctx._entry_unix if ctx._entry_unix > 0 else 0

        # Write to JSONL
        record = asdict(ctx)
        record.pop("_entry_unix", None)  # internal field

        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
            logger.info(
                f"[TradeLog] {ctx.trade_id} closed: "
                f"pnl={ctx.pnl_pct:+.2%} net={ctx.pnl_net_pct:+.4%} "
                f"fee={ctx.fee_actual_usdt:.4f}$ drag={ctx.fee_drag_pct:.3f}% "
                f"reason={exit_reason} bars={ctx.bars_held} "
                f"mfe={ctx.mfe_pct:.2%} mae={ctx.mae_pct:.2%} "
                f"mfe/trail={ctx.mfe_to_trail_ratio:.2f} "
                f"btc={ctx.btc_regime_1h} vs_btc={ctx.trade_vs_btc_regime}"
            )
        except Exception as e:
            logger.error(f"[TradeLog] Write failed: {e}")

        return record

    @property
    def pending_count(self) -> int:
        return len(self._pending)
