"""Strategy D: Asymmetric Sniper — extreme signal + fixed-dollar risk.

Philosophy: "틀려도 $1.5, 맞으면 $4.5+"
- Enter ONLY on 3-sigma-equivalent extreme (Q99 percentile)
- Fixed dollar SL (not ATR-based) -> predictable max loss
- Trailing TP with R:R floor of 1:3
- Funding rate filter: only enter when funding pays you
- Counter-trend direction (extreme flow exhaustion)
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.signals.contract import Signal, Action, Regime
from src.strategies.base import StrategyBase, ROUND_TRIP_FEE_RATE

logger = logging.getLogger("strategy.asymmetric_sniper")


class AsymmetricSniper(StrategyBase):
    """Extreme-only counter-trend entries with fixed-dollar risk and 1:3+ R:R."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track cooldown and daily trade count
        self._last_trade_ts: float = 0.0  # unix timestamp of last trade
        self._daily_trade_count: int = 0
        self._daily_trade_date: str = ""  # YYYY-MM-DD

    # ── Core evaluation ────────────────────────────────────

    async def evaluate(self, coins: list[str]) -> list[Signal]:
        """Override: global-state checks (daily cap, cooldown) before parallel dispatch."""
        cfg = self.config.extra
        max_daily = cfg.get("max_daily_trades", 3)

        # Reset daily counter at day boundary
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_trade_date:
            self._daily_trade_count = 0
            self._daily_trade_date = today

        if self._daily_trade_count >= max_daily:
            return []
        if self._check_cooldown():
            return []

        # Parallel evaluation via base.evaluate (which calls _eval_one_coin in gather)
        signals = await super().evaluate(coins)
        # Patience: only 1 signal per cycle
        return signals[:1]

    async def _eval_one_coin(self, coin: str) -> "Signal | None":
        """Per-coin evaluation — called in parallel by base.evaluate()."""
        cfg = self.config.extra
        cvd_quantile = cfg.get("cvd_quantile", 0.99)
        cvd_sigma_mult = cfg.get("cvd_sigma_mult", 3.0)
        roll_window = cfg.get("cvd_roll_window", 480)
        funding_threshold = cfg.get("funding_threshold", 0.0005)

        df = await self.data_hub.get_ohlcv(coin, "1m", limit=roll_window + 50)
        if df is None or len(df) < roll_window:
            return None

        ticker = await self.data_hub.get_ticker(coin)
        if ticker.get("spread_bps", 0) > 5.0:
            return None

        extreme_result = self._check_cvd_extreme(df, cvd_quantile, cvd_sigma_mult, roll_window)
        if extreme_result is None:
            return None

        direction, strength, z_score, cvd_raw = extreme_result

        funding_ok = await self._check_funding(coin, direction, funding_threshold)
        if not funding_ok:
            return None

        self._log.info(
            f"[{coin}] ASYMMETRIC SNIPER -> {direction} | "
            f"z={z_score:.2f} strength={strength:.3f}"
        )
        return Signal(
            symbol=coin,
            horizon_min=120,
            regime=Regime.VOLATILE,
            pred_return=strength,
            p_up=0.3 if direction == "SHORT" else 0.7,
            confidence=min(abs(z_score) / 5.0, 1.0),
            model_name="asymmetric_sniper",
            strategy_name=self.name,
            action=Action.LONG if direction == "LONG" else Action.SHORT,
            size=1.0,
            ttl_bars=240,
            extra={
                "trigger": "cvd_q99_3sigma",
                "strength": strength,
                "z_score": z_score,
                "cvd_value": cvd_raw,
                "cvd_z_score": z_score,
                "ofi_value": 0.0,  # asymmetric_sniper uses CVD only
                "rr_target": cfg.get("rr_minimum", 3.0),
            },
        )

    # ── CVD extreme detection ──────────────────────────────

    def _check_cvd_extreme(
        self,
        df: pd.DataFrame,
        quantile: float,
        sigma_mult: float,
        window: int,
    ) -> tuple[str, float, float, float] | None:
        """Check if CVD delta is at Q99 AND > 3-sigma extreme.

        Returns (direction, strength, z_score, cvd_raw) or None.
        Direction is COUNTER to the extreme (extreme buy -> SHORT).
        """
        # Compute CVD delta (per-bar, not cumulative)
        cvd_series = self.data_hub.compute_cvd(df)
        if len(cvd_series) < window:
            return None

        cvd_delta = cvd_series.diff().dropna()
        if len(cvd_delta) < window:
            return None

        recent = cvd_delta.iloc[-window:]
        current = cvd_delta.iloc[-1]

        # Rolling statistics
        q_high = recent.quantile(quantile)
        q_low = recent.quantile(1 - quantile)
        roll_mean = recent.mean()
        roll_std = recent.std()

        if roll_std == 0 or np.isnan(roll_std):
            return None

        z_score = float((current - roll_mean) / roll_std)

        # Condition 1: Q99 percentile breach
        # Condition 2: |z_score| > sigma_mult (3.0)
        # Both must pass

        if current > q_high and z_score > sigma_mult:
            # Extreme buying -> expect exhaustion -> SHORT
            strength = float((current - q_high) / (q_high if q_high != 0 else 1))
            return ("SHORT", -strength, z_score, float(current))

        if current < q_low and z_score < -sigma_mult:
            # Extreme selling -> expect exhaustion -> LONG
            strength = float((q_low - current) / (abs(q_low) if q_low != 0 else 1))
            return ("LONG", strength, z_score, float(current))

        return None

    # ── Barriers: fixed-dollar SL, trailing TP ─────────────

    def compute_barriers(
        self, signal: Signal, atr: float, price: float, extra: dict | None = None
    ) -> tuple[float, float]:
        """Fixed-dollar SL, trailing stop TP (tp_price=0).

        SL distance = risk_usdt / notional * price
        For $32 alloc x 5x = $160 notional, $1.5 risk = 0.94% SL distance.
        """
        cfg = extra if extra is not None else self.config.extra
        risk_usdt = cfg.get("risk_per_trade_usdt", 1.5)
        max_notional = self.config.allocation_usdt * self.config.leverage

        if max_notional <= 0:
            return (price * 0.99, 0.0)  # fallback: 1% SL

        # SL as percentage of price
        sl_pct = risk_usdt / max_notional  # e.g. 1.5 / 160 = 0.009375
        sl_dist = price * sl_pct

        # Enforce minimum SL distance = round-trip fee threshold (≈0.19%)
        min_sl_dist = price * ROUND_TRIP_FEE_RATE
        sl_dist = max(sl_dist, min_sl_dist)

        if signal.action == Action.SHORT:
            sl_price = price + sl_dist
        else:
            sl_price = price - sl_dist

        # tp_price=0 signals trailing stop mode
        return (sl_price, 0.0)

    # ── Cooldown check ─────────────────────────────────────

    def _check_cooldown(self) -> bool:
        """Return True if we are in cooldown (should NOT trade).

        After any trade, wait cooldown_bars minutes.
        """
        if self._last_trade_ts == 0.0:
            return False

        cooldown_bars = self.config.extra.get("cooldown_bars", 30)
        cooldown_secs = cooldown_bars * 60  # 1 bar = 1 minute
        elapsed = time.time() - self._last_trade_ts

        if elapsed < cooldown_secs:
            remaining = cooldown_secs - elapsed
            self._log.debug(
                f"Cooldown active: {remaining:.0f}s remaining "
                f"({cooldown_bars} bars)"
            )
            return True
        return False

    # ── Funding rate filter ────────────────────────────────

    async def _check_funding(
        self, coin: str, direction: str, threshold: float
    ) -> bool:
        """Only enter when funding rate pays us.

        SHORT: funding > +threshold (shorts get paid)
        LONG:  funding < -threshold (longs get paid)
        """
        # funding_filter_enabled=false → skip entire funding check (useful for testnet)
        if not self.config.extra.get("funding_filter_enabled", True):
            return True

        funding = await self.data_hub.get_funding_rate(coin)
        if funding is None:
            # funding_skip_on_null=False → allow when rate unavailable (testnet)
            if not self.config.extra.get("funding_skip_on_null", True):
                self._log.debug(f"[{coin}] funding unavailable, allowed")
                return True
            self._log.debug(f"[{coin}] funding rate unavailable, skip")
            return False

        if direction == "SHORT" and funding > threshold:
            return True
        if direction == "LONG" and funding < -threshold:
            return True

        return False

    # ── Override _try_execute for fixed-dollar risk sizing ──
    # The base class sizes by 10% of allocation per trade.
    # We override to use fixed risk_usdt from config + track cooldown.

    async def _try_execute(self, signal: Signal) -> dict | None:
        """Execute with fixed-dollar risk sizing instead of % allocation."""
        coin = signal.symbol
        side = "BUY" if signal.action == Action.LONG else "SELL"

        # Fetch current price + ATR
        ticker = await self.data_hub.get_ticker(coin)
        price = ticker["last"]
        df = await self.data_hub.get_ohlcv(coin, "1m", limit=100)
        atr = self._compute_atr(df, period=14)

        if atr <= 0:
            self._log.warning(f"[{coin}] ATR=0, skip")
            return None

        # Apply per-coin adaptive params — pass as argument, never mutate self.config.extra.
        effective_extra = (
            self.coin_profiles.get_params(coin, self.config.extra)
            if self.coin_profiles else self.config.extra
        )
        cfg = effective_extra

        sl_price, tp_price = self.compute_barriers(signal, atr, price, extra=effective_extra)
        sl_dist = abs(price - sl_price)

        min_sl_dist = price * ROUND_TRIP_FEE_RATE
        if sl_dist < min_sl_dist:
            self._log.warning(
                f"[{coin}] SL too tight: {sl_dist:.6f} < fee_threshold {min_sl_dist:.6f} "
                f"({ROUND_TRIP_FEE_RATE:.3%}) — skip"
            )
            return None

        # Dynamic position sizing
        if self.position_sizer:
            qty_raw, notional, sizing_info = self.position_sizer.compute_qty(
                allocation_usdt=self.config.allocation_usdt,
                leverage=self.config.leverage,
                price=price,
                sl_dist=sl_dist,
                atr=atr,
                spread_bps=ticker.get("spread_bps", 0),
                signal_confidence=signal.confidence,
            )
            if qty_raw <= 0:
                self._log.info(f"[{coin}] PositionSizer skip: {sizing_info}")
                return None
        else:
            # Fallback: fixed-dollar risk
            risk_usdt = cfg.get("risk_per_trade_usdt", 1.5)
            qty_raw = risk_usdt / sl_dist
            notional = qty_raw * price

        # Cap by remaining allocation
        max_notional = self.config.allocation_usdt * self.config.leverage
        used = self.pos_manager.strategy_notional(self.name)
        available = max_notional - used
        if notional > available:
            notional = available
            qty_raw = notional / price

        # Per-position cap: single position cannot exceed 1/max_positions of total budget
        max_pos = max(1, self.config.max_positions)
        per_pos_cap = max_notional / max_pos
        if notional > per_pos_cap:
            notional = per_pos_cap
            qty_raw = notional / price

        if notional < 5.0:
            self._log.info(f"[{coin}] notional {notional:.2f} < $5, skip")
            return None

        # R:R floor check (disable with rr_check_enabled=false for testnet)
        # actual_rr always computed for logging; check only enforced when enabled
        potential_reward = atr * cfg.get("trailing_atr_mult_initial", 2.0)
        actual_rr = potential_reward / sl_dist if sl_dist > 0 else 0.0
        if cfg.get("rr_check_enabled", True):
            rr_min = cfg.get("rr_minimum", 3.0)
            if actual_rr < rr_min * 0.75:
                # R:R must reach at least 75% of target (e.g. 2.25 for rr_min=3.0)
                self._log.info(
                    f"[{coin}] R:R too low: {actual_rr:.1f} "
                    f"(need {rr_min:.1f}), skip"
                )
                return None

        # Fetch funding for portfolio gate
        try:
            funding_rate = await self.data_hub.get_funding_rate(coin)
            if funding_rate is None:
                funding_rate = 0.0
        except Exception:
            funding_rate = 0.0

        # Portfolio-level approval
        async with self._portfolio_lock:
            approved, reason = self.portfolio_risk.approve_entry(
                strategy_name=self.name,
                coin=coin,
                side=side,
                notional=notional,
                funding_rate=funding_rate,
            )
            if not approved:
                self._log.info(f"[{coin}] rejected: {reason}")
                return None

            qty = self.exchange.round_qty(coin, qty_raw)
            if qty <= 0:
                return None

            _is_new_coin = coin not in self._leverage_initialized
            try:
                result = await self._place_entry(coin, side, qty, price)
            except Exception as e:
                self._log.error(f"[{coin}] order failed: {e}")
                return None

            if not result or not result.get("success"):
                return None

            fill_price = result.get("fill_price", price)

            # Record entry to OrderLedger
            if self.ledger:
                try:
                    _order_id = result.get("order_id", "")
                    self.ledger.insert_order(
                        order_link_id=_order_id,
                        symbol=coin,
                        side=side,
                        order_type="limit" if not self.config.paper_mode else "paper",
                        qty=qty,
                        price=fill_price,
                        purpose="entry",
                    )
                    self.ledger.insert_fill(
                        order_link_id=_order_id,
                        fill_price=fill_price,
                        fill_qty=result.get("fill_qty", qty),
                        fee=result.get("fee", 0),
                    )
                except Exception as e:
                    self._log.debug(f"[{coin}] Ledger write failed: {e}")

            # Re-compute barriers from fill price using per-coin params
            sl_price, tp_price = self.compute_barriers(signal, atr, fill_price, extra=effective_extra)

            # Trailing stop config
            trail_initial = cfg.get("trailing_atr_mult_initial", 2.0)
            trail_dist = atr * trail_initial

            # Compute initial TP for exchange display (RR = rr_minimum, default 3.0)
            # Fee offset ensures net RR = rr_minimum after paying round-trip fees
            sl_dist = abs(fill_price - sl_price)
            rr = cfg.get("rr_minimum", 3.0)
            fee_offset = fill_price * ROUND_TRIP_FEE_RATE
            if side == "BUY":
                tp_price = fill_price + sl_dist * rr + fee_offset
            else:
                tp_price = fill_price - sl_dist * rr - fee_offset

            from src.execution.position_store import OpenPosition
            pos = OpenPosition(
                coin=coin,
                side=side,
                entry_price=fill_price,
                qty=qty,
                sl_price=sl_price,
                tp_price=tp_price if tp_price > 0 else sl_price,
                entry_time=signal.ts.isoformat(),
                ttl_bars=signal.ttl_bars or 120,
                strategy_tag=self.name,
                trailing_sl=True,
                trail_distance=trail_dist,
            )
            self.pos_manager.add_position(self.name, pos)
            self._trade_count += 1

            # ── Register exchange-side SL/TP — every position MUST have both ──
            # closePosition=True (FOR POSITION) requires position to exist in Binance DB first.
            # Add 1s initial wait + 5-retry with 1s delay.
            if not self.config.paper_mode:
                sl_side = "SELL" if side == "BUY" else "BUY"
                rounded_sl = self.exchange.round_price(coin, sl_price)
                rounded_tp = self.exchange.round_price(coin, tp_price) if tp_price > 0 else 0

                # 새 종목(margin mode 변경): 3s, 기존 종목: 1s
                _sl_init_wait = 3.0 if _is_new_coin else 1.0
                if _is_new_coin:
                    self._log.info(
                        f"[{coin}] 신규 종목 — SL/TP 등록 전 {_sl_init_wait}s 대기 "
                        f"(margin mode 전파 대기)"
                    )
                await asyncio.sleep(_sl_init_wait)

                # SL retry loop
                sl_result = {"success": False}
                for attempt in range(5):
                    if attempt > 0:
                        await asyncio.sleep(1.0)
                    sl_oid = self.exchange.make_order_id(
                        coin, sl_side, self._trade_count, prefix=f"v8sl{attempt}",
                    )
                    sl_result = await self.exchange.place_protective_stop(
                        symbol=coin, side=sl_side, qty=qty,
                        stop_price=rounded_sl, order_link_id=sl_oid,
                    )
                    if sl_result.get("success"):
                        pos.sl_order_id = sl_oid
                        pos.sl_exchange_id = sl_result.get("exchange_order_id", "")
                        self._log.info(
                            f"[{coin}] Exchange SL registered @ {sl_price:.4f}"
                            + (f" (retry {attempt})" if attempt > 0 else "")
                        )
                        break

                if not sl_result.get("success"):
                    self._log.error(
                        f"[SL_FAIL_CRITICAL] {coin} SL 5회 실패 → 강제청산 (err: {sl_result.get('error')})"
                    )
                    close_side = "SELL" if side == "BUY" else "BUY"
                    close_ok = False
                    try:
                        await self.exchange.market_close(
                            coin, close_side, qty,
                            order_link_id=self.exchange.make_order_id(coin, close_side, prefix="v8emg"),
                        )
                        close_ok = True
                    except Exception as ce:
                        self._log.error(
                            f"[SL_FAIL_CRITICAL] {coin} 강제청산 실패: {ce} "
                            f"— NAKED POSITION 위험! 수동 개입 필요"
                        )
                    if close_ok:
                        self.pos_manager.remove_position(self.name, coin)
                    return None

                # TP retry loop
                if rounded_tp > 0:
                    tp_result = {"success": False}
                    for attempt in range(5):
                        if attempt > 0:
                            await asyncio.sleep(1.0)
                        tp_oid = self.exchange.make_order_id(
                            coin, sl_side, self._trade_count, prefix=f"v8tp{attempt}",
                        )
                        tp_result = await self.exchange.place_take_profit(
                            symbol=coin, side=sl_side, qty=qty,
                            tp_price=rounded_tp, order_link_id=tp_oid,
                        )
                        if tp_result.get("success"):
                            pos.tp_order_id = tp_oid
                            pos.tp_exchange_id = tp_result.get("exchange_order_id", "")
                            self._log.info(
                                f"[{coin}] Exchange TP registered @ {tp_price:.4f}"
                                + (f" (retry {attempt})" if attempt > 0 else "")
                            )
                            break

                    if not tp_result.get("success"):
                        self._log.error(
                            f"[TP_FAIL_CRITICAL] {coin} TP 3회 실패 → 강제청산 (err: {tp_result.get('error')})"
                        )
                        try:
                            if pos.sl_exchange_id:
                                await self.exchange.cancel_order(
                                    coin, exchange_order_id=pos.sl_exchange_id,
                                    order_link_id=pos.sl_order_id,
                                )
                        except Exception:
                            pass
                        close_side = "SELL" if side == "BUY" else "BUY"
                        try:
                            await self.exchange.market_close(
                                coin, close_side, qty,
                                order_link_id=self.exchange.make_order_id(coin, close_side, prefix="v8emg"),
                            )
                        except Exception as ce:
                            self._log.error(
                                f"[TP_FAIL_CRITICAL] {coin} 강제청산 실패: {ce} "
                                f"— NAKED POSITION 위험! 수동 개입 필요"
                            )
                            # Keep in tracker — monitor continues watching
                            return None
                        self.pos_manager.remove_position(self.name, coin)
                        return None

            self.pos_manager._save()

            # ── Trade context logging ──
            actual_risk = sl_dist * qty
            if self.trade_logger:
                try:
                    await self.trade_logger.capture_entry_context(
                        data_hub=self.data_hub,
                        coin=coin,
                        strategy_name=self.name,
                        side=side,
                        signal_extra=signal.extra,
                        fill_price=fill_price,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        qty=qty,
                        notional=notional,
                        leverage=self.config.leverage,
                        atr=atr,
                        trailing_sl=True,
                        trail_distance=trail_dist,
                        risk_usdt=actual_risk,
                        rr_estimate=actual_rr,
                        strategy_params=self.config.extra,
                        paper_mode=self.config.paper_mode,
                    )
                except Exception as e:
                    self._log.warning(f"[{coin}] Trade context capture failed: {e}")

            # Track cooldown and daily count
            self._last_trade_ts = time.time()
            self._daily_trade_count += 1

            fee_usdt = notional * ROUND_TRIP_FEE_RATE
            sl_pct = sl_dist / fill_price * 100
            tp_pct = abs(tp_price - fill_price) / fill_price * 100 if tp_price > 0 else 0
            self._log.info(
                f"[{coin}] ENTRY {side} @ {fill_price:.4f} | "
                f"qty={qty} | SL={sl_price:.4f}(-{sl_pct:.2f}%) | TP={tp_price:.4f}(+{tp_pct:.2f}%) | "
                f"risk=${actual_risk:.2f} | notional=${notional:.2f} | "
                f"fee≈${fee_usdt:.3f}({ROUND_TRIP_FEE_RATE:.3%}) | "
                f"trade #{self._daily_trade_count} today"
            )

            return {
                "strategy": self.name,
                "coin": coin,
                "side": side,
                "price": fill_price,
                "qty": qty,
                "sl": sl_price,
                "tp": tp_price,
                "trailing": True,
                "risk_usdt": actual_risk,
                "rr_estimate": actual_rr,
            }
