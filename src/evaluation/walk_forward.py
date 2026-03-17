"""Walk-Forward Evaluation — purged rolling split + fee-aware 검증.

Signal 리스트와 실제 가격 데이터를 받아서:
1. 시간순 purged rolling window로 분할
2. 각 window에서 signal → forward return 계산
3. 비용(수수료+슬리피지+스프레드) 차감
4. window별/전체 성과 지표 산출

Usage:
    engine = WalkForwardEngine(cost_bps=20)
    results = engine.evaluate(signals, price_data)
    print(results.summary)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from src.signals.contract import Signal, Action, Regime


@dataclass
class TradeResult:
    """하나의 시그널에 대한 실현 결과."""
    signal: Signal
    entry_price: float = 0.0
    exit_price: float = 0.0
    forward_return_pct: float = 0.0     # 비용 전 수익률
    net_return_pct: float = 0.0         # 비용 후 수익률
    hit_tp: bool = False
    hit_sl: bool = False
    expired: bool = False               # TTL 만료
    hold_bars: int = 0
    max_favorable: float = 0.0          # 최대 유리 움직임 (%)
    max_adverse: float = 0.0            # 최대 불리 움직임 (%)

    @property
    def is_win(self) -> bool:
        return self.net_return_pct > 0


@dataclass
class WindowResult:
    """하나의 walk-forward window 결과."""
    window_id: int
    start: datetime
    end: datetime
    trades: list[TradeResult] = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def n_wins(self) -> int:
        return sum(1 for t in self.trades if t.is_win)

    @property
    def win_rate(self) -> float:
        return self.n_wins / self.n_trades if self.n_trades else 0

    @property
    def total_return(self) -> float:
        return sum(t.net_return_pct for t in self.trades)

    @property
    def avg_return(self) -> float:
        return self.total_return / self.n_trades if self.n_trades else 0

    @property
    def expectancy(self) -> float:
        """평균 순수익 (비용 후, 거래당)."""
        return self.avg_return


@dataclass
class WalkForwardResult:
    """전체 walk-forward 평가 결과."""
    windows: list[WindowResult] = field(default_factory=list)
    total_signals: int = 0
    actionable_signals: int = 0

    @property
    def all_trades(self) -> list[TradeResult]:
        return [t for w in self.windows for t in w.trades]

    def summary(self) -> dict:
        trades = self.all_trades
        n = len(trades)
        if n == 0:
            return {"error": "no trades", "total_signals": self.total_signals}

        net_rets = [t.net_return_pct for t in trades]
        wins = [t for t in trades if t.is_win]
        losses = [t for t in trades if not t.is_win]

        # Drawdown
        cumulative = np.cumsum(net_rets)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        max_dd = float(np.min(drawdowns)) if len(drawdowns) else 0

        # Per-regime breakdown
        regime_pnl = {}
        for t in trades:
            r = t.signal.regime.value
            if r not in regime_pnl:
                regime_pnl[r] = []
            regime_pnl[r].append(t.net_return_pct)
        regime_summary = {
            r: {"n": len(v), "avg_ret": np.mean(v), "total_ret": sum(v)}
            for r, v in regime_pnl.items()
        }

        # Per-coin breakdown
        coin_pnl = {}
        for t in trades:
            c = t.signal.symbol
            if c not in coin_pnl:
                coin_pnl[c] = []
            coin_pnl[c].append(t.net_return_pct)
        coin_summary = {
            c: {"n": len(v), "avg_ret": np.mean(v), "total_ret": sum(v)}
            for c, v in coin_pnl.items()
        }

        # Per-window trend
        window_returns = [w.total_return for w in self.windows if w.n_trades > 0]

        avg_win = np.mean([t.net_return_pct for t in wins]) if wins else 0
        avg_loss = np.mean([t.net_return_pct for t in losses]) if losses else 0
        profit_factor = (
            abs(sum(t.net_return_pct for t in wins) /
                sum(t.net_return_pct for t in losses))
            if losses and sum(t.net_return_pct for t in losses) != 0
            else float("inf")
        )

        return {
            "total_signals": self.total_signals,
            "actionable_signals": self.actionable_signals,
            "coverage": self.actionable_signals / self.total_signals if self.total_signals else 0,
            "n_trades": n,
            "n_windows": len(self.windows),
            "win_rate": len(wins) / n,
            "total_return_pct": sum(net_rets),
            "avg_return_pct": np.mean(net_rets),
            "median_return_pct": float(np.median(net_rets)),
            "expectancy_per_trade": np.mean(net_rets),
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_drawdown_pct": max_dd,
            "max_favorable_avg": np.mean([t.max_favorable for t in trades]),
            "max_adverse_avg": np.mean([t.max_adverse for t in trades]),
            "avg_hold_bars": np.mean([t.hold_bars for t in trades]),
            "tp_hit_rate": sum(1 for t in trades if t.hit_tp) / n,
            "sl_hit_rate": sum(1 for t in trades if t.hit_sl) / n,
            "expiry_rate": sum(1 for t in trades if t.expired) / n,
            "turnover_per_window": n / len(self.windows) if self.windows else 0,
            "regime_pnl": regime_summary,
            "coin_pnl": coin_summary,
            "window_returns": window_returns,
        }


class WalkForwardEngine:
    """Purged rolling walk-forward 평가 엔진."""

    def __init__(
        self,
        cost_bps: float = 20.0,
        window_hours: int = 24,
        step_hours: int = 6,
        purge_hours: int = 2,
        bar_minutes: int = 5,
    ):
        self.cost_bps = cost_bps
        self.window_hours = window_hours
        self.step_hours = step_hours
        self.purge_hours = purge_hours
        self.bar_minutes = bar_minutes
        self.round_trip_cost = cost_bps * 2 / 10000  # 양방향 비용 (비율)

    @classmethod
    def from_config(cls) -> "WalkForwardEngine":
        """config/settings.yaml에서 execution tier 파라미터를 로드하여 생성."""
        from src.utils.config import get_execution
        cfg = get_execution()
        wf = cfg.get("walk_forward", {})
        return cls(
            cost_bps=cfg.get("cost_bps", 20.0),
            window_hours=wf.get("window_hours", 24),
            step_hours=wf.get("step_hours", 6),
            purge_hours=wf.get("purge_hours", 2),
            bar_minutes=cfg.get("bar_minutes", 5),
        )

    def evaluate(
        self,
        signals: list[Signal],
        price_data: dict[str, pd.DataFrame],
    ) -> WalkForwardResult:
        """시그널 리스트 + OHLCV → walk-forward 결과.

        Args:
            signals: policy 적용 완료된 Signal 리스트 (시간순)
            price_data: {symbol: DataFrame(OHLCV, DatetimeIndex)}
        """
        if not signals:
            return WalkForwardResult()

        actionable = [s for s in signals if s.is_actionable]
        result = WalkForwardResult(
            total_signals=len(signals),
            actionable_signals=len(actionable),
        )

        if not actionable:
            return result

        # 시간 범위 결정
        ts_list = sorted(s.ts for s in actionable)
        t_start = ts_list[0]
        t_end = ts_list[-1]

        # Rolling windows 생성
        windows = self._make_windows(t_start, t_end)

        for wid, (w_start, w_end) in enumerate(windows):
            purge_end = w_start + timedelta(hours=self.purge_hours)
            w_signals = [
                s for s in actionable
                if purge_end <= s.ts < w_end
            ]

            trades = []
            for sig in w_signals:
                tr = self._simulate_trade(sig, price_data)
                if tr is not None:
                    trades.append(tr)

            result.windows.append(WindowResult(
                window_id=wid, start=w_start, end=w_end, trades=trades,
            ))

        return result

    def _make_windows(
        self, t_start: datetime, t_end: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """Rolling window 리스트 생성."""
        windows = []
        current = t_start
        w_delta = timedelta(hours=self.window_hours)
        s_delta = timedelta(hours=self.step_hours)

        while current + w_delta <= t_end + s_delta:
            windows.append((current, current + w_delta))
            current += s_delta

        if not windows:
            windows.append((t_start, t_end))

        return windows

    def _simulate_trade(
        self, signal: Signal, price_data: dict[str, pd.DataFrame],
    ) -> Optional[TradeResult]:
        """하나의 signal에 대해 forward return 시뮬레이션."""
        sym = signal.symbol
        if sym not in price_data:
            return None

        df = price_data[sym]
        if df.empty:
            return None

        # signal.ts 이후 가장 가까운 bar 찾기
        sig_ts = signal.ts
        if sig_ts.tzinfo is not None:
            sig_ts = sig_ts.replace(tzinfo=None)

        future_idx = df.index[df.index >= sig_ts]
        if len(future_idx) < 2:
            return None

        entry_idx = future_idx[0]
        entry_price = float(df.loc[entry_idx, "Close"])
        if entry_price <= 0:
            return None

        # horizon 기간의 bars
        horizon_bars = max(1, signal.horizon_min // self.bar_minutes)
        ttl = signal.ttl_bars if signal.ttl_bars > 0 else horizon_bars
        max_bars = min(ttl, len(future_idx) - 1)

        if max_bars < 1:
            return None

        # Forward simulation
        is_long = signal.action == Action.LONG
        hit_tp = False
        hit_sl = False
        exit_idx = 0
        exit_price = entry_price
        max_favorable = 0.0
        max_adverse = 0.0

        for i in range(1, max_bars + 1):
            bar_ts = future_idx[i]
            high = float(df.loc[bar_ts, "High"])
            low = float(df.loc[bar_ts, "Low"])
            close = float(df.loc[bar_ts, "Close"])

            if is_long:
                fav = (high / entry_price - 1) * 100
                adv = (low / entry_price - 1) * 100
            else:
                fav = (1 - low / entry_price) * 100
                adv = (1 - high / entry_price) * 100

            max_favorable = max(max_favorable, fav)
            max_adverse = min(max_adverse, adv)

            # TP/SL 체크
            if signal.take_profit > 0 and fav >= signal.take_profit:
                hit_tp = True
                exit_price = entry_price * (1 + signal.take_profit / 100) if is_long \
                    else entry_price * (1 - signal.take_profit / 100)
                exit_idx = i
                break

            if signal.stop_loss > 0 and adv <= -signal.stop_loss:
                hit_sl = True
                exit_price = entry_price * (1 - signal.stop_loss / 100) if is_long \
                    else entry_price * (1 + signal.stop_loss / 100)
                exit_idx = i
                break

            exit_price = close
            exit_idx = i

        # 수익률 계산
        if is_long:
            raw_ret = (exit_price / entry_price - 1) * 100
        else:
            raw_ret = (1 - exit_price / entry_price) * 100

        net_ret = raw_ret - (self.cost_bps * 2 / 100)  # 양방향 비용

        return TradeResult(
            signal=signal,
            entry_price=entry_price,
            exit_price=exit_price,
            forward_return_pct=raw_ret,
            net_return_pct=net_ret,
            hit_tp=hit_tp,
            hit_sl=hit_sl,
            expired=not hit_tp and not hit_sl,
            hold_bars=exit_idx,
            max_favorable=max_favorable,
            max_adverse=max_adverse,
        )
