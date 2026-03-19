"""Sniper Entry — 4h 방향 기반 단기 정밀 진입.

4h 모델이 방향을 결정하면, 1분/5분봉에서 pullback을 감지해
더 좋은 가격에 진입하고 빠르게 수익을 확보.

ML 모델 없이 가격 구조만으로 진입 타이밍 결정:
  BUY 시그널: RSI 과매도 + EMA 지지 근처 pullback
  SELL 시그널: RSI 과매수 + EMA 저항 근처 pullback

SL: 진입 지점 local extreme (4h ATR보다 타이트)
TP: 4h ATR × 1.0 (빠르게 도달 가능한 수준)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("live_bot.sniper")


@dataclass
class SniperSetup:
    """스나이퍼 진입 조건 충족 시 생성."""
    coin: str
    side: str              # 4h 모델 방향
    entry_price: float     # pullback 진입가
    sl_price: float        # local extreme 기반 SL
    tp_price: float        # 4h ATR × 1.0 기반 TP
    sl_pct: float          # SL 거리 (%)
    tp_pct: float          # TP 거리 (%)
    rr: float              # Risk:Reward
    trigger: str           # "rsi_pullback", "ema_bounce", "vwap_touch"


def detect_pullback_entry(
    df_short: pd.DataFrame,
    direction: str,
    atr_4h: float,
    entry_price_4h: float,
    max_sl_pct: float = 0.008,
    min_rr: float = 1.2,
) -> Optional[SniperSetup]:
    """단기 차트에서 pullback 진입 조건 감지.

    Parameters
    ----------
    df_short : pd.DataFrame
        1분 or 5분봉 OHLCV (최소 30 bars)
    direction : str
        "BUY" or "SELL" (4h 모델 방향)
    atr_4h : float
        4h ATR (TP 기준)
    entry_price_4h : float
        4h bar 종가 (기준가)
    max_sl_pct : float
        SL 최대 거리 (0.5% default)
    min_rr : float
        최소 R:R (1.5 default)
    """
    if len(df_short) < 20:
        return None

    close = df_short["close"].values
    high = df_short["high"].values
    low = df_short["low"].values
    current = close[-1]

    # RSI (14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_now = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

    # EMA 빠른/느린
    ema_fast = pd.Series(close).ewm(span=8, adjust=False).mean().iloc[-1]
    ema_slow = pd.Series(close).ewm(span=21, adjust=False).mean().iloc[-1]

    # Local extreme (최근 10 bars)
    recent_low = np.min(low[-10:])
    recent_high = np.max(high[-10:])

    # TP = 4h ATR × 0.5 (현실적으로 빠르게 도달 가능한 수준)
    tp_dist = atr_4h * 0.5

    trigger = None
    entry = current
    sl = 0.0

    if direction == "BUY":
        # BUY pullback 조건:
        # 1) RSI < 40 (과매도 근처)
        # 2) 가격이 EMA slow 근처 또는 아래
        # 3) 가격이 4h 종가보다 아래 (pullback 확인)
        if rsi_now < 40 and current <= ema_slow * 1.003:
            trigger = "rsi_pullback"
            sl = recent_low * 0.997  # local low 아래 0.3% 여유
        elif current <= ema_slow and current > ema_slow * 0.995:
            trigger = "ema_bounce"
            sl = ema_slow * 0.993
        else:
            return None

        tp = entry + tp_dist
        sl_pct = (entry - sl) / entry
        tp_pct = (tp - entry) / entry

    else:  # SELL
        if rsi_now > 60 and current >= ema_slow * 0.997:
            trigger = "rsi_pullback"
            sl = recent_high * 1.003  # local high 위 0.3% 여유
        elif current >= ema_slow and current < ema_slow * 1.005:
            trigger = "ema_bounce"
            sl = ema_slow * 1.007
        else:
            return None

        tp = entry - tp_dist
        sl_pct = (sl - entry) / entry
        tp_pct = (entry - tp) / entry

    # SL 거리 체크
    if sl_pct > max_sl_pct:
        return None  # SL 너무 넓음
    if sl_pct < 0.001:
        return None  # SL 너무 타이트 (노이즈)

    rr = tp_pct / (sl_pct + 1e-10)
    if rr < min_rr:
        return None  # R:R 부족

    return SniperSetup(
        coin="", side=direction,
        entry_price=entry, sl_price=sl, tp_price=tp,
        sl_pct=sl_pct, tp_pct=tp_pct, rr=rr,
        trigger=trigger,
    )


def simulate_sniper_trade(
    df_after: pd.DataFrame,
    setup: SniperSetup,
    max_bars: int = 60,
    cost_pct: float = 0.0015,
) -> dict:
    """스나이퍼 진입 후 결과 시뮬레이션."""
    entry = setup.entry_price
    sl = setup.sl_price
    tp = setup.tp_price

    for i in range(min(len(df_after), max_bars)):
        bar = df_after.iloc[i]

        if setup.side == "BUY":
            if bar["low"] <= sl:
                pnl = (sl - entry) / entry - cost_pct
                return {"pnl": pnl, "exit": "SL", "bars": i + 1}
            if bar["high"] >= tp:
                pnl = (tp - entry) / entry - cost_pct
                return {"pnl": pnl, "exit": "TP", "bars": i + 1}
        else:
            if bar["high"] >= sl:
                pnl = (entry - sl) / entry - cost_pct
                return {"pnl": pnl, "exit": "SL", "bars": i + 1}
            if bar["low"] <= tp:
                pnl = (entry - tp) / entry - cost_pct
                return {"pnl": pnl, "exit": "TP", "bars": i + 1}

    # TTL
    final = df_after["close"].iloc[min(len(df_after) - 1, max_bars - 1)]
    if setup.side == "BUY":
        pnl = (final - entry) / entry - cost_pct
    else:
        pnl = (entry - final) / entry - cost_pct
    return {"pnl": pnl, "exit": "TTL", "bars": max_bars}
