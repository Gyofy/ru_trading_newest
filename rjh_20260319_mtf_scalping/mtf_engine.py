"""Multi-Timeframe Trading Engine.

5분봉이 방향을 결정하고, 1분봉이 진입 타이밍을 결정.
레버리지 매매에서 조기 청산을 방지하기 위해:
  - SL은 5분봉 ATR 기준 (1분봉 노이즈 필터)
  - TP는 3단계 분할 (TP1→BEP, TP2→TP1, TP3→전량)
  - 방향 불일치 시 진입하지 않음

Flow:
  1. 5분봉 모델 예측 → 방향(BUY/SELL) + confidence
  2. 방향이 확정되면 entry_window 동안 대기
  3. 1분봉 모델이 같은 방향 시그널 → 진입
  4. SL/TP는 5분봉 ATR 기반으로 설정
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("live_bot.mtf")


@dataclass
class DirectionSignal:
    """5분봉에서 나온 방향 시그널."""
    coin: str
    side: str              # "BUY" or "SELL"
    p_trade: float
    p_direction: float
    confidence: float
    atr_5m: float          # 5분봉 ATR — SL/TP 계산에 사용
    timestamp: str
    valid_until_bar: int   # 이 bar까지 1분봉 진입 허용


@dataclass
class TimingSignal:
    """1분봉에서 나온 타이밍 시그널."""
    coin: str
    side: str
    p_trade: float
    p_direction: float
    confidence: float
    entry_price: float


def compute_mtf_barriers(
    entry_price: float,
    atr_5m: float,
    side: str,
    cfg: dict,
) -> dict:
    """5분봉 ATR 기반으로 SL/TP 계산 (레버리지 안전).

    Returns dict with sl, tp1, tp2, tp3, tp (=tp3).
    """
    # Fixed pct mode (scalping) — overrides ATR if present
    sl_fixed = cfg.get("sl_fixed_pct")
    tp1_fixed = cfg.get("tp1_fixed_pct")
    tp2_fixed = cfg.get("tp2_fixed_pct")

    if sl_fixed and tp1_fixed:
        # Scalping mode: fixed percentage distances
        sl_dist = entry_price * sl_fixed
        tp1_dist = entry_price * tp1_fixed
        tp2_dist = entry_price * (tp2_fixed or tp1_fixed * 2)
        tp3_dist = tp2_dist  # no TP3
    else:
        # ATR mode (original)
        sl_mult = cfg.get("sl_atr_mult", 3.0)
        tp1_mult = cfg.get("tp1_atr_mult", 2.0)
        tp2_mult = cfg.get("tp2_atr_mult", 4.0)
        tp3_mult = cfg.get("tp3_atr_mult", 6.0)
        min_sl_pct = cfg.get("min_sl_pct", 0.012)
        min_tp_pct = cfg.get("min_tp_pct", 0.015)
        sl_dist = max(sl_mult * atr_5m, entry_price * min_sl_pct)
        tp1_dist = max(tp1_mult * atr_5m, entry_price * min_tp_pct)
        tp2_dist = max(tp2_mult * atr_5m, entry_price * min_tp_pct * 2)
        tp3_dist = max(tp3_mult * atr_5m, entry_price * min_tp_pct * 3)

    if side == "BUY":
        return {
            "sl": entry_price - sl_dist,
            "tp1": entry_price + tp1_dist,
            "tp2": entry_price + tp2_dist,
            "tp3": entry_price + tp3_dist,
            "tp": entry_price + tp3_dist,
            "sl_dist_pct": sl_dist / entry_price,
        }
    else:
        return {
            "sl": entry_price + sl_dist,
            "tp1": entry_price - tp1_dist,
            "tp2": entry_price - tp2_dist,
            "tp3": entry_price - tp3_dist,
            "tp": entry_price - tp3_dist,
            "sl_dist_pct": sl_dist / entry_price,
        }


class MTFDirectionManager:
    """5분봉 방향 시그널 관리.

    5분봉 모델이 시그널을 내면 entry_window 동안 유효.
    1분봉이 같은 방향을 확인하면 진입 허용.
    """

    def __init__(self, entry_window_bars: int = 6):
        self.entry_window = entry_window_bars  # 5분봉 bar 수 (6 = 30분)
        self.active_directions: dict[str, DirectionSignal] = {}
        self._current_5m_bar: int = 0

    def update_5m_bar(self, bar_idx: int) -> None:
        """5분봉 bar 인덱스 업데이트. 만료된 시그널 제거."""
        self._current_5m_bar = bar_idx
        expired = [
            coin for coin, sig in self.active_directions.items()
            if bar_idx > sig.valid_until_bar
        ]
        for coin in expired:
            logger.info(f"[MTF] {coin} direction expired (bar {bar_idx})")
            del self.active_directions[coin]

    def set_direction(self, signal: DirectionSignal) -> None:
        """5분봉 방향 시그널 등록."""
        signal.valid_until_bar = self._current_5m_bar + self.entry_window
        self.active_directions[signal.coin] = signal
        logger.info(
            f"[MTF] {signal.coin} direction={signal.side} "
            f"conf={signal.confidence:.3f} atr_5m={signal.atr_5m:.6f} "
            f"valid until bar {signal.valid_until_bar}"
        )

    def check_1m_entry(
        self,
        coin: str,
        timing: TimingSignal,
        min_5m_conf: float = 0.55,
        min_1m_conf: float = 0.45,
    ) -> Optional[dict]:
        """1분봉 시그널이 5분봉 방향과 일치하는지 확인.

        Returns entry params dict or None.
        """
        direction = self.active_directions.get(coin)
        if direction is None:
            return None  # 5분봉 시그널 없음

        # 방향 일치 확인
        if timing.side != direction.side:
            logger.debug(f"[MTF] {coin}: 1m={timing.side} vs 5m={direction.side} MISMATCH")
            return None

        # Confidence 체크
        if direction.confidence < min_5m_conf:
            return None
        if timing.confidence < min_1m_conf:
            return None

        return {
            "coin": coin,
            "side": direction.side,
            "5m_confidence": direction.confidence,
            "1m_confidence": timing.confidence,
            "combined_confidence": direction.confidence * timing.confidence,
            "atr_5m": direction.atr_5m,
            "entry_price": timing.entry_price,
        }

    def get_active_coins(self) -> list[str]:
        return list(self.active_directions.keys())


def estimate_leverage_risk(
    sl_dist_pct: float,
    leverage: int,
    risk_frac: float,
) -> dict:
    """레버리지 매매에서의 실질 리스크 계산.

    Returns:
        margin_risk_pct: 마진 대비 손실 비율
        equity_risk_pct: equity 대비 손실 비율
        safe: True if equity risk < 2%
    """
    # risk_frac = equity의 몇 %를 이 거래에 걸 것인가
    # 실질 마진 = notional / leverage
    # SL 히트 시 손실 = notional * sl_dist_pct
    # equity 대비 손실 = risk_frac (by design — qty = risk / sl_dist)
    equity_risk = risk_frac  # risk_frac 자체가 equity 대비 손실
    margin_risk = sl_dist_pct * leverage  # 마진 대비 얼마나 위험한가

    return {
        "margin_risk_pct": margin_risk,
        "equity_risk_pct": equity_risk,
        "safe": equity_risk < 0.02,  # DD limit 2% 이내
        "liquidation_buffer": 1.0 / leverage - sl_dist_pct,  # 청산가까지 여유
    }
