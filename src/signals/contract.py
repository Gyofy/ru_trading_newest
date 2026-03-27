"""Signal Contract — 모든 모델 출력을 통일하는 데이터 규약.

모델은 prediction을 내놓고, policy가 action을 결정한다.
이 모듈은 그 둘의 경계를 정의한다.

Usage:
    signal = Signal.from_model(
        symbol="BTC", horizon_min=30, regime="TREND",
        pred_return=0.45, p_up=0.72, model_name="hot_scanner_v1",
    )
    # policy layer fills: action, size, tp, sl, ttl, cost
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class Action(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class Regime(str, Enum):
    TREND = "TREND"
    RANGE = "RANGE"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"


@dataclass
class Signal:
    """하나의 매매 시그널. 모델 출력 + 정책 결정을 모두 담는다."""

    # ── 식별 ──
    signal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: str = ""
    horizon_min: int = 30  # 예측 지평 (분)

    # ── 시장 상태 ──
    regime: Regime = Regime.UNKNOWN

    # ── 모델 출력 (prediction) ──
    pred_return: float = 0.0      # 예상 수익률 (%, e.g. +0.45)
    p_up: float = 0.5             # 상승 확률 (0~1)
    confidence: float = 0.0       # 모델 확신도 (0~1)
    model_name: str = ""          # 출처 모델

    # ── 정책 결정 (policy가 채움) ──
    action: Action = Action.FLAT
    size: float = 0.0             # 포지션 비중 (0~1, 계좌 대비)
    take_profit: float = 0.0      # TP (%, e.g. +2.0)
    stop_loss: float = 0.0        # SL (%, e.g. -1.0)
    ttl_bars: int = 0             # 유효 기간 (바 수, 0=무제한)
    cost_bps: float = 20.0        # 편도 비용 (bps), 기본 20bps = 0.2%

    # ── 메타 ──
    strategy_name: str = ""       # which strategy generated this signal
    extra: dict = field(default_factory=dict)

    # ── 파생 속성 ──
    @property
    def edge(self) -> float:
        """Expected edge = |pred_return| - round-trip cost."""
        return abs(self.pred_return) - (self.cost_bps * 2 / 100)

    @property
    def is_actionable(self) -> bool:
        return self.action != Action.FLAT and self.size > 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ts"] = self.ts.isoformat()
        d["action"] = self.action.value
        d["regime"] = self.regime.value
        d["edge"] = round(self.edge, 4)
        d["is_actionable"] = self.is_actionable
        return d

    @classmethod
    def from_model(
        cls,
        symbol: str,
        horizon_min: int,
        regime: str | Regime,
        pred_return: float,
        p_up: float,
        model_name: str,
        confidence: float | None = None,
        cost_bps: float = 20.0,
        **extra,
    ) -> Signal:
        """모델 출력으로부터 Signal 생성 (action은 아직 FLAT)."""
        if isinstance(regime, str):
            regime = Regime(regime) if regime in Regime.__members__ else Regime.UNKNOWN
        if confidence is None:
            confidence = abs(p_up - 0.5) * 2  # 0.5→0, 1.0→1
        return cls(
            symbol=symbol,
            horizon_min=horizon_min,
            regime=regime,
            pred_return=pred_return,
            p_up=p_up,
            confidence=confidence,
            model_name=model_name,
            cost_bps=cost_bps,
            extra=extra,
        )

    @classmethod
    def from_hot_coin(cls, hc, regime: str | Regime = Regime.UNKNOWN) -> Signal:
        """HotCoin → Signal 변환."""
        p_up = 0.5
        pred_return = 0.0
        if hc.direction == "LONG":
            p_up = 0.5 + hc.direction_confidence * 0.5
            pred_return = hc.expected_range_pct
        elif hc.direction == "SHORT":
            p_up = 0.5 - hc.direction_confidence * 0.5
            pred_return = -hc.expected_range_pct

        return cls.from_model(
            symbol=hc.coin,
            horizon_min=60,  # hot scan은 ~1h 지평
            regime=regime,
            pred_return=pred_return,
            p_up=p_up,
            model_name="hot_scanner_v1",
            confidence=hc.direction_confidence,
            opportunity_score=hc.opportunity_score,
            heat_score=hc.heat_score,
            chart_score=hc.chart_score,
        )

    def __repr__(self) -> str:
        act = self.action.value
        ret = f"{self.pred_return:+.2f}%"
        return (f"Signal({self.symbol} {act} {ret} "
                f"p_up={self.p_up:.2f} conf={self.confidence:.2f} "
                f"edge={self.edge:.2f}% regime={self.regime.value})")
