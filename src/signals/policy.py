"""Policy Layer — prediction을 action으로 변환.

모델이 "얼마나 오를 것 같다"를 말하면, policy가 "그래서 뭘 할 것인가"를 결정한다.
HOLD는 모델 라벨이 아니라 policy의 FLAT 결정이다.

핵심 규칙:
- TREND regime에서 역추세 시그널 차단
- SL은 최소 1% (타이트한 SL → 조기 손절 방지)
- TP는 SL의 2배 이상 (R:R ≥ 2.0)
- edge < 0.3% 시그널은 비용 대비 가치 없음 → FLAT

Usage:
    policy = SignalPolicy()
    signal = policy.decide(signal)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .contract import Signal, Action, Regime


@dataclass
class PolicyConfig:
    """정책 파라미터."""

    # ── Edge 필터 ──
    min_edge_pct: float = 0.30        # 비용 차감 후 최소 edge (0.1→0.3 상향)
    min_confidence: float = 0.35      # 최소 확신도 (0.3→0.35)
    min_p_up_long: float = 0.60       # LONG 최소 p_up (0.58→0.60)
    max_p_up_short: float = 0.40      # SHORT 최대 p_up (0.42→0.40)

    # ── 포지션 사이징 ──
    base_size: float = 0.03           # 기본 포지션 (계좌 3%)
    max_size: float = 0.05            # 최대 (CLAUDE.md: 코인당 5%)
    size_scale_by_confidence: bool = True

    # ── TP/SL (고정 범위 기반) ──
    min_sl_pct: float = 1.0           # SL 최소 1% (너무 타이트하면 noise에 걸림)
    max_sl_pct: float = 3.0           # SL 최대 3%
    min_rr_ratio: float = 2.0         # 최소 R:R 비율 (TP/SL ≥ 2.0)
    min_tp_pct: float = 2.0           # TP 최소 2% (0.5→2.0)
    max_tp_pct: float = 8.0           # TP 최대 8%

    # ── TTL (config: timeframes.execution.default_ttl_bars) ──
    default_ttl_bars: int = 12        # 12 × 5m = 1h (config 기준)

    # ── Regime-specific ──
    regime_overrides: dict = field(default_factory=lambda: {
        "VOLATILE": {
            "min_edge_pct": 0.50,     # 변동성: 높은 edge 요구
            "min_confidence": 0.45,
            "base_size": 0.02,        # 포지션 축소
            "min_sl_pct": 1.5,        # SL 살짝 넓게
            "max_sl_pct": 4.0,
            "min_tp_pct": 3.0,        # TP도 넓게
            "default_ttl_bars": 8,    # TTL 짧게
        },
        "RANGE": {
            "min_edge_pct": 0.20,     # 레인지: 낮은 edge OK (mean-reversion 유리)
            "min_sl_pct": 1.0,
            "max_sl_pct": 2.5,
            "min_tp_pct": 1.5,        # TP 보수적
            "max_tp_pct": 5.0,
            "default_ttl_bars": 8,
        },
        "TREND": {
            "min_edge_pct": 0.40,     # 트렌드: 역추세 방지 → 높은 기준
            "min_confidence": 0.45,
            "min_sl_pct": 1.5,        # 트렌드에서 SL 넓게
            "max_sl_pct": 4.0,
            "min_tp_pct": 3.0,        # TP 넓게 (트렌드 추종)
            "max_tp_pct": 10.0,
            "min_rr_ratio": 2.5,      # R:R 높게
            "default_ttl_bars": 18,   # TTL 넓게
        },
    })

    # ── Regime 방향 필터 ──
    # mean-reversion 모델은 TREND에서 거래 금지
    block_trend_for_meanrev: bool = True
    # UNKNOWN regime도 차단 (regime 판별 불가 → 보수적)
    block_unknown_regime: bool = True


class SignalPolicy:
    """Signal에 action/size/tp/sl/ttl을 채우는 정책 엔진."""

    def __init__(self, config: PolicyConfig | None = None):
        self.config = config or PolicyConfig()

    @classmethod
    def from_config(cls) -> "SignalPolicy":
        """config/settings.yaml에서 execution tier TTL을 반영한 기본 정책 생성."""
        from src.utils.config import get_execution
        cfg = get_execution()
        pc = PolicyConfig(default_ttl_bars=cfg.get("default_ttl_bars", 12))
        return cls(config=pc)

    def _get_param(self, key: str, regime: Regime):
        """regime 오버라이드 적용된 파라미터 반환."""
        overrides = self.config.regime_overrides.get(regime.value, {})
        if key in overrides:
            return overrides[key]
        return getattr(self.config, key)

    def _flat(self, signal: Signal) -> Signal:
        signal.action = Action.FLAT
        signal.size = 0.0
        return signal

    def decide(self, signal: Signal) -> Signal:
        """Signal에 policy 결정을 채운다."""
        regime = signal.regime

        min_edge = self._get_param("min_edge_pct", regime)
        min_conf = self._get_param("min_confidence", regime)
        min_p_long = self._get_param("min_p_up_long", regime)
        max_p_short = self._get_param("max_p_up_short", regime)

        # ── Step 1: Edge 필터 ──
        if signal.edge < min_edge:
            return self._flat(signal)

        if signal.confidence < min_conf:
            return self._flat(signal)

        # ── Step 2: 방향 결정 ──
        if signal.p_up >= min_p_long and signal.pred_return > 0:
            proposed_action = Action.LONG
        elif signal.p_up <= max_p_short and signal.pred_return < 0:
            proposed_action = Action.SHORT
        else:
            return self._flat(signal)

        # ── Step 3: Regime 방향 필터 ──
        # mean-reversion 모델(hot_scanner)은 TREND에서 역행 → 전면 차단
        if self.config.block_trend_for_meanrev and regime == Regime.TREND:
            return self._flat(signal)

        # UNKNOWN regime: 판별 불가 → 보수적으로 FLAT
        if self.config.block_unknown_regime and regime == Regime.UNKNOWN:
            return self._flat(signal)

        signal.action = proposed_action

        # ── Step 4: 포지션 사이징 ──
        base = self._get_param("base_size", regime)
        max_sz = self._get_param("max_size", regime)
        if self._get_param("size_scale_by_confidence", regime):
            # confidence 0.35~1.0 → size base*0.5 ~ base*1.5
            scale = 0.5 + signal.confidence
            signal.size = min(base * scale, max_sz)
        else:
            signal.size = base

        # ── Step 5: TP/SL (고정 범위 기반) ──
        min_sl = self._get_param("min_sl_pct", regime)
        max_sl = self._get_param("max_sl_pct", regime)
        min_tp = self._get_param("min_tp_pct", regime)
        max_tp = self._get_param("max_tp_pct", regime)
        min_rr = self._get_param("min_rr_ratio", regime)

        # SL: pred_return 기반이되 최소/최대 범위 적용
        raw_sl = abs(signal.pred_return) * 0.8
        signal.stop_loss = max(min_sl, min(raw_sl, max_sl))

        # TP: R:R ratio 보장
        min_tp_from_rr = signal.stop_loss * min_rr
        raw_tp = abs(signal.pred_return) * 2.0
        signal.take_profit = max(min_tp, min_tp_from_rr, min(raw_tp, max_tp))

        # ── Step 6: TTL ──
        signal.ttl_bars = self._get_param("default_ttl_bars", regime)

        return signal

    def decide_batch(self, signals: list[Signal]) -> list[Signal]:
        """여러 시그널에 policy 적용."""
        return [self.decide(s) for s in signals]

    def filter_actionable(self, signals: list[Signal]) -> list[Signal]:
        """actionable 시그널만 반환."""
        return [s for s in signals if s.is_actionable]

    def summary(self, signals: list[Signal]) -> dict:
        """시그널 배치 요약 통계."""
        total = len(signals)
        actionable = [s for s in signals if s.is_actionable]
        longs = [s for s in actionable if s.action == Action.LONG]
        shorts = [s for s in actionable if s.action == Action.SHORT]

        return {
            "total_signals": total,
            "actionable": len(actionable),
            "longs": len(longs),
            "shorts": len(shorts),
            "flat": total - len(actionable),
            "coverage": len(actionable) / total if total else 0,
            "avg_edge": (sum(s.edge for s in actionable) / len(actionable)
                         if actionable else 0),
            "avg_size": (sum(s.size for s in actionable) / len(actionable)
                         if actionable else 0),
            "avg_confidence": (sum(s.confidence for s in actionable) / len(actionable)
                               if actionable else 0),
        }
