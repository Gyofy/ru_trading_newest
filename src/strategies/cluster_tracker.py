"""ClusterTracker — 전략 간 시그널 동시 발생 추적.

6개 전략이 독립적으로 돌면서 시그널을 발생시킬 때,
"동시에 몇 개 전략이 같은 방향을 가리키는가?"를 추적.

이 데이터가 Meta-Labeling의 킬러 피처:
- CVD Extreme + Cascade Fade 동시 LONG = 단독보다 WR 20%p+ (가설)
- 3전략 이상 동방향 = 강한 확신 → 사이즈 2배

trade_context.jsonl에 클러스터 피처를 자동 주입하여
향후 메타모델 학습 시 사용.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger("cluster_tracker")


@dataclass
class SignalEvent:
    """하나의 시그널 발생 이벤트."""
    strategy: str
    coin: str
    side: str          # "BUY" or "SELL"
    strength: float    # 시그널 강도 (0~1)
    ts: float          # unix timestamp


class ClusterTracker:
    """Cross-strategy signal co-occurrence tracker.

    각 전략이 시그널 생성 시 publish() 호출.
    진입 직전 query()로 현재 클러스터 상태 조회.
    get_trade_context_features()로 trade_context에 주입할 피처 dict 반환.
    """

    DEFAULT_WINDOW_SEC = 300.0  # 5분 윈도우

    def __init__(self, window_sec: float = DEFAULT_WINDOW_SEC):
        self._window = window_sec
        # coin → list[SignalEvent] (최근 윈도우 내 시그널)
        self._signals: dict[str, list[SignalEvent]] = defaultdict(list)
        # 통계
        self._total_published = 0

    def publish(
        self,
        strategy: str,
        coin: str,
        side: str,
        strength: float = 1.0,
        ts: float | None = None,
    ) -> None:
        """전략이 시그널을 발생시킬 때 호출.

        Args:
            strategy: 전략 이름 (e.g. "cvd_extreme")
            coin: 코인 심볼 (e.g. "SOL")
            side: "BUY" or "SELL"
            strength: 시그널 강도 (0~1, 기본 1.0)
            ts: unix timestamp (None이면 현재 시각)
        """
        now = ts or time.time()
        event = SignalEvent(
            strategy=strategy,
            coin=coin,
            side=side.upper(),
            strength=strength,
            ts=now,
        )
        self._signals[coin].append(event)
        self._total_published += 1

        # 오래된 이벤트 정리
        self._cleanup(coin, now)

    def query(self, coin: str, side: str | None = None) -> dict:
        """현재 클러스터 상태 조회.

        Args:
            coin: 대상 코인
            side: 특정 방향만 조회 (None이면 전체)

        Returns:
            {
                "n_strategies_long": int,
                "n_strategies_short": int,
                "n_strategies_any": int,
                "n_strategies_same_dir": int,  # side 기준 동방향
                "cluster_agreement": float,    # 동방향 비율 (0~1)
                "strategies_long": list[str],
                "strategies_short": list[str],
                "strongest_strength": float,
                "avg_strength": float,
            }
        """
        now = time.time()
        self._cleanup(coin, now)

        events = self._signals.get(coin, [])

        # 전략별 최신 시그널만 (중복 제거 — 리스트 순서로 최신 우선)
        latest: dict[str, SignalEvent] = {}
        for e in events:
            if e.strategy not in latest or e.ts >= latest[e.strategy].ts:
                latest[e.strategy] = e

        longs = [e for e in latest.values() if e.side == "BUY"]
        shorts = [e for e in latest.values() if e.side == "SELL"]
        all_events = list(latest.values())

        n_long = len(longs)
        n_short = len(shorts)
        n_any = len(all_events)

        # side 기준 동방향 수
        if side and side.upper() == "BUY":
            n_same = n_long
        elif side and side.upper() == "SELL":
            n_same = n_short
        else:
            n_same = max(n_long, n_short)

        agreement = n_same / n_any if n_any > 0 else 0.0
        strengths = [e.strength for e in all_events]

        return {
            "n_strategies_long": n_long,
            "n_strategies_short": n_short,
            "n_strategies_any": n_any,
            "n_strategies_same_dir": n_same,
            "cluster_agreement": round(agreement, 4),
            "strategies_long": [e.strategy for e in longs],
            "strategies_short": [e.strategy for e in shorts],
            "strongest_strength": max(strengths) if strengths else 0.0,
            "avg_strength": round(sum(strengths) / len(strengths), 4) if strengths else 0.0,
        }

    def get_trade_context_features(self, coin: str, side: str) -> dict:
        """trade_context.jsonl에 저장할 클러스터 피처 dict.

        base.py의 capture_entry_context()에서 호출하여 trade_context에 주입.
        """
        q = self.query(coin, side)
        return {
            "cluster_n_long": q["n_strategies_long"],
            "cluster_n_short": q["n_strategies_short"],
            "cluster_n_any": q["n_strategies_any"],
            "cluster_n_same_dir": q["n_strategies_same_dir"],
            "cluster_agreement": q["cluster_agreement"],
            "cluster_strongest": q["strongest_strength"],
            "cluster_avg_strength": q["avg_strength"],
            "cluster_strategies_long": ",".join(q["strategies_long"]),
            "cluster_strategies_short": ",".join(q["strategies_short"]),
        }

    def _cleanup(self, coin: str, now: float) -> None:
        """윈도우 밖 이벤트 제거."""
        cutoff = now - self._window
        self._signals[coin] = [
            e for e in self._signals.get(coin, []) if e.ts >= cutoff
        ]
