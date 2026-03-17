"""디스커션 합의 도출 모듈.

<consensus_rules>
1. Claude 내부 리스크매니저 거부권: risk_level=extreme → 무조건 abort
2. 내부 분석(전략+리스크) 기반 1차 합의
3. Gemini 외부 검증으로 보정 (disagree 시 신뢰도 할인)
4. 신뢰도 하한: consensus_confidence < 0.5면 보류
</consensus_rules>
"""

from dataclasses import dataclass, field
from typing import Any

from ..utils.logging import get_logger

logger = get_logger("consensus")

# 내부 역할 가중치 (Claude가 두 역할 모두 수행)
INTERNAL_WEIGHTS = {
    "strategy_lead": 0.55,
    "risk_manager": 0.45,
}

# Gemini 검증 보정 계수
GEMINI_ADJUST = {
    "agree": 1.0,       # 동의 → 신뢰도 유지
    "partial": 0.85,     # 부분 동의 → 15% 할인
    "disagree": 0.6,     # 반대 → 40% 할인
}

# 액션 → 수치 매핑
ACTION_SCORES = {
    "proceed": 1.0,
    "buy": 1.0,
    "bullish": 0.8,
    "reduce_size": 0.4,
    "hold": 0.0,
    "neutral": 0.0,
    "bearish": -0.8,
    "sell": -1.0,
    "abort": -1.0,
}


@dataclass
class DiscussionResult:
    """디스커션 최종 결과."""
    final_action: str                              # proceed|reduce_size|hold|abort
    consensus_confidence: float                    # 0.0 ~ 1.0
    risk_veto: bool                                # 리스크 매니저 거부 여부
    summary: str                                   # 합의 요약
    individual_votes: dict[str, dict[str, Any]] = field(default_factory=dict)
    gemini_verification: dict[str, Any] | None = field(default=None)
    dissenting_opinions: list[str] = field(default_factory=list)


def extract_consensus(
    internal_analyses: dict[str, dict],
    gemini_verification: dict | None,
) -> DiscussionResult:
    """내부 분석 + Gemini 검증으로 합의를 도출합니다."""
    votes: dict[str, dict[str, Any]] = {}
    dissents: list[str] = []

    for role_name, analysis in internal_analyses.items():
        parsed = analysis.get("parsed", {})
        vote = _extract_vote(role_name, parsed)
        votes[role_name] = vote

        logger.info(
            "[%s] 투표: action=%s, confidence=%.2f",
            role_name,
            vote.get("action", "unknown"),
            vote.get("confidence", 0),
        )

    # === 리스크 매니저 거부권 체크 ===
    risk_vote = votes.get("risk_manager", {})
    if risk_vote.get("risk_level") == "extreme":
        logger.warning("리스크 매니저 거부권 발동: risk_level=extreme")
        return DiscussionResult(
            final_action="abort",
            consensus_confidence=1.0,
            risk_veto=True,
            summary="리스크 매니저 거부권 발동 — extreme risk level 감지",
            individual_votes=votes,
            gemini_verification=gemini_verification,
            dissenting_opinions=[
                "리스크매니저: " + risk_vote.get("max_drawdown_scenario", "극단적 위험")
            ],
        )

    # === 내부 가중 합의 계산 ===
    weighted_score = 0.0
    total_weight = 0.0
    confidences = []

    for role_name, vote in votes.items():
        weight = INTERNAL_WEIGHTS.get(role_name, 0.5)
        action = vote.get("action", "hold")
        score = ACTION_SCORES.get(action, 0.0)
        confidence = vote.get("confidence", 0.5)

        weighted_score += score * weight * confidence
        total_weight += weight
        confidences.append(confidence)

        if score < 0:
            objections = vote.get("objections", vote.get("tail_risks", []))
            if isinstance(objections, list):
                for obj in objections:
                    dissents.append(f"[{role_name}] {obj}")

    if total_weight > 0:
        normalized_score = weighted_score / total_weight
    else:
        normalized_score = 0.0

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # === Gemini 검증 보정 ===
    gemini_factor = 1.0
    if gemini_verification:
        verdict = gemini_verification.get("verdict", "partial")
        gemini_factor = GEMINI_ADJUST.get(verdict, 0.85)
        gemini_conf = gemini_verification.get("confidence", 0.5)

        # Gemini가 disagree + high confidence면 추가 할인
        if verdict == "disagree" and gemini_conf > 0.7:
            gemini_factor = 0.4
            blind_spots = gemini_verification.get("blind_spots", [])
            for spot in blind_spots:
                dissents.append(f"[Gemini] {spot}")

        logger.info(
            "[Gemini 보정] verdict=%s, factor=%.2f", verdict, gemini_factor
        )

    # 최종 액션 결정
    final_action = _score_to_action(normalized_score, avg_confidence)

    # 합의 신뢰도 = 일치도 × 평균 확신도 × Gemini 보정
    agreement_ratio = _calculate_agreement(votes)
    consensus_confidence = agreement_ratio * avg_confidence * gemini_factor

    # 신뢰도 하한 체크
    if consensus_confidence < 0.5 and final_action == "proceed":
        final_action = "hold"
        logger.info("합의 신뢰도 %.2f < 0.5 → hold로 전환", consensus_confidence)

    summary = _build_summary(
        votes, gemini_verification, final_action, consensus_confidence, dissents
    )

    return DiscussionResult(
        final_action=final_action,
        consensus_confidence=round(consensus_confidence, 3),
        risk_veto=False,
        summary=summary,
        individual_votes=votes,
        gemini_verification=gemini_verification,
        dissenting_opinions=dissents,
    )


def _extract_vote(role_name: str, parsed: dict) -> dict[str, Any]:
    """파싱된 응답에서 투표 정보를 추출합니다."""
    vote: dict[str, Any] = {}

    rec = (
        parsed.get("recommendation", "")
        or parsed.get("action", "")
        or parsed.get("assessment", "")
    )
    rec_lower = rec.lower() if isinstance(rec, str) else ""

    for key in ACTION_SCORES:
        if key in rec_lower:
            vote["action"] = key
            break
    else:
        vote["action"] = "hold"

    vote["confidence"] = parsed.get("confidence", parsed.get("model_agreement", 0.5))
    if not isinstance(vote["confidence"], (int, float)):
        vote["confidence"] = 0.5

    vote["risk_level"] = parsed.get("risk_level", "medium")
    vote["objections"] = parsed.get("objections", parsed.get("tail_risks", []))
    vote["concerns"] = parsed.get("concerns", [])
    vote["key_points"] = parsed.get("key_points", [])
    vote["max_drawdown_scenario"] = parsed.get("max_drawdown_scenario", "")

    return vote


def _score_to_action(score: float, confidence: float) -> str:
    """가중 점수를 최종 액션으로 변환합니다."""
    if confidence < 0.4:
        return "hold"
    if score > 0.5:
        return "proceed"
    if score > 0.2:
        return "reduce_size"
    if score > -0.3:
        return "hold"
    return "abort"


def _calculate_agreement(votes: dict[str, dict]) -> float:
    """투표 일치도를 계산합니다 (0~1)."""
    if len(votes) < 2:
        return 1.0

    actions = [v.get("action", "hold") for v in votes.values()]
    scores = [ACTION_SCORES.get(a, 0.0) for a in actions]

    if not scores:
        return 0.0

    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)

    return max(0.0, 1.0 - variance)


def _build_summary(
    votes: dict[str, dict],
    gemini_verification: dict | None,
    final_action: str,
    confidence: float,
    dissents: list[str],
) -> str:
    """합의 결과 요약을 생성합니다."""
    parts = [f"최종 판단: {final_action} (합의 신뢰도: {confidence:.1%})"]

    for role_name, vote in votes.items():
        parts.append(
            f"  [{role_name}] {vote.get('action', '?')} "
            f"(확신도: {vote.get('confidence', 0):.0%})"
        )

    if gemini_verification:
        verdict = gemini_verification.get("verdict", "?")
        comment = gemini_verification.get("comment", "")
        parts.append(f"  [Gemini 검증] {verdict}: {comment}")
    else:
        parts.append("  [Gemini] 미사용 — 내부 분석만으로 합의")

    if dissents:
        parts.append("반대 의견:")
        for d in dissents[:3]:
            parts.append(f"  - {d}")

    return "\n".join(parts)
