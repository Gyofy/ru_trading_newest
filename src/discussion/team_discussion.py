"""멀티 AI 팀 디스커션 오케스트레이터.

<design_principles>
- Claude(오케스트레이터)가 전략총괄 + 리스크매니저 두 역할을 직접 수행 (API 비용 0)
- Gemini는 최종 검증 단계에서 1회만 호출 (비용 최소화)
- GPT/Claude API는 사용하지 않음
- 리스크 매니저 거부권(veto)은 Claude가 내부적으로 판단
</design_principles>
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .ai_clients import get_gemini_client, GeminiClient
from .roles import (
    get_internal_role,
    get_discussion_topic_prompt,
    GEMINI_VERIFIER_ROLE,
)
from .consensus import extract_consensus, DiscussionResult
from ..utils.logging import get_logger
from ..utils.config import get_data_path

logger = get_logger("team_discussion")


class TeamDiscussion:
    """멀티 AI 팀 디스커션을 관리합니다.

    <flow>
    1단계: Claude 내부 분석 (전략총괄 + 리스크매니저) — API 비용 0
    2단계: Gemini 외부 검증 (1회 호출, 최소 토큰) — 비용 최소화
    3단계: 합의 도출
    </flow>
    """

    def __init__(self):
        self.gemini = get_gemini_client()
        self.discussion_log: list[dict[str, Any]] = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_discussion(
        self,
        topic_type: str,
        context_data: dict,
        strategy_analysis: dict | None = None,
        risk_analysis: dict | None = None,
    ) -> DiscussionResult:
        """디스커션을 실행합니다.

        Args:
            topic_type: 토론 주제 유형
            context_data: 토론에 필요한 데이터 컨텍스트
            strategy_analysis: Claude가 수행한 전략 분석 결과 (JSON dict)
            risk_analysis: Claude가 수행한 리스크 분석 결과 (JSON dict)

        Returns:
            DiscussionResult: 합의 결과
        """
        logger.info(
            "=== 팀 디스커션 시작 [%s] 주제: %s ===",
            self.session_id,
            topic_type,
        )

        # === 1단계: Claude 내부 분석 ===
        internal_analyses = self._build_internal_analyses(
            topic_type, context_data, strategy_analysis, risk_analysis
        )

        # === 2단계: Gemini 외부 검증 (1회) ===
        gemini_verification = self._run_gemini_verification(
            topic_type, context_data, internal_analyses
        )

        # === 3단계: 합의 도출 ===
        result = extract_consensus(internal_analyses, gemini_verification)

        # === 로그 저장 ===
        self._save_discussion_log(topic_type, context_data, result)

        logger.info(
            "=== 디스커션 완료 [%s] 합의: %s (신뢰도: %.2f) ===",
            self.session_id,
            result.final_action,
            result.consensus_confidence,
        )

        return result

    def _build_internal_analyses(
        self,
        topic_type: str,
        context_data: dict,
        strategy_analysis: dict | None,
        risk_analysis: dict | None,
    ) -> dict[str, dict]:
        """Claude 오케스트레이터의 내부 분석 결과를 구성합니다.

        strategy_analysis와 risk_analysis는 Claude Code가 직접 생성한 것이므로
        별도 API 호출 없이 바로 사용합니다.
        """
        analyses = {}

        # 전략 분석
        if strategy_analysis:
            analyses["strategy_lead"] = {
                "role": "전략 총괄 (Strategy Lead)",
                "parsed": strategy_analysis,
            }
            self.discussion_log.append({
                "step": "internal_analysis",
                "role": "strategy_lead",
                "response": strategy_analysis,
                "timestamp": datetime.now().isoformat(),
            })
            logger.info("[전략총괄] 분석 완료 — assessment: %s", strategy_analysis.get("assessment", "?"))
        else:
            logger.warning("[전략총괄] 분석 결과 미제공 — 기본값 사용")
            analyses["strategy_lead"] = {
                "role": "전략 총괄 (Strategy Lead)",
                "parsed": {"assessment": "neutral", "confidence": 0.5, "recommendation": "hold"},
            }

        # 리스크 분석
        if risk_analysis:
            analyses["risk_manager"] = {
                "role": "리스크 매니저 (Risk Manager)",
                "parsed": risk_analysis,
            }
            self.discussion_log.append({
                "step": "internal_analysis",
                "role": "risk_manager",
                "response": risk_analysis,
                "timestamp": datetime.now().isoformat(),
            })
            logger.info("[리스크매니저] 분석 완료 — risk_level: %s", risk_analysis.get("risk_level", "?"))
        else:
            logger.warning("[리스크매니저] 분석 결과 미제공 — 기본값 사용")
            analyses["risk_manager"] = {
                "role": "리스크 매니저 (Risk Manager)",
                "parsed": {"risk_level": "medium", "confidence": 0.5, "recommendation": "hold"},
            }

        return analyses

    def _run_gemini_verification(
        self,
        topic_type: str,
        context_data: dict,
        internal_analyses: dict[str, dict],
    ) -> dict | None:
        """Gemini에게 내부 분석 결과의 외부 검증을 요청합니다.

        1회만 호출하며, 토큰을 최소화합니다.
        Gemini 미사용 시 None 반환 → 내부 분석만으로 합의.
        """
        if not self.gemini:
            logger.info("Gemini 비활성 — 내부 분석만으로 합의 진행")
            return None

        # 내부 분석 요약을 간결하게 구성 (입력 토큰 절감)
        summary = {
            "topic": topic_type,
            "strategy": internal_analyses.get("strategy_lead", {}).get("parsed", {}),
            "risk": internal_analyses.get("risk_manager", {}).get("parsed", {}),
        }
        summary_json = json.dumps(summary, ensure_ascii=False)

        verification_prompt = f"""<verification_request>
내부 분석팀의 결론을 검증해주세요. 간결하게 JSON으로만 응답하세요.

{summary_json}
</verification_request>"""

        logger.info("[Gemini] 외부 검증 요청 (1회)...")

        raw_response = self.gemini.chat(
            system_prompt=GEMINI_VERIFIER_ROLE["system_prompt"],
            user_message=verification_prompt,
            temperature=0.2,
            max_tokens=512,  # 최소 토큰
        )

        parsed = self._parse_gemini_response(raw_response)

        self.discussion_log.append({
            "step": "gemini_verification",
            "role": "external_verifier",
            "raw_response": raw_response,
            "parsed": parsed,
            "timestamp": datetime.now().isoformat(),
        })

        logger.info(
            "[Gemini] 검증 완료 — verdict: %s",
            parsed.get("verdict", "?"),
        )

        return parsed

    def _parse_gemini_response(self, raw: str) -> dict:
        """Gemini 응답에서 JSON을 추출합니다."""
        import re

        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        try:
            stripped = raw.strip()
            start = stripped.find("{")
            end = stripped.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(stripped[start:end])
        except json.JSONDecodeError:
            pass

        logger.warning("[Gemini] JSON 파싱 실패, raw 텍스트 사용")
        return {"verdict": "partial", "confidence": 0.5, "comment": raw[:200]}

    def _save_discussion_log(
        self,
        topic_type: str,
        context_data: dict,
        result: DiscussionResult,
    ) -> Path:
        """디스커션 전체 로그를 파일로 저장합니다."""
        date_str = datetime.now().strftime("%Y%m%d")
        log_dir = get_data_path("reports", date_str)
        log_path = log_dir / f"discussion_{self.session_id}.json"

        log_data = {
            "session_id": self.session_id,
            "topic_type": topic_type,
            "architecture": "claude_internal + gemini_verify",
            "gemini_enabled": self.gemini is not None,
            "context_data": context_data,
            "discussion_log": self.discussion_log,
            "consensus": {
                "final_action": result.final_action,
                "consensus_confidence": result.consensus_confidence,
                "risk_veto": result.risk_veto,
                "summary": result.summary,
                "individual_votes": result.individual_votes,
            },
            "timestamp": datetime.now().isoformat(),
        }

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        logger.info("디스커션 로그 저장: %s", log_path)
        return log_path


def quick_signal_review(
    signal_data: dict,
    strategy_analysis: dict | None = None,
    risk_analysis: dict | None = None,
) -> DiscussionResult:
    """시그널 리뷰를 위한 간편 함수."""
    team = TeamDiscussion()
    return team.run_discussion(
        "signal_review", signal_data, strategy_analysis, risk_analysis
    )


def quick_portfolio_review(
    portfolio_data: dict,
    strategy_analysis: dict | None = None,
    risk_analysis: dict | None = None,
) -> DiscussionResult:
    """포트폴리오 리뷰를 위한 간편 함수."""
    team = TeamDiscussion()
    return team.run_discussion(
        "portfolio_review", portfolio_data, strategy_analysis, risk_analysis
    )
