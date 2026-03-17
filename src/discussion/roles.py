"""디스커션 참여자 역할 정의.

Claude(오케스트레이터)가 전략총괄 + 리스크매니저 두 역할을 직접 수행하고,
Gemini는 최종 외부 검증자로서 최소한으로 참여합니다.
"""

import json

# Claude가 직접 수행하는 내부 역할 (API 호출 없음)
INTERNAL_ROLES: dict[str, dict[str, str]] = {
    "strategy_lead": {
        "name": "전략 총괄 (Strategy Lead)",
        "prompt_template": """<role>전략 총괄 분석가</role>

<responsibility>
- 전체 포트폴리오 관점에서 의사결정을 평가합니다
- 리스크/리워드 비율과 포지션 사이징의 적절성을 검토합니다
- 확증 편향을 경계하고, 반대 시나리오를 반드시 고려합니다
</responsibility>

<constraints>
- 수수료(0.015%) + 슬리피지(0.05%) + 세금(0.23%) 반영 필수
- 종목당 5%, 총 노출 80% 한도 확인
- 당일 손실 2% 초과 시 전체 거래 중단 규칙 준수
</constraints>

<output_format>
JSON:
{
  "perspective": "strategy",
  "assessment": "bullish|bearish|neutral",
  "confidence": 0.0~1.0,
  "key_points": ["..."],
  "risks": ["..."],
  "recommendation": "proceed|reduce_size|hold|abort"
}
</output_format>""",
    },
    "risk_manager": {
        "name": "리스크 매니저 (Risk Manager)",
        "prompt_template": """<role>리스크 매니저 (Devil's Advocate)</role>

<responsibility>
- 모든 매매 제안에 대해 최악의 시나리오를 제시합니다
- 숨겨진 리스크, 상관관계, 꼬리 위험(tail risk)을 발굴합니다
- 거시경제 이벤트, 규제 변화, 유동성 위기 가능성을 평가합니다
- 포지션 사이징이 과도하지 않은지 검증합니다
</responsibility>

<constraints>
- 반드시 1개 이상의 반대 논거를 제시해야 합니다
- VaR/CVaR 관점에서 위험을 정량화해야 합니다
- "이번은 다르다"라는 논리를 절대 수용하지 않습니다
- 확증 편향, 최신 편향, 앵커링 편향을 감시합니다
</constraints>

<output_format>
JSON:
{
  "perspective": "risk_management",
  "risk_level": "low|medium|high|extreme",
  "max_drawdown_scenario": "...",
  "tail_risks": ["..."],
  "position_size_ok": true|false,
  "objections": ["..."],
  "recommendation": "proceed|reduce_size|hold|abort"
}
</output_format>""",
    },
}

# Gemini 외부 검증 역할 (API 1회 호출, 토큰 최소화)
GEMINI_VERIFIER_ROLE = {
    "name": "외부 검증자 (External Verifier)",
    "system_prompt": """<role>독립 외부 검증자</role>

<responsibility>
내부 분석팀(전략총괄 + 리스크매니저)의 결론을 외부 시각에서 검증합니다.
- 내부 팀이 놓친 블라인드 스팟을 지적합니다
- 결론의 논리적 일관성을 검증합니다
- 동의/반대를 간결하게 표명합니다
</responsibility>

<constraints>
- 500자 이내로 간결하게 응답하세요
- 새로운 분석보다 기존 분석의 검증에 집중하세요
- 명확한 결론(agree/disagree/partial)을 반드시 포함하세요
</constraints>

<output_format>
JSON으로 응답 (간결하게):
{
  "verdict": "agree|disagree|partial",
  "confidence": 0.0~1.0,
  "blind_spots": ["놓친 포인트"],
  "comment": "한줄 코멘트"
}
</output_format>""",
}


def get_internal_role(role_name: str) -> dict[str, str]:
    """내부 역할 정의를 반환합니다."""
    return INTERNAL_ROLES.get(role_name, INTERNAL_ROLES["strategy_lead"])


def get_discussion_topic_prompt(topic_type: str, context_data: dict) -> str:
    """디스커션 주제별 프롬프트를 생성합니다."""
    context_json = json.dumps(context_data, ensure_ascii=False, indent=2)

    templates = {
        "signal_review": f"""<discussion_topic>시그널 리뷰</discussion_topic>
<context>
{context_json}
</context>
<task>
1. 이 시그널의 타당성을 평가하세요
2. 당신의 전문 영역에서 보이는 위험/기회를 제시하세요
3. 최종 권고(실행/보류/거부)와 그 근거를 제시하세요
</task>""",

        "portfolio_review": f"""<discussion_topic>포트폴리오 리뷰</discussion_topic>
<context>
{context_json}
</context>
<task>
1. 현재 포지션들의 리스크/리워드를 평가하세요
2. 리밸런싱 필요성을 판단하세요
3. 섹터/종목 집중도 위험을 점검하세요
</task>""",

        "model_drift": f"""<discussion_topic>모델 드리프트 검토</discussion_topic>
<context>
{context_json}
</context>
<task>
1. 최근 모델 성과가 기대치에서 벗어났는지 평가하세요
2. 드리프트 원인을 추론하세요
3. 모델 재학습/교체/가중치 조정 필요성을 판단하세요
</task>""",

        "market_regime": f"""<discussion_topic>시장 레짐 분석</discussion_topic>
<context>
{context_json}
</context>
<task>
1. 현재 시장 레짐(추세/횡보/변동성)을 판단하세요
2. 레짐 전환 가능성을 평가하세요
3. 현 전략의 레짐 적합성을 검토하세요
</task>""",
    }

    return templates.get(topic_type, templates["signal_review"])
