"""외부 AI API 클라이언트 래퍼.

Gemini만 외부 API로 사용하며, 최종 검증 단계에서 1회만 호출합니다.
Claude는 오케스트레이터로서 전략총괄 + 리스크매니저 역할을 직접 수행합니다.
GPT/Claude API는 사용하지 않습니다 (비용 절감).
"""

import json
import os
from typing import Any

import requests

from ..utils.logging import get_logger

logger = get_logger("ai_clients")


class GeminiClient:
    """Google Gemini API 클라이언트.

    Gemini 2.5 Pro 요금제 기준으로 토큰 사용량을 최소화합니다.
    - max_tokens 기본값 1024 (최소한의 응답)
    - 최종 검증 단계에서만 1회 호출
    """

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.model = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.max_tokens_default = 1024  # 비용 절감용 기본값

        if not self.api_key:
            logger.warning("GEMINI_API_KEY 환경변수 미설정")

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        """Gemini에게 메시지를 보내고 응답을 받습니다.

        Args:
            max_tokens: 응답 최대 토큰. None이면 self.max_tokens_default 사용.
        """
        max_tokens = max_tokens or self.max_tokens_default

        url = (
            f"{self.base_url}/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_message}"}]}
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.error("Gemini API 호출 실패: %s", e)
            return f"[Gemini 응답 실패: {e}]"

    def is_available(self) -> bool:
        return bool(self.api_key)


def get_gemini_client() -> GeminiClient | None:
    """Gemini 클라이언트를 반환합니다. 사용 불가 시 None."""
    client = GeminiClient()
    if client.is_available():
        logger.info("Gemini 클라이언트 활성화 (모델: %s)", client.model)
        return client
    logger.warning("Gemini 클라이언트 비활성 — GEMINI_API_KEY 미설정")
    return None
