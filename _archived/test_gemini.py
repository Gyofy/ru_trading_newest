"""Gemini API 연결 테스트."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")

import requests

API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "gemini-2.5-flash"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

payload = {
    "contents": [
        {"role": "user", "parts": [{"text": "안녕하세요. 간단히 자기소개 한 줄만 해주세요."}]}
    ],
    "generationConfig": {
        "temperature": 0.3,
        "maxOutputTokens": 256,
    },
}

print(f"API Key: {API_KEY[:10]}...{API_KEY[-4:]}")
print(f"Model: {MODEL}")
print("요청 중...")

resp = requests.post(URL, json=payload, timeout=30)
print(f"Status: {resp.status_code}")

if resp.status_code == 200:
    data = resp.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    print(f"\n✅ Gemini 응답:\n{text}")
else:
    print(f"\n❌ 에러:\n{resp.text}")
