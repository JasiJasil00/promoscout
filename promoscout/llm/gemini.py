from __future__ import annotations
import os, requests
from .base import LLMClient, LLMError

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class _Err(LLMError):
    def __init__(self, msg, retryable=True):
        super().__init__(msg); self.retryable = retryable


class GeminiClient(LLMClient):
    """Google Gemini via REST (no SDK needed). Key from env GEMINI_API_KEY; model from GEMINI_MODEL."""
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 90, **kw):
        super().__init__(**kw)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        self.timeout = timeout
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is not set")

    def model_id(self) -> str:
        return f"gemini:{self.model}"

    def _call(self, system: str, user: str) -> str:
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2}}
        try:
            r = requests.post(URL.format(model=self.model), json=body, timeout=self.timeout,
                              headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"})
        except requests.RequestException as e:
            raise _Err(f"network error: {e}")
        if r.status_code in (429, 500, 502, 503, 504):
            raise _Err(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code != 200:
            raise _Err(f"HTTP {r.status_code}: {r.text[:300]}", retryable=False)
        try:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, ValueError) as e:
            raise _Err(f"unexpected response shape: {r.text[:300]}")
