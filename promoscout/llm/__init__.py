from .base import LLMClient, LLMError
from .gemini import GeminiClient
from .mock import MockClient


def make_client(kind: str, cfg: dict, **kw):
    """kind: 'none' | 'gemini' | 'mock'. Add new providers here."""
    llm = cfg.get("llm", {})
    common = dict(min_interval_s=llm.get("min_interval_s", 0), max_retries=llm.get("max_retries", 4))
    if kind in ("none", "", None): return None
    if kind == "gemini": return GeminiClient(**common, **kw)
    if kind == "mock": return MockClient(**kw)
    raise ValueError(f"unknown LLM provider: {kind}")
