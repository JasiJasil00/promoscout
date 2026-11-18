import json, pytest
from promoscout.llm.gemini import GeminiClient
from promoscout.llm.base import LLMError
import promoscout.llm.gemini as g

class R:
    def __init__(self, code, body): self.status_code, self._b, self.text = code, body, json.dumps(body)
    def json(self): return self._b
ok = lambda t: R(200, {"candidates": [{"content": {"parts": [{"text": t}]}}]})

def test_request_shape_and_json_parse(monkeypatch):
    seen = {}
    def post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, body=json, headers=headers); return ok('```json\n{"results": []}\n```')
    monkeypatch.setattr(g.requests, "post", post)
    c = GeminiClient(api_key="k", model="m1", cache_dir=None)
    assert c.generate_json("sys", "usr") == {"results": []}
    assert seen["url"].endswith("models/m1:generateContent") and seen["headers"]["x-goog-api-key"] == "k"
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert "k" not in json.dumps(c.raw_log)                       # key never lands in the raw log

def test_retries_429_but_not_400(monkeypatch):
    seq = [R(429, {}), R(503, {}), ok("{}")]
    monkeypatch.setattr(g.requests, "post", lambda *a, **k: seq.pop(0))
    assert GeminiClient(api_key="k", cache_dir=None, max_retries=4).generate_json("s", "u") == {}
    monkeypatch.setattr(g.requests, "post", lambda *a, **k: R(400, {"error": "bad key"}))
    with pytest.raises(LLMError):
        GeminiClient(api_key="k", cache_dir=None, max_retries=4).generate_json("s", "u")

def test_missing_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMError): GeminiClient(cache_dir=None)

def test_cache_avoids_second_call(monkeypatch, tmp_path):
    n = {"c": 0}
    def post(*a, **k): n["c"] += 1; return ok('{"a": 1}')
    monkeypatch.setattr(g.requests, "post", post)
    c = GeminiClient(api_key="k", cache_dir=str(tmp_path))
    c.generate_json("s", "u"); c.generate_json("s", "u")
    assert n["c"] == 1
