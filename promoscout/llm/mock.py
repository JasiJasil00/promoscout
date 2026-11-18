"""Offline stand-in for tests and demos. NOT a language model: it returns fixed mid-range scores
and quotes the first spoken line of each candidate. Its output must never be reported as model results."""
from __future__ import annotations
import json, re
from .base import LLMClient


class MockClient(LLMClient):
    name = "mock"

    def __init__(self, **kw):
        kw.setdefault("cache_dir", None)
        super().__init__(**kw)

    def _call(self, system: str, user: str) -> str:
        out = []
        for part in re.split(r"\n(?=CANDIDATE )", user):
            m = re.match(r"CANDIDATE (\S+)", part)
            if not m: continue
            lines = re.findall(r"\[(E\d+-S\d+-B\d+)\] [\d.]+-[\d.]+ (?!CAMERA|ACTION|SFX)[^:\n]+: ([^\n]+)", part)
            bid, q = (lines[0] if lines else ("", ""))
            lid, lq = (lines[-1] if lines else ("", ""))
            ev = lambda b, t: {"score": 5, "block_id": b, "evidence_quote": t, "reason": "mock", "open_question": "mock"}
            out.append({"candidate_id": m.group(1), "hook": ev(bid, q), "cliffhanger": ev(lid, lq),
                        "clarity": {"score": 5, "who": "mock", "what": "mock", "why_it_matters": "mock", "missing_context": []},
                        "emotion": ev(bid, q), "one_line_why": "mock", "unsupported_claims": [], "confidence": 0.5,
                        "verdict": "maybe", "pair_coherent": True if "SEGMENT 2" in part else None, "pair_reason": ""})
        return json.dumps({"results": out})
