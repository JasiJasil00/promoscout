"""LLM judging stage. The model never supplies timestamps; its output is parsed and every reference is verified."""
from __future__ import annotations
from pydantic import ValidationError
from .models import Candidate, Episode, LLMJudgment
from .candidates import window_blocks, block_line
from .prompts import SYSTEM_JUDGE, user_prompt


def _items(batch: list[Candidate], eps: dict[int, Episode]) -> list[dict]:
    items = []
    for c in batch:
        segs = []
        for sg in c.segments:
            ep = eps[sg.episode]
            segs.append({"episode": sg.episode, "logline": ep.logline,
                         "lines": [block_line(b) for b in window_blocks(ep, sg.start, sg.end)]})
        items.append({"candidate_id": c.candidate_id, "duration": c.duration, "segments": segs})
    return items


def judge_candidates(cands: list[Candidate], eps: dict[int, Episode], client, title: str, batch_size: int = 8,
                     progress=None) -> list[str]:
    """Fills c.llm in place. Returns human-readable failure notes (also stored as flags by the pipeline)."""
    notes: list[str] = []
    by_id = {c.candidate_id: c for c in cands}

    def run(batch: list[Candidate], tag: str) -> set[str]:
        got: set[str] = set()
        try:
            data = client.generate_json(SYSTEM_JUDGE, user_prompt(title, _items(batch, eps)), tag=tag)
        except Exception as e:
            notes.append(f"{tag}: LLM call failed: {e}"); return got
        results = data.get("results") if isinstance(data, dict) else data
        if not isinstance(results, list):
            notes.append(f"{tag}: response had no 'results' list"); return got
        for r in results:
            try:
                j = LLMJudgment.model_validate(r)
            except ValidationError as e:
                notes.append(f"{tag}: invalid judgment for {r.get('candidate_id') if isinstance(r, dict) else '?'}: {str(e)[:150]}")
                continue
            c = by_id.get(j.candidate_id)
            if c is None or j.candidate_id not in {x.candidate_id for x in batch}:
                notes.append(f"{tag}: model returned unknown candidate_id {j.candidate_id!r}"); continue
            c.llm = j; got.add(j.candidate_id)
        return got

    n = len(cands)
    for i in range(0, n, batch_size):
        batch = cands[i:i + batch_size]
        got = run(batch, f"judge-batch-{i // batch_size + 1}")
        missing = [c for c in batch if c.candidate_id not in got]
        for c in missing:                                       # one individual retry for anything the model skipped
            if c.candidate_id not in run([c], f"judge-retry-{c.candidate_id}"):
                notes.append(f"{c.candidate_id}: no valid judgment after retry")
        if progress: progress(min(i + batch_size, n), n)
    return notes
