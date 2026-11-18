from __future__ import annotations
import datetime as dt
from dataclasses import dataclass, field
from .config import load_aliases, load_config
from .ingest import load_episodes
from .analysis import analyze_episodes
from .candidates import generate_windows, generate_joined
from .validate import validate_candidate
from .judge import judge_candidates
from .scoring import apply_scores, select_and_rank
from .dedupe import mark_duplicates
from .audit import audit_candidate
from .models import AuditFlag, Candidate, Episode, IngestReport


@dataclass
class RunResult:
    episodes: list[Episode]
    ingest: IngestReport
    analysis: dict
    candidates: list[Candidate]
    notes: list[str] = field(default_factory=list)
    raw_llm: list[dict] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def approved(self) -> list[Candidate]:
        return sorted([c for c in self.candidates if c.status == "approved"], key=lambda c: c.rank or 10**9)


def run_pipeline(inputs, cfg: dict | None = None, llm=None, title: str = "Untitled microdrama", progress=None) -> RunResult:
    cfg = cfg or load_config()
    aliases, hidden = load_aliases(cfg)
    episodes, rep = load_episodes(inputs, aliases, hidden)
    notes: list[str] = [f"ingest error: {e}" for e in rep.errors]
    if not episodes:
        return RunResult([], rep, {}, [], notes + ["no episodes could be parsed"], [], {"mode": "empty"})
    eps = {e.number: e for e in episodes}
    analysis = analyze_episodes(episodes)

    cands = generate_windows(episodes, cfg)[: cfg["candidates"]["max_shortlist"]] + generate_joined(episodes, cfg)
    dmin, dmax = cfg["duration"]["min"], cfg["duration"]["max"]
    for c in cands:
        c.flags += validate_candidate(c, eps, dmin, dmax)

    if llm is not None:
        valid = [c for c in cands if not any(f.severity == "error" for f in c.flags)]
        notes += judge_candidates(valid, eps, llm, title, cfg["llm"]["batch_size"], progress)
        for c in valid:
            if c.llm is None:
                c.flags.append(AuditFlag(code="llm_failed", severity="warn", message="no valid LLM judgment; scored by heuristics"))
    for c in cands:
        apply_scores(c, cfg["weights"], cfg["audit"]["disagreement"])
    mark_duplicates(cands, cfg["dedupe"]["time_iou"], cfg["dedupe"]["text_cosine"])
    for c in cands:
        audit_candidate(c, cfg)
    select_and_rank(cands, cfg)
    cands.sort(key=lambda c: (c.rank is None, c.rank or 0, -c.scores.total))
    mode = "llm:" + llm.model_id() if llm is not None else "deterministic"
    return RunResult(episodes, rep, analysis, cands, notes, list(getattr(llm, "raw_log", [])),
                     {"mode": mode, "title": title, "run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                      "weights": cfg["weights"], "duration": cfg["duration"], "episodes": len(episodes)})
