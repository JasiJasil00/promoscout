from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .config import fmt_ts
from .pipeline import RunResult


def seg_str(c) -> str:
    return " | ".join(f"Ep{s.episode} {fmt_ts(s.start)}-{fmt_ts(s.end)}" for s in c.segments)


def candidates_df(r: RunResult) -> pd.DataFrame:
    rows = []
    for c in r.candidates:
        rows.append({
            "rank": c.rank, "tier": c.tier, "status": c.status, "candidate_id": c.candidate_id, "origin": c.origin,
            "episodes": ",".join(str(s.episode) for s in c.segments), "segments": seg_str(c),
            "start_first": c.segments[0].start, "end_last": c.segments[-1].end, "duration_s": c.duration,
            "total": c.scores.total, "hook_score": c.scores.hook, "cliffhanger_score": c.scores.cliffhanger,
            "clarity_score": c.scores.clarity, "emotion_score": c.scores.emotion, "confidence": c.scores.confidence,
            "score_source": c.scores.source, "hook": c.hook, "cliffhanger": c.cliffhanger, "why_it_works": c.why,
            "flags": "; ".join(f"{f.code}" for f in c.flags), "duplicate_of": c.duplicate_of or ""})
    return pd.DataFrame(rows)


def audit_df(r: RunResult) -> pd.DataFrame:
    rows = [{"candidate_id": c.candidate_id, "status": c.status, "rank": c.rank, "episodes": ",".join(str(s.episode) for s in c.segments),
             "segments": seg_str(c), "severity": f.severity, "code": f.code, "message": f.message}
            for c in r.candidates for f in c.flags]
    return pd.DataFrame(rows, columns=["candidate_id", "status", "rank", "episodes", "segments", "severity", "code", "message"])


def review_sheet_df(r: RunResult, n: int = 30) -> pd.DataFrame:
    """Blank human-review benchmark: fill the last columns after reading/watching each clip."""
    df = candidates_df(r)
    df = df[df.status == "approved"].head(n)[["rank", "candidate_id", "segments", "duration_s", "total", "hook", "cliffhanger", "why_it_works"]].copy()
    for col in ("human_hook_ok(Y/N)", "human_cliffhanger_ok(Y/N)", "human_standalone_ok(Y/N)", "would_post(Y/N)", "notes"):
        df[col] = ""
    return df


def write_outputs(r: RunResult, outdir: str | Path) -> dict[str, str]:
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    paths = {}
    def w(name, text):
        p = out / name; p.write_text(text, encoding="utf-8"); paths[name] = str(p)
    w("promos.json", json.dumps({"run": r.info, "approved": [c.model_dump() for c in r.approved],
                                 "all_candidates": [c.model_dump() for c in r.candidates]}, indent=2, ensure_ascii=False))
    w("promos_all.csv", candidates_df(r).to_csv(index=False))
    w("promos_approved.csv", candidates_df(r).query("status == 'approved'").to_csv(index=False))
    w("quality_audit.csv", audit_df(r).to_csv(index=False))
    w("review_sheet.csv", review_sheet_df(r).to_csv(index=False))
    w("episode_analysis.json", json.dumps(r.analysis, indent=2, ensure_ascii=False))
    w("parse_report.json", json.dumps({"ingest": r.ingest.model_dump(), "notes": r.notes,
                                       "episode_warnings": {e.number: e.warnings for e in r.episodes if e.warnings}}, indent=2, ensure_ascii=False))
    w("raw_llm_outputs.jsonl", "\n".join(json.dumps(x, ensure_ascii=False) for x in r.raw_llm))
    return paths
