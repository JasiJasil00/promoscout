from __future__ import annotations
from .models import Candidate, ScoreCard
from .features import weighted_total

DET_CONFIDENCE = 0.35   # fixed, low: heuristic-only scores are weak evidence


def apply_scores(c: Candidate, w: dict, disagreement: float) -> None:
    d, l = c.det, c.llm
    if l is None:
        c.scores = ScoreCard(hook=d.hook, cliffhanger=d.cliffhanger, clarity=d.clarity, emotion=d.emotion,
                             total=weighted_total(d.hook, d.cliffhanger, d.clarity, d.emotion, w),
                             confidence=DET_CONFIDENCE, source="deterministic",
                             evidence={"hook": d.hook_text, "cliffhanger": d.cliff_text})
        c.hook, c.cliffhanger = d.hook_text, d.cliff_text
        c.why = ("Heuristic pick: hook signals = " + "; ".join(d.signals["hook"][:2]) + ". Cliffhanger signals = "
                 + "; ".join(d.signals["cliffhanger"][:2]) + ".")
        return
    h, cf, cl, em = l.hook.score, l.cliffhanger.score, l.clarity.score, l.emotion.score
    gap = max(abs(h - d.hook), abs(cf - d.cliffhanger), abs(cl - d.clarity), abs(em - d.emotion))
    conf = l.confidence * (0.7 if gap >= disagreement else 1.0)
    c.scores = ScoreCard(hook=h, cliffhanger=cf, clarity=cl, emotion=em,
                         total=weighted_total(h, cf, cl, em, w), confidence=round(conf, 2), source="llm",
                         evidence={"hook": l.hook.evidence_quote, "cliffhanger": l.cliffhanger.evidence_quote,
                                   "emotion": l.emotion.evidence_quote})
    c.hook = l.hook.reason or l.hook.evidence_quote
    c.cliffhanger = l.cliffhanger.open_question or l.cliffhanger.reason or l.cliffhanger.evidence_quote
    c.why = l.one_line_why


def select_and_rank(cands: list[Candidate], cfg: dict) -> None:
    """Greedy selection with a diversity penalty. Sets status/rank/tier. Raw totals are never altered."""
    sel = cfg["selection"]; pen = sel["diversity_penalty"]
    pool = [c for c in cands if c.status == "candidate" and not any(f.severity == "error" for f in c.flags)]
    for c in cands:
        if c.status == "candidate" and c not in pool: c.status = "rejected"
    chosen: list[Candidate] = []
    scene_of = lambda c: {(s.episode, b.split("-")[1]) for s in c.segments for b in s.block_ids}
    while pool and len(chosen) < sel["approved_target"]:
        def adj(c):
            eps = {s.episode for s in c.segments}
            a = c.scores.total
            a -= pen["same_episode"] * sum(1 for x in chosen if eps & {s.episode for s in x.segments})
            a -= pen["same_scene"] * sum(1 for x in chosen if scene_of(c) & scene_of(x))
            return a
        best = max(pool, key=adj); pool.remove(best)
        if best.scores.total < sel["min_total"]:
            best.status = "rejected"; continue
        if best.llm is not None and best.llm.verdict == "reject":
            best.status = "rejected"; continue
        if best.origin == "joined" and best.llm is not None and best.llm.pair_coherent is False:
            best.status = "rejected"; continue
        chosen.append(best)
    tiers = sel["tiers"]
    for i, c in enumerate(chosen, 1):
        c.status, c.rank = "approved", i
        c.tier = "A" if i <= tiers["A"] else ("B" if i <= tiers["A"] + tiers["B"] else "C")
