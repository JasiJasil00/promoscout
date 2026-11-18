"""Independent validation of candidates against the PARSED screenplay (never trusts model output)."""
from __future__ import annotations
from .models import AuditFlag, Candidate, Episode

EPS = 0.05


def validate_candidate(c: Candidate, episodes: dict[int, Episode], dmin: float, dmax: float) -> list[AuditFlag]:
    f: list[AuditFlag] = []
    err = lambda code, msg: f.append(AuditFlag(code=code, severity="error", message=msg))
    if not c.segments:
        err("invalid_timestamp", "candidate has no segments"); return f
    total = 0.0
    prev = None
    for i, sg in enumerate(c.segments):
        ep = episodes.get(sg.episode)
        if ep is None:
            err("invalid_timestamp", f"segment {i+1}: episode {sg.episode} does not exist"); continue
        if not (0 <= sg.start < sg.end <= ep.duration + EPS):
            err("invalid_timestamp", f"segment {i+1}: {sg.start}-{sg.end} outside episode 0-{ep.duration}")
        total += sg.end - sg.start
        byid = {b.block_id: b for b in ep.blocks}
        missing = [x for x in sg.block_ids if x not in byid]
        if missing: err("invalid_timestamp", f"segment {i+1}: unknown block ids {missing[:3]}")
        for b in ep.blocks:
            if b.kind == "dialogue" and ((b.start + EPS < sg.start < b.end - EPS) or (b.start + EPS < sg.end < b.end - EPS)):
                f.append(AuditFlag(code="cut_inside_dialogue", severity="error",
                                   message=f"segment {i+1} cut at {sg.start if b.start < sg.start < b.end else sg.end} falls inside line {b.block_id}"))
                break
        if not sg.block_ids: err("invalid_timestamp", f"segment {i+1}: contains no screenplay blocks")
        if prev is not None and (sg.episode, sg.start) < (prev.episode, prev.end):
            err("invalid_timestamp", f"segment {i+1} is out of order or overlaps the previous segment")
        prev = sg
    if abs(total - c.duration) > 0.1:
        err("invalid_timestamp", f"stated duration {c.duration} != segment total {total:.2f}")
    if not (dmin - 1e-6 <= total <= dmax + 1e-6):
        err("bad_duration", f"duration {total:.1f}s is outside {dmin:.0f}-{dmax:.0f}s")
    return f
