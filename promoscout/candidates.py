"""Candidate generation. Code owns every timestamp: cuts snap to block edges and never fall inside a spoken line."""
from __future__ import annotations
import bisect
from .models import Block, Candidate, DetFeatures, Episode, Segment
from .features import score_blocks, weighted_total

EPS = 0.05


def block_line(b: Block) -> str:
    who = f"{b.speaker_raw}" + (f" ({b.tone})" if b.tone else "") if b.kind == "dialogue" else b.kind.upper()
    return f"[{b.block_id}] {b.start:.1f}-{b.end:.1f} {who}: {b.text}"


def _valid_cut(t: float, spans: list[tuple[float, float]]) -> bool:
    return not any(a + EPS < t < b - EPS for a, b in spans)


def episode_windows(ep: Episode, dmin: float, dmax: float, min_dialogue: int) -> list[tuple[float, float]]:
    blocks = ep.blocks
    dspans = [(b.start, b.end) for b in blocks if b.kind == "dialogue"]
    dstarts = sorted(a for a, _ in dspans)
    starts = {round(a, 2) for a in dstarts}
    for b in blocks:
        if b.kind != "dialogue":
            i = bisect.bisect_left(dstarts, b.start - EPS)
            if i < len(dstarts) and dstarts[i] - b.start <= 3.0 and _valid_cut(b.start, dspans):
                starts.add(round(b.start, 2))
    ends = {round(b.end, 2) for b in blocks if b.kind == "dialogue"} | {round(sc.end, 2) for sc in ep.scenes}
    ends = sorted(x for x in ends if x <= ep.duration + EPS and _valid_cut(x, dspans))
    out = []
    for s in sorted(starts):
        if not _valid_cut(s, dspans):
            continue
        lo = bisect.bisect_left(ends, s + dmin - 1e-9); hi = bisect.bisect_right(ends, s + dmax + 1e-9)
        for e in ends[lo:hi]:
            n = sum(1 for a, _ in dspans if s - EPS <= a < e - EPS)
            if n >= min_dialogue:
                out.append((s, e))
    return out


def window_blocks(ep: Episode, s: float, e: float) -> list[Block]:
    return [b for b in ep.blocks if b.start >= s - EPS and b.start < e - EPS]


def make_segment(ep: Episode, s: float, e: float) -> Segment:
    return Segment(episode=ep.number, start=round(s, 2), end=round(e, 2),
                   block_ids=[b.block_id for b in window_blocks(ep, s, e)])


def build_candidate(eps_segments: list[tuple[Episode, float, float]], det: DetFeatures, origin: str) -> Candidate:
    segs, lines = [], []
    for ep, s, e in eps_segments:
        segs.append(make_segment(ep, s, e))
        lines += [block_line(b) for b in window_blocks(ep, s, e)]
    cid = ("C-" if origin == "window" else "J-") + "+".join(f"E{x.episode:03d}_{int(x.start*100):05d}_{int(x.end*100):05d}" for x in segs)
    return Candidate(candidate_id=cid, segments=segs, origin=origin, duration=round(sum(x.duration for x in segs), 2),
                     det=det, text="\n".join(lines))


def generate_windows(episodes: list[Episode], cfg: dict) -> list[Candidate]:
    w = cfg["weights"]; dmin, dmax = cfg["duration"]["min"], cfg["duration"]["max"]
    cc = cfg["candidates"]
    allc: list[tuple[float, Candidate]] = []
    for ep in episodes:
        scene_starts = [sc.start for sc in ep.scenes]
        scored = []
        for s, e in episode_windows(ep, dmin, dmax, cc["min_dialogue_lines"]):
            det = score_blocks(window_blocks(ep, s, e), s, e, ep, scene_starts)
            scored.append((weighted_total(det.hook, det.cliffhanger, det.clarity, det.emotion, w), s, e, det))
        scored.sort(key=lambda x: -x[0])
        kept: list[tuple[float, float]] = []
        for tot, s, e, det in scored:                     # per-episode non-max suppression
            if len(kept) >= cc["per_episode_cap"]: break
            if all(max(0, min(e, ke) - max(s, ks)) / (max(e, ke) - min(s, ks)) <= 0.5 for ks, ke in kept):
                kept.append((s, e)); allc.append((tot, build_candidate([(ep, s, e)], det, "window")))
    allc.sort(key=lambda x: -x[0])
    return [c for _, c in allc]


def generate_joined(episodes: list[Episode], cfg: dict) -> list[Candidate]:
    """Multi-segment promos: a hook-strong short segment followed by a cliffhanger-strong short segment
    (same episode, later in time, or the next episode). Pairing is code; coherence is checked later by the LLM."""
    w = cfg["weights"]; dmin, dmax = cfg["duration"]["min"], cfg["duration"]["max"]
    sm = cfg["candidates"]["short_segment"]; byn = {e.number: e for e in episodes}
    A: list[tuple[Episode, float, float, DetFeatures]] = []; B = []
    for ep in episodes:
        scene_starts = [sc.start for sc in ep.scenes]
        segs = []
        for s, e in episode_windows(ep, sm["min"], sm["max"], 2):
            segs.append((s, e, score_blocks(window_blocks(ep, s, e), s, e, ep, scene_starts)))
        def top(key, n):
            out = []
            for s, e, d in sorted(segs, key=lambda x: -getattr(x[2], key)):
                if all(e <= o[0] or s >= o[1] for o in out_iv(out)): out.append((s, e, d))
                if len(out) >= n: break
            return out
        out_iv = lambda lst: [(x[0], x[1]) for x in lst]
        A += [(ep, s, e, d) for s, e, d in top("hook", 4)]
        B += [(ep, s, e, d) for s, e, d in top("cliffhanger", 4)]
    pairs = []
    for ea, sa, ea_, da in A:
        for eb, sb, eb_, db in B:
            same = ea.number == eb.number
            if not ((same and sb >= ea_ + 5.0) or eb.number == ea.number + 1):
                continue
            dur = (ea_ - sa) + (eb_ - sb)
            if not (dmin <= dur <= dmax): continue
            ca = {b.character for b in window_blocks(ea, sa, ea_) if b.character}
            cb = {b.character for b in window_blocks(eb, sb, eb_) if b.character}
            if len(ca & cb) < 2: continue                 # needs the same people on both sides
            det = DetFeatures(
                hook=da.hook, cliffhanger=db.cliffhanger, clarity=round(max(0, (da.clarity + db.clarity) / 2 - 0.5), 2),
                emotion=round((da.emotion + db.emotion) / 2, 2),
                signals={"hook": da.signals["hook"], "cliffhanger": db.signals["cliffhanger"],
                         "clarity": ["multi-segment join (-0.5 clarity)"] + da.signals["clarity"][:2],
                         "emotion": da.signals["emotion"][:2] + db.signals["emotion"][:2]},
                hook_text=da.hook_text, hook_block=da.hook_block, cliff_text=db.cliff_text, cliff_block=db.cliff_block)
            tot = weighted_total(det.hook, det.cliffhanger, det.clarity, det.emotion, w)
            pairs.append((tot, [(ea, sa, ea_), (eb, sb, eb_)], det))
    pairs.sort(key=lambda x: -x[0])
    seen, out = set(), []
    for tot, segs, det in pairs:
        key = tuple((x[0].number, x[1]) for x in segs)
        if key in seen: continue
        seen.add(key); out.append(build_candidate(segs, det, "joined"))
        if len(out) >= cfg["candidates"]["joined_max"]: break
    return out
