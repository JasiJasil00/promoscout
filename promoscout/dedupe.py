from __future__ import annotations
import math, re
from collections import Counter
from .models import Candidate
from .features import STOP


def _iou(a: Candidate, b: Candidate) -> float:
    inter = union = 0.0
    for sa in a.segments:
        for sb in b.segments:
            if sa.episode == sb.episode:
                inter += max(0.0, min(sa.end, sb.end) - max(sa.start, sb.start))
    union = a.duration + b.duration - inter
    return inter / union if union > 0 else 0.0


def _vec(text: str) -> Counter:
    # use only the spoken/action content, not ids or timestamps
    body = " ".join(l.split(": ", 1)[1] for l in text.split("\n") if ": " in l)
    return Counter(w for w in re.findall(r"[a-z']+", body.lower()) if w not in STOP and len(w) > 2)


def _tfidf(vecs: list[Counter]) -> list[dict]:
    n = len(vecs); df = Counter(w for v in vecs for w in v)
    out = []
    for v in vecs:
        d = {w: tf * (math.log((1 + n) / (1 + df[w])) + 1.0) for w, tf in v.items()}
        norm = math.sqrt(sum(x * x for x in d.values())) or 1.0
        out.append({w: x / norm for w, x in d.items()})
    return out


def _cos(a: dict, b: dict) -> float:
    if len(a) > len(b): a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())


def mark_duplicates(cands: list[Candidate], iou_thr: float, cos_thr: float) -> None:
    """Higher-scoring candidate wins. Duplicate = heavy time overlap OR near-identical text."""
    order = sorted(range(len(cands)), key=lambda i: -cands[i].scores.total)
    vecs = _tfidf([_vec(c.text) for c in cands])
    kept: list[int] = []
    for i in order:
        c = cands[i]
        for j in kept:
            k = cands[j]
            iou = _iou(c, k); cs = _cos(vecs[i], vecs[j])
            if iou >= iou_thr or cs >= cos_thr:
                c.status, c.duplicate_of = "duplicate", k.candidate_id
                from .models import AuditFlag
                c.flags.append(AuditFlag(code="duplicate", severity="warn",
                                         message=f"overlaps {k.candidate_id} (time IoU {iou:.2f}, text cosine {cs:.2f})"))
                break
        else:
            kept.append(i)
