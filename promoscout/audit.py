"""Quality auditor: turns scores and judgments into explicit, reviewable flags."""
from __future__ import annotations
import re
from .models import AuditFlag, Candidate

norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[“”\"]", "", s)).strip().lower()


def audit_candidate(c: Candidate, cfg: dict) -> None:
    a = cfg["audit"]; s = c.scores; add = lambda code, sev, msg: c.flags.append(AuditFlag(code=code, severity=sev, message=msg))
    if s.hook < a["weak_hook"]: add("weak_hook", "warn", f"hook score {s.hook:.1f} < {a['weak_hook']}")
    if s.cliffhanger < a["weak_cliffhanger"]: add("missing_cliffhanger", "warn", f"cliffhanger score {s.cliffhanger:.1f} < {a['weak_cliffhanger']}")
    if s.clarity < a["poor_clarity"]: add("poor_clarity", "warn", f"standalone clarity {s.clarity:.1f} < {a['poor_clarity']}")
    d = c.det
    if c.llm is not None:
        l = c.llm
        if l.clarity.missing_context: add("missing_context", "warn", "newcomer lacks: " + "; ".join(l.clarity.missing_context[:3]))
        if l.unsupported_claims: add("unsupported_interpretation", "warn", "model flagged unsupported claims: " + "; ".join(l.unsupported_claims[:2]))
        text_n = norm(c.text)
        for dim, ev in (("hook", l.hook.evidence_quote), ("cliffhanger", l.cliffhanger.evidence_quote), ("emotion", l.emotion.evidence_quote)):
            if ev and norm(ev) not in text_n:
                add("unsupported_interpretation", "error", f"{dim} evidence quote is not in the clip text (possible fabrication): '{ev[:60]}'")
        ids = {b for sg in c.segments for b in sg.block_ids}
        for dim, bid in (("hook", l.hook.block_id), ("cliffhanger", l.cliffhanger.block_id)):
            if bid and bid not in ids: add("invalid_reference", "error", f"{dim} cites block {bid}, which is not in this clip")
        gap = {k: abs(getattr(l, k).score - getattr(d, k)) for k in ("hook", "cliffhanger", "clarity", "emotion")}
        worst = max(gap, key=gap.get)
        if gap[worst] >= a["disagreement"]:
            add("llm_heuristic_disagreement", "info", f"LLM and heuristic differ by {gap[worst]:.1f} on {worst}")
        if l.verdict == "reject" and s.total >= 50: add("possible_false_positive", "warn", "scores are high but the model verdict is 'reject'")
        if c.origin == "joined" and l.pair_coherent is False: add("incoherent_join", "error", "model judged the two segments do not form a coherent promo: " + l.pair_reason[:120])
    if s.hook >= 7 and s.clarity < 5: add("possible_false_positive", "warn", "strong hook but a newcomer may not understand it")
    if s.source == "deterministic": add("heuristic_only", "info", "scored by keyword heuristics only; no LLM judgment")
    if c.origin == "joined": add("multi_segment", "info", "joined from separate moments; check the cut feels natural")
