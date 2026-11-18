import json
from helpers import *
from promoscout.candidates import generate_windows
from promoscout.judge import judge_candidates
from promoscout.llm.base import LLMClient, LLMError
from promoscout.llm.mock import MockClient
from promoscout.scoring import apply_scores
from promoscout.audit import audit_candidate
from promoscout.features import score_blocks
from promoscout.analysis import analyze_episodes

def setup(n=1):
    eps = [episode_from(dialogue_rows(40, names=("JULIANA", "GERALD", "CLARISSA")), num=i + 1, scene_end=125) for i in range(n)]
    return eps, {e.number: e for e in eps}, generate_windows(eps, CFG)[:3]

class Scripted(LLMClient):
    name = "scripted"
    def __init__(self, fn, **kw): super().__init__(cache_dir=None, **kw); self.fn = fn; self.calls = 0
    def _call(self, s, u): self.calls += 1; return self.fn(self.calls, u)

def good(u, mutate=None):
    out = json.loads(MockClient()._call("", u))
    if mutate: mutate(out["results"])
    return json.dumps(out)

def test_mock_roundtrip_fills_judgments():
    _, em, cs = setup(); judge_candidates(cs, em, MockClient(), "T", 8)
    assert all(c.llm is not None for c in cs)

def test_fabricated_quote_and_bad_block_reference_flagged_as_errors():
    _, em, cs = setup()
    def mut(r): r[0]["hook"]["evidence_quote"] = "A line the screenplay never contains"; r[0]["cliffhanger"]["block_id"] = "E099-S01-B001"
    cl = Scripted(lambda n, u: good(u, mut)); judge_candidates(cs, em, cl, "T", 8)
    apply_scores(cs[0], CFG["weights"], 4); audit_candidate(cs[0], CFG)
    codes = {f.code for f in cs[0].flags if f.severity == "error"}
    assert codes == {"unsupported_interpretation", "invalid_reference"}

def test_model_supplied_timestamps_are_ignored():
    _, em, cs = setup(); before = [c.segments[0].model_dump() for c in cs]
    def mut(r):
        for x in r: x["start"] = 1.0; x["end"] = 999.0; x["segments"] = [{"start": 5, "end": 500}]
    judge_candidates(cs, em, Scripted(lambda n, u: good(u, mut)), "T", 8)
    assert [c.segments[0].model_dump() for c in cs] == before

def test_retry_on_transient_failure_then_success():
    _, em, cs = setup(); state = {"n": 0}
    def fn(n, u):
        if n < 3: raise LLMError("HTTP 429")
        return good(u)
    cl = Scripted(fn, max_retries=4); judge_candidates(cs, em, cl, "T", 8)
    assert all(c.llm for c in cs) and cl.calls == 3

def test_permanent_failure_reported_not_raised():
    _, em, cs = setup(); cl = Scripted(lambda n, u: "not json at all", max_retries=2)
    notes = judge_candidates(cs, em, cl, "T", 8)
    assert notes and all(c.llm is None for c in cs)

def test_candidate_skipped_by_model_gets_individual_retry():
    _, em, cs = setup()
    def fn(n, u):
        if n == 1: return good(u, lambda r: r.pop())     # drop the last candidate in the first batch
        return good(u)
    cl = Scripted(fn); judge_candidates(cs, em, cl, "T", 8)
    assert all(c.llm for c in cs) and cl.calls == 2

def test_invalid_schema_score_out_of_range_rejected():
    _, em, cs = setup()
    def mut(r): r[0]["hook"]["score"] = 42
    notes = judge_candidates(cs, em, Scripted(lambda n, u: good(u, mut), max_retries=1), "T", 8)
    assert any("invalid judgment" in n for n in notes)

# ---- narrative-context tests
def test_continuation_episode_lowers_clarity_at_episode_open():
    rows = dialogue_rows(20, names=("JULIANA", "GERALD"))
    a = episode_from(rows, scene_end=65, continuity="First episode.")
    b = episode_from(rows, scene_end=65, continuity="Picks up immediately from Episode 4.")
    f = lambda ep: score_blocks([x for x in ep.blocks if x.start < 40], 0.0, 40.0, ep, [0.0])
    assert f(b).clarity < f(a).clarity

def test_unnamed_speaker_and_alias_resolution():
    rows = [("DLG", "THE FORMER GROOM (Cold)", 1, 3, "I changed my mind."), ("DLG", "LADY IN GOLD GOWN", 4, 6, "Because he is marrying me."),
            ("DLG", "MYSTERIOUS BEARDED MAN", 7, 9, "Stand back.")]
    ep = episode_from(rows, scene_end=10)
    assert [b.character for b in ep.blocks] == ["Gerald", "Clarissa", "Alexander"]
    assert [b.hidden_identity for b in ep.blocks] == [False, False, True]

def test_episode_analysis_tags_moments_and_tracks_characters_across_episodes():
    r1 = [("DLG", "JULIANA", 1, 3, "You betrayed me and lied to me.")]
    r2 = [("DLG", "JULIANA", 1, 3, "The truth is a secret.")]
    eps = [episode_from(r1, num=1, scene_end=5), episode_from(r2, num=2, scene_end=5)]
    a = analyze_episodes(eps)
    assert a["episodes"][0]["moment_counts"]["betrayal"] >= 1 and a["episodes"][1]["moment_counts"]["secret/revelation"] >= 1
    assert a["characters"]["Juliana"]["episodes"] == [1, 2]
