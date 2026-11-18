from helpers import *
from promoscout.candidates import generate_windows
from promoscout.dedupe import mark_duplicates
from promoscout.scoring import apply_scores, select_and_rank
from promoscout.features import weighted_total
from promoscout.models import AuditFlag

def prep(n=2):
    eps = [episode_from(dialogue_rows(40, names=("JULIANA", "GERALD", "CLARISSA")), num=i + 1, scene_end=125) for i in range(n)]
    cs = generate_windows(eps, CFG)
    for c in cs: apply_scores(c, CFG["weights"], 4)
    return eps, cs

def test_weighted_total_matches_hand_calculation():
    # 0.30*8 + 0.30*6 + 0.25*5 + 0.15*4 = 2.4+1.8+1.25+0.6 = 6.05 -> 60.5
    assert weighted_total(8, 6, 5, 4, CFG["weights"]) == 60.5
    assert weighted_total(10, 10, 10, 10, CFG["weights"]) == 100.0 and weighted_total(0, 0, 0, 0, CFG["weights"]) == 0.0

def test_time_overlap_marked_duplicate_keeps_higher_score():
    _, cs = prep(1)
    a = cs[0]; b = a.model_copy(deep=True); b.candidate_id = "dup"; b.scores.total = a.scores.total - 1
    mark_duplicates([a, b], 0.3, 0.99)
    assert b.status == "duplicate" and b.duplicate_of == a.candidate_id and a.status == "candidate"

def test_identical_text_in_different_episodes_is_duplicate():
    _, cs = prep(2)
    a = cs[0]; b = a.model_copy(deep=True); b.candidate_id = "x"; b.segments[0].episode = 2; b.scores.total -= 1
    mark_duplicates([a, b], 0.3, 0.6)
    assert b.status == "duplicate"

def test_distinct_candidates_not_duplicates():
    _, cs = prep(2)
    a = next(c for c in cs if c.segments[0].episode == 1)
    far = next(c for c in cs if c.segments[0].episode == 2)
    far = far.model_copy(deep=True); far.text = "[E002-S01-B001] 0-1 X: entirely different words appear here regarding crown estate inheritance"
    mark_duplicates([a, far], 0.3, 0.6)
    assert far.status == "candidate"

def test_ranking_orders_by_score_with_diversity_and_tiers():
    _, cs = prep(2)
    select_and_rank(cs, CFG)   # dedupe is tested separately; identical synthetic episodes would all be duplicates
    app = sorted([c for c in cs if c.status == "approved"], key=lambda c: c.rank)
    assert app and [c.rank for c in app] == list(range(1, len(app) + 1))
    assert app[0].scores.total == max(c.scores.total for c in cs if c.status == "approved")
    assert app[0].tier == "A"
    assert {c.segments[0].episode for c in app[:2]} == {1, 2}        # diversity penalty spreads across episodes

def test_error_flag_candidates_are_never_approved():
    _, cs = prep(1)
    cs[0].flags.append(AuditFlag(code="invalid_timestamp", severity="error", message="x"))
    select_and_rank(cs, CFG)
    assert cs[0].status == "rejected" and cs[0].rank is None
