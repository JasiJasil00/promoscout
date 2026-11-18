import copy
from helpers import *
from promoscout.candidates import generate_windows, generate_joined, make_segment, build_candidate
from promoscout.validate import validate_candidate
from promoscout.features import score_blocks
from promoscout.models import Candidate, Segment

def eps_of(n=3, **kw):
    return [episode_from(dialogue_rows(40, names=("JULIANA", "GERALD", "CLARISSA")), num=i + 1, scene_end=125, **kw) for i in range(n)]

def test_all_windows_obey_duration_and_never_cut_dialogue():
    eps = eps_of(); cands = generate_windows(eps, CFG); em = {e.number: e for e in eps}
    assert cands
    for c in cands:
        assert 20 <= c.duration <= 60
        assert not [f for f in validate_candidate(c, em, 20, 60) if f.severity == "error"], c.candidate_id

def test_validator_catches_duration_and_fabricated_timestamps():
    eps = eps_of(1); em = {1: eps[0]}
    c = generate_windows(eps, CFG)[0]
    short = c.model_copy(deep=True); short.segments[0].end = short.segments[0].start + 10; short.duration = 10
    assert any(f.code == "bad_duration" for f in validate_candidate(short, em, 20, 60))
    fake = c.model_copy(deep=True); fake.segments[0].end = 9999; fake.duration = 9999 - fake.segments[0].start
    codes = {f.code for f in validate_candidate(fake, em, 20, 60)}
    assert "invalid_timestamp" in codes
    ghost = c.model_copy(deep=True); ghost.segments[0].block_ids = ["E001-S01-B999"]
    assert any("unknown block" in f.message for f in validate_candidate(ghost, em, 20, 60))
    nodup = c.model_copy(deep=True); nodup.segments[0].episode = 42
    assert any("does not exist" in f.message for f in validate_candidate(nodup, em, 20, 60))

def test_validator_catches_cut_inside_spoken_line():
    eps = eps_of(1); em = {1: eps[0]}; c = generate_windows(eps, CFG)[0].model_copy(deep=True)
    d = next(b for b in eps[0].blocks if b.kind == "dialogue" and b.start > c.segments[0].start + 5)
    c.segments[0].end = round((d.start + d.end) / 2, 2); c.duration = c.segments[0].end - c.segments[0].start
    assert any(f.code == "cut_inside_dialogue" for f in validate_candidate(c, em, 1, 99))

def test_multi_segment_valid_and_order_and_total():
    eps = eps_of(3); em = {e.number: e for e in eps}
    j = generate_joined(eps, CFG)
    assert j and all(c.origin == "joined" and len(c.segments) == 2 for c in j)
    for c in j:
        assert not [f for f in validate_candidate(c, em, 20, 60) if f.severity == "error"], c.candidate_id
        assert 20 <= c.duration <= 60
    bad = j[0].model_copy(deep=True); bad.segments = bad.segments[::-1]
    assert any("out of order" in f.message for f in validate_candidate(bad, em, 20, 60))
    over = j[0].model_copy(deep=True); over.segments[1] = over.segments[0].model_copy(); 
    assert any(f.severity == "error" for f in validate_candidate(over, em, 20, 60))

def test_stated_duration_mismatch_flagged():
    eps = eps_of(1); c = generate_windows(eps, CFG)[0].model_copy(deep=True); c.duration += 5
    assert any("stated duration" in f.message for f in validate_candidate(c, {1: eps[0]}, 20, 60))
