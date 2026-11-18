from helpers import *
from promoscout.parser import parse_episode

def test_basic_parse_and_traceability():
    ep = episode_from(dialogue_rows(6), scene_end=20)
    assert ep.number == 1 and len(ep.scenes) == 1
    d = [b for b in ep.blocks if b.kind == "dialogue"]
    assert d[0].speaker_raw == "JULIANA" and d[0].tone == "Furious" and d[0].character == "Juliana"
    assert d[0].text.startswith("Line number") and d[0].src_line > 0 and d[0].block_id.startswith("E001-S01-B")

def test_uppercase_labels_episode6_variant():
    ep = episode_from(dialogue_rows(6), scene_end=20, upper=True)
    assert len(ep.blocks) > 0 and ep.scenes[0].end == 20 and ep.scenes[0].heading

def test_missing_timestamps_block_skipped_with_warning():
    t = script(1, dialogue_rows(4), scene_end=12).replace("Start: (00:00.20) | End: (00:02.40)", "")
    ep = parse_episode(t, "x.txt", 1)
    assert any("no valid Start/End" in w for w in ep.warnings)
    assert len(ep.blocks) >= 1

def test_end_before_start_skipped():
    rows = dialogue_rows(4) + [("DLG", "JULIANA", 9.0, 8.0, "Backwards line.")]
    ep = episode_from(rows, scene_end=12)
    assert not any(b.text == "Backwards line." for b in ep.blocks)
    assert any("end (" in w and "< start" in w for w in ep.warnings)

def test_missing_scene_times_inferred():
    t = script(1, dialogue_rows(4), scene_end=12).replace("Scene end:", "Scene xend:")
    ep = parse_episode(t, "x.txt", 1)
    assert any("missing Scene end" in w for w in ep.warnings) and ep.duration > 0

def test_block_outside_scene_warns():
    rows = dialogue_rows(4) + [("DLG", "GERALD", 20.0, 25.0, "Too late.")]
    ep = episode_from(rows, scene_end=12)
    assert any("outside scene" in w for w in ep.warnings)

def test_scene_gap_warns():
    sc = [(1, "A", 0.0, 10.0, dialogue_rows(3)), (2, "B", 12.0, 20.0, dialogue_rows(3, start=12.0))]
    ep = parse_episode(script(1, scenes=sc), "x.txt", 1)
    assert any("gap/overlap" in w for w in ep.warnings)

def test_garbage_and_empty_do_not_crash():
    for txt in ["", "random words\nmore words", "**CAMERA:**\nnot a timing line\nfoo"]:
        ep = parse_episode(txt, "bad.txt", 5)
        assert ep.blocks == [] and ep.warnings

def test_speaker_label_with_parentheticals():
    rows = [("DLG", "MATRIARCH (O.S.) (Firm)", 1.0, 3.0, "Clarissa is the jewel."), ("DLG", "THE BEARDED STRANGER (Calm)", 4.0, 6.0, "Hello.")]
    ep = episode_from(rows, scene_end=8)
    a, b = ep.blocks
    assert a.speaker_raw == "MATRIARCH" and a.tone == "Firm"
    assert b.character == "Alexander" and b.hidden_identity

def test_blocks_sorted_by_start_even_if_file_order_is_not():
    rows = [("DLG", "JULIANA", 5.0, 6.0, "second"), ("DLG", "GERALD", 1.0, 2.0, "first")]
    ep = episode_from(rows, scene_end=8)
    assert [b.text for b in ep.blocks] == ["first", "second"]
