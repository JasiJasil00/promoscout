"""Integration test on the real pack. Skipped unless PROMOSCOUT_PACK points at the extracted pack folder or ZIP."""
import os, pytest
from helpers import *
from promoscout.pipeline import run_pipeline

PACK = os.environ.get("PROMOSCOUT_PACK")
pytestmark = pytest.mark.skipif(not PACK, reason="set PROMOSCOUT_PACK to run")

def test_full_pack_deterministic():
    r = run_pipeline(PACK, CFG, None, "pack")
    assert len(r.episodes) == 50 and not r.ingest.errors
    assert len(r.approved) == CFG["selection"]["approved_target"]
    assert all(20 <= c.duration <= 60 for c in r.candidates)
    assert not [c for c in r.approved if any(f.severity == "error" for f in c.flags)]
