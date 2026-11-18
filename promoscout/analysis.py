"""Episode-level narrative analysis (deterministic). Tags dramatic moments and tracks characters across episodes."""
from __future__ import annotations
import re
from collections import defaultdict
from .models import Episode

CATS = {
    "betrayal": r"betray|traitor|lied to|deceiv|you knew|behind my back|abandon|cheat",
    "secret/revelation": r"secret|truth|reveal|hidden|confess|identity|true name|all along|never told|discover",
    "confrontation": r"how dare|get out|enough|you will pay|stand down|face me|accus|defy|silence|guards?|seize",
    "emotional breakdown": r"tears|sobb|weep|cry|crying|collaps|trembl|broken|shatter|beg|plead",
    "romantic tension": r"kiss|embrace|heartbeat|closer|lips|touch|beloved|my wife|my husband|devotion|hold you",
    "suspense": r"footsteps|shadow|lurk|whisper|creak|silence|watching|sneak|dagger|poison|arsenic|antidote",
    "unexpected event": r"suddenly|bursts|crash|gasp|shock|stuns|appears|arrives|ambush|barges",
    "power shift": r"duke|duchess|title|throne|heir|inherit|command|authority|council|crown|rank|kneel",
}
CAT_RE = {k: re.compile(v, re.I) for k, v in CATS.items()}


def analyze_episodes(episodes: list[Episode]) -> dict:
    per, chars = [], defaultdict(lambda: {"first_episode": None, "last_episode": None, "episodes": set(), "lines": 0})
    for ep in episodes:
        moments, c_lines = [], defaultdict(int)
        for b in ep.blocks:
            if b.kind == "dialogue" and b.character:
                c = chars[b.character]
                c["first_episode"] = ep.number if c["first_episode"] is None else c["first_episode"]
                c["last_episode"] = ep.number; c["episodes"].add(ep.number); c["lines"] += 1; c_lines[b.character] += 1
            hay = b.text + (" " + b.tone if b.tone else "")
            for cat, rx in CAT_RE.items():
                if b.kind in ("dialogue", "action", "sfx") and rx.search(hay):
                    moments.append({"block_id": b.block_id, "start": b.start, "end": b.end, "category": cat,
                                    "quote": b.text[:160]})
        last_d = next((b for b in reversed(ep.blocks) if b.kind == "dialogue"), None)
        per.append({
            "episode": ep.number, "logline": ep.logline, "continuity": ep.continuity, "duration": ep.duration,
            "scenes": [{"index": s.index, "heading": s.heading, "start": s.start, "end": s.end} for s in ep.scenes],
            "characters": dict(sorted(c_lines.items(), key=lambda x: -x[1])),
            "moments": moments, "moment_counts": {k: sum(1 for m in moments if m["category"] == k) for k in CATS},
            "ending_line": (last_d.text if last_d else ""), "ending_is_question": bool(last_d and last_d.text.strip().endswith("?")),
            "warnings": ep.warnings,
        })
    story = {k: {**v, "episodes": sorted(v["episodes"])} for k, v in chars.items()}
    return {"episodes": per, "characters": story, "method": "deterministic keyword tagging (weak; see README)"}
