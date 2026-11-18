from promoscout.parser import parse_episode
from promoscout.ingest import load_episodes
from promoscout.config import load_config, load_aliases

def ts(t):
    m, s = divmod(t, 60); return f"{int(m):02d}:{s:05.2f}"

def script(num=1, rows=None, scene_end=None, heading="INT. HALL – DAY", upper=False, continuity="First episode.", scenes=None):
    """rows: (kind, label, start, end, text). kind in CAMERA/ACTION/SOUND FX/DLG."""
    H = (lambda s: s.upper()) if upper else (lambda s: s)
    out = [f"EPISODE NUMBER: Episode {num}", "EPISODE LOGLINE: A test logline.", f"CONTINUITY NOTES: {continuity}", "",
           "--- SCRIPT START ---", ""]
    scenes = scenes or [(1, heading, 0.0, scene_end, rows)]
    for idx, head, s0, s1, rws in scenes:
        end = s1 if s1 is not None else max(r[3] for r in rws)
        out += [f"SCENE {idx:02d}", f"{H('Scene heading')}: {head}", f"{H('Scene start')}: {ts(s0)}", f"{H('Scene end')}: {ts(end)}", "",
                H("Actions and dialogue") + ":", ""]
        for kind, label, a, b, text in rws:
            lab = {"CAMERA": "CAMERA", "ACTION": "ACTION", "SOUND FX": "SOUND FX"}.get(kind, label)
            body = f'"{text}"' if kind == "DLG" else text
            out += [f"**{lab}:**", f"Start: ({ts(a)}) | End: ({ts(b)})", body, ""]
    out.append("--- SCRIPT END ---")
    return "\n".join(out)

def dialogue_rows(n=24, step=3.0, names=("JULIANA", "GERALD"), start=0.0):
    rows = []
    for i in range(n):
        a = start + i * step
        q = "Why did you lie to me?" if i % 5 == 4 else f"Line number {i} about the betrayal and the secret."
        rows.append(("DLG", names[i % len(names)] + " (Furious)", a + 0.2, a + 2.4, q))
        if i % 4 == 0:
            rows.append(("CAMERA", "", a, a + step, f"Close-up {i}."))
    return rows

def episode_from(rows, num=1, **kw):
    cfg = load_config(); al, hid = load_aliases(cfg)
    return parse_episode(script(num, rows, **kw), f"episode-{num:03d}.txt", num, al, hid)

CFG = load_config()
