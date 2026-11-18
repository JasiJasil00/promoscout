"""Tolerant screenplay parser. Never raises on bad input: problems become Episode.warnings."""
from __future__ import annotations
import re
from .models import Block, Scene, Episode

TS = r"(\d{1,2}):(\d{2}(?:\.\d+)?)"
RE_TIMING = re.compile(rf"^\s*start:\s*\(?{TS}\)?\s*\|\s*end:\s*\(?{TS}\)?\s*$", re.I)
RE_BLOCK = re.compile(r"^\s*\*\*(.+?):?\*\*:?\s*$")
RE_SCENE = re.compile(r"^\s*scene\s+(\d+)\s*$", re.I)
RE_HEADING = re.compile(r"^\s*scene heading:\s*(.*)$", re.I)
RE_SSTART = re.compile(rf"^\s*scene start:\s*{TS}\s*$", re.I)
RE_SEND = re.compile(rf"^\s*scene end:\s*{TS}\s*$", re.I)
RE_EPNUM = re.compile(r"^\s*episode number:\s*(?:episode\s*)?(\d+)", re.I)
RE_LOG = re.compile(r"^\s*episode logline:\s*(.*)$", re.I)
RE_CONT = re.compile(r"^\s*continuity notes:\s*(.*)$", re.I)
RE_ACTDLG = re.compile(r"^\s*actions? and dialogue:?\s*$", re.I)
RE_STARTMARK = re.compile(r"^\s*-{2,}\s*script start\s*-{2,}\s*$", re.I)
RE_ENDMARK = re.compile(r"^\s*-{2,}\s*script end\s*-{2,}\s*$", re.I)
NONTONE = {"O.S.", "V.O.", "CONT'D", "OFF-SCREEN", "OFFSCREEN", "O.C."}
EPS = 0.05


def to_sec(m: str, s: str) -> float:
    return int(m) * 60 + float(s)


def _split_label(label: str) -> tuple[str, str | None]:
    """'MATRIARCH (O.S.) (Firm)' -> ('MATRIARCH', 'Firm')."""
    parens = re.findall(r"\(([^)]*)\)", label)
    name = re.sub(r"\([^)]*\)", "", label).strip()
    tones = [p.strip() for p in parens if p.strip().upper() not in NONTONE]
    return name, (tones[-1] if tones else None)


def parse_episode(text: str, filename: str, fallback_number: int | None = None,
                  aliases: dict[str, str] | None = None, hidden: set[str] | None = None) -> Episode:
    aliases, hidden = aliases or {}, hidden or set()
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ep = Episode(number=fallback_number or 0, filename=filename)
    warn = ep.warnings.append
    got_num = False
    scene: Scene | None = None
    scene_declared_end: dict[int, float | None] = {}
    scene_declared_start: dict[int, float | None] = {}
    cur_idx = 0
    in_script = False
    saw_start_marker = False
    bcount = 0
    i, n = 0, len(lines)

    def ensure_scene(lineno: int) -> Scene:
        nonlocal scene, cur_idx
        if scene is None:
            cur_idx += 1
            warn(f"line {lineno}: content before any 'SCENE NN' marker; created implicit scene {cur_idx}")
            scene = Scene(episode=ep.number, index=cur_idx, heading="(unknown)", start=0.0, end=0.0)
            scene_declared_start[cur_idx] = None
            scene_declared_end[cur_idx] = None
            ep.scenes.append(scene)
        return scene

    while i < n:
        raw = lines[i]
        ln = i + 1
        m = RE_EPNUM.match(raw)
        if m and not in_script:
            ep.number = int(m.group(1)); got_num = True
        elif RE_LOG.match(raw) and not in_script:
            ep.logline = RE_LOG.match(raw).group(1).strip()
        elif RE_CONT.match(raw) and not in_script:
            ep.continuity = RE_CONT.match(raw).group(1).strip()
        elif RE_STARTMARK.match(raw):
            in_script = True; saw_start_marker = True
        elif RE_ENDMARK.match(raw):
            in_script = False
        elif RE_SCENE.match(raw):
            in_script = True
            cur_idx = int(RE_SCENE.match(raw).group(1))
            scene = Scene(episode=ep.number, index=cur_idx, heading="", start=0.0, end=0.0)
            scene_declared_start[cur_idx] = None; scene_declared_end[cur_idx] = None
            ep.scenes.append(scene)
        elif RE_HEADING.match(raw) and scene is not None:
            scene.heading = RE_HEADING.match(raw).group(1).strip()
        elif RE_SSTART.match(raw) and scene is not None:
            mm = RE_SSTART.match(raw); scene_declared_start[cur_idx] = to_sec(mm.group(1), mm.group(2))
        elif RE_SEND.match(raw) and scene is not None:
            mm = RE_SEND.match(raw); scene_declared_end[cur_idx] = to_sec(mm.group(1), mm.group(2))
        elif RE_ACTDLG.match(raw):
            pass
        elif RE_BLOCK.match(raw):
            label = RE_BLOCK.match(raw).group(1).strip()
            sc = ensure_scene(ln)
            # next non-empty line must be the timing line
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            tm = RE_TIMING.match(lines[j]) if j < n else None
            # collect text lines until blank/next marker
            k = (j + 1) if tm else (i + 1)
            body: list[str] = []
            while k < n:
                t = lines[k]
                if (not t.strip() or RE_BLOCK.match(t) or RE_SCENE.match(t) or RE_ENDMARK.match(t)
                        or RE_HEADING.match(t) or RE_SSTART.match(t) or RE_SEND.match(t)):
                    break
                body.append(t.strip()); k += 1
            if not tm:
                warn(f"line {ln}: block '{label}' has no valid Start/End line; block skipped")
                i = max(k, i + 1); continue
            start = to_sec(tm.group(1), tm.group(2)); end = to_sec(tm.group(3), tm.group(4))
            text_ = " ".join(body).strip()
            if end < start:
                warn(f"line {ln}: block '{label}' end ({end}) < start ({start}); block skipped")
                i = k; continue
            if not text_:
                warn(f"line {ln}: block '{label}' has empty text; block skipped")
                i = k; continue
            base, tone = _split_label(label)
            bu = base.upper()
            if bu in ("CAMERA", "ACTION", "SOUND FX"):
                kind = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sfx"}[bu]; speaker = None
            else:
                kind = "dialogue"; speaker = bu
                if len(text_) >= 2 and text_[0] == '"' and text_[-1] == '"':
                    text_ = text_[1:-1]
            bcount += 1
            canon = None
            if speaker:
                canon = aliases.get(speaker) or speaker.title()
            sc.blocks.append(Block(
                block_id=f"E{ep.number:03d}-S{sc.index:02d}-B{len(sc.blocks)+1:03d}", episode=ep.number,
                scene=sc.index, kind=kind, speaker_raw=speaker, character=canon,
                hidden_identity=bool(speaker and speaker in hidden), tone=tone, start=start, end=end,
                text=text_, src_line=ln))
            i = k; continue
        i += 1

    # ---- finalize / validate
    if not got_num:
        warn("missing 'EPISODE NUMBER'; using number from filename" if fallback_number else "missing 'EPISODE NUMBER'")
    if not saw_start_marker:
        warn("missing '--- SCRIPT START ---' marker")
    if not ep.scenes:
        warn("no scenes found; episode is empty")
    for sc in ep.scenes:
        sc.episode = ep.number
        for b in sc.blocks:
            b.episode = ep.number
            b.block_id = f"E{ep.number:03d}-S{sc.index:02d}-B{int(b.block_id[-3:]):03d}"
        ds, de = scene_declared_start.get(sc.index), scene_declared_end.get(sc.index)
        bs = min((b.start for b in sc.blocks), default=0.0)
        be = max((b.end for b in sc.blocks), default=0.0)
        if not sc.heading:
            warn(f"scene {sc.index}: missing heading"); sc.heading = "(unknown)"
        if ds is None:
            warn(f"scene {sc.index}: missing Scene start; inferred {bs:.2f}"); ds = bs
        if de is None:
            warn(f"scene {sc.index}: missing Scene end; inferred {be:.2f}"); de = be
        if de < ds:
            warn(f"scene {sc.index}: end < start; using block extent"); ds, de = bs, be
        sc.start, sc.end = ds, de
        for b in sc.blocks:
            if b.start < sc.start - EPS or b.end > sc.end + EPS:
                warn(f"{b.block_id}: timing {b.start}-{b.end} outside scene {sc.start}-{sc.end}")
        sc.blocks.sort(key=lambda b: (b.start, b.end))
    ep.scenes.sort(key=lambda s: s.start)
    for a, b in zip(ep.scenes, ep.scenes[1:]):
        if abs(b.start - a.end) > EPS:
            warn(f"scene {a.index}->{b.index}: gap/overlap ({a.end:.2f} vs {b.start:.2f})")
    ep.duration = max((s.end for s in ep.scenes), default=0.0)
    return ep
