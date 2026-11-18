"""Deterministic (no-LLM) scoring signals. These are CRUDE, lexicon-based heuristics.
They exist to (1) shortlist candidates cheaply, (2) work with no API key, and
(3) give an independent baseline to compare the LLM against. Every score carries its reasons."""
from __future__ import annotations
import re
from .models import Block, DetFeatures, Episode

W = lambda s: set(s.split())
SHOCK = W("betray betrayed lie lied lies liar secret truth married marry pregnant poison poisoned kill killed dead death "
          "murder traitor divorce arrest arrested guilty impostor swear swore knew bastard whore wedding cheat cheated "
          "betrayal confess confession fraud steal stole blood scar duke duchess forbid banish banished leave")
THREAT = W("kill die death regret warn warning dare destroy ruin expose reveal truth secret poison arrest hang pay "
           "never again punish beg mercy")
RELATION = W("sister brother husband wife mother father fiancé fiance daughter son duke duchess king queen lord lady "
             "bride groom family son-in-law mistress lover")
STAKES = W("inherit inheritance title throne marry wedding death trial council hearing war crown estate heir antidote "
           "arsenic poison fortune land")
INTENSE_TONE = W("furious outraged desperate sobbing crying screaming shaken trembling devastated enraged horrified "
                 "broken pleading begging shocked defiant cold threatening gasping anguished heartbroken betrayed raw "
                 "menacing terrified panicked venomous seething vicious stunned incredulous commanding piercing "
                 "accusing wounded fierce")
ACTION_SHOCK = W("slap slaps slapped strikes struck draws drew blade sword dagger blood collapses falls gasps stumbles "
                 "shoves shoved grabs seizes tears bleeding clutches trembles clenches kneels kicks")
TENSION_FX = W("gasp ominous cliffhanger freezes frozen stunned shock sudden footsteps slam thud scream drone swells "
               "black silence darkness shadows lurks horrified")
RESOLVED = ("thank you", "goodnight", "farewell", "i love you", "sleep well", "at last", "forgive me")
GENERIC_WORDS = W("NOBLEMAN NOBLEWOMAN MAID LORD JUDGE LADY MAN WOMAN GUARD OFFICIAL INFORMANT THUG FIGHTER VILLAGER "
                  "MOTHER HOUSEKEEPER CONSTABLE PRIEST MATRIARCH ARISTOCRATIC ELDER BLOND HIGH ROYAL CROWN PEASANT")
STOP = W("the a an and or but to of in on at for with is are was were be been it this that you your i me my we our he him "
         "his she her they them their not no yes do does did have has had will would can could shall as by from so if then "
         "than there what who why how when where just very too")

tok = lambda s: re.findall(r"[a-zA-Z'’\-]+", s.lower())
clip = lambda x: max(0.0, min(10.0, x))


def _is_proper(label: str | None) -> bool:
    return bool(label) and not (set(label.split()) & GENERIC_WORDS)


def score_blocks(blocks: list[Block], s: float, e: float, ep: Episode, scene_starts: list[float]) -> DetFeatures:
    dlg = [b for b in blocks if b.kind == "dialogue"]
    sig: dict[str, list[str]] = {"hook": [], "cliffhanger": [], "clarity": [], "emotion": []}

    # ---------------- HOOK: first 3 seconds
    first = [b for b in blocks if b.start < s + 3.0]
    fd = [b for b in first if b.kind == "dialogue"]
    hook = 1.0
    hook_b = fd[0] if fd else (next((b for b in first if b.kind == "action"), first[0] if first else None))
    if fd:
        t = fd[0].text; words = tok(t)
        hook += 2.5; sig["hook"].append("dialogue begins within first 3s")
        if re.search(r"[?!]", t): hook += 1.2; sig["hook"].append("opening line is a question/exclamation")
        hits = set(words) & SHOCK
        if hits: hook += min(2.5, 1.2 * len(hits)); sig["hook"].append(f"shock vocabulary: {sorted(hits)[:3]}")
        if len(words) <= 8: hook += 0.8; sig["hook"].append("short punchy opening line")
        if fd[0].tone and set(tok(fd[0].tone)) & INTENSE_TONE: hook += 1.5; sig["hook"].append(f"intense tone: {fd[0].tone}")
    else:
        sig["hook"].append("no dialogue in first 3s")
    ac = set(w for b in first if b.kind == "action" for w in tok(b.text)) & ACTION_SHOCK
    if ac: hook += 1.5; sig["hook"].append(f"physical/shock action: {sorted(ac)[:3]}")

    # ---------------- CLIFFHANGER: last ~6 seconds
    tail = [b for b in blocks if b.end > e - 6.0]
    td = [b for b in tail if b.kind == "dialogue"]
    cliff = 1.0
    cliff_b = td[-1] if td else (tail[-1] if tail else None)
    if td:
        last = td[-1].text.strip(); lw = set(tok(last))
        if last.endswith("?"): cliff += 3.0; sig["cliffhanger"].append("last line is an unanswered question")
        elif last.endswith(("...", "—", "-")): cliff += 2.0; sig["cliffhanger"].append("last line trails off")
        elif last.endswith("!"): cliff += 1.2; sig["cliffhanger"].append("last line is an exclamation")
        if lw & THREAT: cliff += 1.8; sig["cliffhanger"].append(f"threat/reveal vocabulary: {sorted(lw & THREAT)[:3]}")
        if e - td[-1].end <= 3.0: cliff += 1.0; sig["cliffhanger"].append("cut lands within 3s of the last line")
        if any(r in last.lower() for r in RESOLVED): cliff -= 2.0; sig["cliffhanger"].append("last line sounds resolved")
    fx = set(w for b in tail if b.kind in ("sfx", "action", "camera") for w in tok(b.text)) & TENSION_FX
    if fx: cliff += min(2.5, 1.2 * len(fx)); sig["cliffhanger"].append(f"tension cues at the cut: {sorted(fx)[:3]}")
    if not sig["cliffhanger"]: sig["cliffhanger"].append("no open question or tension cue at the end")

    # ---------------- CLARITY
    clar = 2.0
    speakers = {b.character for b in dlg if b.character}
    named = {c for c in speakers if _is_proper(c.upper())}
    text_l = " ".join(b.text for b in dlg)
    mentioned = {c for c in {x for x in (b.character for b in ep.blocks if b.kind == "dialogue") if x and _is_proper(x.upper())}
                 if c.split()[0].lower() in text_l.lower()}
    n_named = len(named | mentioned)
    if n_named: clar += min(2.5, 1.25 * n_named); sig["clarity"].append(f"named characters: {sorted(named | mentioned)[:4]}")
    else: sig["clarity"].append("no named characters in the clip")
    rel = set(tok(text_l)) & RELATION
    if rel: clar += 1.5; sig["clarity"].append(f"relationship words: {sorted(rel)[:3]}")
    if 2 <= len(speakers) <= 4: clar += 2.0; sig["clarity"].append(f"{len(speakers)} speakers (easy to follow)")
    elif len(speakers) == 1: clar += 0.5; sig["clarity"].append("single speaker")
    elif len(speakers) > 5: clar -= 1.5; sig["clarity"].append(f"{len(speakers)} speakers (crowded)")
    st = set(tok(text_l)) & STAKES
    if st: clar += 1.0; sig["clarity"].append(f"stakes words: {sorted(st)[:3]}")
    if any(s <= x < e - 5 for x in scene_starts if x > s - 0.01): clar += 0.5; sig["clarity"].append("scene start inside clip (setting established)")
    if s < 20 and re.search(r"picks up|directly follows|immediately follows|continues", ep.continuity, re.I):
        clar -= 1.5; sig["clarity"].append("opens early in an episode that continues the previous one (context gap)")
    if any(b.hidden_identity for b in dlg): clar -= 0.5; sig["clarity"].append("speaker identity not yet revealed to audience")

    # ---------------- EMOTION
    emo = 1.0
    if dlg:
        toned = [b for b in dlg if b.tone and set(tok(b.tone)) & INTENSE_TONE]
        share = len(toned) / len(dlg)
        emo += 5 * share
        if toned: sig["emotion"].append(f"{len(toned)}/{len(dlg)} lines carry intense tone labels")
        ex = sum(b.text.count("!") for b in dlg) / len(dlg)
        if ex: emo += min(2.0, 3 * ex); sig["emotion"].append(f"exclamation density {ex:.2f}/line")
    ea = set(w for b in blocks if b.kind == "action" for w in tok(b.text)) & ACTION_SHOCK
    if ea: emo += min(1.5, 0.5 * len(ea)); sig["emotion"].append(f"physical action: {sorted(ea)[:3]}")
    if any(b.kind == "sfx" for b in blocks): emo += 0.5; sig["emotion"].append("sound effect present")
    if not sig["emotion"]: sig["emotion"].append("no intensity cues found")

    return DetFeatures(hook=round(clip(hook), 2), cliffhanger=round(clip(cliff), 2), clarity=round(clip(clar), 2),
                       emotion=round(clip(emo), 2), signals=sig,
                       hook_text=(hook_b.text if hook_b else ""), hook_block=(hook_b.block_id if hook_b else ""),
                       cliff_text=(cliff_b.text if cliff_b else ""), cliff_block=(cliff_b.block_id if cliff_b else ""))


def weighted_total(h: float, c: float, cl: float, em: float, w: dict) -> float:
    return round(100 * (w["hook"] * h + w["cliffhanger"] * c + w["clarity"] * cl + w["emotion"] * em) / 10, 2)
