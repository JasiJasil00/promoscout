# PromoScout

Finds 20-60 second promo clips in microdrama screenplays (Kuku AI Generalist assignment, Problem 1).
Code owns every timestamp; an LLM (optional) only judges clips that code has already cut.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then put your key in .env (never commit it)
```
Python 3.10+. Only Gemini needs a key; everything else works offline.

## Run
```bash
# 0. verify your Gemini key/model with one tiny call
python -m promoscout llm-check

# 1. no key needed: heuristic scoring only
python -m promoscout run path/to/pack.zip --out outputs/run_heuristic --title "I Married a Broken Soldier"

# 2. with Gemini (free tier): same command, adds LLM judgments
python -m promoscout run path/to/pack.zip --out outputs/run_gemini --llm gemini --title "I Married a Broken Soldier"

# 3. dashboard
streamlit run app.py

# 4. tests (the pack test only runs if you point it at the pack)
python -m pytest -q
PROMOSCOUT_PACK=path/to/pack.zip python -m pytest -q        # Windows PowerShell: $env:PROMOSCOUT_PACK="..."
```
Inputs accepted: the 50-episode ZIP, any other ZIP of `.txt` episodes, a folder, or individual `.txt` files.
Interview tip: a new ZIP needs no code change. Show-specific names live in `aliases.yaml` (optional; unknown speakers fall back to their label).

## Pipeline
`ingest -> parse -> validate -> candidates (windows + joined) -> [LLM judge] -> score -> dedupe -> audit -> select -> export`

| Stage | Module | Owner |
|---|---|---|
| ZIP/folder/file ingest, bad files reported not fatal | `ingest.py` | code |
| Tolerant parser (case-insensitive labels, warnings) | `parser.py` | code |
| 20-60s windows, cuts snapped to block edges, never inside a spoken line; two-segment joins | `candidates.py` | code |
| Timestamp / duration / block-id validation | `validate.py` | code |
| Hook, cliffhanger, clarity, emotion judgments + evidence quotes | `judge.py`, `prompts.py` | LLM |
| Heuristic baseline and fallback | `features.py` | code |
| Weighted score (30/30/25/15, editable in `config.yaml` or the sidebar) | `scoring.py` | code |
| TF-IDF + time-overlap duplicate detection | `dedupe.py` | code |
| Flags: weak hook, missing cliffhanger, fabricated quote, etc. | `audit.py` | code |

Swap the model: subclass `LLMClient` in `promoscout/llm/` (implement `_call`) and register it in `llm/__init__.py`.
Model name comes from `GEMINI_MODEL`. Responses are cached in `.llm_cache/` so re-runs cost nothing.

## Outputs (`--out` folder)
`promos.json` (full), `promos_all.csv`, `promos_approved.csv`, `quality_audit.csv`, `review_sheet.csv` (blank human benchmark),
`episode_analysis.json`, `parse_report.json`, `raw_llm_outputs.jsonl` (unedited request/response log).

## Guarantees
- Candidates are built from parsed blocks; the LLM never returns timestamps, and any it returns are ignored.
- Evidence quotes from the model must be exact substrings of the clip, else an error flag rejects the candidate.
- Every segment is re-validated against the parsed screenplay (episode exists, within duration, no mid-line cuts, ordered, 20-60s total).

## Known limits (read before trusting scores)
- No accuracy figure is claimed. Quality is measured only after a human fills `review_sheet.csv`.
- Heuristic scores are keyword-based and crude. The "episode analysis" moment tags are keyword tags, not understanding.
- Hooks are judged from screenplay text; the real visual/audio impact of the first 3 seconds is invisible to the tool.
- Joined (two-segment) candidates take the best hook and the best cliffhanger, so heuristic scoring favours them. Check them by eye.
- Alias table and lexicons are tuned to English period melodrama; other genres need new lexicons and aliases.
- Confidential data: do not commit the pack or outputs that quote it (`data/` and `outputs/` are git-ignored).

## Scaling to hundreds of shows
Episodes are independent, so parsing/candidate generation is embarrassingly parallel (one job per episode or show).
Cost is controlled by the heuristic shortlist: the LLM sees ~190 clips per 50-episode show, not every window.
For thousands of episodes: queue per-show jobs, store parsed episodes and judgments in a database keyed by content hash
(the cache already works this way), use a cheaper model for the shortlist and a stronger one for the final top 30,
and keep a human approval step before posting.
