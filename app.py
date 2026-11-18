"""PromoScout dashboard. Run:  streamlit run app.py"""
from __future__ import annotations
import json, os, tempfile
from pathlib import Path
import pandas as pd
import streamlit as st

from promoscout.config import fmt_ts, load_config
from promoscout.export import audit_df, candidates_df, review_sheet_df, seg_str
from promoscout.llm import make_client
from promoscout.pipeline import run_pipeline

st.set_page_config(page_title="PromoScout", layout="wide")
st.title("PromoScout: promo finder for microdrama screenplays")
cfg = load_config()

with st.sidebar:
    st.header("Input")
    up = st.file_uploader("ZIP or .txt episode files", type=["zip", "txt"], accept_multiple_files=True)
    path = st.text_input("...or a local ZIP / folder path", "")
    title = st.text_input("Show title", "Untitled microdrama")
    st.header("Model")
    provider = st.selectbox("Judge", ["none (heuristics only)", "gemini", "mock (offline test double)"])
    model = st.text_input("Gemini model", os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    key = st.text_input("Gemini API key (kept in this session only)", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
    st.header("Weights (initial assumptions)")
    w = {k: st.slider(k, 0.0, 1.0, float(v), 0.05) for k, v in cfg["weights"].items()}
    tot = sum(w.values()) or 1.0
    cfg["weights"] = {k: v / tot for k, v in w.items()}
    st.caption(f"normalised: " + ", ".join(f"{k} {v:.0%}" for k, v in cfg["weights"].items()))
    cfg["selection"]["approved_target"] = st.number_input("Approved promos", 3, 100, cfg["selection"]["approved_target"])
    go = st.button("Run analysis", type="primary")

if go:
    inputs = [(f.name, f.getvalue()) for f in up] if up else ([path] if path else [])
    if not inputs:
        st.error("Provide a ZIP/.txt upload or a path."); st.stop()
    try:
        kind = {"none (heuristics only)": "none", "gemini": "gemini"}.get(provider, "mock")
        llm = make_client(kind, cfg, **({"api_key": key, "model": model} if kind == "gemini" else {}))
    except Exception as e:
        st.error(f"Could not start model: {e}"); st.stop()
    bar = st.progress(0.0, text="Working...")
    st.session_state["res"] = run_pipeline(inputs, cfg, llm, title, progress=lambda i, n: bar.progress(i / n, text=f"LLM judged {i}/{n}"))
    bar.empty()

res = st.session_state.get("res")
if res is None:
    st.info("Upload the screenplay pack in the sidebar and click **Run analysis**. With no key the app uses transparent keyword heuristics only.")
    st.stop()
if not res.episodes:
    st.error("No episodes could be parsed."); st.write(res.notes); st.stop()

st.caption(f"Mode: **{res.info['mode']}**" + ("  |  heuristic scores are crude fallbacks, not model judgments" if res.info["mode"] == "deterministic" else ""))
tabs = st.tabs(["Overview", "Episode analysis", "Promo discovery", "Score breakdown", "Quality audit", "Export"])
cdf = candidates_df(res)
approved = cdf[cdf.status == "approved"]

with tabs[0]:
    a, b, c, d = st.columns(4)
    a.metric("Episodes processed", len(res.episodes))
    b.metric("Candidate promos", len(res.candidates))
    c.metric("Approved promos", len(approved))
    d.metric("Avg score (approved)", f"{approved.total.mean():.1f}" if len(approved) else "n/a")
    st.write(f"Files seen: {res.ingest.files_seen}, parsed: {res.ingest.episodes_parsed}, skipped: {len(res.ingest.skipped)}, errors: {len(res.ingest.errors)}")
    warns = {e.number: e.warnings for e in res.episodes if e.warnings}
    if warns or res.notes or res.ingest.errors:
        with st.expander(f"Parse warnings and notes ({len(warns)} episodes with warnings)"):
            st.json({"episode_warnings": warns, "notes": res.notes, "ingest_errors": res.ingest.errors, "skipped": res.ingest.skipped})
    st.bar_chart(cdf.groupby("status").size().rename("candidates"))

with tabs[1]:
    nums = [e["episode"] for e in res.analysis["episodes"]]
    n = st.selectbox("Episode", nums)
    e = next(x for x in res.analysis["episodes"] if x["episode"] == n)
    st.markdown(f"**Logline:** {e['logline']}\n\n**Continuity:** {e['continuity']}")
    st.write(f"Duration {e['duration']:.1f}s, {len(e['scenes'])} scene(s). Ends on a question: {e['ending_is_question']}")
    st.write("**Characters (lines of dialogue):**", e["characters"])
    st.write("**Dramatic moments** (keyword-tagged; see README for limits)")
    mdf = pd.DataFrame(e["moments"])
    if len(mdf):
        mdf["time"] = mdf.start.map(fmt_ts); st.dataframe(mdf[["time", "category", "quote", "block_id"]], width="stretch", hide_index=True)
    st.write("**Story context across episodes**")
    st.dataframe(pd.DataFrame(res.analysis["characters"]).T.sort_values("lines", ascending=False), width="stretch")

with tabs[2]:
    show = st.radio("Show", ["approved", "all candidates"], horizontal=True)
    df = approved if show == "approved" else cdf
    st.dataframe(df[["rank", "tier", "status", "episodes", "segments", "duration_s", "total", "hook", "cliffhanger", "why_it_works"]],
                 width="stretch", hide_index=True)
    st.subheader("Segment details and source text")
    for c in (res.approved if show == "approved" else res.candidates)[:40]:
        with st.expander(f"#{c.rank or '-'} [{c.tier or '-'}] {seg_str(c)}  |  {c.duration:.0f}s  |  score {c.scores.total:.1f}"):
            st.write(f"**Hook:** {c.hook}"); st.write(f"**Cliffhanger:** {c.cliffhanger}"); st.write(f"**Why it works:** {c.why}")
            for s in c.segments:
                st.caption(f"Episode {s.episode}: {fmt_ts(s.start)} to {fmt_ts(s.end)}  ({len(s.block_ids)} screenplay blocks)")
            st.code(c.text, language=None)

with tabs[3]:
    opts = {f"#{c.rank or '-'} {c.candidate_id}": c for c in res.candidates}
    c = opts[st.selectbox("Candidate", list(opts))]
    sc = c.scores
    st.metric("Overall (weighted, /100)", f"{sc.total:.1f}", help=f"source: {sc.source}")
    st.write(f"Confidence: **{sc.confidence:.2f}** ({'model self-report, reduced if it disagrees with heuristics' if sc.source == 'llm' else 'fixed low value: heuristics only'})")
    st.dataframe(pd.DataFrame([
        {"dimension": k, "weight": f"{cfg['weights'][k]:.0%}", "LLM/used score": v, "heuristic score": getattr(c.det, k)}
        for k, v in (("hook", sc.hook), ("cliffhanger", sc.cliffhanger), ("clarity", sc.clarity), ("emotion", sc.emotion))]),
        hide_index=True, width="stretch")
    st.write("**Evidence quotes:**", sc.evidence)
    st.write("**Heuristic reasons:**", c.det.signals)
    if c.llm:
        st.write("**Model judgment:**"); st.json(c.llm.model_dump())

with tabs[4]:
    adf = audit_df(res)
    st.write("Flag counts:", adf.code.value_counts().to_dict() if len(adf) else "none")
    sev = st.multiselect("Severity", ["error", "warn", "info"], ["error", "warn"])
    code = st.multiselect("Flag type", sorted(adf.code.unique()) if len(adf) else [])
    f = adf[adf.severity.isin(sev)]
    if code: f = f[f.code.isin(code)]
    st.dataframe(f, width="stretch", hide_index=True)
    st.write("**Approved promos that carry warnings (weak recommendations):**")
    weak = adf[(adf.status == "approved") & (adf.severity == "warn")]
    st.dataframe(weak, width="stretch", hide_index=True)

with tabs[5]:
    st.download_button("promos.json (full)", json.dumps({"run": res.info, "approved": [c.model_dump() for c in res.approved],
                       "all_candidates": [c.model_dump() for c in res.candidates]}, indent=2), "promos.json", "application/json")
    st.download_button("promos_all.csv", cdf.to_csv(index=False), "promos_all.csv", "text/csv")
    st.download_button("promos_approved.csv", approved.to_csv(index=False), "promos_approved.csv", "text/csv")
    st.download_button("quality_audit.csv", audit_df(res).to_csv(index=False), "quality_audit.csv", "text/csv")
    st.download_button("review_sheet.csv (blank human benchmark)", review_sheet_df(res).to_csv(index=False), "review_sheet.csv", "text/csv")
    st.download_button("raw_llm_outputs.jsonl (unedited)", "\n".join(json.dumps(x) for x in res.raw_llm), "raw_llm_outputs.jsonl")
    st.caption("`python -m promoscout run <zip> --out outputs/run --llm gemini` writes every file, including episode_analysis.json and parse_report.json.")
