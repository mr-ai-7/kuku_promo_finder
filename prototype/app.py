"""Kuku Promo Finder v2 Streamlit application.
Run from project root: python -m streamlit run prototype/app.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from promo_finder import run_analysis, selected_dialogue, relevant_events, ts_to_s, s_to_ts

st.set_page_config(page_title="Promo Finder v2", page_icon="🎬", layout="wide")
st.title("🎬 Promo Finder v2")
st.caption("Find and audit 20–60-second microdrama promotional moments from timestamped screenplays.")
st.info("Free offline analysis. Scores are editorial heuristics, not view or conversion predictions. A human must check the footage.")

with st.sidebar:
    st.header("Analysis settings")
    target = st.slider("Number of ranked promos", min_value=1, max_value=75, value=20, step=1)
    per_ep = st.slider("Maximum promos from one episode", min_value=1, max_value=3, value=1)
    st.divider()
    st.caption("Supports the documented Kuku-style timed TXT format (ZIP or single TXT).")
    st.caption("Privacy: do not publish proprietary screenplay files or raw excerpts in a public repository.")

uploaded = st.file_uploader("Upload a ZIP of timestamped scripts, or a single TXT", type=["zip", "txt"])
signature = None
if uploaded is not None:
    signature = (hashlib.sha256(uploaded.getvalue()).hexdigest(), int(target), int(per_ep))
    st.caption(f"Selected: {uploaded.name} · {len(uploaded.getvalue()) / 1024:.1f} KB")
    st.caption("Note: 'episodes processed' is the number of parsed scripts; 'ranked promos' is the number selected by the slider.")
    if st.button("Analyze all uploaded episodes", type="primary"):
        try:
            with st.spinner("Parsing timestamps, checking boundaries and ranking candidate clips..."):
                with tempfile.TemporaryDirectory() as td:
                    ext = ".zip" if uploaded.name.lower().endswith(".zip") else ".txt"
                    path = Path(td) / ("uploaded" + ext)
                    path.write_bytes(uploaded.getvalue())
                    out = Path(td) / "outputs"
                    payload, episodes, candidates = run_analysis(path, out, target, per_ep)
                    data = {
                        "signature": signature,
                        "payload": payload,
                        "episodes": {e.number: e for e in episodes},
                        "downloads": {f.name: f.read_bytes() for f in out.iterdir() if f.is_file()},
                    }
                st.session_state["analysis"] = data
                st.success("Analysis complete. All recognized script files were validated and processed.")
        except Exception as exc:
            st.session_state.pop("analysis", None)
            st.error(f"Analysis failed: {exc}")

analysis = st.session_state.get("analysis")
if analysis is not None and signature != analysis["signature"]:
    analysis = None
    st.warning("The uploaded file or ranking settings changed. Click Analyze again to avoid using stale results.")

if analysis is None:
    st.markdown("### How to use")
    st.write("1. Upload a compatible screenplay ZIP or TXT.  2. Analyze.  3. Inspect opening, ending and warnings.  4. Export raw results.")
else:
    result = analysis["payload"]
    a, b, c, d = st.columns(4)
    a.metric("Episodes parsed", result["processed_episode_count"])
    b.metric("Candidate clips checked", f"{result['total_candidate_count']:,}")
    c.metric("Ranked suggestions", result["selected_promo_count"])
    d.metric("With quality warnings", sum(bool(p["quality_warnings"]) for p in result["selected_promos"]))
    if result["selected_promo_count"] < result["requested_promo_count"]:
        st.warning("Fewer valid, diverse promos were found than requested. This is not a processing error.")

    st.subheader("Ranked promo suggestions")
    data = pd.read_csv(io.BytesIO(analysis["downloads"]["promos_ranked.csv"]))
    show = ["rank", "episode", "start", "end", "duration_seconds", "overall_score", "scene_transitions", "quality_warnings"]
    st.dataframe(data[show], hide_index=True, use_container_width=True)
    st.caption("One continuous cut per promo. Timestamps come from parsed script events; final video cuts require editor verification.")

    if result["selected_promos"]:
        options = list(range(len(result["selected_promos"])))
        selected = st.selectbox("Inspect a promo", options, format_func=lambda i: (
            f"#{result['selected_promos'][i]['rank']:02d} — Episode {result['selected_promos'][i]['episode']:02d} — "
            f"{result['selected_promos'][i]['segments'][0]['start']} to {result['selected_promos'][i]['segments'][0]['end']}"
        ))
        promo = result["selected_promos"][selected]
        one, two = st.columns(2)
        with one:
            st.markdown("**Opening within the first 3 seconds**")
            st.write(promo["hook"])
            st.markdown("**Proposed ending / cliffhanger**")
            st.write(promo["cliffhanger"])
        with two:
            st.markdown("**Why it was ranked**")
            st.write(promo["why_it_works"])
            st.caption(f"Hook {promo['hook_score']:.2f}/5 · Cliffhanger {promo['cliffhanger_score']:.2f}/5 · Standalone proxy {promo['self_contained_score']:.2f}/5")
            if promo["quality_warnings"]:
                for warning in promo["quality_warnings"]:
                    st.warning(warning)
            else:
                st.caption("No automatic warning. This does not mean an editor approved the clip.")
        with st.expander("Inspect dialogue/actions inside this exact cut"):
            seg = promo["segments"][0]
            ep = analysis["episodes"][promo["episode"]]
            events = relevant_events(ep, ts_to_s(seg["start"]), ts_to_s(seg["end"]), {"DIALOGUE", "ACTION"})
            for e in events:
                st.write(f"{s_to_ts(e.start)}–{s_to_ts(e.end)} | {e.label}: {e.text}")

        st.subheader("Human editor review")
        st.caption("This assessment is saved only in the current browser session. It never changes the raw model outputs.")
        prefix = "editor_" + signature[0][:12] + f"_{target}_{per_ep}_" + promo["promo_id"]
        st.selectbox("Decision", ["Not reviewed", "Approve", "Reject", "Needs footage check"], key=prefix + "_decision")
        st.text_area("Editor comments (optional)", key=prefix + "_notes", height=80,
                     placeholder="e.g. opening too confusing; ending resolves too early")

        review_rows = []
        for p in result["selected_promos"]:
            prefix_i = "editor_" + signature[0][:12] + f"_{target}_{per_ep}_" + p["promo_id"]
            sg = p["segments"][0]
            review_rows.append({
                "promo_id": p["promo_id"], "episode": p["episode"], "start": sg["start"], "end": sg["end"],
                "decision": st.session_state.get(prefix_i + "_decision", "Not reviewed"),
                "notes": st.session_state.get(prefix_i + "_notes", ""),
            })
        buf = io.StringIO(); writer = csv.DictWriter(buf, fieldnames=list(review_rows[0])); writer.writeheader(); writer.writerows(review_rows)
        st.download_button("Download your human review CSV", buf.getvalue().encode("utf-8-sig"),
                           file_name="my_editor_review.csv", mime="text/csv")

    st.divider()
    st.subheader("Download raw outputs")
    for filename, label in [
        ("promos_ranked.json", "Ranked promos — JSON"),
        ("promos_ranked.csv", "Ranked promos — CSV"),
        ("episode_coverage.csv", "All-episode coverage — CSV"),
        ("candidate_scores_top3_per_episode.csv", "Top 3 candidate clips per episode — CSV"),
        ("editor_review_template.csv", "Blank editor evaluation sheet — CSV"),
        ("run_summary.json", "Run summary — JSON"),
    ]:
        st.download_button(label, analysis["downloads"][filename], file_name=filename,
                           mime="application/json" if filename.endswith(".json") else "text/csv",
                           key="dl_" + filename)
    st.caption("These files are from the active run. Keep them unchanged when submitting raw results.")

with st.expander("What Version 2 improves and what remains uncertain"):
    st.markdown("""
    **Improvements over Version 1:** no show-specific names in scoring, hooks evaluated in the first three seconds,
    stronger warnings for partial sentences, possible resolution, exposition, rhetorical questions and scene transitions;
    input validation; near-duplicate filtering; stale-result protection; optional human review CSV.

    **Limitations:** outputs are continuous clips only. Semantic understanding is limited by rules and TF-IDF;
    the tool cannot see actual video, estimate views, validate sound or subtitles, or guarantee independent viewer comprehension.
    No external LLM or paid API is required for this prototype.
    """)
