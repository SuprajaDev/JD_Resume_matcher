import json
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, "src")

import embedding_chunked
import tfidf_baseline
from hybrid import rescale
from load_data import load_texts
from match_lines import requirement_lines

st.title("Resume - JD Matcher")


def short(text, limit=110):
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


@st.cache_data
def get_hybrid_scores():
    tfidf = rescale(tfidf_baseline.get_scores())
    chunked = rescale(embedding_chunked.get_scores())
    return (tfidf + chunked) / 2


@st.cache_data
def get_jds():
    return load_texts("data/jds")


@st.cache_data
def get_saved():
    try:
        with open("data/explanations.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


scores = get_hybrid_scores()
jds = get_jds()
saved = get_saved()

jd_id = st.selectbox("Choose a job description", list(jds.keys()))
with st.expander("Show job description"):
    st.text(jds[jd_id])

ranked = scores.loc[jd_id].sort_values(ascending=False)
st.subheader("Ranked resumes")
st.caption("Relative order for this job only. The best resume gets 1.0, so this is not a percentage fit.")
st.table(pd.DataFrame({"resume": ranked.index, "relative match": ranked.round(2).values}))

st.subheader("Why this resume ranks here")
if not requirement_lines(jds[jd_id]):
    st.info("This job description has no bullet-point requirements to check.")
else:
    resume_id = st.selectbox("Choose a resume", list(ranked.index[:3]))
    results = saved.get(jd_id + "|" + resume_id)
    if not results:
        st.info("No saved explanation for this pair yet. Try the top 2 resumes.")
    else:
        matches = [r for r in results if r["verdict"] == "MATCH"]
        partials = [r for r in results if r["verdict"] == "PARTIAL"]
        gaps = [r for r in results if r["verdict"] == "GAP"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Match", len(matches))
        col2.metric("Partial", len(partials))
        col3.metric("Gap", len(gaps))

        shown = matches + partials
        if shown:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Verdict": [r["verdict"] for r in shown],
                        "Job requirement": [short(r["requirement"], 90) for r in shown],
                        "Proof from the resume": [short(r["quote"]) for r in shown],
                    }
                ),
                hide_index=True,
            )

        with st.expander("Requirements not shown on the resume (" + str(len(gaps)) + ")"):
            for r in gaps:
                st.write("- " + r["requirement"])

        st.caption("Every proof line is checked to exist in the resume. The match judgments come from an AI model and may be imperfect.")