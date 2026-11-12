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
st.text_area("Job description", jds[jd_id], height=200)

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
        matched = sum(r["verdict"] == "MATCH" for r in results)
        partial = sum(r["verdict"] == "PARTIAL" for r in results)
        gaps = sum(r["verdict"] == "GAP" for r in results)
        st.write("MATCH: " + str(matched) + "  |  PARTIAL: " + str(partial) + "  |  GAP: " + str(gaps))
        for r in results:
            st.markdown("**" + r["verdict"] + "** - " + r["requirement"])
            if r["quote"]:
                st.caption("Proof from the resume: " + r["quote"])
        st.caption("Every proof line is checked to exist in the resume. The match judgments come from an AI model and may be imperfect.")