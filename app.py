import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, "src")

import embedding_chunked
import tfidf_baseline
from hybrid import rescale
from load_data import load_texts
from match_lines import match_resume, requirement_lines

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
def get_resumes():
    return load_texts("data/resumes")


scores = get_hybrid_scores()
jds = get_jds()
resumes = get_resumes()

jd_id = st.selectbox("Choose a job description", list(jds.keys()))
st.text_area("Job description", jds[jd_id], height=200)

ranked = scores.loc[jd_id].sort_values(ascending=False)
st.subheader("Ranked resumes")
st.table(pd.DataFrame({"resume": ranked.index, "match score": ranked.round(2).values}))

st.subheader("Closest resume lines for the top 3")
st.caption("For each job requirement, the resume line that is most similar in meaning. A low score means the resume probably does not show that skill.")

if not requirement_lines(jds[jd_id]):
    st.info("This job description has no bullet-point requirements to match.")
else:
    for resume_id in ranked.index[:3]:
        with st.expander(resume_id + "  (match score " + str(round(float(ranked[resume_id]), 2)) + ")"):
            for req, candidates in match_resume(jds[jd_id], resumes[resume_id]):
                line, score = candidates[0]
                st.markdown("**" + req + "**")
                st.write("Closest line (" + str(round(score, 2)) + "): " + line)