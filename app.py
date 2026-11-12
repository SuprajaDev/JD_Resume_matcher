import json
import sys

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "src")

from load_data import load_texts

st.title("Resume - JD Matcher")


def short(text, limit=110):
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def requirement_lines(jd_text):
    lines = []
    for line in jd_text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            lines.append(line[2:].strip())
        elif line.startswith("•"):
            lines.extend(p.strip() for p in line.split("•") if len(p.split()) >= 3)
    return lines


def read_upload(file):
    if file.name.lower().endswith(".pdf"):
        import pdfplumber

        with pdfplumber.open(file) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)
    return file.read().decode("utf-8", errors="ignore")


@st.cache_data
def get_scores():
    return pd.read_csv("data/scores.csv", index_col=0)


@st.cache_data
def get_jds():
    return load_texts("data/jds")


@st.cache_data
def get_resumes():
    return load_texts("data/resumes")


@st.cache_data
def get_saved():
    try:
        with open("data/explanations.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


scores = get_scores()
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
if not any(key.startswith(jd_id + "|") for key in saved):
    st.info("No saved explanations for this job yet. Try another job.")
else:
    resume_id = st.selectbox("Choose a resume", list(ranked.index[:3]))
    results = saved.get(jd_id + "|" + resume_id)
    if not results:
        st.info("No saved explanation for this resume. Try one of the top 2.")
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

st.divider()
st.subheader("Try your own resume")
st.caption(
    "Upload a PDF or text resume to compare it with the job chosen above. "
    "This uses word matching only (no AI model). The file is not stored and is not sent to any AI service."
)
upload = st.file_uploader("Upload a resume", type=["pdf", "txt"])

if upload is not None:
    text = read_upload(upload)
    st.write("Read " + str(len(text.split())) + " words from " + upload.name)
    if len(text.split()) < 30:
        st.warning("Could not read enough text. Scanned PDFs (photos of a page) are not supported. Try a text-based PDF or a .txt file.")
    else:
        samples = get_resumes()
        corpus = list(jds.values()) + list(samples.values()) + [text]
        vectorizer = TfidfVectorizer(stop_words="english").fit(corpus)

        jd_vec = vectorizer.transform([jds[jd_id]])
        sample_scores = cosine_similarity(jd_vec, vectorizer.transform(list(samples.values())))[0]
        your_score = float(cosine_similarity(jd_vec, vectorizer.transform([text]))[0][0])
        rank = int((sample_scores > your_score).sum()) + 1

        st.metric("Your rank among 11 resumes (10 samples plus yours)", "#" + str(rank))
        st.caption(
            "Word-matching score: " + str(round(your_score, 2)) + ". "
            "Word matching favors long resumes and misses synonyms, so treat this as a rough guide."
        )

        reqs = requirement_lines(jds[jd_id])
        lines = [l.strip().lstrip("-").strip() for l in text.splitlines()]
        lines = [l for l in lines if len(l.split()) >= 3]
        if reqs and lines:
            sims = cosine_similarity(vectorizer.transform(reqs), vectorizer.transform(lines))

            order = sorted(range(len(reqs)), key=lambda i: -float(sims[i].max()))
            used = {}
            picks = {}
            for i in order:
                for j in sims[i].argsort()[::-1]:
                    j = int(j)
                    if used.get(j, 0) < 2:
                        picks[i] = (j, float(sims[i][j]))
                        used[j] = used.get(j, 0) + 1
                        break

            rows = []
            unmatched = []
            for i, req in enumerate(reqs):
                j, score = picks.get(i, (0, 0.0))
                if score >= 0.10:
                    rows.append(
                        {
                            "Job requirement": short(req, 90),
                            "Closest line in your resume": short(lines[j]),
                            "Word overlap": round(score, 2),
                        }
                    )
                else:
                    unmatched.append(req)

            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True)
            with st.expander("Requirements with no clear match (" + str(len(unmatched)) + ")"):
                for req in unmatched:
                    st.write("- " + req)
        else:
            st.info("This job has no bullet-point requirements to compare line by line.")
