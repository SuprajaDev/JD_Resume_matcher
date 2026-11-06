import pandas as pd
from sentence_transformers import SentenceTransformer, util

from load_data import load_texts


def split_chunks(text, size=80):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def get_scores(model_name="all-MiniLM-L6-v2", size=80):
    jds = load_texts("data/jds")
    resumes = load_texts("data/resumes")
    model = SentenceTransformer(model_name)

    jd_chunks = {k: model.encode(split_chunks(t, size)) for k, t in jds.items()}
    resume_chunks = {k: model.encode(split_chunks(t, size)) for k, t in resumes.items()}

    scores = pd.DataFrame(index=list(jds.keys()), columns=list(resumes.keys()), dtype=float)
    for jd_id, jd_vecs in jd_chunks.items():
        for resume_id, res_vecs in resume_chunks.items():
            sims = util.cos_sim(jd_vecs, res_vecs).numpy()
            scores.loc[jd_id, resume_id] = sims.max(axis=1).mean()
    return scores


if __name__ == "__main__":
    print(get_scores().round(2))