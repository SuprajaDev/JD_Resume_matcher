import pandas as pd
from sentence_transformers import SentenceTransformer, util

from load_data import load_texts


def get_scores(model_name="all-MiniLM-L6-v2"):
    jds = load_texts("data/jds")
    resumes = load_texts("data/resumes")

    model = SentenceTransformer(model_name)
    jd_vectors = model.encode(list(jds.values()))
    resume_vectors = model.encode(list(resumes.values()))

    scores = util.cos_sim(jd_vectors, resume_vectors).numpy()
    return pd.DataFrame(scores, index=jds.keys(), columns=resumes.keys())


if __name__ == "__main__":
    print(get_scores().round(2))