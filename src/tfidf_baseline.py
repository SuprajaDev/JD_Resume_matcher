import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from load_data import load_texts


def get_scores():
    jds = load_texts("data/jds")
    resumes = load_texts("data/resumes")

    all_texts = list(jds.values()) + list(resumes.values())
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    vectorizer.fit(all_texts)

    jd_vectors = vectorizer.transform(jds.values())
    resume_vectors = vectorizer.transform(resumes.values())

    scores = cosine_similarity(jd_vectors, resume_vectors)
    return pd.DataFrame(scores, index=jds.keys(), columns=resumes.keys())


if __name__ == "__main__":
    print(get_scores().round(2))