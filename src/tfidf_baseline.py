import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from load_data import load_texts

jds = load_texts("data/jds")
resumes = load_texts("data/resumes")

# 1. Learn the vocabulary from all texts, then turn each text into numbers
all_texts = list(jds.values()) + list(resumes.values())
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
vectorizer.fit(all_texts)

jd_vectors = vectorizer.transform(jds.values())
resume_vectors = vectorizer.transform(resumes.values())

# 2. Compare every JD with every resume
scores = cosine_similarity(jd_vectors, resume_vectors)

# 3. Show the scores as a table (rows = JDs, columns = resumes)
table = pd.DataFrame(scores, index=jds.keys(), columns=resumes.keys())
print(table.round(2))