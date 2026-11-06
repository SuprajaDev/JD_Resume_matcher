import pandas as pd

import embedding_baseline
import tfidf_baseline
from evaluate import evaluate

labels = pd.read_csv("data/labels/labels.csv")

print("=== TF-IDF ===")
evaluate(tfidf_baseline.get_scores(), labels)

print()
print("=== Embeddings (all-MiniLM-L6-v2) ===")
evaluate(embedding_baseline.get_scores(), labels)