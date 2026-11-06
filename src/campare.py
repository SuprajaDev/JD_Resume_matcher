import pandas as pd

import embedding_baseline
import embedding_chunked
import tfidf_baseline
from evaluate import evaluate

labels = pd.read_csv("data/labels/labels.csv")

print("=== TF-IDF ===")
evaluate(tfidf_baseline.get_scores(), labels)

print()
print("=== Embeddings (plain) ===")
evaluate(embedding_baseline.get_scores(), labels)

print()
print("=== Embeddings with chunking ===")
evaluate(embedding_chunked.get_scores(), labels)