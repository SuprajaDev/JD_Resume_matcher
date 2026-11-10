import pandas as pd

import embedding_baseline
import embedding_chunked
import tfidf_baseline
from evaluate import evaluate


def rescale(scores):
    # squeeze each job's scores into 0-1 so two methods can be averaged fairly
    low = scores.min(axis=1)
    high = scores.max(axis=1)
    return scores.sub(low, axis=0).div(high - low, axis=0)


if __name__ == "__main__":
    labels = pd.read_csv("data/labels/labels.csv")

    tfidf = rescale(tfidf_baseline.get_scores())
    plain = rescale(embedding_baseline.get_scores())
    chunked = rescale(embedding_chunked.get_scores())

    print("=== Hybrid: TF-IDF + plain embeddings ===")
    evaluate((tfidf + plain) / 2, labels)

    print()
    print("=== Hybrid: TF-IDF + chunked embeddings ===")
    evaluate((tfidf + chunked) / 2, labels)
