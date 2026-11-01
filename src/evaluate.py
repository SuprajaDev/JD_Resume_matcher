import pandas as pd
from sklearn.metrics import ndcg_score

from tfidf_baseline import get_scores

TOP_K = 3


def evaluate(scores, labels, k=TOP_K):
    hits = {rank: 0 for rank in range(1, k + 1)}
    ndcgs = []

    for jd_id in scores.index:
        jd_labels = labels[labels["jd_id"] == jd_id].set_index("resume_id")["label"]
        jd_labels = jd_labels.loc[scores.columns]
        best_resumes = jd_labels[jd_labels == jd_labels.max()].index

        ranked = list(scores.loc[jd_id].sort_values(ascending=False).index)
        first_hit = min(ranked.index(r) + 1 for r in best_resumes)
        for rank in hits:
            if first_hit <= rank:
                hits[rank] += 1

        ndcg = ndcg_score([jd_labels.values], [scores.loc[jd_id].values], k=k)
        ndcgs.append(ndcg)
        print(jd_id, "best resume ranked at position", first_hit, "| NDCG@" + str(k) + ":", round(ndcg, 2))

    print()
    for rank in hits:
        print("Top-" + str(rank), "accuracy:", hits[rank], "of", len(scores.index))
    print("Mean NDCG@" + str(k) + ":", round(sum(ndcgs) / len(ndcgs), 2))


if __name__ == "__main__":
    labels = pd.read_csv("data/labels/labels.csv")
    evaluate(get_scores(), labels)