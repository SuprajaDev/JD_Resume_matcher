import sys

import pandas as pd

import embedding_baseline
import embedding_chunked
import tfidf_baseline

labels = pd.read_csv("data/labels/labels.csv")
methods = {"tfidf": tfidf_baseline, "plain": embedding_baseline, "chunked": embedding_chunked}

jd_id = sys.argv[1]
jd_labels = labels[labels["jd_id"] == jd_id].set_index("resume_id")["label"]

for name, module in methods.items():
    ranked = module.get_scores().loc[jd_id].sort_values(ascending=False)
    print(name)
    for position, (resume_id, score) in enumerate(ranked.items(), start=1):
        print("  ", position, resume_id, round(float(score), 2), "label", jd_labels[resume_id])
    print()
