import pandas as pd

from load_data import load_texts
from tfidf_baseline import get_scores


def get_line(text, prefix):
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.replace(prefix, "").strip()
    return "unknown"


def show_rankings(scores, labels, top_n=3):
    jds = load_texts("data/jds")
    for jd_id in scores.index:
        company = get_line(jds[jd_id], "Company:")
        role = get_line(jds[jd_id], "Role:")
        print(jd_id, "|", company, "|", role)

        jd_labels = labels[labels["jd_id"] == jd_id].set_index("resume_id")["label"]
        ranked = scores.loc[jd_id].sort_values(ascending=False).head(top_n)

        for position, (resume_id, score) in enumerate(ranked.items(), start=1):
            label = jd_labels[resume_id]
            best = "  <- best by your labels" if label == jd_labels.max() else ""
            print("  Top", position, ":", resume_id, "| match score", round(score, 2), "| your label", label, best)
        print()


if __name__ == "__main__":
    labels = pd.read_csv("data/labels/labels.csv")
    show_rankings(get_scores(), labels)