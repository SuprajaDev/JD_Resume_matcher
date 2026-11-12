import json

import embedding_chunked
import tfidf_baseline
from explain import explain
from hybrid import rescale
from load_data import load_texts
from match_lines import requirement_lines
import threading


def explain_with_limit(jd_text, resume_text, seconds=240):
    box = {}

    def work():
        try:
            box["result"] = explain(jd_text, resume_text)
        except Exception as error:
            box["error"] = error

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimeoutError("took longer than " + str(seconds) + " seconds")
    if "error" in box:
        raise box["error"]
    return box["result"]

OUT = "data/explanations.json"

jds = load_texts("data/jds")
resumes = load_texts("data/resumes")
scores = (rescale(tfidf_baseline.get_scores()) + rescale(embedding_chunked.get_scores())) / 2

try:
    with open(OUT) as f:
        saved = json.load(f)
except FileNotFoundError:
    saved = {}

for jd_id in scores.index:
    if not requirement_lines(jds[jd_id]):
        continue
    for resume_id in scores.loc[jd_id].sort_values(ascending=False).index[:2]:
        key = jd_id + "|" + resume_id
        if key in saved:
            continue
        try:
                       saved[key] = explain_with_limit(jds[jd_id], resumes[resume_id]) 
        except Exception as error:
            print("failed", key, type(error).__name__)
            continue
        with open(OUT, "w") as f:
            json.dump(saved, f, indent=1)
        print("saved", key)

print("done:", len(saved), "explanations saved")