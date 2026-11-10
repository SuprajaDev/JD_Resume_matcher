import sys

from sentence_transformers import SentenceTransformer, util

from load_data import load_texts

MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def requirement_lines(jd_text):
    return [line[2:].strip() for line in jd_text.splitlines() if line.startswith("- ")]


def resume_lines(resume_text):
    lines = [line.strip().lstrip("- ").strip() for line in resume_text.splitlines()]
    return [l for l in lines if len(l.split()) >= 3 and not l.startswith(("Source:", "Profile:"))]


def match_resume(jd_text, resume_text, top_n=3):
    reqs = requirement_lines(jd_text)
    lines = resume_lines(resume_text)
    sims = util.cos_sim(MODEL.encode(reqs), MODEL.encode(lines)).numpy()
    results = []
    for i, req in enumerate(reqs):
        best = sims[i].argsort()[::-1][:top_n]
        results.append((req, [(lines[j], float(sims[i][j])) for j in best]))
    return results


if __name__ == "__main__":
    jd_id, resume_id = sys.argv[1], sys.argv[2]
    jds = load_texts("data/jds")
    resumes = load_texts("data/resumes")
    for req, candidates in match_resume(jds[jd_id], resumes[resume_id]):
        print("REQ:", req)
        for line, score in candidates:
            print("   ", round(score, 2), "|", line)
