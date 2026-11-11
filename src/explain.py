import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

from load_data import load_texts
from match_lines import match_resume

load_dotenv()
CLIENT = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
MODEL_NAME = os.environ["OPENROUTER_MODEL"]


def build_prompt(matches):
    blocks = []
    for i, (req, candidates) in enumerate(matches, start=1):
        lines = "\n".join("   - " + line for line, score in candidates)
        blocks.append(str(i) + ". REQUIREMENT: " + req + "\n   RESUME LINES:\n" + lines)
    return (
        "You are checking a resume against job requirements.\n"
        "For each requirement, look ONLY at the resume lines listed under it.\n"
        "Answer MATCH if a line clearly shows the skill, PARTIAL if it shows part of it, "
        "GAP if no line shows it.\n"
        "If MATCH or PARTIAL, copy the supporting line EXACTLY, word for word, into quote. "
        "Otherwise quote is an empty string.\n"
        "Reply with ONLY a JSON list like "
        '[{"n": 1, "verdict": "MATCH", "quote": "exact line"}]\n\n'
        + "\n\n".join(blocks)
    )


def explain(jd_text, resume_text):
    matches = match_resume(jd_text, resume_text)
    reply = CLIENT.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[{"role": "user", "content": build_prompt(matches)}],
    )
    text = reply.choices[0].message.content
    answers = json.loads(text[text.index("["):text.rindex("]") + 1])

    results = []
    for answer in answers:
        n = answer.get("n", 0)
        if n < 1 or n > len(matches):
            continue
        req, candidates = matches[n - 1]
        allowed = [line for line, score in candidates]
        verdict = answer.get("verdict", "GAP")
        quote = answer.get("quote", "")
        fake = False
        if verdict in ("MATCH", "PARTIAL") and quote not in allowed:
            verdict, quote, fake = "GAP", "", True
        results.append({"requirement": req, "verdict": verdict, "quote": quote, "fake_quote": fake})
    return results


if __name__ == "__main__":
    jd_id, resume_id = sys.argv[1], sys.argv[2]
    jds = load_texts("data/jds")
    resumes = load_texts("data/resumes")
    results = explain(jds[jd_id], resumes[resume_id])
    for r in results:
        print(r["verdict"], "|", r["requirement"])
        if r["quote"]:
            print("      proof:", r["quote"])
        if r["fake_quote"]:
            print("      (the model quoted a line that is not in the resume, so it was downgraded)")
    fakes = sum(r["fake_quote"] for r in results)
    print()
    print("Requirements checked:", len(results), "| invented quotes caught:", fakes)