# Project log

## Oct 27
- Planned the workflow and created the repo.
## Oct 28
 - Pushed 5 JDs and 5 resumes, next: labeling
## Oct 31
 - TF-IDF baseline ran. Top-1 correct on 3 of 5 JDs. Misses: jd_01 and jd_03 (long real resume_01 outranked focused ones)
 ## Nov 1
 - add ranking output s ripts
## Nov 4
- Day 4: labeled all 100 pairs (batch 1 and 2). TF-IDF baseline on 10x10: top-1 8/10, top-2 10/10, NDCG@3 0.90
## Nov 5
- Day 5: installed sentence-transformers, ran plain embeddings. Top-1 6/10 vs TF-IDF 8/10. Cause: long texts get cut off, so next is chunking
## Nov 6
- failure inspection and findings
## Nov 9
 - Day 6: length test (real resume 460 words vs about 125 for synthetic). Hybrid TF-IDF + chunked embeddings: NDCG@3 0.91, top-2 10/10. Chosen for the demo
## Nov 10
 - line matching, Streamlit app, API setup, LLM explanations with quote check (0 invented quotes on jd_08 test)
 ## Nov 11
 - precomputed LLM explanations, fixed a too-strict quote check (it was downgrading real matches to GAP), 16 pairs saved with 0 invented quotes