# Results

| Method | Top-1 | Top-2 | Top-3 | Mean NDCG@3 | Notes |
|---|---|---|---|---|---|
| TF-IDF baseline | 8 of 10 (80%) | 10 of 10 | 10 of 10 | 0.90 | Misses at jd_01 and jd_03: long real resume_01 ranks first |
| Embeddings (plain, MiniLM) | 6 of 10 (60%) | 9 of 10 | 10 of 10 | 0.88 | Lost to TF-IDF: texts longer than the model's 256-piece limit |

Test: 10 JDs x 10 resumes, 100 labeled pairs.
Limits: small test set, 9 of 10 resumes are synthetic, and the labels were drafted with AI help and reviewed by me.