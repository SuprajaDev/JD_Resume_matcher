# Results

| Method | Top-1 | Top-2 | Top-3 | Mean NDCG@3 | Notes |
|---|---|---|---|---|---|
| TF-IDF baseline | 8 of 10 | 10 of 10 | 10 of 10 | 0.90 | Strong baseline. Misses jd_01 and jd_03 |
| Embeddings (plain, MiniLM) | 6 of 10 | 9 of 10 | 10 of 10 | 0.88 | Long texts cut off at the model's limit |
| Embeddings (chunked, 80 words) | 7 of 10 | 8 of 10 | 9 of 10 | 0.87 | Fixes jd_01, drops on jd_07 |
| Hybrid: TF-IDF + plain | 6 of 10 | 10 of 10 | 10 of 10 | 0.88 | No gain over TF-IDF alone |
| Hybrid: TF-IDF + chunked | 7 of 10 | 10 of 10 | 10 of 10 | 0.91 | Best NDCG and best top-2. Chosen for the demo |

Test: 10 JDs x 10 resumes, 100 labeled pairs.
Limits: small test set. 9 of 10 resumes are synthetic (about 125 words) while the one real resume is 460 words, which inflates its scores. Labels were drafted with AI help and reviewed by me. Differences of one job (10 points of top-1) are within noise.