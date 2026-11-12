import embedding_chunked
import tfidf_baseline
from hybrid import rescale

scores = (rescale(tfidf_baseline.get_scores()) + rescale(embedding_chunked.get_scores())) / 2
scores.to_csv("data/scores.csv")
print(scores.round(2))