from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = ["ML engineer", "machine learning engineer", "pastry chef"]
vectors = model.encode(texts)

print("ML engineer vs machine learning engineer:", float(util.cos_sim(vectors[0], vectors[1])))
print("ML engineer vs pastry chef:", float(util.cos_sim(vectors[0], vectors[2])))