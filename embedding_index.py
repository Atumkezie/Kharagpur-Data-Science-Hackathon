from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.Client()

# Load novel text
doc = open("data/sample_dataset.txt").read()
chunks = doc.split("\n")

# Store embeddings in Vector DB
for chunk in chunks:
    if chunk.strip():
        emb = model.encode(chunk).tolist()
        db.add(embedding=emb, text=chunk)

print("Novel embeddings indexed into Vector DB ✔️")
