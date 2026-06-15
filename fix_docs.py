import pickle
import os
from app.dataset import load_documents

print("Re-running dataset loader with preserved newlines...")
docs, labels, names = load_documents()

print(f"Loaded {len(docs)} documents. Checking first doc for newlines:")
print(repr(docs[0][:100]))

os.makedirs("app/data", exist_ok=True)
with open("app/data/documents.pkl", "wb") as f:
    pickle.dump(docs, f)

print("Saved documents.pkl successfully!")
