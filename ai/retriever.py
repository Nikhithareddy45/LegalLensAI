import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from config import (
    EMBED_MODEL,
    FAISS_INDEX_PATH,
    METADATA_CSV,
    CHUNK_JSONL
)


class ContractRetriever:

    def __init__(self):
        print("🔍 Loading FAISS index...")
        self.index = faiss.read_index(FAISS_INDEX_PATH)

        print("📄 Loading metadata...")
        self.meta = pd.read_csv(METADATA_CSV)

        print("🧠 Loading embedding model...")
        self.model = SentenceTransformer(EMBED_MODEL)

        print("📦 Loading chunk texts...")
        self.chunks = {}
        with open(CHUNK_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self.chunks[row["chunk_id"]] = row["text"]

    def embed_query(self, query):
        emb = self.model.encode([query], convert_to_numpy=True)
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        return emb.astype("float32")

    def search(self, query, top_k=5):
        print("🔎 Searching for:", query)

        q_emb = self.embed_query(query)
        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            row = self.meta.iloc[idx].to_dict()
            row["score"] = float(score)
            row["text"] = self.chunks.get(row["chunk_id"], "")
            results.append(row)

        return results


if __name__ == "__main__":
    r = ContractRetriever()
    res = r.search("termination clause notice period", top_k=3)

    for r in res:
        print("\n--- RESULT ---")
        print("Chunk ID:", r["chunk_id"])
        print("Score:", r["score"])
        print("Text:", r["text"][:300], "...")
