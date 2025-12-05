import numpy as np
from sentence_transformers import SentenceTransformer
from ai.retriever import ContractRetriever
from ai.config import EMBED_MODEL


class ContractSummarizer:

    def __init__(self):
        print("🔍 Loading summarization model...")
        self.model = SentenceTransformer(EMBED_MODEL)
        self.retriever = ContractRetriever()

    def summarize_document(self, doc_id, num_chunks=5):
        """
        Summarize a contract by choosing the top N representative chunks.
        """
        print(f"📝 Summarizing document: {doc_id}")

        # Get all chunks belonging to this doc
        doc_chunks = {
            cid: text
            for cid, text in self.retriever.chunks.items()
            if cid.startswith(doc_id)
        }

        if not doc_chunks:
            return "❌ Document not found."

        texts = list(doc_chunks.values())

        # Embed chunks
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

        # Compute centroid of the document
        centroid = embeddings.mean(axis=0, keepdims=True)

        # Score by similarity to centroid
        scores = (embeddings @ centroid.T).flatten()

        # Pick top chunks
        top_idx = np.argsort(scores)[-num_chunks:][::-1]

        selected_chunks = [texts[i] for i in top_idx]

        # Build readable summary
        summary = "\n\n".join(chunk.strip() for chunk in selected_chunks)

        return summary


if __name__ == "__main__":
    s = ContractSummarizer()

    # Grab a sample document ID from retriever metadata
    import pandas as pd
    meta = pd.read_csv("..\\outputs\\metadata.csv")
    sample_doc = meta.iloc[0]["doc_id"]

    print("\n### SUMMARY (first document) ###\n")
    print(s.summarize_document(sample_doc))
