import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from config import (
    OUTPUT_DIR, CHUNK_JSONL, METADATA_CSV,
    EMBED_MODEL, FAISS_INDEX_PATH, EMBEDDINGS_NPY
)


def load_chunks():
    chunks = []
    with open(CHUNK_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_index():
    print("📥 Loading chunks...")
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]

    print(f"Total chunks: {len(texts)}")
    print("🔍 Generating embeddings using:", EMBED_MODEL)

    model = SentenceTransformer(EMBED_MODEL)

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=16,
    )

    # Normalize embeddings (important for cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)

    print("💾 Saving embeddings to:", EMBEDDINGS_NPY)
    np.save(EMBEDDINGS_NPY, embeddings)

    # FAISS index
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype("float32"))

    print("💾 Saving FAISS index to:", FAISS_INDEX_PATH)
    faiss.write_index(index, FAISS_INDEX_PATH)

    print("🚀 Index build complete!")


if __name__ == "__main__":
    build_index()
