import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "CUAD_v1")
TXT_DIR = os.path.join(DATA_DIR, "full_contract_txt")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)

CHUNK_SIZE = 1800       # characters
CHUNK_OVERLAP = 300     # characters

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, "faiss.index")
EMBEDDINGS_NPY = os.path.join(OUTPUT_DIR, "embeddings.npy")
CHUNK_JSONL = os.path.join(OUTPUT_DIR, "chunks.jsonl")
METADATA_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")
