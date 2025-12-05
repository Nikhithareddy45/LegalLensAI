import os
import json
import pandas as pd
from tqdm import tqdm
from ai.config import TXT_DIR, OUTPUT_DIR, CHUNK_JSONL, METADATA_CSV, CHUNK_SIZE, CHUNK_OVERLAP


def clean_text(text: str) -> str:
    """Basic cleaning for contract text."""
    text = text.replace("\r", "")
    # Replace redacted markers
    text = text.replace("[* * *]", "<REDACTED>").replace("***", "<REDACTED>")
    return text.strip()


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split long documents into overlapping chunks."""
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]

        chunks.append({
            "start": start,
            "end": end,
            "text": chunk
        })

        if end == length:
            break

        start = end - overlap

    return chunks


def preprocess_contracts():
    print("📄 Loading TXT files from:", TXT_DIR)

    txt_files = [f for f in os.listdir(TXT_DIR) if f.endswith(".txt")]
    print(f"Found {len(txt_files)} contract TXT files.")
    
    metadata = []
    jsonl_file = open(CHUNK_JSONL, "w", encoding="utf-8")

    for file in tqdm(txt_files):
        filepath = os.path.join(TXT_DIR, file)
        doc_id = os.path.splitext(file)[0]

        # Read file
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        clean = clean_text(raw_text)
        chunks = chunk_text(clean)

        # Save chunks
        for i, ch in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"

            record = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": ch["text"],
                "start": ch["start"],
                "end": ch["end"]
            }

            jsonl_file.write(json.dumps(record) + "\n")

            metadata.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "start": ch["start"],
                "end": ch["end"]
            })

    jsonl_file.close()
    pd.DataFrame(metadata).to_csv(METADATA_CSV, index=False)

    print("✅ Preprocessing complete!")
    print("➡ Chunks saved to:", CHUNK_JSONL)
    print("➡ Metadata saved to:", METADATA_CSV)


if __name__ == "__main__":
    preprocess_contracts()

