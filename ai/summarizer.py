import numpy as np
from ai.config import TXT_DIR
from ai.preprocess import chunk_text
from pathlib import Path
import re


class ContractSummarizer:

    def __init__(self):
        pass

    def summarize_document(self, doc_id, num_chunks=5):
        """
        Summarize a contract by choosing the top N representative chunks.
        """
        print(f"📝 Summarizing document: {doc_id}")

        txt_path = Path(TXT_DIR) / f"{doc_id}.txt"
        if not txt_path.exists():
            return "❌ Document not found."

        with open(txt_path, "r", encoding="utf-8") as f:
            raw = f.read()
        # Sentence-based summarization: select 7–15 key sentences
        t = raw
        sents = re.split(r"(?<=[\.\!\?])\s+", t.strip())
        sents = [s.strip() for s in sents if s.strip()]
        keywords = [
            "termination","notice","payment","net","fee","indemnity",
            "liability","confidentiality","renewal","automatic","governing law",
            "jurisdiction","warranty","breach","penalty","obligation"
        ]
        def score(s):
            ls = s.lower()
            return sum(1 for k in keywords if k in ls) + min(len(s)//120,1)
        scored = sorted(((score(s), s) for s in sents), key=lambda x: -x[0])
        selected = [s for sc, s in scored[:15]]
        if len(selected) < 7:
            extra = [s for s in sents if s not in selected]
            selected += extra[:(7-len(selected))]
        selected = [s[:220] for s in selected]
        summary = "\n".join([f"- {p}" for p in selected])

        return summary


if __name__ == "__main__":
    s = ContractSummarizer()

    # Grab a sample document ID from retriever metadata
    import pandas as pd
    meta = pd.read_csv("..\\outputs\\metadata.csv")
    sample_doc = meta.iloc[0]["doc_id"]

    print("\n### SUMMARY (first document) ###\n")
    print(s.summarize_document(sample_doc))
