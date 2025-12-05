from ai.retriever import ContractRetriever
from ai.config import TXT_DIR
from pathlib import Path
import re


class LegalQASystem:

    def __init__(self):
        pass

    def answer(self, question, top_k=3, doc_id=None):
        q = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", question)]
        stop = {"what","which","where","when","that","this","with","from","into","about","does","is","are","the","and","for"}
        q = [w for w in q if w not in stop]

        def score_sent(s):
            words = set(re.findall(r"\b[a-zA-Z]{3,}\b", s.lower()))
            return sum(1 for w in q if w in words)

        texts = []
        if doc_id:
            txt_path = Path(TXT_DIR) / f"{doc_id}.txt"
            if not txt_path.exists():
                return []
            with open(txt_path, "r", encoding="utf-8") as f:
                t = f.read()
            texts = [t]
        else:
            retriever = ContractRetriever()
            results = retriever.search(question, top_k=top_k)
            texts = [r["text"] for r in results]

        best = []
        for t in texts:
            sents = re.split(r"(?<=[\.\!\?])\s+", t)
            scored = [(score_sent(s), s.strip()) for s in sents if s.strip()]
            scored.sort(key=lambda x: -x[0])
            for sc, s in scored:
                if sc <= 0:
                    continue
                best.append(s)
                break

        if not best:
            return ["No relevant information found in the document."]

        ans = best[0]
        ans = re.sub(r"\s+", " ", ans).strip()
        if len(ans) > 200:
            ans = ans[:200]
        return [ans]


if __name__ == "__main__":
    qa = LegalQASystem()
    res = qa.answer("What is the notice period for termination?", top_k=5)

    for a in res[:3]:
        print("\n--- ANSWER ---")
        print("Text:", a["answer"])
        print("Score:", a["score"])
        print("Source:", a["source"])
        print("Context:", a["context"])
