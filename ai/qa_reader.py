from transformers import pipeline
from retriever import ContractRetriever


class LegalQASystem:

    def __init__(self):
        print("🤖 Loading QA model (roberta-base-squad2)...")
        self.qa = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            tokenizer="deepset/roberta-base-squad2",
        )
        self.retriever = ContractRetriever()

    def answer(self, question, top_k=5):
        print(f"\n❓ QUESTION: {question}\n")

        results = self.retriever.search(question, top_k=top_k)

        answers = []

        for chunk in results:
            context = chunk["text"]

            try:
                out = self.qa(question=question, context=context)
                answers.append({
                    "answer": out.get("answer"),
                    "score": float(out.get("score")),
                    "source": chunk["chunk_id"],
                    "context": context[:500] + "..."
                })
            except Exception as e:
                pass

        # Rank by score
        answers = sorted(answers, key=lambda x: -x["score"])

        return answers


if __name__ == "__main__":
    qa = LegalQASystem()
    res = qa.answer("What is the notice period for termination?", top_k=5)

    for a in res[:3]:
        print("\n--- ANSWER ---")
        print("Text:", a["answer"])
        print("Score:", a["score"])
        print("Source:", a["source"])
        print("Context:", a["context"])
