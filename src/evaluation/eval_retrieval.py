import pandas as pd
from pathlib import Path
from src.query_system.retriever import ClauseRetriever
from src.data_loading.data_loader import CUADDataLoader
from src.data_loading.data_processor import DataProcessor

OUT_CSV = Path("results/metrics/retrieval.csv")


def _build_test_corpus_and_labels(contracts):
    corpus = []
    queries = []
    gold_ids = []
    # Build corpus first to define IDs
    for contract in contracts:
        for paragraph in contract.get("paragraphs", []):
            ctx = paragraph.get("context", "")
            if ctx and len(ctx) > 20:
                corpus.append(ctx)
    ctx_to_id = {c: i for i, c in enumerate(corpus)}
    # Build query → gold id pairs
    for contract in contracts:
        for paragraph in contract.get("paragraphs", []):
            ctx = paragraph.get("context", "")
            if ctx and len(ctx) > 20 and ctx in ctx_to_id:
                for qa in paragraph.get("qas", []):
                    q = qa.get("question", "")
                    if q:
                        queries.append(q)
                        gold_ids.append(ctx_to_id[ctx])
    return corpus, queries, gold_ids


def evaluate():
    loader = CUADDataLoader()
    loader.load()
    contracts = loader.get_contracts()

    processor = DataProcessor(seed=42)
    splits = processor.create_splits(contracts)
    test_contracts = splits["test"]

    corpus, queries, gold_ids = _build_test_corpus_and_labels(test_contracts)

    retriever = ClauseRetriever()
    retriever.build_index(corpus)

    results = {1: [], 3: [], 5: []}
    reciprocal_ranks = []

    for q, gold in zip(queries, gold_ids):
        retrieved = retriever.search(q, top_k=5)
        ids = [r["id"] for r in retrieved]
        for k in results:
            results[k].append(1 if gold in ids[:k] else 0)
        rr = 0.0
        if gold in ids:
            rank = ids.index(gold) + 1
            rr = 1.0 / rank
        reciprocal_ranks.append(rr)

    df = pd.DataFrame(
        {
            "K": [1, 3, 5],
            "Precision@K": [
                sum(results[1]) / max(len(results[1]), 1),
                sum(results[3]) / max(len(results[3]), 1),
                sum(results[5]) / max(len(results[5]), 1),
            ],
            "MRR": [
                sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1),
                "",
                "",
            ],
        }
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print("✓ Retrieval evaluation saved")


if __name__ == "__main__":
    evaluate()
