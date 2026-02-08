import sys
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from rouge_score import rouge_scorer
from src.data_loading.data_loader import CUADDataLoader
from src.data_loading.data_processor import DataProcessor
from src.summarization.fusion import FusionSummarizer

OUT_CSV = Path("results/metrics/summarization_rouge.csv")


def _build_text_ref_pairs(contracts, max_pairs=500):
    pairs = []
    count = 0
    for contract in contracts:
        for paragraph in contract.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                answers = qa.get("answers", [])
                if answers:
                    ref = answers[0].get("text", "").strip()
                    if ref and len(ref) > 10:
                        pairs.append((context, ref))
                        count += 1
                        if count >= max_pairs:
                            return pairs
    return pairs


def evaluate():
    loader = CUADDataLoader()
    loader.load()
    contracts = loader.get_contracts()

    processor = DataProcessor(seed=42)
    splits = processor.create_splits(contracts)

    pairs = _build_text_ref_pairs(splits["test"], max_pairs=300)
    if not pairs:
        print("No reference answers found; skipping summarization evaluation.")
        return

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    summarizer = FusionSummarizer(use_abstractive=False)

    agg = {"rouge1": [], "rouge2": [], "rougeL": []}
    for text, ref in pairs:
        pred = summarizer.summarize(text)
        result = scorer.score(ref, pred)
        for k in agg:
            agg[k].append(result[k].fmeasure)

    df = pd.DataFrame(
        {
            "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
            "Value": [
                sum(agg["rouge1"]) / len(agg["rouge1"]),
                sum(agg["rouge2"]) / len(agg["rouge2"]),
                sum(agg["rougeL"]) / len(agg["rougeL"]),
            ],
        }
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"✓ Summarization evaluation saved to {OUT_CSV}")


if __name__ == "__main__":
    evaluate()
