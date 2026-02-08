import sys
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.data_loading.data_loader import CUADDataLoader

OUT_CSV = Path("results/metrics/dataset_stats.csv")


def compute_dataset_stats(cuad_path: str = "data/raw/CUAD_v1.json"):
    loader = CUADDataLoader(data_path=cuad_path)
    loader.load()
    stats = loader.get_statistics()
    df = pd.DataFrame(
        {
            "Metric": [
                "Number of Contracts",
                "Number of Clauses",
                "Number of QA Pairs",
                "Number of Clause Types",
                "Avg QAs per Contract",
            ],
            "Value": [
                stats["total_contracts"],
                stats["total_paragraphs"],
                stats["total_qas"],
                stats["unique_clause_types"],
                stats["avg_qas_per_contract"],
            ],
        }
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"✓ Dataset stats saved to {OUT_CSV}")


if __name__ == "__main__":
    compute_dataset_stats()
