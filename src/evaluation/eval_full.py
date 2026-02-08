import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluation.dataset_stats import compute_dataset_stats
from src.evaluation.eval_classification import evaluate as eval_cls
from src.evaluation.eval_summarization import evaluate as eval_sum
from src.evaluation.eval_risk import evaluate as eval_risk
from src.evaluation.eval_retrieval import evaluate as eval_ret


def main():
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    try:
        compute_dataset_stats()
    except Exception as e:
        print(f"[WARN] Dataset stats failed: {e}")
    try:
        eval_cls()
    except Exception as e:
        print(f"[WARN] Classification eval failed: {e}")
    try:
        eval_sum()
    except Exception as e:
        print(f"[WARN] Summarization eval failed: {e}")
    try:
        eval_risk()
    except Exception as e:
        print(f"[WARN] Risk eval failed: {e}")
    try:
        eval_ret()
    except Exception as e:
        print(f"[WARN] Retrieval eval failed: {e}")
    print("✓ eval_full completed")


if __name__ == "__main__":
    main()
