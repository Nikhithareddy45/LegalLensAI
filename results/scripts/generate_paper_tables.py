import pandas as pd
from pathlib import Path


def to_latex(csv_path: str, caption: str):
    df = pd.read_csv(csv_path)
    label = f"tab:{caption.lower().replace(' ', '_').replace('-', '_')}"
    print(df.to_latex(index=False, float_format="%.3f", caption=caption, label=label))


def main():
    tables = [
        ("results/metrics/dataset_stats.csv", "Dataset Statistics (CUAD)"),
        ("results/metrics/classification.csv", "Classification Performance"),
        ("results/metrics/summarization_rouge.csv", "Summarization Metrics (ROUGE)"),
        ("results/metrics/risk_detection.csv", "Risk Detection Performance"),
        ("results/metrics/retrieval.csv", "Retrieval Metrics"),
        ("results/ablation_results.csv", "Ablation Study Results"),
        ("results/statistical_significance.csv", "Statistical Significance Tests"),
    ]
    for p, c in tables:
        if Path(p).exists():
            to_latex(p, c)
        else:
            print(f"Missing {p}")


if __name__ == "__main__":
    main()
