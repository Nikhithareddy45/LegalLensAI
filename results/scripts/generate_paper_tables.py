import pandas as pd
from pathlib import Path


def to_latex(csv_path: str, caption: str):
    df = pd.read_csv(csv_path)
    label = f"tab:{caption.lower().replace(' ', '_')}"
    print(df.to_latex(index=False, float_format="%.3f", caption=caption, label=label))


def main():
    paths = [
        ("results/metrics/per_category_f1.csv", "Per-Category F1 Scores (41 clauses)"),
        ("results/ablation_results.csv", "Ablation Study Results"),
        ("results/statistical_significance.csv", "Statistical Significance Tests"),
    ]
    for p, c in paths:
        if Path(p).exists():
            to_latex(p, c)
        else:
            print(f"Missing {p}")


if __name__ == "__main__":
    main()
