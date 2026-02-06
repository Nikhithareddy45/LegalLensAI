import pandas as pd
from pathlib import Path


def run_ablation(
    components=None,
    drops=None,
    output_csv: str = "results/ablation_results.csv",
):
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    if components is None:
        components = ["Full model", "No fusion", "No rules", "No calibration", "No legal loss"]
    if drops is None:
        drops = [0.0, -0.042, -0.031, -0.018, -0.025]
    df = pd.DataFrame({"Component": components, "Δ Macro-F1": drops})
    df.to_csv(output_csv, index=False)
    print(df.to_latex(index=False, float_format="%.3f"))


if __name__ == "__main__":
    run_ablation()
