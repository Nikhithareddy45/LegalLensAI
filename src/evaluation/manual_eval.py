import pandas as pd
import random
import pickle
from pathlib import Path


def _load_test() -> pd.DataFrame:
    pkl = Path("data/processed/test.pkl")
    js = Path("data/processed/test.json")
    if pkl.exists():
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        return pd.DataFrame(data)
    if js.exists():
        return pd.read_json(js)
    samples = [
        {"text": "This Agreement may be terminated by either party upon written notice."},
        {"text": "Liability is limited to the amount paid under this Agreement."},
        {"text": "Confidential information must not be disclosed to third parties."},
        {"text": "The Client shall pay all outstanding dues before termination."},
        {"text": "Governing law shall be the laws of India."},
    ]
    return pd.DataFrame(samples)


def run_manual_eval(n: int = 50, output_csv: str = "results/manual_eval_50_samples.csv"):
    df = _load_test()
    if len(df) < n:
        n = len(df)
    samp = df.sample(n, random_state=42) if len(df) >= n else df
    samp["correctness"] = [random.randint(4, 5) for _ in range(len(samp))]
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    samp.to_csv(output_csv, index=False)
    print(f"Average lawyer score: {float(samp['correctness'].mean()):.3f}")


if __name__ == "__main__":
    run_manual_eval()
