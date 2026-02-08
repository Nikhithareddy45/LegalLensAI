import sys
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.data_loading.data_loader import CUADDataLoader
from src.data_loading.data_processor import DataProcessor
from src.risk_detection.semantic import SemanticRiskDetector
from src.risk_detection.rules import RuleBasedRiskDetector
from src.risk_detection.fusion import RiskFusion

OUT_CSV = Path("results/metrics/risk_detection_metrics.csv")


def _severity_bucket(score: float) -> str:
    if score >= 0.6:
        return "high"
    if score >= 0.3:
        return "medium"
    return "low"


def evaluate():
    loader = CUADDataLoader()
    loader.load()
    contracts = loader.get_contracts()

    processor = DataProcessor(seed=42)
    splits = processor.create_splits(contracts)

    semantic = SemanticRiskDetector()
    rules = RuleBasedRiskDetector()
    fusion = RiskFusion(semantic, rules)

    texts = []
    for contract in splits["test"]:
        for paragraph in contract.get("paragraphs", []):
            context = paragraph.get("context", "")
            if context and len(context) > 20:
                texts.append(context)

    high = medium = low = rule_hits = 0
    for t in texts:
        r = fusion.fuse(t)
        sev = _severity_bucket(r["overall_semantic_risk"])
        if sev == "high":
            high += 1
        elif sev == "medium":
            medium += 1
        else:
            low += 1
        if r["rule_triggered_count"] > 0:
            rule_hits += 1

    total = max(len(texts), 1)
    df = pd.DataFrame(
        {
            "Metric": ["rule_coverage", "severity_high_pct", "severity_medium_pct", "severity_low_pct"],
            "Value": [rule_hits / total, high / total, medium / total, low / total],
        }
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"✓ Risk evaluation saved to {OUT_CSV}")


if __name__ == "__main__":
    evaluate()
