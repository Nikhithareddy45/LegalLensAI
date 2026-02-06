from src.risk_detection.risk_analyzer import RiskAnalyzer

TEXT = "Liability is limited to the amount paid under this Agreement"
CLAUSE_TYPE = "Limitation of Liability"

analyzer = RiskAnalyzer()
result = analyzer.analyze(CLAUSE_TYPE, TEXT)

print("\n--- RISK ANALYSIS ---")
for k, v in result.items():
    print(f"{k}: {v}")
