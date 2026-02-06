from src.query_system.query_engine import QueryEngine
from src.evaluation.retrieval_metrics import RetrievalMetrics
from src.evaluation.risk_metrics_eval import RiskEvaluation


def main():
    print("\n" + "=" * 60)
    print("CHECKPOINT 5: SYSTEM EVALUATION")
    print("=" * 60)

    clauses = [
        "This Agreement may be terminated by either party upon written notice.",
        "Liability is limited to the amount paid under this Agreement.",
        "Confidential information must not be disclosed to third parties.",
        "The Client shall pay all outstanding dues before termination."
    ]

    query = "What happens if the contract is terminated?"
    relevant_clauses = [
        "This Agreement may be terminated by either party upon written notice.",
        "The Client shall pay all outstanding dues before termination."
    ]

    engine = QueryEngine()
    results = engine.process_query(query, clauses)

    # ---- Retrieval Evaluation ----
    retrieved_texts = [r["clause"] for r in results]
    p_at_3 = RetrievalMetrics.precision_at_k(
        retrieved_texts, relevant_clauses, k=3
    )

    print(f"\n🔹 Retrieval Metrics")
    print(f"Precision@3: {p_at_3:.2f}")

    # ---- Risk Evaluation ----
    risk_results = [r["risk"] for r in results]

    coverage = RiskEvaluation.rule_coverage(risk_results)
    distribution = RiskEvaluation.risk_distribution(risk_results)

    print(f"\n🔹 Risk Detection Metrics")
    print(f"Rule Coverage: {coverage:.2f}")
    print("Risk Distribution:")
    print(f"  Low: {distribution['low']}")
    print(f"  Medium: {distribution['medium']}")
    print(f"  High: {distribution['high']}")

    print("\n" + "=" * 60)
    print("✓ CHECKPOINT 5 PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
