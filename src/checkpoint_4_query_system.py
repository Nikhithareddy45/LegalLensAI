from src.query_system.query_engine import QueryEngine


def main():
    print("\n" + "=" * 60)
    print("CHECKPOINT 4: QUERY SYSTEM TESTING")
    print("=" * 60)

    clauses = [
        "This Agreement may be terminated by either party upon written notice.",
        "Liability is limited to the amount paid under this Agreement.",
        "Confidential information must not be disclosed to third parties.",
        "The Client shall pay all outstanding dues before termination."
    ]

    query = "What happens if the agreement is terminated?"

    engine = QueryEngine()
    results = engine.process_query(query, clauses)

    print(f"\nUser Query: {query}\n")

    for i, res in enumerate(results, 1):
        print(f"Result {i}")
        print("-" * 40)
        print(f"Clause: {res['clause']}")
        print(f"Similarity Score: {res['similarity_score']:.3f}")
        print(f"Summary: {res['summary']}")
        print(f"Overall Risk: {res['risk']['overall_semantic_risk']:.3f}")
        print()

    print("=" * 60)
    print("✓ CHECKPOINT 4 PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
