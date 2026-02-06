class RetrievalMetrics:
    """
    Evaluate clause retrieval quality
    """

    @staticmethod
    def precision_at_k(retrieved_clauses, relevant_clauses, k=3):
        """
        Precision@K = relevant_retrieved / K
        """
        retrieved_top_k = retrieved_clauses[:k]
        relevant_count = sum(
            1 for clause in retrieved_top_k
            if clause in relevant_clauses
        )
        return relevant_count / k
