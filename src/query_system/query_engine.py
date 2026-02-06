from src.query_system.retriever import ClauseRetriever
from src.summarization.fusion import FusionSummarizer
from src.risk_detection.semantic import SemanticRiskDetector
from src.risk_detection.rules import RuleBasedRiskDetector
from src.risk_detection.fusion import RiskFusion
from src.query_system.qa_resolver import generate_answer


class QueryEngine:
    """
    End-to-end query processing system
    """

    def __init__(self):
        self.retriever = ClauseRetriever()
        self.summarizer = FusionSummarizer()

        semantic = SemanticRiskDetector()
        rules = RuleBasedRiskDetector()
        self.risk_engine = RiskFusion(semantic, rules)

    def process_query(self, query, clauses):
        """
        Process a user query and return results
        """
        retrieved = self.retriever.retrieve(query, clauses)

        results = []
        for item in retrieved:
            clause_text = item["clause"]

            summary = self.summarizer.summarize(clause_text)
            risk = self.risk_engine.fuse(clause_text)

            results.append({
                "clause": clause_text,
                "similarity_score": item["score"],
                "summary": summary,
                "risk": risk
            })

        return results

    def answer(self, query, clauses, full_text: str = ""):
        retrieved = self.process_query(query, clauses)
        ans, conf = generate_answer(query, retrieved, full_text)
        evidence = [
            {"clause": r["clause"], "score": r["similarity_score"]}
            for r in sorted(retrieved, key=lambda x: x["similarity_score"], reverse=True)[:3]
        ]
        return ans, evidence, conf
