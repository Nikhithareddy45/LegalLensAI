class RiskEvaluation:
    """
    Evaluate risk detection behavior
    """

    @staticmethod
    def rule_coverage(results):
        """
        Percentage of samples where at least one rule triggered
        """
        triggered = sum(
            1 for r in results
            if r["rule_triggered_count"] > 0
        )
        return triggered / len(results)

    @staticmethod
    def risk_distribution(results):
        """
        Count low / medium / high risk samples
        """
        dist = {"low": 0, "medium": 0, "high": 0}

        for r in results:
            score = r["overall_semantic_risk"]
            if score < 0.3:
                dist["low"] += 1
            elif score < 0.6:
                dist["medium"] += 1
            else:
                dist["high"] += 1

        return dist
