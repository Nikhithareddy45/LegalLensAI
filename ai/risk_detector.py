# ai/risk_detector.py

import re

class RiskDetector:

    HIGH_RISK = [
        ("termination", "High"),
        ("penalty", "High"),
        ("breach", "High"),
        ("liability", "High"),
        ("non-compete", "High"),
    ]

    MEDIUM_RISK = [
        ("renewal", "Medium"),
        ("confidentiality", "Medium"),
        ("indemnity", "Medium"),
        ("payment terms", "Medium"),
    ]

    LOW_RISK = [
        ("jurisdiction", "Low"),
        ("governing law", "Low"),
        ("notice period", "Low"),
        ("service clauses", "Low"),
    ]

    def analyze(self, text: str):
        risks = []

        def find_risks(patterns):
            local_risks = []
            for pat, weight in patterns:
                if re.search(pat, text, re.IGNORECASE):
                    excerpt = self.extract_context(text, pat)
                    local_risks.append({
                        "type": pat,
                        "weight": weight,
                        "context": excerpt
                    })
            return local_risks

        risks += find_risks(self.HIGH_RISK)
        risks += find_risks(self.MEDIUM_RISK)
        risks += find_risks(self.LOW_RISK)

        return risks

    def extract_context(self, text, keyword):
        idx = text.lower().find(keyword.lower())
        if idx == -1:
            return ""
        start = max(0, idx - 80)
        end = min(len(text), idx + 150)
        return text[start:end]
