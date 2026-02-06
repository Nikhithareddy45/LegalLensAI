"""
Rule-based Risk Detection
"""

import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class RuleBasedRiskDetector:
    """
    Rule-based risk detection using predefined regex patterns
    """

    def __init__(self):
        # Rule name : regex pattern
        self.rules = {
            "no_liability": r"(no|without|exclude|excluding)\s+(any\s+)?liability",
            "liability_limitation": r"limitation\s+of\s+liability|liability\s+shall\s+be\s+limited",
            "unreasonable_payment": r"(unlimited|excessive|unreasonable)\s+(fees?|payment|charges?)",
            "payment_obligation": r"(shall|must)\s+pay\s+(all|any|outstanding)",
            "forced_termination": r"(immediate|summary|without\s+notice)\s+(termination|cancellation)",
            "termination_clause": r"(termination|terminate)\s+by\s+(either|any)\s+party",
            "confidentiality_risk": r"(confidential|non[-\s]?disclosure|nda)",
            "ip_concerns": r"(intellectual\s+property|patent|trademark|copyright)",
            "data_protection": r"(gdpr|data\s+protection|privacy|personal\s+data)",
            "non_compete": r"non[-\s]?compete|non[-\s]?solicitation",
            "indemnity": r"(indemnify|indemnification|hold\s+harmless)"
        }

    def detect(self, text: str) -> Dict:
        """
        Detect risks in text using regex rules
        """

        risks = {}
        text_lower = text.lower()

        for rule_name, pattern in self.rules.items():
            try:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                triggered = len(matches) > 0

                # Severity logic
                if triggered:
                    if rule_name in ["no_liability", "liability_limitation", "indemnity"]:
                        severity = 5
                    elif rule_name in ["forced_termination", "non_compete"]:
                        severity = 4
                    else:
                        severity = 3
                else:
                    severity = 1

                risks[rule_name] = {
                    "triggered": triggered,
                    "occurrences": len(matches),
                    "severity": severity
                }

            except Exception as e:
                logger.error(f"Rule processing failed for {rule_name}: {e}")
                risks[rule_name] = {
                    "triggered": False,
                    "occurrences": 0,
                    "severity": 0
                }

        return risks
