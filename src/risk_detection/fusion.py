"""
Risk Detection Fusion
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

class RiskFusion:
    """Fuse semantic and rule-based risk detection"""
    
    def __init__(self, semantic_detector, rule_detector):
        self.semantic = semantic_detector
        self.rules = rule_detector
    
    def fuse(self, text: str) -> Dict:
        """Fuse predictions from both models"""
        
        # Get predictions from both models
        semantic_risks = self.semantic.detect(text)
        rule_risks = self.rules.detect(text)
        
        # Fuse results
        fused_results = {
            'semantic_risks': semantic_risks,
            'rule_risks': rule_risks,
            'overall_semantic_risk': self.semantic.compute_overall_risk(semantic_risks),
            'rule_triggered_count': sum(1 for risk in rule_risks.values() if risk['triggered']),
            'high_risk_categories': []
        }
        
        # Identify high-risk categories
        for category, risk in semantic_risks.items():
            if risk['score'] > 0.6:
                fused_results['high_risk_categories'].append(category)
        
        return fused_results