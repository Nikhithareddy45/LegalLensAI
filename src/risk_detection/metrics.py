"""
Risk Detection Evaluation Metrics
"""

import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)

class RiskMetrics:
    """Compute risk detection metrics"""
    
    @staticmethod
    def compute_f1(tp: int, fp: int, fn: int) -> float:
        """Compute F1 score"""
        if tp + fp == 0 or tp + fn == 0:
            return 0.0
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def compute_risk_statistics(all_results: List[Dict]) -> Dict:
        """Compute overall risk statistics"""
        
        overall_risks = [r['overall_semantic_risk'] for r in all_results]
        triggered_counts = [r['rule_triggered_count'] for r in all_results]
        
        stats = {
            'avg_overall_risk': np.mean(overall_risks),
            'max_risk': np.max(overall_risks),
            'min_risk': np.min(overall_risks),
            'avg_rules_triggered': np.mean(triggered_counts),
            'samples_with_high_risk': sum(1 for r in overall_risks if r > 0.6),
            'risk_distribution': {
                'low': sum(1 for r in overall_risks if r < 0.3),
                'medium': sum(1 for r in overall_risks if 0.3 <= r < 0.6),
                'high': sum(1 for r in overall_risks if r >= 0.6)
            }
        }
        
        return stats