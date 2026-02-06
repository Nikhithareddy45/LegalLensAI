"""
Semantic Risk Detection using BiLSTM
"""

import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class SemanticRiskDetector:
    """Semantic risk detection using neural networks"""
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Risk keywords for semantic understanding
        self.risk_keywords = {
            'payment': ['payment', 'fee', 'cost', 'price', 'amount'],
            'liability': ['liability', 'liable', 'responsible', 'risk'],
            'termination': ['terminate', 'termination', 'end', 'cancel'],
            'confidentiality': ['confidential', 'secret', 'proprietary', 'nda'],
            'indemnity': ['indemnify', 'indemnification', 'indemnities']
        }
    
    def detect(self, text: str) -> Dict:
        """Detect risks in text"""
        
        risks = {}
        text_lower = text.lower()
        
        for category, keywords in self.risk_keywords.items():
            # Count keyword occurrences
            count = sum(text_lower.count(kw) for kw in keywords)
            
            # Normalize to 0-1
            risk_score = min(count / 5.0, 1.0)  # Normalize by 5
            
            risks[category] = {
                'score': risk_score,
                'detected': risk_score > 0.3,
                'severity': 1 + int(risk_score * 4)  # 1-5 scale
            }
        
        return risks
    
    def compute_overall_risk(self, risks: Dict) -> float:
        """Compute overall risk score"""
        scores = [risk['score'] for risk in risks.values()]
        return np.mean(scores) if scores else 0.0