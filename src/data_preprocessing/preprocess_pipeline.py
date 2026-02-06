"""
Preprocessing pipeline for CUAD
Converts contracts → QA samples usable by the model
"""

from typing import List, Dict


class PreprocessPipeline:
    @staticmethod
    def process_contracts(contracts: List[Dict]) -> List[Dict]:
        """
        Convert contracts into flat QA samples
        Output format MUST match Dataset expectations
        """

        samples = []

        for contract in contracts:
            for paragraph in contract.get("paragraphs", []):
                context = paragraph.get("context", "")

                for qa in paragraph.get("qas", []):
                    question = qa.get("question", "")
                    is_impossible = qa.get("is_impossible", False)

                    # Binary label: 1 = clause present, 0 = not present
                    label = 0 if is_impossible else 1

                    samples.append({
                        "question": question,
                        "context": context,
                        "label": label
                    })

        return samples
