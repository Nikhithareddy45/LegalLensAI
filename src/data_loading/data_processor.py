"""
Data processor for CUAD dataset
- Creates train / validation / test splits
- Extracts QA pairs
"""

import pickle
from pathlib import Path
from typing import Dict, List

from sklearn.model_selection import train_test_split

from src.utils.logger import setup_logger

logger = setup_logger(__name__, "data_processing.log")


class DataProcessor:
    """
    Process CUAD contracts into splits and QA pairs
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def create_splits(
        self,
        contracts: List[Dict],
        train_size: float = 0.75,
        val_size: float = 0.15,
        test_size: float = 0.10,
    ) -> Dict:
        """
        Create train / val / test splits
        """

        logger.info("Creating train/val/test splits...")

        assert abs(train_size + val_size + test_size - 1.0) < 0.01, (
            "Split sizes must sum to 1.0"
        )

        # First split: train+val vs test
        train_val, test = train_test_split(
            contracts,
            test_size=test_size,
            random_state=self.seed,
            shuffle=True,
        )

        # Second split: train vs val
        val_ratio = val_size / (train_size + val_size)

        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=self.seed,
            shuffle=True,
        )

        self.train_data = train
        self.val_data = val
        self.test_data = test

        logger.info("✓ Splits created successfully")
        logger.info(f"  - Train: {len(train)} contracts")
        logger.info(f"  - Val:   {len(val)} contracts")
        logger.info(f"  - Test:  {len(test)} contracts")

        return {
            "train": train,
            "val": val,
            "test": test,
            "sizes": {
                "train": len(train),
                "val": len(val),
                "test": len(test),
                "total": len(contracts),
            },
        }

    def extract_qa_pairs(self, contracts: List[Dict]) -> List[Dict]:
        """
        Extract QA pairs from CUAD contracts
        """

        logger.info(f"Extracting QA pairs from {len(contracts)} contracts...")

        qa_pairs = []

        for contract in contracts:
            contract_id = contract.get("title", "Unknown")

            for paragraph in contract.get("paragraphs", []):
                context = paragraph.get("context", "")

                for qa in paragraph.get("qas", []):
                    question = qa.get("question", "")
                    clause_type = (
                        question.split("(")[0].strip()
                        if "(" in question
                        else question
                    )

                    qa_pairs.append(
                        {
                            "contract_id": contract_id,
                            "question": question,
                            "clause_type": clause_type,
                            "context": context,
                            "answers": qa.get("answers", []),
                            "is_impossible": qa.get("is_impossible", False),
                            "id": qa.get("id", ""),
                        }
                    )

        logger.info(f"✓ Extracted {len(qa_pairs)} QA pairs")
        return qa_pairs

    def save_splits(self, output_dir: str = "data/processed"):
        """
        Save splits to disk
        """

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving splits to {output_dir}...")

        with open(output_path / "train.pkl", "wb") as f:
            pickle.dump(self.train_data, f)

        with open(output_path / "val.pkl", "wb") as f:
            pickle.dump(self.val_data, f)

        with open(output_path / "test.pkl", "wb") as f:
            pickle.dump(self.test_data, f)

        logger.info("✓ Splits saved successfully")
