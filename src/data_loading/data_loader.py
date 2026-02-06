"""
Data loader for CUAD dataset
"""

import json
from pathlib import Path
from typing import Dict, List

from src.utils.logger import setup_logger
from src.data_loading.data_processor import DataProcessor

logger = setup_logger(__name__, "data_loading.log")


class CUADDataLoader:
    """
    Loads the CUAD (Contract Understanding Atticus Dataset) from JSON
    """

    def __init__(self, data_path: str = "data/raw/CUAD_v1.json"):
        self.data_path = Path(data_path)
        self.data = None

        if not self.data_path.exists():
            logger.error(f"Dataset not found at {self.data_path}")
            raise FileNotFoundError(f"CUAD dataset not found at {self.data_path}")

    def load(self) -> Dict:
        """
        Load CUAD JSON file
        """
        logger.info(f"Loading CUAD dataset from {self.data_path}...")

        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

            logger.info("✓ CUAD dataset loaded successfully")
            return self.data

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while loading dataset: {e}")
            raise

    def get_contracts(self) -> List[Dict]:
        """
        Return all contracts
        """
        if self.data is None:
            self.load()

        return self.data.get("data", [])

    def get_statistics(self) -> Dict:
        """
        Compute basic dataset statistics
        """
        if self.data is None:
            self.load()

        contracts = self.get_contracts()

        logger.info("Computing dataset statistics...")

        total_paragraphs = 0
        total_qas = 0
        clause_types = set()

        for contract in contracts:
            for paragraph in contract.get("paragraphs", []):
                total_paragraphs += 1

                for qa in paragraph.get("qas", []):
                    total_qas += 1

                    question = qa.get("question", "")
                    clause_type = (
                        question.split("(")[0].strip()
                        if "(" in question
                        else question
                    )
                    clause_types.add(clause_type)

        stats = {
            "total_contracts": len(contracts),
            "total_paragraphs": total_paragraphs,
            "total_qas": total_qas,
            "unique_clause_types": len(clause_types),
            "clause_types": sorted(list(clause_types)),
            "avg_qas_per_contract": (
                total_qas / len(contracts) if contracts else 0
            ),
        }

        logger.info("✓ Dataset statistics computed")
        logger.info(f"  - Contracts: {stats['total_contracts']}")
        logger.info(f"  - Paragraphs: {stats['total_paragraphs']}")
        logger.info(f"  - QA pairs: {stats['total_qas']}")
        logger.info(f"  - Clause types: {stats['unique_clause_types']}")

        return stats


def load_qa_corpus_for_index() -> list:
    """
    Build the test corpus (paragraph contexts) used for retrieval indexing.
    """
    loader = CUADDataLoader()
    loader.load()
    contracts = loader.get_contracts()
    processor = DataProcessor(seed=42)  # type: ignore
    splits = processor.create_splits(contracts)
    corpus = []
    for contract in splits["test"]:
        for paragraph in contract.get("paragraphs", []):
            ctx = paragraph.get("context", "")
            if ctx and len(ctx) > 20:
                corpus.append(ctx)
    return corpus


def load_qa_test_data():
    """
    Return (queries, gold_ids) for QA retrieval evaluation on test split.
    gold_ids reference positions in the test corpus built by load_qa_corpus_for_index.
    """
    loader = CUADDataLoader()
    loader.load()
    contracts = loader.get_contracts()
    processor = DataProcessor(seed=42)  # type: ignore
    splits = processor.create_splits(contracts)
    # Build corpus to define IDs
    corpus = []
    for contract in splits["test"]:
        for paragraph in contract.get("paragraphs", []):
            ctx = paragraph.get("context", "")
            if ctx and len(ctx) > 20:
                corpus.append(ctx)
    ctx_to_id = {c: i for i, c in enumerate(corpus)}
    # Build queries and gold ids
    queries = []
    gold_ids = []
    for contract in splits["test"]:
        for paragraph in contract.get("paragraphs", []):
            ctx = paragraph.get("context", "")
            if ctx and len(ctx) > 20 and ctx in ctx_to_id:
                for qa in paragraph.get("qas", []):
                    q = qa.get("question", "")
                    if q:
                        queries.append(q)
                        gold_ids.append(ctx_to_id[ctx])
    return queries, gold_ids
