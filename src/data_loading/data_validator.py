"""
Data validator for CUAD dataset
Ensures structural integrity and checks missing values
"""

from typing import Dict
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "data_validation.log")


class DataValidator:
    """
    Validates CUAD dataset structure and content
    """

    @staticmethod
    def validate_structure(data: Dict) -> bool:
        """
        Validate JSON structure
        """
        logger.info("Validating dataset structure...")

        try:
            assert "data" in data, "Missing 'data' key"

            for i, contract in enumerate(data["data"]):
                assert "title" in contract, f"Contract {i} missing title"
                assert "paragraphs" in contract, f"Contract {i} missing paragraphs"

                for j, para in enumerate(contract["paragraphs"]):
                    assert "context" in para, f"Contract {i}, Para {j} missing context"
                    assert "qas" in para, f"Contract {i}, Para {j} missing qas"

                    for k, qa in enumerate(para["qas"]):
                        assert "question" in qa, f"Missing question at {i}-{j}-{k}"
                        assert "answers" in qa, f"Missing answers at {i}-{j}-{k}"
                        assert "id" in qa, f"Missing id at {i}-{j}-{k}"

            logger.info("✓ Dataset structure validation PASSED")
            return True

        except AssertionError as e:
            logger.error(f"Structure validation FAILED: {e}")
            return False

    @staticmethod
    def check_missing_values(data: Dict) -> Dict:
        """
        Check missing or empty values
        """
        logger.info("Checking missing values...")

        stats = {
            "total_qas": 0,
            "missing_questions": 0,
            "missing_context": 0,
            "missing_answers": 0,
            "empty_answers": 0,
        }

        for contract in data["data"]:
            for para in contract["paragraphs"]:
                for qa in para["qas"]:
                    stats["total_qas"] += 1

                    if not qa.get("question"):
                        stats["missing_questions"] += 1
                    if not para.get("context"):
                        stats["missing_context"] += 1
                    if qa.get("answers") is None:
                        stats["missing_answers"] += 1
                    elif len(qa["answers"]) == 0:
                        stats["empty_answers"] += 1

        logger.info("✓ Missing value check completed")
        for k, v in stats.items():
            logger.info(f"  - {k}: {v}")

        return stats

    @staticmethod
    def validate_all(data: Dict) -> Dict:
        """
        Run all validations
        """
        structure_ok = DataValidator.validate_structure(data)
        missing_stats = DataValidator.check_missing_values(data)

        status = "PASSED" if structure_ok else "FAILED"

        logger.info(f"✓ Overall validation status: {status}")

        return {
            "status": status,
            "structure_valid": structure_ok,
            "missing_values": missing_stats,
        }
