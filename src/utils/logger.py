"""
Central logging utility for the LegalLensAI project
Used across data loading, preprocessing, training, and evaluation
"""

import logging
import sys
from pathlib import Path


# Base logs directory
LOG_DIR = Path("results/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Create and configure a logger

    Args:
        name (str): Logger name (usually __name__)
        log_file (str, optional): Log file name

    Returns:
        logging.Logger
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers (very important)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Log format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = LOG_DIR / log_file
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
