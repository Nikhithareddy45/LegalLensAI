"""
Fusion summarization: Extractive + Abstractive
"""

from src.summarization.extractive import ExtractiveSummarizer
from src.summarization.abstractive import AbstractiveSummarizer


class FusionSummarizer:
    def __init__(self):
        self.extractive = ExtractiveSummarizer()
        self.abstractive = AbstractiveSummarizer()

    def summarize(self, text):
        extracted = self.extractive.summarize(text)
        joined_text = " ".join(extracted)
        return self.abstractive.summarize(joined_text)
