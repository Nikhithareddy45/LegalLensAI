"""
Fusion summarization: Extractive + Abstractive
"""

from src.summarization.extractive import ExtractiveSummarizer
from src.summarization.abstractive import AbstractiveSummarizer


class FusionSummarizer:
    def __init__(self, use_abstractive: bool = True):
        self.extractive = ExtractiveSummarizer()
        self.abstractive = None
        if use_abstractive:
            try:
                self.abstractive = AbstractiveSummarizer()
            except OSError:
                self.abstractive = None
            except Exception:
                self.abstractive = None

    def summarize(self, text):
        extracted = self.extractive.summarize(text)
        joined_text = " ".join(extracted)
        if self.abstractive is None or not joined_text:
            return joined_text
        return self.abstractive.summarize(joined_text)
