"""
Text cleaning and normalization utilities
"""

import re


class TextCleaner:
    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""

        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z0-9.,;:()\- ]", "", text)

        return text.strip()
