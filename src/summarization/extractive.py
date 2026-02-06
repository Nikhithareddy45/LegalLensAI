"""
Extractive summarization using TF-IDF
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class ExtractiveSummarizer:
    def __init__(self, top_k=5):
        self.top_k = top_k

    def summarize(self, text: str):
        """
        text: full legal document (string)
        returns: list of top-k important sentences
        """
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 30]

        if len(sentences) <= self.top_k:
            return sentences

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(sentences)

        scores = np.asarray(tfidf.sum(axis=1)).ravel()
        top_indices = scores.argsort()[-self.top_k:][::-1]

        summary = [sentences[i] for i in top_indices]
        return summary
