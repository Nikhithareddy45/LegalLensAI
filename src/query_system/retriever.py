import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


class ClauseRetriever:
    """
    Retrieves relevant clauses using LegalBERT embeddings
    """

    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _embed(self, texts):
        """Generate embeddings for a list of texts"""
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            return embeddings.numpy()

    def retrieve(self, query, clauses, top_k=3):
        """
        Retrieve top-k relevant clauses
        """
        query_emb = self._embed([query])
        clause_embs = self._embed(clauses)

        similarities = cosine_similarity(query_emb, clause_embs)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "clause": clauses[idx],
                "score": float(similarities[idx])
            })

        return results
