"""
Abstractive summarization using BART
"""

import torch
from transformers import BartForConditionalGeneration, BartTokenizer


class AbstractiveSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def summarize(self, text, max_length=150, min_length=50):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                num_beams=4,
                max_length=120,
                min_length=40,
                no_repeat_ngram_size=3,
                forced_bos_token_id=self.tokenizer.bos_token_id,
                early_stopping=True
            )

        return self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )
