"""
Evaluation metrics
"""

import torch
from sklearn.metrics import precision_recall_fscore_support


class Evaluator:
    @staticmethod
    def evaluate(model, dataloader, device):
        model.eval()

        preds = []
        labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                label = batch["labels"].cpu().numpy()

                output = model(input_ids, attention_mask)
                prediction = (output.cpu().numpy() > 0.5).astype(int)

                preds.extend(prediction.flatten())
                labels.extend(label)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
