"""
Inference script – verifies trained model
"""

import torch
from transformers import AutoTokenizer
from src.modeling.model import ClauseClassifier

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MODEL_PATH = "models/query_system/legalbert_clause.pt"

SAMPLE_TEXT = """
The Client may terminate this Agreement at any time by providing 15 days written notice.
"""


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = ClauseClassifier(MODEL_NAME).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    inputs = tokenizer(
        SAMPLE_TEXT,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device)
        )

    logits = outputs["logits"]
    prediction = torch.argmax(logits, dim=1)

    print("✅ Inference successful")
    print("Logits:", logits)
    print("Predicted class:", prediction.item())


if __name__ == "__main__":
    main()
