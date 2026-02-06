import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support
from pathlib import Path
from src.data_loading.data_loader import CUADDataLoader
from src.data_loading.data_processor import DataProcessor
from src.data_preprocessing.preprocess_pipeline import PreprocessPipeline
from src.modeling.model import ClauseClassifier
from src.modeling.dataset import ClauseDataset


MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MODEL_PATH = "models/query_system/legalbert_clause.pt"
OUT_CSV = Path("results/metrics/classification.csv")


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    loader = CUADDataLoader()
    loader.load()
    contracts = loader.get_contracts()

    processor = DataProcessor(seed=42)
    splits = processor.create_splits(contracts)
    test_data = PreprocessPipeline.process_contracts(splits["test"])

    dataset = ClauseDataset(test_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    model = ClauseClassifier(MODEL_NAME).to(device)
    if Path(MODEL_PATH).exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["labels"].cpu().numpy()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"].cpu().numpy()
            pred = logits.argmax(axis=1)
            preds.extend(pred.tolist())
            labels.extend(y.tolist())

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )

    df = pd.DataFrame(
        {
            "Metric": ["Precision_macro", "Recall_macro", "F1_macro", "Precision_micro", "Recall_micro", "F1_micro"],
            "Value": [precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro],
        }
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"✓ Classification evaluation saved to {OUT_CSV}")


if __name__ == "__main__":
    evaluate()
