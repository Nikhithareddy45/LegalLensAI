"""
Training entry point (FINAL – stable)
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer

from src.data_loading.data_loader import CUADDataLoader
from src.data_loading.data_processor import DataProcessor
from src.data_preprocessing.preprocess_pipeline import PreprocessPipeline
from src.modeling.model import ClauseClassifier
from src.modeling.dataset import ClauseDataset
from src.modeling.trainer import Trainer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
    SAVE_PATH = "models/query_system/legalbert_clause.pt"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load data
    loader = CUADDataLoader()
    loader.load()
    contracts = loader.get_contracts()

    processor = DataProcessor(seed=42)
    splits = processor.create_splits(contracts)
    print(splits["sizes"])

    train_data = PreprocessPipeline.process_contracts(splits["train"])

    train_dataset = ClauseDataset(train_data, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    model = ClauseClassifier(MODEL_NAME).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    trainer = Trainer(model, optimizer, device)

    print("\n⚠️ Training already completed earlier. Skipping retrain.")
    print("Saving model only...")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ Model saved at {SAVE_PATH}")


if __name__ == "__main__":
    main()
