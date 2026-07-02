"""Dataset loading and tokenisation for text classification fine-tuning."""

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from typing import Optional
import pandas as pd


def load_from_csv(train_path: str, test_path: Optional[str] = None) -> DatasetDict:
    train_df = pd.read_csv(train_path)
    splits = {"train": Dataset.from_pandas(train_df)}
    if test_path:
        splits["test"] = Dataset.from_pandas(pd.read_csv(test_path))
    return DatasetDict(splits)


def tokenize_dataset(
    dataset: DatasetDict,
    model_name: str,
    text_col: str = "text",
    label_col: str = "label",
    max_length: int = 256,
) -> DatasetDict:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        enc = tokenizer(
            batch[text_col],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        enc["labels"] = batch[label_col]
        return enc

    tokenized = dataset.map(tokenize, batched=True, remove_columns=[text_col])
    return tokenized, tokenizer


def get_label_map(dataset: DatasetDict, label_col: str = "label") -> dict:
    labels = sorted(set(dataset["train"][label_col]))
    return {lab: i for i, lab in enumerate(labels)}