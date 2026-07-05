"""
Dataset preparation for text classification fine-tuning.

This module handles the boring-but-critical part of any NLP project:
getting data into the right shape before the model sees it. A lot of
fine-tuning failures come from subtle tokenization issues (wrong padding,
truncation cutting off the label signal, etc.) — so we keep this layer
explicit and easy to inspect.
"""

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from typing import Optional
import pandas as pd


def load_from_csv(train_path: str, test_path: Optional[str] = None) -> DatasetDict:
    """
    Load train (and optionally test) CSVs into a HuggingFace DatasetDict.

    We use DatasetDict instead of plain DataFrames because HuggingFace
    Trainer expects this format, and it handles batched tokenization
    more efficiently.

    Args:
        train_path: Path to the training CSV. Must have at least a text
            column and a label column (see tokenize_dataset for defaults).
        test_path: Optional path to a held-out test CSV.

    Returns:
        DatasetDict with 'train' split (and 'test' if provided).
    """
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
) -> tuple:
    """
    Tokenize all splits in the dataset using the model's own tokenizer.

    We use padding="max_length" (rather than dynamic padding) because
    the Trainer with fp16 works most stably with fixed-size tensors.
    max_length=256 is a conscious trade-off: most classification tasks
    don't need full 512-token context, and halving it roughly doubles
    the throughput.

    The text column is removed after tokenization — the model only needs
    input_ids, attention_mask and labels.

    Args:
        dataset: DatasetDict from load_from_csv().
        model_name: HuggingFace model name or local path.
        text_col: Name of the column containing the input text.
        label_col: Name of the column containing integer or string labels.
        max_length: Maximum number of tokens per sample.

    Returns:
        Tuple of (tokenized DatasetDict, tokenizer instance).
    """
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
    """
    Build a deterministic label-to-integer mapping from the training set.

    Sorting the labels alphabetically ensures the mapping is reproducible
    across runs — important for evaluation, where predicted class IDs must
    match the training mapping exactly.

    Args:
        dataset: DatasetDict (only the 'train' split is inspected).
        label_col: Name of the label column.

    Returns:
        Dict mapping label string/int → integer class index.
    """
    labels = sorted(set(dataset["train"][label_col]))
    return {lab: i for i, lab in enumerate(labels)}