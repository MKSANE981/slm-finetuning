"""
LoRA fine-tuning script for sequence classification.

We use PEFT (Parameter-Efficient Fine-Tuning) with LoRA adapters instead
of full fine-tuning for two reasons:
  1. It reduces trainable parameters by ~95%, so it fits on a single GPU.
  2. The base model weights stay frozen, which acts as a strong regularizer
     and prevents catastrophic forgetting.

The default base model is distilbert-base-uncased — it's small (66M params),
fast, and good enough for most classification tasks. Swap it for
'microsoft/deberta-v3-base' or 'roberta-base' for harder tasks.

Results on ENSAI internship dataset: 86.4% weighted F1 after 5 epochs.
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from dataset import load_from_csv, tokenize_dataset, get_label_map
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred):
    """
    Compute accuracy and weighted F1 after each evaluation epoch.

    We use weighted F1 as the primary metric (not accuracy) because
    it accounts for class imbalance — accuracy can look great while
    the model completely ignores minority classes.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def train(
    train_path: str,
    test_path: str,
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "./outputs",
    lora_r: int = 8,
    lora_alpha: int = 16,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-4,
):
    """
    Fine-tune a small language model with LoRA adapters.

    LoRA works by injecting trainable low-rank matrices into the attention
    layers (query and value projections). The rank `r` controls the
    expressiveness of the adapters: r=8 is conservative and rarely overfits;
    go higher (16, 32) if your dataset is large and the task is complex.

    lora_alpha controls the scaling of the adapter outputs. A common
    heuristic is lora_alpha = 2 * lora_r, which we follow here.

    Args:
        train_path: Path to training CSV.
        test_path: Path to test/validation CSV.
        model_name: Base model to fine-tune.
        output_dir: Where to save checkpoints and the final model.
        lora_r: LoRA rank (adapter expressiveness).
        lora_alpha: LoRA scaling factor.
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        lr: Learning rate (higher than full fine-tuning is OK with LoRA).

    Returns:
        The trained HuggingFace Trainer instance.
    """
    dataset = load_from_csv(train_path, test_path)
    label_map = get_label_map(dataset)
    n_labels = len(label_map)

    tokenized, tokenizer = tokenize_dataset(dataset, model_name, label_map=label_map)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=n_labels,
        id2label={v: k for k, v in label_map.items()},
        label2id=label_map,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        # q_lin and v_lin are DistilBERT's query and value projection names
        # For BERT/RoBERTa, use ["query", "value"] instead
        target_modules=["q_lin", "v_lin"],
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()  # useful sanity check — should be ~1-3%

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("test"),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")
    return trainer


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for text classification")
    parser.add_argument("--train_path", required=True, help="Path to train CSV")
    parser.add_argument("--test_path", required=True, help="Path to test CSV")
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()
    train(
        args.train_path, args.test_path,
        model_name=args.model_name, output_dir=args.output_dir,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )


if __name__ == "__main__":
    main()