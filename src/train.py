"""LoRA fine-tuning with PEFT + HuggingFace Trainer."""

import argparse
from dataset import load_from_csv, tokenize_dataset, get_label_map
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(eval_pred):
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
    dataset = load_from_csv(train_path, test_path)
    label_map = get_label_map(dataset)
    n_labels = len(label_map)

    tokenized, tokenizer = tokenize_dataset(dataset, model_name)

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
        target_modules=["q_lin", "v_lin"],
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("test"),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")
    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", required=True)
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