"""Evaluation and inference on fine-tuned model."""

import argparse
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_pipeline(model_dir: str):
    config = PeftConfig.from_pretrained(model_dir)
    clf = pipeline(
        "text-classification",
        model=model_dir,
        tokenizer=config.base_model_name_or_path,
        device=-1,
    )
    return clf


def evaluate(clf, texts: list, labels: list) -> dict:
    preds = [clf(t)[0]["label"] for t in texts]
    report = classification_report(labels, preds, output_dict=True)
    print(classification_report(labels, preds))
    return {"preds": preds, "report": report}


def plot_confusion(labels: list, preds: list, label_names: list, save_path: str = None):
    cm = confusion_matrix(labels, preds, labels=label_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_names, yticklabels=label_names)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="label")
    args = parser.parse_args()

    df = pd.read_csv(args.test_path)
    clf = load_pipeline(args.model_dir)
    results = evaluate(clf, df[args.text_col].tolist(), df[args.label_col].tolist())
    print(f"\nWeighted F1: {results['report']['weighted avg']['f1-score']:.4f}")


if __name__ == "__main__":
    main()