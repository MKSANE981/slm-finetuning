"""
Evaluation and inference for the fine-tuned classification model.

This module is intentionally separate from training so you can run
evaluation on a saved checkpoint without re-training. It also exposes
a simple inference function (load_pipeline) that you can import and
call from a notebook or a downstream service.
"""

import argparse
import numpy as np
import pandas as pd
from transformers import pipeline
from peft import PeftConfig
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_pipeline(model_dir: str):
    """
    Load a saved PEFT model as a HuggingFace text-classification pipeline.

    The pipeline handles tokenization and softmax internally, so you just
    pass raw text strings and get back predicted labels and scores.

    We read PeftConfig to find the base model name automatically — you
    don't need to specify it separately, even after saving.

    Args:
        model_dir: Path to the directory saved by train.py.

    Returns:
        A HuggingFace pipeline object (callable on strings or lists of strings).
    """
    config = PeftConfig.from_pretrained(model_dir)
    clf = pipeline(
        "text-classification",
        model=model_dir,
        tokenizer=config.base_model_name_or_path,
        device=-1,  # -1 = CPU; set to 0 for GPU
    )
    return clf


def evaluate(clf, texts: list, labels: list) -> dict:
    """
    Run inference on a list of texts and compute classification metrics.

    Prints a full per-class classification report (precision, recall, F1)
    and returns the raw predictions for further analysis.

    Args:
        clf: A pipeline loaded by load_pipeline().
        texts: List of raw text strings.
        labels: Corresponding ground-truth labels.

    Returns:
        Dict with 'preds' (predicted labels) and 'report' (sklearn dict).
    """
    preds = [clf(t)[0]["label"] for t in texts]
    report = classification_report(labels, preds, output_dict=True)
    print(classification_report(labels, preds))
    return {"preds": preds, "report": report}


def plot_confusion(labels: list, preds: list, label_names: list, save_path: str = None):
    """
    Plot a confusion matrix heatmap to spot systematic misclassifications.

    The diagonal is what you want to be dark. Off-diagonal cells reveal
    which classes the model confuses — useful for deciding whether to
    collect more data for specific categories.

    Args:
        labels: Ground-truth label list.
        preds: Predicted label list.
        label_names: Ordered list of class names (must match training order).
        save_path: If provided, save the figure to this path instead of showing it.
    """
    cm = confusion_matrix(labels, preds, labels=label_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_names, yticklabels=label_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned classification model")
    parser.add_argument("--model_dir", required=True, help="Path to saved model directory")
    parser.add_argument("--test_path", required=True, help="Path to test CSV")
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="label")
    args = parser.parse_args()

    df = pd.read_csv(args.test_path)
    clf = load_pipeline(args.model_dir)
    results = evaluate(clf, df[args.text_col].tolist(), df[args.label_col].tolist())
    print(f"\nWeighted F1: {results['report']['weighted avg']['f1-score']:.4f}")


if __name__ == "__main__":
    main()