# SLM Fine-tuning for Text Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Stack](https://img.shields.io/badge/Stack-PyTorch%20%7C%20HuggingFace%20%7C%20LoRA%2FPEFT-orange)

Fine-tuning a small language model (Qwen / DistilBERT-family) on domain-specific text
classification tasks using LoRA (PEFT) for parameter-efficient training.

Originally developed for insurance document classification. Adapted here for public
demonstration using the AG News dataset.

## Key Results (insurance domain — proprietary data)

| Metric | Value |
|---|---|
| Accuracy | **86.4%** |
| Inference speed | **2.1× faster** than GPT-4o-mini |
| Model size | 0.6B parameters |
| Deployment | Local / sovereign (no external API) |

## Approach

1. **Dataset preparation** — domain-specific corpus, label balancing, train/val/test split
2. **LoRA fine-tuning** — low-rank adapters via PEFT, targeting attention layers
3. **Evaluation** — accuracy, F1, confusion matrix, calibration
4. **Export** — merged weights, ONNX export option for fast inference

## Tech Stack

```
PyTorch · HuggingFace Transformers · PEFT (LoRA) · Datasets · Evaluate · Accelerate
```

## Quick Start

```bash
pip install -r requirements.txt

# Train on AG News (public demo)
python src/train.py --model distilbert-base-uncased --dataset ag_news --epochs 3

# Evaluate
python src/evaluate.py --checkpoint ./outputs/checkpoint-final
```

## Project Structure

```
slm-finetuning/
├── src/
│   ├── dataset.py        # Data loading, tokenisation, splits
│   ├── train.py          # LoRA fine-tuning loop
│   └── evaluate.py       # Metrics + confusion matrix
├── outputs/              # Checkpoints (gitignored)
├── requirements.txt
└── README.md
```