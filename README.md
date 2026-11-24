# SLM Fine-tuning for Text Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Stack](https://img.shields.io/badge/Stack-PyTorch%20%7C%20HuggingFace%20%7C%20LoRA%2FPEFT-orange)

Fine-tuning a small language model on a domain-specific text classification task
using LoRA (PEFT) for parameter-efficient training.

The method was first applied to insurance document classification in a professional
setting (proprietary data and model, not published here). This repository is a public
demonstration of the same approach using DistilBERT and the AG News dataset.

## Results

| Setting | Result |
|---|---|
| Insurance documents (proprietary, not reproducible here) | **86.4% accuracy** on a dedicated test set of 81 documents, against 74.1% for o4-mini on the same set |
| This public demo (AG News subset) | run `src/evaluate.py` to obtain your own figures |

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

## Pipeline Interconnections

Steps must run in order — each produces outputs consumed by the next:

```
prepare_data.py
  └─ downloads fancyzhx/ag_news from HuggingFace Hub
  └─ saves data/train.csv (8 000 rows) + data/test.csv (1 600 rows)
        ↓
src/dataset.py  →  tokenize_dataset()
  ├─ label_map {"World":0, "Sports":1, "Business":2, "Sci/Tech":3}
  │  must match the CSV label column exactly — mismatch raises ValueError
  ├─ tokenization truncates to max_length=128; longer articles are clipped
  └─ DataCollatorWithPadding pads to batch max length at training time
        ↓
src/train.py  →  TrainingArguments + Trainer
  ├─ fp16=torch.cuda.is_available() — disabled on CPU, enabled on GPU only
  ├─ LoRA rank r=8, alpha=16, targeting ["q_lin","v_lin"] attention layers
  └─ checkpoints saved to outputs/ (gitignored; ~200 MB per checkpoint)
        ↓
src/evaluate.py  →  accuracy, F1, confusion matrix
  └─ loads checkpoint from outputs/; path must match what train.py wrote
```

## Platform Notes

| Note | Detail |
|------|--------|
| **CPU-only training** | `fp16=False` on CPU — mixed precision requires CUDA. Expect ~15 min/epoch on CPU for 8 000 examples with DistilBERT. Use a GPU or reduce `num_train_epochs`. |
| **Model download** | First run downloads DistilBERT (~250 MB) and AG News (~30 MB) from HuggingFace Hub. Subsequent runs use the local cache. |
| **Proprietary results** | The 86.4% accuracy figure is from the insurance document classification task on proprietary data. AG News results will differ and are not comparable. |

## Project Structure

```
slm-finetuning/
├── src/
│   ├── dataset.py        # Data loading, tokenisation, splits
│   ├── train.py          # LoRA fine-tuning loop (fp16 auto-detected)
│   └── evaluate.py       # Metrics + confusion matrix
├── data/
│   ├── train.csv         # AG News subset — 8 000 rows
│   └── test.csv          # AG News subset — 1 600 rows
├── prepare_data.py       # Downloads and saves data/
├── outputs/              # Checkpoints (gitignored)
├── requirements.txt
└── README.md
```