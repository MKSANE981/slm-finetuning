"""
Download ag_news from HuggingFace and save train/test subsets as CSV.

ag_news has 4 classes:
  0 = World, 1 = Sports, 2 = Business, 3 = Sci/Tech

We take a subset to keep training fast on CPU (adjust N_TRAIN / N_TEST as needed).
"""

from datasets import load_dataset
import pandas as pd
import os

N_TRAIN = 2000   # samples per class for training
N_TEST  = 400    # samples per class for testing

LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "SciTech"}

def main():
    print("Downloading ag_news...")
    ds = load_dataset("fancyzhx/ag_news")

    os.makedirs("data", exist_ok=True)

    for split, n in [("train", N_TRAIN), ("test", N_TEST)]:
        rows = []
        for label_id, label_name in LABEL_NAMES.items():
            subset = ds[split].filter(lambda x: x["label"] == label_id)
            subset = subset.select(range(min(n, len(subset))))
            for item in subset:
                rows.append({"text": item["text"], "label": label_name})

        df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
        path = f"data/{split}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} rows → {path}")

    print("Done. Run training with:")
    print("  python src/train.py --train_path data/train.csv --test_path data/test.csv --epochs 3")

if __name__ == "__main__":
    main()