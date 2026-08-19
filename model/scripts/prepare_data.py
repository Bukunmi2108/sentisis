"""Inspect and prepare the TweetEval sentiment dataset."""

from collections import Counter
from typing import Any

from model.data import (
    DATASET_CONFIG,
    DATASET_NAME,
    LABELS,
    load_splits,
    prepare_split,
)


def main() -> None:
    """Print dataset sizes, class counts, and cleaning examples."""
    splits = load_splits()
    print(f"dataset: {DATASET_NAME}/{DATASET_CONFIG}")
    for split in ("train", "validation", "test"):
        rows: Any = splits[split]
        labels = [int(label) for label in rows["label"]]
        counts = Counter(labels)
        distribution = ", ".join(f"{LABELS[label]}={counts[label]}" for label in LABELS)
        print(f"{split}: {len(labels)} rows ({distribution})")

    train_rows: Any = splits["train"]
    raw_texts = [str(text) for text in train_rows["text"][:3]]
    normalized, _ = prepare_split(splits, "train", bow=False)
    bow, _ = prepare_split(splits, "train", bow=True)
    print("samples:")
    for index, raw in enumerate(raw_texts):
        print(f"raw: {raw}")
        print(f"normalized: {normalized[index]}")
        print(f"bow: {bow[index]}")


if __name__ == "__main__":
    main()
