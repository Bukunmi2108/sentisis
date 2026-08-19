"""TweetEval loading and preparation helpers."""

from typing import Any

from datasets import DatasetDict, load_dataset  # pyright: ignore[reportUnknownVariableType]

from model.preprocess import normalize, to_bow

LABELS: dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}
DATASET_NAME = "tweet_eval"
DATASET_CONFIG = "sentiment"


def load_splits() -> DatasetDict:
    """Load the published TweetEval sentiment splits."""
    return load_dataset(DATASET_NAME, DATASET_CONFIG)


def prepare(split: str, *, bow: bool) -> tuple[list[str], list[int]]:
    """Return cleaned text and labels for one TweetEval split."""
    return prepare_split(load_splits(), split, bow=bow)


def prepare_split(
    splits: DatasetDict,
    split: str,
    *,
    bow: bool,
) -> tuple[list[str], list[int]]:
    """Prepare one split from an already loaded dataset."""
    if split not in splits:
        available = ", ".join(sorted(str(name) for name in splits))
        raise ValueError(f"Unknown split {split!r}; available splits: {available}")

    rows: Any = splits[split]
    texts = [str(text) for text in rows["text"]]
    labels = [int(label) for label in rows["label"]]
    cleaner = to_bow if bow else normalize
    return [cleaner(text) for text in texts], labels
