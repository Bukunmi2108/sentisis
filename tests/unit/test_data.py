from typing import Any

import pytest

from model.data import LABELS, prepare_split


def test_labels_are_fixed_to_tweet_eval_encoding() -> None:
    assert LABELS == {0: "negative", 1: "neutral", 2: "positive"}


def test_prepare_split_returns_normalized_text_and_labels() -> None:
    splits: Any = {"train": {"text": ["WOW!!! @alice"], "label": [2]}}

    texts, labels = prepare_split(splits, "train", bow=False)

    assert texts == ["WOW!!! [user]"]
    assert labels == [2]


def test_prepare_split_returns_bag_of_words_text() -> None:
    splits: Any = {"train": {"text": ["WOW!!! @alice"], "label": [2]}}

    texts, labels = prepare_split(splits, "train", bow=True)

    assert texts == ["wow user"]
    assert labels == [2]


def test_prepare_split_rejects_unknown_split() -> None:
    splits: Any = {"train": {"text": [], "label": []}}

    with pytest.raises(ValueError, match="Unknown split"):
        prepare_split(splits, "test", bow=False)
