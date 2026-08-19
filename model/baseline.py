"""TF-IDF into logistic regression: the bar the transformer has to clear."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import joblib
from sklearn.feature_extraction.text import (  # pyright: ignore[reportUnknownVariableType]
    TfidfVectorizer,  # pyright: ignore[reportUnknownVariableType]
)
from sklearn.linear_model import (  # pyright: ignore[reportUnknownVariableType]
    LogisticRegression,  # pyright: ignore[reportUnknownVariableType]
)
from sklearn.pipeline import Pipeline  # pyright: ignore[reportUnknownVariableType]

from model.evaluate import Report, evaluate

MODEL_NAME = "tfidf-logreg"
SEED = 42


def build_pipeline() -> Pipeline:
    """Build the vectorizer and classifier with the configuration fixed before training."""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        max_features=200_000,
        lowercase=False,
        token_pattern=r"\b\w+\b",
    )
    classifier = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
    )
    return Pipeline([("tfidf", vectorizer), ("logreg", classifier)])


def fit(texts: Sequence[str], labels: Sequence[int]) -> Pipeline:
    """Fit a fresh pipeline on already-cleaned bag-of-words text."""
    pipeline = build_pipeline()
    cast(Any, pipeline).fit(list(texts), list(labels))
    return pipeline


def predict(pipeline: Pipeline, texts: Sequence[str]) -> list[int]:
    """Predict label ids for already-cleaned bag-of-words text."""
    predicted = cast(Any, pipeline).predict(list(texts))
    return [int(label) for label in predicted]


def train(
    texts: Sequence[str],
    labels: Sequence[int],
    *,
    eval_texts: Sequence[str],
    eval_labels: Sequence[int],
    model_name: str = MODEL_NAME,
) -> tuple[Pipeline, Report]:
    """Fit on the training split and score against a held-out split."""
    pipeline = fit(texts, labels)
    report = evaluate(eval_labels, predict(pipeline, eval_texts), model_name=model_name)
    return pipeline, report


def save(pipeline: Pipeline, path: Path) -> None:
    """Persist a fitted pipeline, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)  # pyright: ignore[reportUnknownMemberType]


def load(path: Path) -> Pipeline:
    """Load a pipeline saved by `save`."""
    return cast(Pipeline, joblib.load(path))  # pyright: ignore[reportUnknownMemberType]


def top_terms(pipeline: Pipeline, label_id: int, count: int = 20) -> list[tuple[str, float]]:
    """Return the terms pushing hardest towards one class."""
    fitted = cast(Any, pipeline)
    names = [str(name) for name in fitted.named_steps["tfidf"].get_feature_names_out()]
    coefficients = fitted.named_steps["logreg"].coef_[label_id]
    ranked = sorted(zip(names, coefficients, strict=True), key=lambda pair: -pair[1])
    return [(term, float(weight)) for term, weight in ranked[:count]]
