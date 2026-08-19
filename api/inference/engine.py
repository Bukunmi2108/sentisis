"""Load the int8 model and turn text into predictions."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import onnxruntime as ort

from api.core.config import Settings
from model.onnx_runner import OnnxSentimentModel, softmax
from model.preprocess import normalize

LABELS: tuple[str, ...] = ("negative", "neutral", "positive")
CONFIDENCE_DIGITS = 4
CANARY_TEXT = "the api is up"


@dataclass(frozen=True)
class Prediction:
    """One classified text."""

    label: str
    confidence: float
    probabilities: dict[str, float]


def to_prediction(probabilities: Sequence[float]) -> Prediction:
    """Build a prediction from one row of class probabilities."""
    if len(probabilities) != len(LABELS):
        raise ValueError(f"expected {len(LABELS)} probabilities, got {len(probabilities)}")
    values = [float(value) for value in probabilities]
    best = max(range(len(values)), key=lambda index: values[index])
    return Prediction(
        label=LABELS[best],
        confidence=round(values[best], CONFIDENCE_DIGITS),
        probabilities={
            name: round(value, CONFIDENCE_DIGITS)
            for name, value in zip(LABELS, values, strict=True)
        },
    )


class Engine:
    """Owns the ONNX session for the life of the process."""

    def __init__(self, settings: Settings) -> None:
        options: Any = cast(Any, ort).SessionOptions()
        options.intra_op_num_threads = settings.onnx_threads
        options.inter_op_num_threads = settings.onnx_threads
        self.model_version = settings.model_version
        self._model = OnnxSentimentModel(
            settings.model_path,
            settings.tokenizer_path,
            max_length=settings.max_length,
            session_options=options,
        )

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        """Classify every text, returning one prediction per input in request order."""
        if not texts:
            return []
        cleaned = [normalize(text) for text in texts]
        unique = list(dict.fromkeys(cleaned))
        probabilities = softmax(self._model.logits(unique))
        by_text = {
            text: to_prediction(row) for text, row in zip(unique, probabilities, strict=True)
        }
        return [by_text[text] for text in cleaned]

    def canary(self) -> Prediction:
        """Run one fixed prediction, so readiness proves the session actually works."""
        return self.predict([CANARY_TEXT])[0]
