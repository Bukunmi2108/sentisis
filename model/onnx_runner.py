"""Run the exported int8 model with base dependencies only."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from tokenizers import Tokenizer

from model.preprocess import normalize

DEFAULT_MAX_LENGTH = 96
DEFAULT_BATCH_SIZE = 64


class OnnxSentimentModel:
    """Tokenize, run the ONNX session, and turn logits into labels."""

    def __init__(
        self,
        model_path: Path,
        tokenizer_path: Path,
        *,
        max_length: int = DEFAULT_MAX_LENGTH,
        session_options: Any = None,
    ) -> None:
        self.session: Any = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer: Any = _load_tokenizer(tokenizer_path, max_length)
        self.max_length = max_length

    def logits(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """Return raw logits for already-normalized text."""
        encodings = self.tokenizer.encode_batch(list(texts))
        input_ids = np.array([encoding.ids for encoding in encodings], dtype=np.int64)
        attention_mask = np.array(
            [encoding.attention_mask for encoding in encodings], dtype=np.int64
        )
        outputs = self.session.run(
            ["logits"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        return np.asarray(outputs[0], dtype=np.float32)

    def predict_proba(
        self, texts: Sequence[str], *, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> npt.NDArray[np.float32]:
        """Return per-class probabilities, cleaning the text first."""
        cleaned = [normalize(text) for text in texts]
        probabilities = np.zeros((len(cleaned), 3), dtype=np.float32)
        for start in range(0, len(cleaned), batch_size):
            chunk = self.logits(cleaned[start : start + batch_size])
            probabilities[start : start + len(chunk)] = softmax(chunk)
        return probabilities

    def predict(self, texts: Sequence[str], *, batch_size: int = DEFAULT_BATCH_SIZE) -> list[int]:
        """Return predicted label ids, cleaning the text first."""
        if not texts:
            return []
        return [int(index) for index in self.predict_proba(texts, batch_size=batch_size).argmax(-1)]


def softmax(logits: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Turn a batch of logit rows into probability rows."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return np.asarray(exponentiated / exponentiated.sum(axis=-1, keepdims=True), dtype=np.float32)


def _load_tokenizer(path: Path, max_length: int) -> Any:
    """Load a fast tokenizer configured for truncation and padding.

    `tokenizers` ships no type information, so the untyped boundary is confined here.
    """
    tokenizer = cast(Any, Tokenizer).from_file(str(path))
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding()
    return tokenizer
