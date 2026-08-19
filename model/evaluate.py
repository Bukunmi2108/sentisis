"""Metrics every model in this project is scored by."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (  # pyright: ignore[reportUnknownVariableType]
    accuracy_score,  # pyright: ignore[reportUnknownVariableType]
    confusion_matrix,  # pyright: ignore[reportUnknownVariableType]
    precision_recall_fscore_support,  # pyright: ignore[reportUnknownVariableType]
)

from model.data import LABELS

LABEL_IDS: list[int] = sorted(LABELS)
LABEL_NAMES: list[str] = [LABELS[label_id] for label_id in LABEL_IDS]


@dataclass(frozen=True)
class ClassMetrics:
    """Precision, recall and F1 for one class, or one average across classes."""

    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class Report:
    """Everything measured about one model on one split."""

    model_name: str
    n_samples: int
    accuracy: float
    per_class: dict[str, ClassMetrics]
    macro: ClassMetrics
    weighted: ClassMetrics
    confusion: npt.NDArray[np.int64]


def _accuracy(truth: npt.NDArray[np.int64], predicted: npt.NDArray[np.int64]) -> float:
    return float(cast(Any, accuracy_score(truth, predicted)))


def _confusion(
    truth: npt.NDArray[np.int64], predicted: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    matrix = cast(Any, confusion_matrix(truth, predicted, labels=LABEL_IDS))
    return np.asarray(matrix, dtype=np.int64)


def _prf_per_class(
    truth: npt.NDArray[np.int64], predicted: npt.NDArray[np.int64]
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
]:
    """Per-class precision, recall, F1 and support."""
    precision, recall, f1, support = cast(
        tuple[Any, Any, Any, Any],
        precision_recall_fscore_support(
            truth,
            predicted,
            labels=LABEL_IDS,
            zero_division=0,  # pyright: ignore[reportArgumentType]
        ),
    )
    return (
        np.asarray(precision, dtype=np.float64),
        np.asarray(recall, dtype=np.float64),
        np.asarray(f1, dtype=np.float64),
        np.asarray(support, dtype=np.int64),
    )


def _prf_average(
    truth: npt.NDArray[np.int64],
    predicted: npt.NDArray[np.int64],
    average: str,
) -> ClassMetrics:
    precision, recall, f1, _ = cast(
        tuple[Any, Any, Any, Any],
        precision_recall_fscore_support(
            truth,
            predicted,
            labels=LABEL_IDS,
            average=average,
            zero_division=0,  # pyright: ignore[reportArgumentType]
        ),
    )
    return ClassMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        support=int(len(truth)),
    )


def evaluate(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    model_name: str,
) -> Report:
    """Score predictions against truth."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} rows but y_pred has {len(y_pred)}")
    if not len(y_true):
        raise ValueError("cannot evaluate an empty set of predictions")

    truth = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)

    precision, recall, f1, support = _prf_per_class(truth, predicted)
    per_class = {
        name: ClassMetrics(
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            support=int(support[index]),
        )
        for index, name in enumerate(LABEL_NAMES)
    }

    return Report(
        model_name=model_name,
        n_samples=len(truth),
        accuracy=_accuracy(truth, predicted),
        per_class=per_class,
        macro=_prf_average(truth, predicted, "macro"),
        weighted=_prf_average(truth, predicted, "weighted"),
        confusion=_confusion(truth, predicted),
    )


def format_report(report: Report) -> str:
    """Render one report as a fixed-width table."""
    lines = [
        f"{report.model_name}  ({report.n_samples:,} samples)",
        "",
        f"{'class':<10} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>9}",
    ]
    for name in LABEL_NAMES:
        metrics = report.per_class[name]
        lines.append(
            f"{name:<10} {metrics.precision:>10.4f} {metrics.recall:>10.4f} "
            f"{metrics.f1:>10.4f} {metrics.support:>9,}"
        )
    for name, metrics in (("macro", report.macro), ("weighted", report.weighted)):
        lines.append(
            f"{name:<10} {metrics.precision:>10.4f} {metrics.recall:>10.4f} "
            f"{metrics.f1:>10.4f} {metrics.support:>9,}"
        )
    lines += [
        "",
        f"accuracy   {report.accuracy:.4f}",
        "",
        "confusion (rows = true, columns = predicted)",
        f"{'':<10}" + "".join(f"{name:>10}" for name in LABEL_NAMES),
    ]
    for index, name in enumerate(LABEL_NAMES):
        row = report.confusion[index]
        lines.append(f"{name:<10}" + "".join(f"{int(count):>10,}" for count in row))
    return "\n".join(lines)


def compare(reports: Sequence[Report]) -> str:
    """Render several reports side by side, ranked by macro-F1."""
    if not reports:
        raise ValueError("nothing to compare")
    ranked = sorted(reports, key=lambda report: report.macro.f1, reverse=True)
    lines = [
        f"{'model':<24} {'macro f1':>10} {'weighted f1':>12} {'accuracy':>10}",
    ]
    for report in ranked:
        lines.append(
            f"{report.model_name:<24} {report.macro.f1:>10.4f} "
            f"{report.weighted.f1:>12.4f} {report.accuracy:>10.4f}"
        )
    lines += ["", f"winner on macro-F1: {ranked[0].model_name}"]
    return "\n".join(lines)


def to_dict(report: Report) -> dict[str, Any]:
    """Convert one report to plain JSON-serialisable types."""
    return {
        "n_samples": report.n_samples,
        "accuracy": report.accuracy,
        "per_class": {
            name: {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "support": metrics.support,
            }
            for name, metrics in report.per_class.items()
        },
        "macro": {
            "precision": report.macro.precision,
            "recall": report.macro.recall,
            "f1": report.macro.f1,
            "support": report.macro.support,
        },
        "weighted": {
            "precision": report.weighted.precision,
            "recall": report.weighted.recall,
            "f1": report.weighted.f1,
            "support": report.weighted.support,
        },
        "confusion": report.confusion.tolist(),
    }


def from_dict(model_name: str, payload: Mapping[str, Any]) -> Report:
    """Rebuild a report from the JSON written by `save_metrics`."""
    return Report(
        model_name=model_name,
        n_samples=int(payload["n_samples"]),
        accuracy=float(payload["accuracy"]),
        per_class={name: ClassMetrics(**metrics) for name, metrics in payload["per_class"].items()},
        macro=ClassMetrics(**payload["macro"]),
        weighted=ClassMetrics(**payload["weighted"]),
        confusion=np.asarray(payload["confusion"], dtype=np.int64),
    )


def save_metrics(
    path: Path,
    reports: Mapping[str, Report],
    *,
    model_name: str,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write every split's metrics for one model into a single JSON file."""
    if not reports:
        raise ValueError("nothing to save")
    payload: dict[str, Any] = {
        "model_name": model_name,
        "labels": LABEL_NAMES,
        "splits": {name: to_dict(report) for name, report in reports.items()},
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_metrics(path: Path) -> dict[str, Report]:
    """Read a metrics bundle back, keyed by split name."""
    payload = json.loads(path.read_text())
    model_name = str(payload["model_name"])
    return {
        name: from_dict(model_name, split_payload)
        for name, split_payload in payload["splits"].items()
    }
