"""Score the downloaded int8 ONNX model and write its metrics bundle."""

import json
from time import perf_counter
from typing import Any

from model import ARTIFACTS_DIR, METRICS_DIR
from model.data import load_splits
from model.evaluate import Report, evaluate, format_report, save_metrics
from model.onnx_runner import OnnxSentimentModel

MODEL_NAME = "distilbert-int8"
DISTILBERT_DIR = ARTIFACTS_DIR / "distilbert"


def main() -> None:
    """Run the int8 session over validation and test, then persist one metrics bundle."""
    model_path = DISTILBERT_DIR / "model_int8.onnx"
    tokenizer_path = DISTILBERT_DIR / "tokenizer" / "tokenizer.json"
    for path in (model_path, tokenizer_path):
        if not path.exists():
            raise SystemExit(f"missing {path}. Unzip the Kaggle artifacts into {DISTILBERT_DIR}.")

    summary_path = DISTILBERT_DIR / "training_summary.json"
    summary: dict[str, Any] = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    model = OnnxSentimentModel(model_path, tokenizer_path)
    splits = load_splits()

    reports: dict[str, Report] = {}
    for split in ("validation", "test"):
        rows: Any = splits[split]
        texts = [str(text) for text in rows["text"]]
        labels = [int(label) for label in rows["label"]]

        started = perf_counter()
        predicted = model.predict(texts)
        elapsed = perf_counter() - started
        print(
            f"{split:11s} {len(texts):6,} rows in {elapsed:6.1f}s "
            f"({len(texts) / elapsed:,.0f} rows/s)"
        )

        reports[split] = evaluate(labels, predicted, model_name=MODEL_NAME)

    save_metrics(
        METRICS_DIR / "distilbert.json",
        reports,
        model_name=MODEL_NAME,
        extra={
            "fp32_reference": summary.get("fp32_scores", {}),
            "train_config": summary.get("config", {}),
        },
    )

    print()
    for report in reports.values():
        print(format_report(report))
        print()

    fp32 = summary.get("fp32_scores", {}).get("test", {}).get("macro_f1")
    if fp32 is not None:
        drop = fp32 - reports["test"].macro.f1
        verdict = "within" if abs(drop) <= 0.01 else "OUTSIDE"
        print(
            f"fp32 test macro-F1 {fp32:.4f}  int8 {reports['test'].macro.f1:.4f}  "
            f"drop {drop:+.4f}  -> {verdict} note 05's 1.0-point gate"
        )


if __name__ == "__main__":
    main()
