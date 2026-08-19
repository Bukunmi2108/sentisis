"""Model training, evaluation and export for the sentisis sentiment API."""

from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
BASELINE_PATH = ARTIFACTS_DIR / "baseline.joblib"
METRICS_PATH = METRICS_DIR / "baseline.json"
