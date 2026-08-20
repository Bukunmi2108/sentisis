import subprocess
import sys

HEAVY = ("torch", "transformers", "datasets", "sklearn", "pandas", "scipy", "joblib")

PROBE = """
import importlib, sys
importlib.import_module("{module}")
loaded = sorted(name for name in sys.modules if name.split(".")[0] in {heavy})
print(",".join(loaded))
"""


def modules_pulled_in_by(module: str) -> list[str]:
    """Import one module in a fresh interpreter and report which heavy packages came with it."""
    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(module=module, heavy=HEAVY)],
        capture_output=True,
        text=True,
        check=True,
    )
    return [name for name in result.stdout.strip().split(",") if name]


def test_the_shared_cleaner_stays_on_base_dependencies() -> None:
    assert modules_pulled_in_by("model.preprocess") == []


def test_the_onnx_runner_stays_on_base_dependencies() -> None:
    assert modules_pulled_in_by("model.onnx_runner") == []


def test_the_application_stays_on_base_dependencies() -> None:
    assert modules_pulled_in_by("api.main") == []
