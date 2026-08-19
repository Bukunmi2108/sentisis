"""Print every model's measured metrics side by side."""

from model import METRICS_DIR
from model.evaluate import Report, compare, format_report, load_metrics

SPLIT = "test"


def main() -> None:
    """Read every model's metrics bundle and rank them by macro-F1 on the test split."""
    if not METRICS_DIR.exists():
        raise SystemExit(f"no metrics yet: {METRICS_DIR} does not exist. Run `make train` first.")

    paths = sorted(METRICS_DIR.glob("*.json"))
    if not paths:
        raise SystemExit(f"no metrics bundles in {METRICS_DIR}. Run `make train` first.")

    reports: list[Report] = []
    for path in paths:
        splits = load_metrics(path)
        if SPLIT not in splits:
            raise SystemExit(f"{path.name} has no {SPLIT!r} split: found {sorted(splits)}")
        reports.append(splits[SPLIT])

    for report in reports:
        print(format_report(report))
        print()
    print(compare(reports))


if __name__ == "__main__":
    main()
