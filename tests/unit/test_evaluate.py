import json
from fractions import Fraction
from pathlib import Path

import pytest

from model.evaluate import compare, evaluate, format_report, load_metrics, save_metrics

TOLERANCE = 1e-12

HAND_TRUE = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
HAND_PRED = [0, 0, 1, 1, 1, 2, 2, 2, 2, 0]


def test_per_class_metrics_match_hand_worked_arithmetic() -> None:
    report = evaluate(HAND_TRUE, HAND_PRED, model_name="hand")

    expected = {
        "negative": (Fraction(2, 3), Fraction(2, 3), Fraction(2, 3), 3),
        "neutral": (Fraction(2, 3), Fraction(1, 2), Fraction(4, 7), 4),
        "positive": (Fraction(1, 2), Fraction(2, 3), Fraction(4, 7), 3),
    }
    for name, (precision, recall, f1, support) in expected.items():
        metrics = report.per_class[name]
        assert metrics.precision == pytest.approx(float(precision), abs=TOLERANCE)
        assert metrics.recall == pytest.approx(float(recall), abs=TOLERANCE)
        assert metrics.f1 == pytest.approx(float(f1), abs=TOLERANCE)
        assert metrics.support == support


def test_averages_match_hand_worked_arithmetic() -> None:
    report = evaluate(HAND_TRUE, HAND_PRED, model_name="hand")

    macro_f1 = (Fraction(2, 3) + Fraction(4, 7) + Fraction(4, 7)) / 3
    weighted_f1 = (3 * Fraction(2, 3) + 4 * Fraction(4, 7) + 3 * Fraction(4, 7)) / 10

    assert report.accuracy == pytest.approx(0.6, abs=TOLERANCE)
    assert report.macro.f1 == pytest.approx(float(macro_f1), abs=TOLERANCE)
    assert report.weighted.f1 == pytest.approx(float(weighted_f1), abs=TOLERANCE)
    assert report.n_samples == 10


def test_confusion_rows_are_truth_not_predictions() -> None:
    report = evaluate([0, 0, 0], [1, 1, 1], model_name="orientation")

    assert int(report.confusion[0][1]) == 3
    assert int(report.confusion[1][0]) == 0


def test_confusion_matches_hand_built_matrix() -> None:
    report = evaluate(HAND_TRUE, HAND_PRED, model_name="hand")

    assert report.confusion.tolist() == [[2, 1, 0], [0, 2, 2], [1, 0, 2]]


def test_perfect_classifier_scores_one_everywhere() -> None:
    labels = [0, 1, 2, 0, 1, 2]
    report = evaluate(labels, labels, model_name="perfect")

    assert report.accuracy == pytest.approx(1.0, abs=TOLERANCE)
    assert report.macro.f1 == pytest.approx(1.0, abs=TOLERANCE)
    for name in ("negative", "neutral", "positive"):
        assert report.per_class[name].f1 == pytest.approx(1.0, abs=TOLERANCE)


def test_class_never_predicted_scores_zero_without_raising() -> None:
    report = evaluate([0, 1, 2, 1], [1, 1, 1, 1], model_name="lazy")

    assert report.per_class["negative"].f1 == 0.0
    assert report.per_class["positive"].f1 == 0.0
    assert report.per_class["neutral"].recall == pytest.approx(1.0, abs=TOLERANCE)


def test_macro_penalises_the_rare_class_more_than_weighted() -> None:
    truth = [1] * 90 + [0] * 10
    predicted = [1] * 100
    report = evaluate(truth, predicted, model_name="majority")

    assert report.macro.f1 < report.weighted.f1
    assert report.accuracy == pytest.approx(0.9, abs=TOLERANCE)


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="y_pred"):
        evaluate([0, 1], [0], model_name="bad")


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        evaluate([], [], model_name="bad")


def test_compare_ranks_by_macro_f1() -> None:
    weak = evaluate([0, 1, 2, 1], [1, 1, 1, 1], model_name="weak")
    strong = evaluate([0, 1, 2, 1], [0, 1, 2, 1], model_name="strong")

    table = compare([weak, strong])

    assert table.index("strong") < table.index("weak")
    assert "winner on macro-F1: strong" in table


def test_format_report_labels_confusion_orientation() -> None:
    text = format_report(evaluate(HAND_TRUE, HAND_PRED, model_name="hand"))

    assert "rows = true, columns = predicted" in text
    assert "hand" in text


def test_save_metrics_writes_one_file_per_model(tmp_path: Path) -> None:
    validation = evaluate(HAND_TRUE, HAND_PRED, model_name="hand")
    test = evaluate([0, 1, 2], [0, 1, 2], model_name="hand")
    path = tmp_path / "metrics" / "hand.json"

    save_metrics(
        path,
        {"validation": validation, "test": test},
        model_name="hand",
        extra={"top_terms": {"negative": [["awful", 1.5]]}},
    )
    payload = json.loads(path.read_text())

    assert payload["model_name"] == "hand"
    assert payload["labels"] == ["negative", "neutral", "positive"]
    assert sorted(payload["splits"]) == ["test", "validation"]
    assert payload["splits"]["validation"]["confusion"] == [[2, 1, 0], [0, 2, 2], [1, 0, 2]]
    assert payload["top_terms"]["negative"] == [["awful", 1.5]]


def test_metrics_survive_a_round_trip(tmp_path: Path) -> None:
    original = evaluate(HAND_TRUE, HAND_PRED, model_name="hand")
    path = tmp_path / "hand.json"

    save_metrics(path, {"test": original}, model_name="hand")
    restored = load_metrics(path)["test"]

    assert restored.model_name == original.model_name
    assert restored.n_samples == original.n_samples
    assert restored.accuracy == pytest.approx(original.accuracy, abs=TOLERANCE)
    assert restored.macro.f1 == pytest.approx(original.macro.f1, abs=TOLERANCE)
    assert restored.per_class["neutral"].f1 == pytest.approx(
        original.per_class["neutral"].f1, abs=TOLERANCE
    )
    assert restored.confusion.tolist() == original.confusion.tolist()


def test_save_metrics_rejects_an_empty_bundle(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nothing to save"):
        save_metrics(tmp_path / "empty.json", {}, model_name="hand")
