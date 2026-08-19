from pathlib import Path
from typing import Any, cast

from model.baseline import build_pipeline, fit, load, predict, save, top_terms

CORPUS = [
    "i love this so much wonderful",
    "absolutely brilliant and lovely",
    "great fun and lovely day",
    "the meeting is at three today",
    "he said the train leaves later",
    "a report about the weather today",
    "this is terrible and awful",
    "i hate this awful mess",
    "dreadful terrible experience",
]
LABELS_FOR_CORPUS = [2, 2, 2, 1, 1, 1, 0, 0, 0]


def test_pipeline_has_the_expected_two_steps() -> None:
    pipeline = cast(Any, build_pipeline())

    assert list(pipeline.named_steps) == ["tfidf", "logreg"]


def test_vectorizer_does_not_lowercase_because_to_bow_already_did() -> None:
    pipeline = cast(Any, build_pipeline())

    assert pipeline.named_steps["tfidf"].lowercase is False


def test_token_pattern_has_no_unicode_flag_so_onnx_runtime_can_load_it() -> None:
    pipeline = cast(Any, build_pipeline())

    assert "(?u)" not in pipeline.named_steps["tfidf"].token_pattern


def test_token_pattern_keeps_single_character_tokens() -> None:
    import re

    pattern = cast(Any, build_pipeline()).named_steps["tfidf"].token_pattern

    assert re.findall(pattern, "i am a person") == ["i", "am", "a", "person"]


def test_classifier_is_balanced_and_seeded() -> None:
    classifier = cast(Any, build_pipeline()).named_steps["logreg"]

    assert classifier.class_weight == "balanced"
    assert classifier.random_state == 42


def test_fit_and_predict_on_a_tiny_corpus() -> None:
    pipeline = fit(CORPUS, LABELS_FOR_CORPUS)

    predicted = predict(pipeline, CORPUS)

    assert predicted == LABELS_FOR_CORPUS
    assert all(isinstance(label, int) for label in predicted)


def test_joblib_round_trip_preserves_predictions(tmp_path: Path) -> None:
    pipeline = fit(CORPUS, LABELS_FOR_CORPUS)
    before = predict(pipeline, CORPUS)
    path = tmp_path / "nested" / "baseline.joblib"

    save(pipeline, path)
    after = predict(load(path), CORPUS)

    assert after == before


def test_top_terms_for_positive_class_are_positive_words() -> None:
    pipeline = fit(CORPUS, LABELS_FOR_CORPUS)

    terms = [term for term, _ in top_terms(pipeline, 2, count=6)]

    assert {"lovely"} & set(terms)


def test_top_terms_are_ranked_by_descending_weight() -> None:
    pipeline = fit(CORPUS, LABELS_FOR_CORPUS)

    weights = [weight for _, weight in top_terms(pipeline, 0, count=10)]

    assert weights == sorted(weights, reverse=True)
