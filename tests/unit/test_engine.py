import pytest

from api.core.config import Settings
from api.inference.engine import LABELS, Engine, Prediction, to_prediction
from model.data import LABELS as TRAINING_LABELS


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine(Settings())


def test_labels_match_the_training_encoding() -> None:
    assert dict(enumerate(LABELS)) == TRAINING_LABELS


def test_label_is_the_argmax() -> None:
    assert to_prediction([0.1, 0.2, 0.7]).label == "positive"
    assert to_prediction([0.7, 0.2, 0.1]).label == "negative"
    assert to_prediction([0.2, 0.7, 0.1]).label == "neutral"


def test_confidence_is_the_winning_probability() -> None:
    prediction = to_prediction([0.0121, 0.0467, 0.9412])
    assert prediction.confidence == 0.9412
    assert prediction.confidence == prediction.probabilities[prediction.label]


def test_probabilities_are_keyed_by_label_name() -> None:
    assert to_prediction([0.1, 0.2, 0.7]).probabilities == {
        "negative": 0.1,
        "neutral": 0.2,
        "positive": 0.7,
    }


def test_a_tie_resolves_to_the_lowest_index() -> None:
    assert to_prediction([0.5, 0.5, 0.0]).label == "negative"


@pytest.mark.parametrize("row", [[], [0.5, 0.5], [0.25, 0.25, 0.25, 0.25]])
def test_wrong_width_rows_are_rejected(row: list[float]) -> None:
    with pytest.raises(ValueError, match="expected 3 probabilities"):
        to_prediction(row)


def test_predictions_are_frozen() -> None:
    prediction = to_prediction([0.1, 0.2, 0.7])
    assert isinstance(prediction, Prediction)
    with pytest.raises(AttributeError):
        prediction.label = "neutral"  # pyright: ignore[reportAttributeAccessIssue]


def test_obvious_sentiment_is_classified_correctly(engine: Engine) -> None:
    predictions = engine.predict(
        ["i absolutely love this, best day ever", "worst experience of my life, i hate it"]
    )
    assert [prediction.label for prediction in predictions] == ["positive", "negative"]


def test_probabilities_sum_to_one(engine: Engine) -> None:
    prediction = engine.predict(["it was fine"])[0]
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    assert set(prediction.probabilities) == set(LABELS)


def test_a_batch_keeps_request_order(engine: Engine) -> None:
    texts = ["worst thing ever", "i love it so much", "the meeting is at noon"]
    batched = engine.predict(texts)
    assert [prediction.label for prediction in batched] == [
        engine.predict([text])[0].label for text in texts
    ]


def test_repeated_texts_collapse_to_one_row(engine: Engine) -> None:
    once = engine.predict(["i love this"])
    many = engine.predict(["i love this"] * 5)
    assert many == once * 5


def test_texts_differing_only_before_cleaning_share_a_row(engine: Engine) -> None:
    predictions = engine.predict(["great @alice", "great @bob"])
    assert predictions[0] == predictions[1]


def test_an_empty_batch_returns_nothing(engine: Engine) -> None:
    assert engine.predict([]) == []


def test_emoji_only_input_still_classifies(engine: Engine) -> None:
    assert engine.predict(["🔥🔥🔥"])[0].label in LABELS


def test_non_english_input_returns_a_label(engine: Engine) -> None:
    assert engine.predict(["me encanta este lugar"])[0].label in LABELS


def test_null_bytes_do_not_crash(engine: Engine) -> None:
    assert engine.predict(["good\x00news"])[0].label in LABELS


def test_text_far_longer_than_max_length_is_truncated_not_rejected(engine: Engine) -> None:
    assert engine.predict(["spam " * 500])[0].label in LABELS


def test_canary_runs_a_real_prediction(engine: Engine) -> None:
    assert engine.canary().label in LABELS
