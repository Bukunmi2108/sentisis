import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response

pytestmark = pytest.mark.integration


def unique_text(prefix: str = "sentisis test") -> str:
    """A text no earlier run can have cached, so hit and miss assertions mean something."""
    return f"{prefix} {uuid.uuid4().hex}"


def json_of(response: Response) -> dict[str, Any]:
    """Read a response body at a type pyright can see."""
    body: dict[str, Any] = response.json()
    return body


def predict(client: TestClient, text: str) -> dict[str, Any]:
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200, response.text
    return json_of(response)


def test_health_never_touches_the_model(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert json_of(response)["status"] == "ok"


def test_ready_reports_a_loaded_model(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    body = json_of(response)
    assert body["status"] == "ready"
    assert body["model_loaded"] is True


def test_metrics_are_exposed(client: TestClient) -> None:
    assert "sentisis_cache_hits_total" in client.get("/metrics").text


def test_obvious_sentiment_is_classified(client: TestClient) -> None:
    assert predict(client, "i absolutely love this, best day ever")["label"] == "positive"
    assert predict(client, "worst experience of my life, i hate it")["label"] == "negative"


def test_a_prediction_carries_the_full_contract(client: TestClient) -> None:
    body = predict(client, unique_text())
    assert set(body) == {
        "label",
        "confidence",
        "probabilities",
        "model_version",
        "latency_ms",
        "cached",
    }
    assert body["confidence"] == body["probabilities"][body["label"]]
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_every_response_carries_a_request_id(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "hello"})
    assert response.headers["X-Request-ID"]


def test_a_supplied_request_id_is_echoed(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "hello"}, headers={"X-Request-ID": "known-id"})
    assert response.headers["X-Request-ID"] == "known-id"


def test_the_second_identical_call_is_served_from_the_cache(client: TestClient) -> None:
    text = unique_text()
    assert predict(client, text)["cached"] is False
    second = predict(client, text)
    assert second["cached"] is True


def test_a_cached_answer_matches_the_computed_one(client: TestClient) -> None:
    text = unique_text()
    first = predict(client, text)
    second = predict(client, text)
    assert first["label"] == second["label"]
    assert first["probabilities"] == second["probabilities"]


def test_batch_returns_results_in_request_order(client: TestClient) -> None:
    texts = ["worst thing ever", "i love it so much", "the meeting is at noon"]
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 200
    body = json_of(response)
    assert body["count"] == 3
    assert [result["label"] for result in body["results"]] == [
        predict(client, text)["label"] for text in texts
    ]


def test_batch_mixes_cached_and_fresh_results(client: TestClient) -> None:
    warm, cold = unique_text("warm"), unique_text("cold")
    predict(client, warm)
    body = json_of(client.post("/predict/batch", json={"texts": [warm, cold]}))
    assert [result["cached"] for result in body["results"]] == [True, False]
    assert body["cached_count"] == 1


def test_a_duplicated_text_appears_once_per_position(client: TestClient) -> None:
    text = unique_text()
    body = json_of(client.post("/predict/batch", json={"texts": [text] * 4}))
    assert body["count"] == 4
    assert len({result["confidence"] for result in body["results"]}) == 1


def test_a_full_batch_is_accepted(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"texts": ["fine"] * 128})
    assert response.status_code == 200
    assert json_of(response)["count"] == 128


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"text": ""}, "empty_text"),
        ({"text": "   "}, "empty_text"),
        ({"text": "a" * 5001}, "text_too_long"),
        ({"txet": "misspelled field"}, "validation_error"),
        ({"text": 123}, "validation_error"),
    ],
)
def test_bad_predict_requests_are_rejected(
    client: TestClient, payload: dict[str, Any], code: str
) -> None:
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    error = json_of(response)["error"]
    assert error["code"] == code
    assert error["retryable"] is False
    assert error["request_id"]


def test_text_at_exactly_the_limit_is_accepted(client: TestClient) -> None:
    assert client.post("/predict", json={"text": "a" * 5000}).status_code == 200


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ({"texts": []}, "batch_size_invalid"),
        ({"texts": ["fine"] * 129}, "batch_size_invalid"),
        ({"texts": ["fine", "  "]}, "empty_text"),
    ],
)
def test_bad_batch_requests_are_rejected(
    client: TestClient, payload: dict[str, Any], code: str
) -> None:
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422
    assert json_of(response)["error"]["code"] == code


def test_one_bad_item_rejects_the_whole_batch(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"texts": ["good", "", "also good"]})
    assert response.status_code == 422
    assert "position 1" in json_of(response)["error"]["message"]


@pytest.mark.parametrize(
    "text",
    [
        "🔥🔥🔥",
        "me encanta este lugar",
        "good\x00news",
        "@alice",
        "https://example.com",
        "spam " * 400,
    ],
)
def test_awkward_input_still_returns_a_label(client: TestClient, text: str) -> None:
    assert predict(client, text)["label"] in {"negative", "neutral", "positive"}


def test_predictions_survive_an_unreachable_cache(client_without_redis: TestClient) -> None:
    text = unique_text()
    first = predict(client_without_redis, text)
    second = predict(client_without_redis, text)
    assert first["cached"] is False
    assert second["cached"] is False
    assert first["label"] == second["label"]


def test_readiness_is_honest_about_an_unreachable_cache(client_without_redis: TestClient) -> None:
    body = json_of(client_without_redis.get("/ready"))
    assert body["status"] == "ready"
    assert body["model_loaded"] is True
    assert body["cache_reachable"] is False


def test_predictions_work_with_caching_switched_off(client_without_cache: TestClient) -> None:
    text = unique_text()
    assert predict(client_without_cache, text)["cached"] is False
    assert predict(client_without_cache, text)["cached"] is False


def test_docs_render_with_the_worked_examples(client: TestClient) -> None:
    schema = json_of(client.get("/openapi.json"))
    example = schema["components"]["schemas"]["PredictRequest"]["examples"][0]
    assert isinstance(example["text"], str) and example["text"].strip()


def test_the_demo_page_is_served_at_the_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "sentisis" in response.text


def test_the_page_does_not_shadow_the_api(client: TestClient) -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.post("/predict", json={"text": "hello"}).status_code == 200
