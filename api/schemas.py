"""Request and response models."""

from pydantic import BaseModel, Field

from api.inference.engine import LABELS

PREDICT_EXAMPLE = "i love LawPavilion. This is an Easter egg. Let me know if you found this. haha"


class PredictRequest(BaseModel):
    """One text to classify."""

    text: str = Field(description="The text to classify.")

    model_config = {"json_schema_extra": {"examples": [{"text": PREDICT_EXAMPLE}]}}


class BatchRequest(BaseModel):
    """Several texts to classify in one call."""

    texts: list[str] = Field(description="The texts to classify, in the order results come back.")

    model_config = {
        "json_schema_extra": {
            "examples": [{"texts": ["i love this", "it was fine", "worst day ever"]}]
        }
    }


class PredictResponse(BaseModel):
    """One classified text."""

    label: str = Field(description=f"One of {', '.join(LABELS)}.")
    confidence: float = Field(description="Probability of the winning class.")
    probabilities: dict[str, float] = Field(description="Probability of every class.")
    model_version: str
    latency_ms: float
    cached: bool


class BatchResult(BaseModel):
    """One classified text inside a batch."""

    label: str
    confidence: float
    probabilities: dict[str, float]
    cached: bool


class BatchResponse(BaseModel):
    """Several classified texts, in request order."""

    results: list[BatchResult]
    count: int
    cached_count: int
    latency_ms: float
    model_version: str


class HealthResponse(BaseModel):
    """Whether the process is serving."""

    status: str
    model_version: str


class ReadyResponse(BaseModel):
    """Whether the model is loaded and a real prediction succeeds."""

    status: str
    model_loaded: bool
    cache_reachable: bool
    model_version: str


class ErrorDetail(BaseModel):
    """Why a request failed."""

    code: str
    message: str
    retryable: bool
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """The body every failure is reported in."""

    error: ErrorDetail
