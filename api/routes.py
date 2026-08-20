"""Transport only: validate the request, call the engine, serialise the result."""

import time

from fastapi import APIRouter

from api.core.config import Settings
from api.deps import CacheDep, EngineDep, SettingsDep
from api.errors import SentisisError
from api.schemas import (
    BatchRequest,
    BatchResponse,
    BatchResult,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)
from model.preprocess import normalize

router = APIRouter()


def clean_or_reject(text: str, settings: Settings, *, position: int | None = None) -> str:
    """Validate one text and return its cleaned form."""
    where = "" if position is None else f" at position {position}"
    if len(text) > settings.max_text_length:
        raise SentisisError(
            "text_too_long",
            f"text{where} exceeds the {settings.max_text_length} character limit",
            status_code=422,
        )
    if not text.strip():
        raise SentisisError("empty_text", f"text{where} is empty", status_code=422)
    return normalize(text)


@router.post("/predict", response_model=PredictResponse, tags=["prediction"])
async def predict(
    payload: PredictRequest, engine: EngineDep, cache: CacheDep, settings: SettingsDep
) -> PredictResponse:
    """Classify one text."""
    cleaned = clean_or_reject(payload.text, settings)
    started = time.perf_counter()
    predictions, cached = await engine.predict_with_cache([cleaned], cache)
    elapsed_ms = (time.perf_counter() - started) * 1000
    prediction = predictions[0]
    return PredictResponse(
        label=prediction.label,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        model_version=engine.model_version,
        latency_ms=round(elapsed_ms, 2),
        cached=cached[0],
    )


@router.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
async def predict_batch(
    payload: BatchRequest, engine: EngineDep, cache: CacheDep, settings: SettingsDep
) -> BatchResponse:
    """Classify several texts, returning results in request order."""
    if not payload.texts:
        raise SentisisError("batch_size_invalid", "batch is empty", status_code=422)
    if len(payload.texts) > settings.max_batch_size:
        raise SentisisError(
            "batch_size_invalid",
            f"batch of {len(payload.texts)} exceeds the {settings.max_batch_size} item limit",
            status_code=422,
        )
    cleaned = [
        clean_or_reject(text, settings, position=index) for index, text in enumerate(payload.texts)
    ]

    started = time.perf_counter()
    predictions, cached = await engine.predict_with_cache(cleaned, cache)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return BatchResponse(
        results=[
            BatchResult(
                label=prediction.label,
                confidence=prediction.confidence,
                probabilities=prediction.probabilities,
                cached=was_cached,
            )
            for prediction, was_cached in zip(predictions, cached, strict=True)
        ],
        count=len(predictions),
        cached_count=sum(cached),
        latency_ms=round(elapsed_ms, 2),
        model_version=engine.model_version,
    )


@router.get("/health", response_model=HealthResponse, tags=["operations"])
async def health(settings: SettingsDep) -> HealthResponse:
    """Report that the process is serving. Touches no model, so it is always cheap."""
    return HealthResponse(status="ok", model_version=settings.model_version)


@router.get("/ready", response_model=ReadyResponse, tags=["operations"])
async def ready(engine: EngineDep, cache: CacheDep) -> ReadyResponse:
    """Report readiness, proving the model works by running one real prediction."""
    engine.canary()
    cache_reachable = await cache.ping() if cache is not None else False
    return ReadyResponse(
        status="ready",
        model_loaded=True,
        cache_reachable=cache_reachable,
        model_version=engine.model_version,
    )
