"""The application: lifespan, error handlers, middleware, routes and metrics."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from api.core.config import get_settings
from api.core.logging import configure_logging, request_id_var
from api.errors import SentisisError
from api.inference.cache import PredictionCache
from api.inference.engine import Engine
from api.middleware import RequestContextMiddleware
from api.routes import router
from api.schemas import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parents[1] / "web"

DESCRIPTION = """
Three-class sentiment analysis over short social text, served by a quantised DistilBERT
exported to ONNX.

Text is cleaned exactly as it was during training before it reaches the model, so what you send
is scored the same way the reported metrics were measured.
"""


def error_response(failure: SentisisError) -> JSONResponse:
    """Render one failure in the API's error shape."""
    return JSONResponse(
        status_code=failure.status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                code=failure.code,
                message=failure.message,
                retryable=failure.retryable,
                request_id=request_id_var.get(),
            )
        ).model_dump(),
    )


async def handle_sentisis_error(_: Request, error: Exception) -> JSONResponse:
    """Render a deliberate failure with its own status, code and retry hint."""
    failure = (
        error if isinstance(error, SentisisError) else SentisisError("internal_error", str(error))
    )
    logger.warning("request rejected", extra={"code": failure.code})
    return error_response(failure)


async def handle_malformed_body(_: Request, __: Exception) -> JSONResponse:
    """Render a body that does not match the schema in the same shape as everything else."""
    return error_response(
        SentisisError(
            "validation_error",
            "the request body does not match the expected shape",
            status_code=422,
        )
    )


async def handle_unexpected(_: Request, __: Exception) -> JSONResponse:
    """Render anything unforeseen without leaking internals to the caller."""
    logger.exception("unhandled error")
    return error_response(
        SentisisError("internal_error", "the request could not be completed", retryable=True)
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Load the model and open the cache once, and let the app serve either way."""
    settings = get_settings()
    configure_logging(settings.log_level)

    app.state.engine = None
    app.state.cache = None

    try:
        app.state.engine = Engine(settings)
        logger.info("model loaded", extra={"model_path": str(settings.model_path)})
    except Exception as error:
        logger.error(
            "model failed to load",
            extra={"model_path": str(settings.model_path), "error": str(error)},
        )

    if settings.cache_enabled:
        app.state.cache = PredictionCache(
            aioredis.from_url(settings.redis_url, decode_responses=True),
            model_version=settings.model_version,
            ttl=settings.cache_ttl,
        )

    try:
        yield
    finally:
        cache: PredictionCache | None = app.state.cache
        if cache is not None:
            await cache.close()


def create_app() -> FastAPI:
    """Build the application."""
    app = FastAPI(
        title="sentisis",
        description=DESCRIPTION,
        version=get_settings().model_version,
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)

    app.add_exception_handler(SentisisError, handle_sentisis_error)
    app.add_exception_handler(RequestValidationError, handle_malformed_body)
    app.add_exception_handler(Exception, handle_unexpected)

    app.include_router(router)
    Instrumentator().instrument(app).expose(app, tags=["operations"])
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
