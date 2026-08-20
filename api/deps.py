"""Access to the objects the app builds once at startup."""

from typing import Annotated

from fastapi import Depends, Request

from api.core.config import Settings, get_settings
from api.errors import SentisisError
from api.inference.cache import PredictionCache
from api.inference.engine import Engine


def get_engine(request: Request) -> Engine:
    """Return the loaded engine, or fail the request if the model never loaded."""
    engine: Engine | None = getattr(request.app.state, "engine", None)
    if engine is None:
        raise SentisisError(
            "model_unavailable", "the model is not loaded", status_code=503, retryable=True
        )
    return engine


def get_cache(request: Request) -> PredictionCache | None:
    """Return the cache, or None when caching is switched off."""
    cache: PredictionCache | None = getattr(request.app.state, "cache", None)
    return cache


EngineDep = Annotated[Engine, Depends(get_engine)]
CacheDep = Annotated[PredictionCache | None, Depends(get_cache)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
