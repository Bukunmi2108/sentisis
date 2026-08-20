"""Redis cache for predictions, which degrades to a miss instead of failing a request."""

import contextlib
import json
import logging
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any, cast

from prometheus_client import Counter
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

CACHE_HITS = Counter("sentisis_cache_hits_total", "Predictions served from the cache")
CACHE_MISSES = Counter("sentisis_cache_misses_total", "Predictions that had to be computed")
CACHE_ERRORS = Counter("sentisis_cache_errors_total", "Cache operations that failed")


def cache_key(model_version: str, cleaned_text: str) -> str:
    """Build the Redis key for one cleaned text under one model version."""
    digest = sha256(cleaned_text.encode("utf-8")).hexdigest()
    return f"sentisis:{model_version}:{digest}"


class PredictionCache:
    """Reads and writes probability rows, and never raises."""

    def __init__(self, client: Any, *, model_version: str, ttl: int) -> None:
        self._client = client
        self._model_version = model_version
        self._ttl = ttl

    async def get_many(self, cleaned_texts: Sequence[str]) -> dict[str, list[float]]:
        """Return the probability rows already cached, keyed by cleaned text."""
        if not cleaned_texts:
            return {}
        unique = list(dict.fromkeys(cleaned_texts))
        try:
            raw = cast(
                list[str | None],
                await self._client.mget([cache_key(self._model_version, t) for t in unique]),
            )
        except (RedisError, OSError) as error:
            CACHE_ERRORS.inc()
            logger.warning("cache read failed", extra={"error": str(error)})
            return {}

        found: dict[str, list[float]] = {}
        for text, value in zip(unique, raw, strict=True):
            row = _decode(value)
            if row is not None:
                found[text] = row
        CACHE_HITS.inc(len(found))
        CACHE_MISSES.inc(len(unique) - len(found))
        return found

    async def set_many(self, rows: Mapping[str, Sequence[float]]) -> None:
        """Store probability rows against their cleaned text, ignoring any failure."""
        if not rows:
            return
        try:
            pipeline = self._client.pipeline()
            for text, row in rows.items():
                key = cache_key(self._model_version, text)
                pipeline.set(key, json.dumps(list(row)), ex=self._ttl)
            await pipeline.execute()
        except (RedisError, OSError) as error:
            CACHE_ERRORS.inc()
            logger.warning("cache write failed", extra={"error": str(error)})

    async def ping(self) -> bool:
        """Report whether Redis is currently reachable."""
        try:
            return bool(await self._client.ping())
        except (RedisError, OSError):
            return False

    async def close(self) -> None:
        """Release the connection pool, ignoring any failure."""
        with contextlib.suppress(RedisError, OSError):
            await self._client.aclose()


def _decode(value: str | None) -> list[float] | None:
    """Read back one stored row, treating anything unusable as a miss."""
    if value is None:
        return None
    try:
        return [float(item) for item in json.loads(value)]
    except (ValueError, TypeError):
        return None
