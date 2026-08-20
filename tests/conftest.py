from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from api.core.config import get_settings
from api.main import create_app


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """A client against the real model and the real cache, built once for the whole run."""
    get_settings.cache_clear()
    with TestClient(create_app()) as running:
        yield running


@pytest.fixture
def client_without_redis(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A client pointed at a port nothing is listening on, to prove the cache degrades."""
    monkeypatch.setenv("SENTISIS_REDIS_URL", "redis://127.0.0.1:6399/0")
    get_settings.cache_clear()
    with TestClient(create_app()) as running:
        yield running
    get_settings.cache_clear()


@pytest.fixture
def client_without_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A client with caching switched off entirely."""
    monkeypatch.setenv("SENTISIS_CACHE_ENABLED", "false")
    get_settings.cache_clear()
    with TestClient(create_app()) as running:
        yield running
    get_settings.cache_clear()
