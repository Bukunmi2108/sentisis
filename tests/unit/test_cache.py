import asyncio
from collections.abc import Coroutine, Sequence
from typing import Any, TypeVar

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from api.inference.cache import PredictionCache, cache_key

ROW = [0.1, 0.2, 0.7]

T = TypeVar("T")


def run(coroutine: Coroutine[Any, Any, T]) -> T:
    """Drive one coroutine to completion without needing an async test plugin."""
    return asyncio.run(coroutine)


class FakePipeline:
    def __init__(self, store: dict[str, str]) -> None:
        self._store = store
        self._pending: list[tuple[str, str]] = []

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._pending.append((key, value))

    async def execute(self) -> None:
        self._store.update(self._pending)


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def mget(self, keys: Sequence[str]) -> list[str | None]:
        return [self.store.get(key) for key in keys]

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self.store)

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


class BrokenRedis:
    async def mget(self, keys: Sequence[str]) -> list[str | None]:
        raise RedisConnectionError("redis is down")

    def pipeline(self) -> Any:
        raise RedisConnectionError("redis is down")

    async def ping(self) -> bool:
        raise RedisConnectionError("redis is down")

    async def aclose(self) -> None:
        raise RedisConnectionError("redis is down")


def build(client: Any) -> PredictionCache:
    return PredictionCache(client, model_version="v1", ttl=60)


def test_the_key_carries_the_model_version() -> None:
    assert cache_key("v1", "hello").startswith("sentisis:v1:")
    assert cache_key("v1", "hello") != cache_key("v2", "hello")


def test_different_texts_get_different_keys() -> None:
    assert cache_key("v1", "hello") != cache_key("v1", "goodbye")


@pytest.mark.parametrize("stored", ["not json", '{"label": "positive"}', "null"])
def test_an_unusable_entry_reads_back_as_a_miss(stored: str) -> None:
    client = FakeRedis()
    client.store[cache_key("v1", "hello")] = stored
    assert run(build(client).get_many(["hello"])) == {}


def test_a_written_row_reads_back() -> None:
    cache = build(FakeRedis())
    run(cache.set_many({"hello": ROW}))
    assert run(cache.get_many(["hello"])) == {"hello": ROW}


def test_an_unknown_text_is_a_miss() -> None:
    assert run(build(FakeRedis()).get_many(["never seen"])) == {}


def test_repeated_texts_are_looked_up_once() -> None:
    cache = build(FakeRedis())
    run(cache.set_many({"hello": ROW}))
    assert run(cache.get_many(["hello", "hello", "hello"])) == {"hello": ROW}


def test_a_read_failure_is_a_miss_not_an_error() -> None:
    assert run(build(BrokenRedis()).get_many(["hello"])) == {}


def test_a_write_failure_is_swallowed() -> None:
    run(build(BrokenRedis()).set_many({"hello": ROW}))


def test_ping_reports_false_when_redis_is_down() -> None:
    assert run(build(BrokenRedis()).ping()) is False


def test_close_never_raises() -> None:
    run(build(BrokenRedis()).close())


def test_empty_input_touches_nothing() -> None:
    cache = build(BrokenRedis())
    assert run(cache.get_many([])) == {}
    run(cache.set_many({}))
