import os

import pytest
from pydantic import ValidationError

from api.core.config import Settings, get_settings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [key for key in os.environ if key.upper().startswith("SENTISIS_")]:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()


def test_defaults_point_at_the_committed_artifacts() -> None:
    settings = Settings()
    assert settings.model_path.exists()
    assert settings.tokenizer_path.exists()


def test_max_length_matches_what_the_model_was_trained_with() -> None:
    assert Settings().max_length == 96


def test_log_level_is_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTISIS_LOG_LEVEL", "debug")
    assert Settings().log_level == "DEBUG"


def test_an_unknown_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTISIS_LOG_LEVEL", "chatty")
    with pytest.raises(ValidationError, match="log_level"):
        Settings()


def test_get_settings_parses_once() -> None:
    assert get_settings() is get_settings()
