"""The one typed settings object the API reads its configuration from."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DISTILBERT_DIR = REPO_ROOT / "model" / "artifacts" / "distilbert"
LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"})


class Settings(BaseSettings):
    """Every value the API reads from the environment, validated once at startup."""

    model_config = SettingsConfigDict(env_prefix="SENTISIS_", extra="forbid", frozen=True)

    model_path: Path = DISTILBERT_DIR / "model_int8.onnx"
    tokenizer_path: Path = DISTILBERT_DIR / "tokenizer" / "tokenizer.json"
    model_version: str = "distilbert-int8-v1"
    max_length: int = Field(default=96, gt=0)
    max_text_length: int = Field(default=5_000, gt=0)
    max_batch_size: int = Field(default=128, gt=0)
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = Field(default=86_400, gt=0)
    cache_enabled: bool = True
    log_level: str = "INFO"
    onnx_threads: int = Field(default=1, gt=0)

    @field_validator("log_level", mode="after")
    @classmethod
    def _known_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in LOG_LEVELS:
            raise ValueError(f"log_level must be one of {sorted(LOG_LEVELS)}, got {value!r}")
        return level


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, parsed once."""
    return Settings()
