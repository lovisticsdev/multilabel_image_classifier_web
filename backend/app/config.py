from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Multi-Label Image Classifier API"
    app_env: str = "development"
    debug: bool = True
    api_prefix: str = "/api"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    model_artifact_dir: Path = PROJECT_ROOT / "checkpoints" / "exported"
    prediction_top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_csv(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            if value.startswith("["):
                return value
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("model_artifact_dir", mode="before")
    @classmethod
    def resolve_artifact_dir(cls, value: str | Path) -> Path:
        return _resolve_path(value)

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() in {"development", "dev", "local"}

    @model_validator(mode="after")
    def finalize(self) -> Self:
        if not self.is_development and self.debug:
            raise ValueError("DEBUG must be false outside development.")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
