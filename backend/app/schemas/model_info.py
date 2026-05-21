from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ModelInfoResponse(BaseModel):
    available: bool
    architecture: str | None = None
    num_classes: int | None = None
    classes: list[str] = []
    image_size: int | None = None
    thresholds: dict[str, float] = {}
    validation_metrics: dict[str, Any] = {}
    message: str | None = None
