from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ImageInfo(BaseModel):
    filename: str
    content_type: str
    size_bytes: int
    width: int
    height: int


class ClassPrediction(BaseModel):
    class_name: str
    probability: float
    threshold: float
    is_predicted: bool


class PredictionResponse(BaseModel):
    image: ImageInfo
    predicted_labels: list[ClassPrediction]
    all_scores: list[ClassPrediction]
    top_k: list[ClassPrediction]
    model: dict[str, Any]
    message: str | None = None
