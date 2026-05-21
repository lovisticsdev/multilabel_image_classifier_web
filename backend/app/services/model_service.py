from __future__ import annotations

from functools import lru_cache

from fastapi import HTTPException, status

from app.config import get_settings
from app.schemas.model_info import ModelInfoResponse
from multilabel_classifier.inference.artifact import ModelMetadata
from multilabel_classifier.inference.predictor import ImagePredictor


class ModelService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.predictor = ImagePredictor(self.settings.model_artifact_dir)
        self._load_error: str | None = None

    def _metadata_or_error(self) -> ModelMetadata | None:
        try:
            return self.predictor.model_info()
        except Exception as exc:
            self._load_error = str(exc)
            return None

    def require_predictor(self) -> ImagePredictor:
        metadata = self._metadata_or_error()
        if metadata is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "model_not_available",
                    "message": "No exported model artifact is available. Train/export a model first.",
                    "details": self._load_error,
                },
            )
        return self.predictor

    def info_response(self) -> ModelInfoResponse:
        metadata = self._metadata_or_error()
        if metadata is None:
            return ModelInfoResponse(
                available=False,
                message="No exported model artifact is available. Train/export a model first.",
            )
        return ModelInfoResponse(
            available=True,
            architecture=metadata.architecture,
            num_classes=metadata.num_classes,
            classes=metadata.classes,
            image_size=metadata.image_size,
            thresholds=metadata.thresholds,
            validation_metrics=metadata.validation_metrics,
        )


@lru_cache
def get_model_service() -> ModelService:
    return ModelService()
