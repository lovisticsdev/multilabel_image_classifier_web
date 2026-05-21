from __future__ import annotations

from functools import lru_cache

from PIL import Image

from app.config import get_settings
from app.schemas.prediction import ClassPrediction, ImageInfo, PredictionResponse
from app.services.model_service import get_model_service


class PredictionService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.model_service = get_model_service()

    def predict(self, image: Image.Image, image_info: ImageInfo) -> PredictionResponse:
        predictor = self.model_service.require_predictor()
        scores = predictor.predict(image)
        all_scores = [ClassPrediction(**score.__dict__) for score in scores]
        predicted = [score for score in all_scores if score.is_predicted]
        top_k = all_scores[: self.settings.prediction_top_k]
        metadata = predictor.model_info()
        message = None if predicted else "No label exceeded its configured threshold."
        return PredictionResponse(
            image=image_info,
            predicted_labels=predicted,
            all_scores=all_scores,
            top_k=top_k,
            model={
                "architecture": metadata.architecture,
                "num_classes": metadata.num_classes,
                "image_size": metadata.image_size,
            },
            message=message,
        )


@lru_cache
def get_prediction_service() -> PredictionService:
    return PredictionService()
