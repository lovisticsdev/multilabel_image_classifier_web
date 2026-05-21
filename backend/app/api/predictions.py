from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.schemas.prediction import PredictionResponse
from app.services.image_validation import ImageValidationService
from app.services.prediction_service import PredictionService, get_prediction_service

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict(
    file: UploadFile = File(...),
    validator: ImageValidationService = Depends(ImageValidationService),
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    image, image_info = validator.validate_upload(file)
    return service.predict(image, image_info)
