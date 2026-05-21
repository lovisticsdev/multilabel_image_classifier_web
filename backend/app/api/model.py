from __future__ import annotations

from fastapi import APIRouter, Depends

from app.schemas.model_info import ModelInfoResponse
from app.services.model_service import ModelService, get_model_service

router = APIRouter()


@router.get("/model", response_model=ModelInfoResponse)
def model_info(service: ModelService = Depends(get_model_service)) -> ModelInfoResponse:
    return service.info_response()
