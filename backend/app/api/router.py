from __future__ import annotations

from fastapi import APIRouter

from app.api import health, model, predictions
from app.config import get_settings

settings = get_settings()
api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(model.router, prefix=settings.api_prefix, tags=["model"])
api_router.include_router(predictions.router, prefix=settings.api_prefix, tags=["predictions"])
