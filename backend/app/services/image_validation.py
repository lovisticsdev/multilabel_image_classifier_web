from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import get_settings
from app.schemas.prediction import ImageInfo

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ImageValidationService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def validate_upload(self, file: UploadFile) -> tuple[Image.Image, ImageInfo]:
        filename = Path(file.filename or "upload").name
        extension = Path(filename).suffix.lower()
        content_type = file.content_type or ""
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "unsupported_extension", "message": "Upload JPEG, PNG, or WebP."},
            )
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "unsupported_content_type", "message": "Upload JPEG, PNG, or WebP."},
            )
        data = file.file.read()
        if not data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "empty_file", "message": "Uploaded file is empty."},
            )
        if len(data) > self.settings.max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={"code": "file_too_large", "message": "Uploaded file exceeds max size."},
            )
        try:
            image = Image.open(BytesIO(data))
            image = ImageOps.exif_transpose(image).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "invalid_image", "message": "Uploaded file is not a readable image."},
            ) from exc
        width, height = image.size
        if width < 32 or height < 32:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "image_too_small", "message": "Image must be at least 32x32."},
            )
        return image, ImageInfo(
            filename=filename,
            content_type=content_type,
            size_bytes=len(data),
            width=width,
            height=height,
        )
