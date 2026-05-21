from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from multilabel_classifier.constants import IMAGENET_MEAN, IMAGENET_STD
from multilabel_classifier.models.factory import create_model
from multilabel_classifier.training.checkpoints import load_checkpoint


class NormalizationMetadata(BaseModel):
    mean: list[float] = Field(default_factory=lambda: list(IMAGENET_MEAN))
    std: list[float] = Field(default_factory=lambda: list(IMAGENET_STD))


class ModelMetadata(BaseModel):
    architecture: str
    num_classes: int
    classes: list[str]
    image_size: int
    thresholds: dict[str, float]
    normalization: NormalizationMetadata = Field(default_factory=NormalizationMetadata)
    trained_at: str | None = None
    validation_metrics: dict[str, Any] = Field(default_factory=dict)


def export_artifact(
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    architecture: str,
    classes: list[str],
    image_size: int,
    thresholds: list[float],
    validation_metrics: dict,
    dropout: float = 0.3,
) -> None:
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    metadata = ModelMetadata(
        architecture=architecture,
        num_classes=len(classes),
        classes=classes,
        image_size=image_size,
        thresholds={class_name: float(thresholds[index]) for index, class_name in enumerate(classes)},
        trained_at=datetime.now(timezone.utc).isoformat(),
        validation_metrics=validation_metrics,
    )
    model = create_model(
        architecture=architecture,
        num_classes=len(classes),
        pretrained=False,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    torch.save(model.state_dict(), output_dir / "model.pt")
    (output_dir / "metadata.json").write_text(metadata.model_dump_json(indent=2), encoding="utf-8")


def load_artifact(artifact_dir: str | Path, device):
    import torch

    artifact_dir = Path(artifact_dir)
    metadata_path = artifact_dir / "metadata.json"
    weights_path = artifact_dir / "model.pt"
    if not metadata_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Missing model artifact. Expected {metadata_path} and {weights_path}."
        )
    metadata = ModelMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    model = create_model(
        architecture=metadata.architecture,
        num_classes=metadata.num_classes,
        pretrained=False,
        dropout=0.0,
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, metadata
