from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from multilabel_classifier.constants import VOC20_CLASSES, VOC8_LEGACY_CLASSES


class DatasetConfig(BaseModel):
    train_images: Path
    val_images: Path
    train_annotations: Path
    val_annotations: Path
    image_name_column: str = "image_name"
    class_mode: Literal["all", "legacy8", "explicit"] = "all"
    classes: list[str] | None = None

    @model_validator(mode="after")
    def resolve_classes(self) -> "DatasetConfig":
        if self.class_mode == "all":
            self.classes = list(VOC20_CLASSES)
        elif self.class_mode == "legacy8":
            self.classes = list(VOC8_LEGACY_CLASSES)
        elif not self.classes:
            raise ValueError("classes must be provided when class_mode='explicit'.")
        return self


class ModelConfig(BaseModel):
    architecture: Literal["efficientnet_b0", "efficientnet_b2", "efficientnet_b3"] = "efficientnet_b2"
    pretrained: bool = True
    image_size: int = Field(default=256, ge=64, le=1024)
    dropout: float = Field(default=0.3, ge=0.0, le=0.8)


class TrainingConfig(BaseModel):
    batch_size: int = Field(default=32, ge=1)
    epochs: int = Field(default=30, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    early_stopping_patience: int = Field(default=7, ge=1)
    num_workers: int = Field(default=2, ge=0)
    seed: int = 42
    mixed_precision: Literal["auto", "true", "false"] = "auto"
    device: Literal["auto", "cpu", "cuda", "tpu"] = "auto"
    compute_pos_weight: bool = True


class ThresholdConfig(BaseModel):
    tune: bool = True
    metric: Literal["f1"] = "f1"
    default_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    search_min: float = Field(default=0.05, ge=0.0, le=1.0)
    search_max: float = Field(default=0.95, ge=0.0, le=1.0)
    search_steps: int = Field(default=19, ge=2, le=200)


class OutputConfig(BaseModel):
    checkpoint_dir: Path = Path("checkpoints")
    export_dir: Path = Path("checkpoints/exported")
    report_dir: Path = Path("outputs")


class ProjectConfig(BaseModel):
    project_name: str = "multilabel-image-classifier"
    dataset: DatasetConfig
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    outputs: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("project_name")
    @classmethod
    def non_empty_project_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("project_name cannot be empty")
        return value


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file_obj:
        raw = yaml.safe_load(file_obj) or {}
    return ProjectConfig.model_validate(raw)
