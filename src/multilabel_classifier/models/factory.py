from __future__ import annotations

from multilabel_classifier.models.efficientnet import create_efficientnet_classifier


def create_model(*, architecture: str, num_classes: int, pretrained: bool, dropout: float):
    if architecture.startswith("efficientnet_"):
        return create_efficientnet_classifier(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported model architecture: {architecture}")
