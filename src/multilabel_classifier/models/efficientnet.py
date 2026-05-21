from __future__ import annotations


def create_efficientnet_classifier(
    *,
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3,
):
    import torch.nn as nn
    from torchvision import models

    model_builders = {
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b2": models.efficientnet_b2,
        "efficientnet_b3": models.efficientnet_b3,
    }
    weight_enums = {
        "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
        "efficientnet_b2": models.EfficientNet_B2_Weights.DEFAULT,
        "efficientnet_b3": models.EfficientNet_B3_Weights.DEFAULT,
    }
    if architecture not in model_builders:
        raise ValueError(f"Unsupported architecture: {architecture}")

    weights = weight_enums[architecture] if pretrained else None
    model = model_builders[architecture](weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model
