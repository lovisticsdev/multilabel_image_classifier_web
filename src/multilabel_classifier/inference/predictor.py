from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from multilabel_classifier.inference.artifact import ModelMetadata, load_artifact
from multilabel_classifier.inference.preprocessing import preprocess_image
from multilabel_classifier.utils.device import select_device


@dataclass(frozen=True)
class ClassScore:
    class_name: str
    probability: float
    threshold: float
    is_predicted: bool


class ImagePredictor:
    def __init__(self, artifact_dir: str | Path, device_name: str = "auto") -> None:
        self.artifact_dir = Path(artifact_dir)
        self.device = select_device(device_name)
        self.model = None
        self.metadata: ModelMetadata | None = None

    def load(self) -> None:
        self.model, self.metadata = load_artifact(self.artifact_dir, self.device)

    def model_info(self) -> ModelMetadata:
        if self.metadata is None:
            self.load()
        assert self.metadata is not None
        return self.metadata

    def predict(self, image: Image.Image) -> list[ClassScore]:
        import torch

        if self.model is None or self.metadata is None:
            self.load()
        assert self.model is not None and self.metadata is not None
        tensor = preprocess_image(image, self.metadata.image_size).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        scores = []
        for index, class_name in enumerate(self.metadata.classes):
            threshold = float(self.metadata.thresholds.get(class_name, 0.5))
            probability = float(probabilities[index])
            scores.append(
                ClassScore(
                    class_name=class_name,
                    probability=probability,
                    threshold=threshold,
                    is_predicted=probability >= threshold,
                )
            )
        scores.sort(key=lambda item: item.probability, reverse=True)
        return scores
