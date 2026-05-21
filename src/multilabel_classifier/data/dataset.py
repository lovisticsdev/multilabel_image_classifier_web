from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from multilabel_classifier.data.validation import load_validated_annotations


class MultiLabelImageDataset:
    def __init__(
        self,
        *,
        image_dir: str | Path,
        annotation_file: str | Path,
        classes: list[str],
        image_name_column: str = "image_name",
        transform=None,
        split_name: str = "dataset",
        frame: pd.DataFrame | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file)
        self.classes = list(classes)
        self.image_name_column = image_name_column
        self.transform = transform
        self.split_name = split_name
        self.frame = frame if frame is not None else load_validated_annotations(
            image_dir=self.image_dir,
            annotation_file=self.annotation_file,
            image_name_column=self.image_name_column,
            classes=self.classes,
            split_name=self.split_name,
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        import torch

        row = self.frame.iloc[index]
        image_name = str(row[self.image_name_column])
        image_path = self.image_dir / image_name
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        labels = torch.tensor(row[self.classes].to_numpy(dtype="float32"), dtype=torch.float32)
        return image, labels
