from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from multilabel_classifier.config import ProjectConfig
from multilabel_classifier.data.dataset import MultiLabelImageDataset
from multilabel_classifier.data.transforms import build_eval_transform, build_train_transform
from multilabel_classifier.data.validation import load_validated_annotations


@dataclass(frozen=True)
class DataLoaders:
    train: object
    val: object
    classes: list[str]
    train_size: int
    val_size: int


def create_dataloaders(config: ProjectConfig) -> DataLoaders:
    import torch
    from torch.utils.data import DataLoader

    classes = list(config.dataset.classes or [])
    if not classes:
        raise ValueError("No classes resolved from dataset config.")

    train_frame = load_validated_annotations(
        image_dir=Path(config.dataset.train_images),
        annotation_file=Path(config.dataset.train_annotations),
        image_name_column=config.dataset.image_name_column,
        classes=classes,
        split_name="train",
    )
    val_frame = load_validated_annotations(
        image_dir=Path(config.dataset.val_images),
        annotation_file=Path(config.dataset.val_annotations),
        image_name_column=config.dataset.image_name_column,
        classes=classes,
        split_name="val",
    )

    train_dataset = MultiLabelImageDataset(
        image_dir=config.dataset.train_images,
        annotation_file=config.dataset.train_annotations,
        image_name_column=config.dataset.image_name_column,
        classes=classes,
        transform=build_train_transform(config.model.image_size),
        split_name="train",
        frame=train_frame,
    )
    val_dataset = MultiLabelImageDataset(
        image_dir=config.dataset.val_images,
        annotation_file=config.dataset.val_annotations,
        image_name_column=config.dataset.image_name_column,
        classes=classes,
        transform=build_eval_transform(config.model.image_size),
        split_name="val",
        frame=val_frame,
    )

    num_workers = int(config.training.num_workers)
    loader_kwargs = {
        "batch_size": config.training.batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    return DataLoaders(
        train=train_loader,
        val=val_loader,
        classes=classes,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
    )
