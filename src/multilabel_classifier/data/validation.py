from __future__ import annotations

from pathlib import Path

import pandas as pd


class DatasetValidationError(ValueError):
    """Raised when a dataset CSV or image directory is invalid."""


def validate_dataset_paths(*, image_dir: Path, annotation_file: Path, split_name: str) -> None:
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"{split_name} image directory not found: {image_dir}")
    if not annotation_file.exists() or not annotation_file.is_file():
        raise FileNotFoundError(f"{split_name} annotation file not found: {annotation_file}")


def validate_annotation_frame(
    frame: pd.DataFrame,
    *,
    image_dir: Path,
    image_name_column: str,
    classes: list[str],
    split_name: str,
) -> pd.DataFrame:
    required_columns = [image_name_column, *classes]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise DatasetValidationError(
            f"{split_name} annotations are missing required columns: {missing_columns}"
        )

    cleaned = frame[required_columns].copy()
    cleaned[image_name_column] = cleaned[image_name_column].astype(str).map(lambda value: Path(value).name)

    duplicate_names = cleaned[image_name_column][cleaned[image_name_column].duplicated()].unique().tolist()
    if duplicate_names:
        preview = duplicate_names[:10]
        raise DatasetValidationError(
            f"{split_name} annotations contain duplicate image names. Examples: {preview}"
        )

    if cleaned[image_name_column].isna().any() or (cleaned[image_name_column].str.strip() == "").any():
        raise DatasetValidationError(f"{split_name} annotations contain empty image names.")

    missing_images = [
        image_name
        for image_name in cleaned[image_name_column].tolist()
        if not (image_dir / image_name).is_file()
    ]
    if missing_images:
        preview = missing_images[:10]
        raise DatasetValidationError(
            f"{split_name} split references {len(missing_images)} missing images. Examples: {preview}"
        )

    for class_name in classes:
        values = pd.to_numeric(cleaned[class_name], errors="coerce")
        if values.isna().any():
            raise DatasetValidationError(
                f"{split_name}.{class_name} contains non-numeric or missing label values."
            )
        invalid = sorted(set(values.astype(int).tolist()) - {0, 1})
        if invalid:
            raise DatasetValidationError(
                f"{split_name}.{class_name} must contain only 0/1 labels; found {invalid}."
            )
        cleaned[class_name] = values.astype("float32")

    return cleaned.reset_index(drop=True)


def load_validated_annotations(
    *,
    image_dir: Path,
    annotation_file: Path,
    image_name_column: str,
    classes: list[str],
    split_name: str,
) -> pd.DataFrame:
    validate_dataset_paths(image_dir=image_dir, annotation_file=annotation_file, split_name=split_name)
    frame = pd.read_csv(annotation_file)
    return validate_annotation_frame(
        frame,
        image_dir=image_dir,
        image_name_column=image_name_column,
        classes=classes,
        split_name=split_name,
    )
