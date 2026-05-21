from __future__ import annotations

import argparse
import csv
import os
import shutil
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def safe_extract_tar(tar_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r") as archive:
        for member in archive.getmembers():
            target_path = (destination / member.name).resolve()
            if not str(target_path).startswith(str(destination)):
                raise RuntimeError(f"Unsafe tar member path detected: {member.name}")
        try:
            archive.extractall(destination, filter="data")
        except TypeError:
            archive.extractall(destination)


def read_image_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_annotation(xml_path: Path) -> dict[str, int]:
    labels = {class_name: 0 for class_name in VOC_CLASSES}

    if not xml_path.exists():
        raise FileNotFoundError(f"Missing annotation XML: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is None or not name_node.text:
            continue

        class_name = name_node.text.strip()
        if class_name in labels:
            labels[class_name] = 1

    return labels


def copy_or_link_image(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        return

    if mode == "copy":
        shutil.copy2(source, destination)
        return

    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            shutil.copy2(source, destination)
            return

    raise ValueError(f"Unsupported mode: {mode}")


def write_split(
    *,
    split_name: str,
    image_ids: list[str],
    voc_root: Path,
    output_root: Path,
    file_mode: str,
) -> None:
    image_source_dir = voc_root / "JPEGImages"
    annotation_dir = voc_root / "Annotations"
    output_image_dir = output_root / split_name / "images"
    output_csv = output_root / f"{split_name}_annotations.csv"

    output_image_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int]] = []
    class_counts = {class_name: 0 for class_name in VOC_CLASSES}
    missing_images: list[str] = []
    missing_annotations: list[str] = []

    for image_id in image_ids:
        image_name = f"{image_id}.jpg"
        image_path = image_source_dir / image_name
        xml_path = annotation_dir / f"{image_id}.xml"

        if not image_path.exists():
            missing_images.append(image_name)
            continue

        if not xml_path.exists():
            missing_annotations.append(f"{image_id}.xml")
            continue

        labels = parse_annotation(xml_path)

        for class_name, value in labels.items():
            class_counts[class_name] += int(value)

        copy_or_link_image(
            source=image_path,
            destination=output_image_dir / image_name,
            mode=file_mode,
        )

        rows.append(
            {
                "image_name": image_name,
                **labels,
            }
        )

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name", *VOC_CLASSES])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{split_name.upper()} split")
    print(f"  Images listed:       {len(image_ids)}")
    print(f"  Usable rows written: {len(rows)}")
    print(f"  Output images:       {output_image_dir}")
    print(f"  Output CSV:          {output_csv}")

    if missing_images:
        print(f"  Missing images:      {len(missing_images)}")

    if missing_annotations:
        print(f"  Missing annotations: {len(missing_annotations)}")

    print("  Class counts:")
    for class_name in VOC_CLASSES:
        print(f"    {class_name:12s} {class_counts[class_name]}")


def find_voc2012_root(extract_root: Path) -> Path:
    candidates = [
        extract_root / "VOCdevkit" / "VOC2012",
        extract_root / "VOC2012",
    ]

    for candidate in candidates:
        if (
            (candidate / "JPEGImages").is_dir()
            and (candidate / "Annotations").is_dir()
            and (candidate / "ImageSets" / "Main").is_dir()
        ):
            return candidate

    matches = list(extract_root.rglob("VOC2012"))
    for candidate in matches:
        if (
            (candidate / "JPEGImages").is_dir()
            and (candidate / "Annotations").is_dir()
            and (candidate / "ImageSets" / "Main").is_dir()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not find VOC2012 root after extraction. Expected JPEGImages, "
        "Annotations, and ImageSets/Main directories."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Pascal VOC 2012 train/val data for multi-label classification."
    )
    parser.add_argument(
        "--tar",
        type=Path,
        default=Path("raw/VOCtrainval_11-May-2012.tar"),
        help="Path to VOCtrainval_11-May-2012.tar.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/voc2012_processed"),
        help="Output directory for processed train/val images and CSV files.",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("raw/extracted"),
        help="Directory where the VOC tar will be extracted.",
    )
    parser.add_argument(
        "--file-mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="Use copy for portability or hardlink to save disk space.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Extract even if the VOC2012 directory already exists.",
    )

    args = parser.parse_args()

    tar_path = args.tar.resolve()
    output_root = args.output.resolve()
    extract_root = args.extract_dir.resolve()

    if not tar_path.exists():
        raise FileNotFoundError(f"VOC tar file not found: {tar_path}")

    existing_voc_root = None
    try:
        existing_voc_root = find_voc2012_root(extract_root)
    except FileNotFoundError:
        existing_voc_root = None

    if args.force_extract or existing_voc_root is None:
        print(f"Extracting {tar_path} to {extract_root}")
        safe_extract_tar(tar_path, extract_root)

    voc_root = find_voc2012_root(extract_root)
    image_sets_dir = voc_root / "ImageSets" / "Main"

    train_ids = read_image_ids(image_sets_dir / "train.txt")
    val_ids = read_image_ids(image_sets_dir / "val.txt")

    output_root.mkdir(parents=True, exist_ok=True)

    write_split(
        split_name="train",
        image_ids=train_ids,
        voc_root=voc_root,
        output_root=output_root,
        file_mode=args.file_mode,
    )

    write_split(
        split_name="val",
        image_ids=val_ids,
        voc_root=voc_root,
        output_root=output_root,
        file_mode=args.file_mode,
    )

    print("\nDone.")
    print(f"Processed dataset root: {output_root}")
    print("\nUse these paths in configs/voc20.yaml:")
    print(f"  train_images: {output_root / 'train' / 'images'}")
    print(f"  val_images: {output_root / 'val' / 'images'}")
    print(f"  train_annotations: {output_root / 'train_annotations.csv'}")
    print(f"  val_annotations: {output_root / 'val_annotations.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())