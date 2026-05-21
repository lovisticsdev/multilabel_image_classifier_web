from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from multilabel_classifier.inference.predictor import ImagePredictor


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict labels for one image.")
    parser.add_argument("image")
    parser.add_argument("--artifact-dir", default="checkpoints/exported")
    args = parser.parse_args()

    predictor = ImagePredictor(args.artifact_dir)
    with Image.open(Path(args.image)) as image:
        scores = predictor.predict(image)
    for score in scores:
        mark = "*" if score.is_predicted else " "
        print(f"{mark} {score.class_name:12s} p={score.probability:.4f} threshold={score.threshold:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
