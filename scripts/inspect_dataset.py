from __future__ import annotations

import argparse

import pandas as pd

from multilabel_classifier.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect class counts for a dataset config.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    classes = config.dataset.classes or []
    for split, annotation in [("train", config.dataset.train_annotations), ("val", config.dataset.val_annotations)]:
        frame = pd.read_csv(annotation)
        print(f"\n{split}: {len(frame)} rows")
        print(frame[classes].sum().sort_values(ascending=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
