from __future__ import annotations

import argparse

from multilabel_classifier.config import load_config
from multilabel_classifier.inference.artifact import export_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a checkpoint to web inference artifact format.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    classes = config.dataset.classes or []
    thresholds = [config.thresholds.default_threshold for _ in classes]
    export_artifact(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        architecture=config.model.architecture,
        classes=classes,
        image_size=config.model.image_size,
        thresholds=thresholds,
        validation_metrics={},
        dropout=config.model.dropout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
