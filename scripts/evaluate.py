from __future__ import annotations

import argparse
from pathlib import Path

from multilabel_classifier.config import load_config
from multilabel_classifier.data.datamodule import create_dataloaders
from multilabel_classifier.evaluation.metrics import compute_multilabel_metrics
from multilabel_classifier.inference.artifact import load_artifact
from multilabel_classifier.training.trainer import collect_probabilities
from multilabel_classifier.utils.device import select_device


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an exported model artifact on the validation split.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--artifact-dir", default=None, help="Exported artifact directory. Defaults to config outputs.export_dir.")
    args = parser.parse_args()

    config = load_config(args.config)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else config.outputs.export_dir
    device = select_device(config.training.device)
    loaders = create_dataloaders(config)
    model, metadata = load_artifact(artifact_dir, device)

    if metadata.classes != loaders.classes:
        raise ValueError(
            "Artifact class order does not match dataset config. "
            f"artifact={metadata.classes}, config={loaders.classes}"
        )

    probabilities, labels = collect_probabilities(model, loaders.val, device)
    thresholds = [metadata.thresholds.get(class_name, config.thresholds.default_threshold) for class_name in metadata.classes]

    import numpy as np

    metrics = compute_multilabel_metrics(
        probabilities,
        labels,
        np.asarray(thresholds, dtype=float),
        metadata.classes,
    )

    print(f"artifact_dir: {artifact_dir}")
    print(f"macro_f1: {metrics['macro_f1']:.4f}")
    print(f"micro_f1: {metrics['micro_f1']:.4f}")
    print("\nPer-class metrics:")
    for item in metrics["per_class"]:
        print(
            f"{item['class_name']:12s} "
            f"precision={item['precision']:.4f} "
            f"recall={item['recall']:.4f} "
            f"f1={item['f1']:.4f} "
            f"threshold={item['threshold']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
