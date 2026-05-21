from __future__ import annotations

import argparse
from pathlib import Path

from multilabel_classifier.config import load_config
from multilabel_classifier.data.datamodule import create_dataloaders
from multilabel_classifier.inference.artifact import export_artifact
from multilabel_classifier.models.factory import create_model
from multilabel_classifier.training.losses import compute_pos_weight
from multilabel_classifier.training.trainer import train_model
from multilabel_classifier.utils.device import select_device
from multilabel_classifier.utils.files import ensure_dir
from multilabel_classifier.utils.reproducibility import set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a multi-label image classifier.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.training.seed)
    ensure_dir(config.outputs.checkpoint_dir)
    ensure_dir(config.outputs.export_dir)
    ensure_dir(config.outputs.report_dir)

    device = select_device(config.training.device)
    loaders = create_dataloaders(config)
    classes = loaders.classes

    model = create_model(
        architecture=config.model.architecture,
        num_classes=len(classes),
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
    ).to(device)

    import torch

    pos_weight = None
    if config.training.compute_pos_weight:
        pos_weight = compute_pos_weight(str(config.dataset.train_annotations), classes).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    checkpoint_path = Path(config.outputs.checkpoint_dir) / "best_model.pt"

    result = train_model(
        model=model,
        train_loader=loaders.train,
        val_loader=loaders.val,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        classes=classes,
        epochs=config.training.epochs,
        patience=config.training.early_stopping_patience,
        checkpoint_path=checkpoint_path,
        threshold_config=config.thresholds,
    )

    export_artifact(
        checkpoint_path=result.best_checkpoint,
        output_dir=config.outputs.export_dir,
        architecture=config.model.architecture,
        classes=classes,
        image_size=config.model.image_size,
        thresholds=result.thresholds.tolist(),
        validation_metrics=result.metrics,
        dropout=config.model.dropout,
    )
    print(f"Exported model artifact to {config.outputs.export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
