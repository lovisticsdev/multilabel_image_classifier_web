from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from multilabel_classifier.evaluation.metrics import compute_multilabel_metrics
from multilabel_classifier.training.checkpoints import save_checkpoint
from multilabel_classifier.training.thresholds import tune_thresholds


@dataclass
class TrainingResult:
    best_checkpoint: Path
    thresholds: np.ndarray
    metrics: dict
    train_losses: list[float]


def collect_probabilities(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    probabilities = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            logits = model(images)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    return np.concatenate(probabilities, axis=0), np.concatenate(labels, axis=0)


def train_model(
    *,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    classes: list[str],
    epochs: int,
    patience: int,
    checkpoint_path: str | Path,
    threshold_config,
) -> TrainingResult:
    import torch

    checkpoint_path = Path(checkpoint_path)
    train_losses: list[float] = []
    best_macro_f1 = -1.0
    best_thresholds = np.full(len(classes), threshold_config.default_threshold, dtype=float)
    best_metrics: dict = {}
    epochs_without_improvement = 0

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = images.size(0)
            total_loss += float(loss.detach().cpu()) * batch_size
            seen += batch_size
            progress.set_postfix(loss=total_loss / max(seen, 1))

        train_loss = total_loss / max(seen, 1)
        train_losses.append(train_loss)

        val_probabilities, val_labels = collect_probabilities(model, val_loader, device)
        if threshold_config.tune:
            thresholds = tune_thresholds(
                val_probabilities,
                val_labels,
                search_min=threshold_config.search_min,
                search_max=threshold_config.search_max,
                search_steps=threshold_config.search_steps,
                default_threshold=threshold_config.default_threshold,
            )
        else:
            thresholds = np.full(len(classes), threshold_config.default_threshold, dtype=float)
        metrics = compute_multilabel_metrics(val_probabilities, val_labels, thresholds, classes)
        macro_f1 = metrics["macro_f1"]
        if scheduler is not None:
            scheduler.step(macro_f1)
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} macro_f1={macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_thresholds = thresholds
            best_metrics = metrics
            save_checkpoint(model, checkpoint_path, {"classes": classes, "macro_f1": macro_f1})
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

    return TrainingResult(
        best_checkpoint=checkpoint_path,
        thresholds=best_thresholds,
        metrics=best_metrics,
        train_losses=train_losses,
    )
