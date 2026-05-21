from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from multilabel_classifier.training.thresholds import apply_thresholds


def compute_multilabel_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
    classes: list[str],
) -> dict:
    predictions = apply_thresholds(probabilities, thresholds)
    per_class = []
    for index, class_name in enumerate(classes):
        per_class.append(
            {
                "class_name": class_name,
                "precision": float(precision_score(labels[:, index], predictions[:, index], zero_division=0)),
                "recall": float(recall_score(labels[:, index], predictions[:, index], zero_division=0)),
                "f1": float(f1_score(labels[:, index], predictions[:, index], zero_division=0)),
                "threshold": float(thresholds[index]),
            }
        )
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "per_class": per_class,
    }
