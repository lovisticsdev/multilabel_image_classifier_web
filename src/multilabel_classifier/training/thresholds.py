from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def apply_thresholds(probabilities: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (probabilities >= thresholds.reshape(1, -1)).astype(int)


def tune_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    search_min: float = 0.05,
    search_max: float = 0.95,
    search_steps: int = 19,
    default_threshold: float = 0.5,
) -> np.ndarray:
    if probabilities.shape != labels.shape:
        raise ValueError("probabilities and labels must have the same shape")

    thresholds = np.full(probabilities.shape[1], default_threshold, dtype=float)
    candidates = np.linspace(search_min, search_max, search_steps)
    for class_index in range(probabilities.shape[1]):
        y_true = labels[:, class_index]
        if y_true.sum() == 0:
            continue
        best_score = -1.0
        best_threshold = default_threshold
        for threshold in candidates:
            y_pred = (probabilities[:, class_index] >= threshold).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        thresholds[class_index] = best_threshold
    return thresholds
