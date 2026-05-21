from __future__ import annotations

import pandas as pd


def compute_pos_weight(annotation_file: str, classes: list[str]):
    import torch

    frame = pd.read_csv(annotation_file)
    positives = frame[classes].sum(axis=0).astype(float)
    negatives = len(frame) - positives
    weights = negatives / positives.clip(lower=1.0)
    return torch.tensor(weights.values, dtype=torch.float32)
