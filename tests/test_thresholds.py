from __future__ import annotations

import numpy as np

from multilabel_classifier.training.thresholds import apply_thresholds, tune_thresholds


def test_apply_thresholds():
    probs = np.array([[0.2, 0.8], [0.7, 0.4]])
    thresholds = np.array([0.5, 0.5])
    assert apply_thresholds(probs, thresholds).tolist() == [[0, 1], [1, 0]]


def test_tune_thresholds_shape():
    probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.7, 0.6]])
    labels = np.array([[0, 1], [1, 0], [1, 1]])
    thresholds = tune_thresholds(probs, labels, search_steps=5)
    assert thresholds.shape == (2,)
