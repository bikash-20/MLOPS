"""Reusable classification metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute accuracy, precision, recall, and F1 (macro-averaged)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    accuracy = float(np.mean(y_true == y_pred))

    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for c in classes:
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
    }


def per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, float]:
    """Return ``{class_name: accuracy}`` for each class index."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out: dict[str, float] = {}
    for i, name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() == 0:
            out[name] = 0.0
            continue
        out[name] = float(np.mean(y_pred[mask] == y_true[mask]))
    return out


__all__ = ["compute_metrics", "per_class_accuracy"]
