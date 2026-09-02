"""Higher-level evaluation routines that print per-class metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.evaluation.metrics import compute_metrics, per_class_accuracy
from src.utils import get_logger

logger = get_logger(__name__)


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str] | None = None,
    label: str = "test",
) -> dict[str, Any]:
    """Compute metrics, log them, and return the result dict."""
    metrics = compute_metrics(y_true, y_pred)
    logger.info(
        "[%s] accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        label,
        metrics["accuracy"],
        metrics["precision_macro"],
        metrics["recall_macro"],
        metrics["f1_macro"],
    )
    if class_names is not None:
        pca = per_class_accuracy(y_true, y_pred, class_names)
        for name, acc in pca.items():
            logger.info("  %s: %.4f", name, acc)
        metrics["per_class_accuracy"] = pca
    metrics["label"] = label
    return metrics


__all__ = ["evaluate_classifier"]
