"""Tests for shared evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics, per_class_accuracy


def test_perfect_predictions_yield_unit_metrics():
    y = np.array([0, 0, 1, 1, 2, 2])
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["precision_macro"] == 1.0
    assert m["recall_macro"] == 1.0
    assert m["f1_macro"] == 1.0


def test_per_class_accuracy_values():
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 0, 2])
    pca = per_class_accuracy(y_true, y_pred, ["a", "b", "c"])
    assert pca["a"] == pytest.approx(2 / 3)
    assert pca["b"] == pytest.approx(0.5)
    assert pca["c"] == pytest.approx(1.0)


def test_empty_class_returns_zero():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    pca = per_class_accuracy(y_true, y_pred, ["a", "b"])
    assert pca["a"] == 1.0
    assert pca["b"] == 1.0
