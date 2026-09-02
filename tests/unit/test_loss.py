"""Tests for the cross-entropy loss."""

import numpy as np

from src.models.iris_nn import NeuralNetwork


def test_perfect_prediction_has_near_zero_loss():
    y_true = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    y_pred = y_true.copy()
    loss = NeuralNetwork.cross_entropy_loss(y_pred, y_true)
    # Should be effectively zero (clipped at 1 - 1e-15).
    assert loss < 1e-6


def test_random_prediction_loss_is_finite():
    rng = np.random.default_rng(0)
    y_pred = rng.dirichlet(alpha=[1, 1, 1], size=10).T
    y_true = np.eye(3)[rng.integers(0, 3, 10)].T
    loss = NeuralNetwork.cross_entropy_loss(y_pred, y_true)
    assert np.isfinite(loss)
    assert loss > 0


def test_clipping_prevents_log_zero():
    # If a probability is exactly 0, log(0) -> -inf. Clipping must prevent that.
    y_pred = np.array([[1.0, 0.0, 0.0]])
    y_true = np.array([[1.0, 0.0, 0.0]])
    loss = NeuralNetwork.cross_entropy_loss(y_pred, y_true)
    assert np.isfinite(loss)
