"""Gradient shape and numerical-correctness tests."""

import numpy as np

from src.models.iris_nn import NeuralNetwork


def test_backward_returns_correct_shapes(sample_iris_data):
    X, Y = sample_iris_data
    nn = NeuralNetwork(input_size=4, hidden_size=8, output_size=3, learning_rate=0.1)
    z1, a1, z2, a2 = nn.forward(X)
    dW1, db1, dW2, db2 = nn.backward(X, Y, z1, a1, z2, a2)

    assert dW1.shape == nn.W1.shape
    assert db1.shape == nn.b1.shape
    assert dW2.shape == nn.W2.shape
    assert db2.shape == nn.b2.shape


def test_numerical_gradient_matches_analytic(sample_iris_data):
    # Tiny network so the numerical check is fast and reliable.
    X, Y = sample_iris_data
    nn = NeuralNetwork(input_size=4, hidden_size=3, output_size=3, learning_rate=0.01)
    z1, a1, z2, a2 = nn.forward(X)
    dW1, _db1, _dW2, _db2 = nn.backward(X, Y, z1, a1, z2, a2)

    eps = 1e-5
    # Spot-check one weight in W1.
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, nn.W1.shape[0]), rng.integers(0, nn.W1.shape[1])

    original = nn.W1[i, j]
    nn.W1[i, j] = original + eps
    _, _, _, a_plus = nn.forward(X)
    loss_plus = nn.cross_entropy_loss(a_plus, Y)

    nn.W1[i, j] = original - eps
    _, _, _, a_minus = nn.forward(X)
    loss_minus = nn.cross_entropy_loss(a_minus, Y)

    numerical = (loss_plus - loss_minus) / (2 * eps)
    analytical = dW1[i, j]
    nn.W1[i, j] = original

    # Relative error should be small.
    assert abs(numerical - analytical) < 1e-5
