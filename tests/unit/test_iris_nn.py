"""Tests for the NeuralNetwork class shape and forward-pass behaviour."""

import numpy as np

from src.models.iris_nn import NeuralNetwork


def test_default_construction_shapes():
    nn = NeuralNetwork()
    assert nn.W1.shape == (10, 4)
    assert nn.b1.shape == (10, 1)
    assert nn.W2.shape == (3, 10)
    assert nn.b2.shape == (3, 1)


def test_forward_returns_expected_shapes(sample_iris_data):
    X, _ = sample_iris_data
    nn = NeuralNetwork(input_size=4, hidden_size=10, output_size=3)
    z1, a1, z2, a2 = nn.forward(X)
    assert z1.shape == (10, X.shape[1])
    assert a1.shape == z1.shape
    assert z2.shape == (3, X.shape[1])
    assert a2.shape == z2.shape


def test_predict_proba_sums_to_one(sample_iris_data):
    X, _ = sample_iris_data
    nn = NeuralNetwork(input_size=4, hidden_size=10, output_size=3)
    probs = nn.predict_proba(X)
    np.testing.assert_allclose(probs.sum(axis=0), np.ones(X.shape[1]), rtol=1e-7)


def test_predict_returns_indices_in_range(sample_iris_data):
    X, _ = sample_iris_data
    nn = NeuralNetwork(input_size=4, hidden_size=10, output_size=3)
    preds = nn.predict(X)
    assert preds.shape == (X.shape[1],)
    assert set(preds.tolist()).issubset({0, 1, 2})


def test_training_decreases_loss(sample_iris_data):
    X, Y = sample_iris_data
    nn = NeuralNetwork(input_size=4, hidden_size=10, output_size=3, learning_rate=0.1)
    initial_loss = nn.cross_entropy_loss(nn.forward(X)[3], Y)
    nn.train(X, Y, epochs=300, verbose=False, log_every=100)
    assert nn.loss_history[-1] < initial_loss
