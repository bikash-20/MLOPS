"""Tests for activation functions used in the Iris NumPy model."""

import numpy as np

from src.models.iris_nn import NeuralNetwork


class TestSoftmax:
    def test_outputs_sum_to_one(self):
        z = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 4.0]])
        out = NeuralNetwork.softmax(z)
        np.testing.assert_allclose(out.sum(axis=0), np.ones(3), rtol=1e-7)

    def test_numerical_stability_with_large_values(self):
        # Without max-subtraction, exp(1000) -> inf, divide gives NaN.
        z = np.array([[1000.0, 1000.0], [1000.0, 1000.0]])
        out = NeuralNetwork.softmax(z)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()
        np.testing.assert_allclose(out.sum(axis=0), np.ones(2), rtol=1e-7)

    def test_output_is_non_negative(self):
        z = np.array([[-5.0, 0.0, 1.0], [10.0, -10.0, 3.0]])
        out = NeuralNetwork.softmax(z)
        assert (out >= 0).all()


class TestReLU:
    def test_positive_passes_through(self):
        z = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(NeuralNetwork.relu(z), z)

    def test_negative_clipped_to_zero(self):
        z = np.array([[-1.0, -2.0], [-3.0, 0.5]])
        expected = np.array([[0.0, 0.0], [0.0, 0.5]])
        np.testing.assert_array_equal(NeuralNetwork.relu(z), expected)

    def test_derivative(self):
        z = np.array([[-1.0, 0.0, 1.0], [2.0, -3.0, 0.001]])
        expected = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
        np.testing.assert_array_equal(NeuralNetwork.relu_derivative(z), expected)
