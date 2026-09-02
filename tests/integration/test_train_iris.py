"""Integration test for the Iris training pipeline (runs in seconds)."""

import pytest

from src.data.iris_dataset import load_iris_dataset
from src.models.iris_nn import NeuralNetwork
from src.utils import set_seed


@pytest.mark.integration
def test_iris_trains_to_high_accuracy():
    set_seed(42)
    data = load_iris_dataset(test_size=0.2, random_state=42)
    nn = NeuralNetwork(input_size=4, hidden_size=10, output_size=3, learning_rate=0.1)
    nn.train(data.X_train, data.Y_train, epochs=500, verbose=False, log_every=100)

    preds = nn.predict(data.X_test)
    accuracy = (preds == data.y_test).mean()
    assert accuracy >= 0.85, f"Expected >=0.85, got {accuracy:.4f}"
    assert nn.loss_history[-1] < nn.loss_history[0]
