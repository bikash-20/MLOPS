"""Integration test for the Wine training pipeline (5 epochs only)."""

import pytest
import torch
from torch import nn as nn_torch

from src.data.wine_dataset import load_wine_dataset
from src.models.wine_nn import WineNet
from src.utils import set_seed


@pytest.mark.integration
def test_wine_pipeline_runs_5_epochs(sample_wine_tensor=None):
    """Train a tiny model on a small subset for 5 epochs to verify pipeline."""
    set_seed(42)
    data = load_wine_dataset(binary=True, test_size=0.5, random_state=42)

    # Subsample to 64 train rows for speed.
    X_train = data.X_train[:64]
    y_train = data.y_train[:64]

    model = WineNet(input_size=11, hidden_sizes=[8, 4], output_size=2, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn_torch.CrossEntropyLoss()

    initial_loss = None
    for epoch in range(5):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        if epoch == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < initial_loss, "Loss should decrease across 5 epochs"
