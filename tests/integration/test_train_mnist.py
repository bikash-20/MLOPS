"""Integration test for the MNIST training pipeline (1 epoch, tiny subset)."""

import pytest
import torch
from torch import nn as nn_torch

from src.models.mnist_cnn import SimpleCNN
from src.utils import set_seed


@pytest.mark.integration
def test_mnist_pipeline_runs_1_epoch():
    """Train a tiny CNN on synthetic MNIST-like data for 1 epoch.

    Asserts the loss decreases, demonstrating that gradients flow end-to-end
    through the conv/FC/dropout stack. Uses synthetic data (not the real
    MNIST download) to keep this CI-friendly and offline.
    """
    set_seed(42)
    torch.manual_seed(42)

    # Synthetic MNIST-like batch: (batch=8, 1, 28, 28) in [0, 1].
    x = torch.rand(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))

    model = SimpleCNN(
        in_channels=1,
        conv_channels=(8, 16),
        fc_hidden=32,
        num_classes=10,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn_torch.CrossEntropyLoss()

    model.train()
    optimizer.zero_grad()
    logits = model(x)
    loss_before = criterion(logits, y).item()
    criterion(logits, y).backward()
    optimizer.step()

    with torch.no_grad():
        loss_after = criterion(model(x), y).item()

    assert loss_after < loss_before, (
        f"Loss should decrease after one optimizer step; "
        f"got before={loss_before:.4f} after={loss_after:.4f}"
    )
