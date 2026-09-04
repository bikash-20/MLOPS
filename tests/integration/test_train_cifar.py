"""Integration test for the CIFAR training pipeline (1 epoch, synthetic data).

This mirrors the MNIST integration test: rather than downloading the real
CIFAR-10 dataset (which adds 170MB and slow CI), we feed a synthetic
``(batch=8, 3, 32, 32)`` batch through the ``CifarResNet`` and assert
that one optimizer step reduces the loss.

Covers:
- Forward pass through the stem + 4 stages + GAP + linear head.
- Backward gradients flow end-to-end.
- Loss decreases after one SGD step (basic sanity, not convergence).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn as nn_torch

from src.models.cifar_resnet import CifarResNet
from src.utils import set_seed


@pytest.mark.integration
def test_cifar_pipeline_runs_1_epoch():
    """Train a tiny CIFAR ResNet on synthetic RGB data for 1 step."""
    set_seed(42)
    torch.manual_seed(42)

    # Synthetic CIFAR-like batch: (batch=8, 3, 32, 32) in [0, 1].
    x = torch.rand(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))

    # Use a small variant to keep the test fast.
    model = CifarResNet(
        in_channels=3,
        num_classes=10,
        base_channels=16,
        num_blocks_per_stage=2,
        dropout=0.0,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn_torch.CrossEntropyLoss()

    model.train()
    optimizer.zero_grad()
    logits = model(x)
    assert logits.shape == (8, 10)

    loss_before = criterion(logits, y).item()
    criterion(logits, y).backward()
    optimizer.step()

    with torch.no_grad():
        loss_after = criterion(model(x), y).item()

    assert loss_after < loss_before, (
        f"Loss should decrease after one optimizer step; "
        f"got before={loss_before:.4f} after={loss_after:.4f}"
    )


@pytest.mark.integration
def test_cifar_resnet_onecycle_scheduler_steps():
    """Verify OneCycleLR with the canonical CIFAR config doesn't error."""
    set_seed(0)
    model = CifarResNet(base_channels=16, num_blocks_per_stage=2, dropout=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        epochs=2,
        steps_per_epoch=4,  # batch is 8, so 1 step = 8/8
    )
    x = torch.rand(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    criterion = nn_torch.CrossEntropyLoss()

    initial_lr = optimizer.param_groups[0]["lr"]
    for _ in range(4):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        scheduler.step()

    final_lr = optimizer.param_groups[0]["lr"]
    assert final_lr != initial_lr, "OneCycle should change the LR each step"
