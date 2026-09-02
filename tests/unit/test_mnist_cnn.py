"""Tests for the SimpleCNN PyTorch model."""

import torch

from src.models.mnist_cnn import SimpleCNN


def test_default_architecture_is_sequential():
    model = SimpleCNN()
    assert isinstance(model.conv_block1, torch.nn.Sequential)
    assert isinstance(model.conv_block2, torch.nn.Sequential)
    assert isinstance(model.classifier, torch.nn.Sequential)


def test_forward_shape_default():
    """Default config: 2-pool stack on 28x28 -> 7x7 -> 64*7*7 -> 128 -> 10."""
    model = SimpleCNN()
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)


def test_forward_shape_batch_one():
    """Single-sample inference still works (no batch-size assumptions)."""
    model = SimpleCNN()
    x = torch.randn(1, 1, 28, 28)
    out = model(x)
    assert out.shape == (1, 10)


def test_param_count_within_expected_range():
    """The default model has ~225k trainable params (well under 1M)."""
    model = SimpleCNN()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 100_000 < n_params < 500_000


def test_dropout_inactive_in_eval_mode():
    """In ``eval()`` mode, dropout is disabled, so two forward passes match."""
    torch.manual_seed(0)
    model = SimpleCNN(dropout=0.9)
    model.eval()
    x = torch.randn(8, 1, 28, 28)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    torch.testing.assert_close(out1, out2)


def test_custom_channels_propagate_through_classifier():
    """Custom conv output channels widen the FC input correctly."""
    model = SimpleCNN(in_channels=1, conv_channels=(16, 32), fc_hidden=64, num_classes=5)
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    assert out.shape == (2, 5)
    # First Linear layer should expect 32 * 7 * 7 inputs.
    first_linear = model.classifier[1]
    assert isinstance(first_linear, torch.nn.Linear)
    assert first_linear.in_features == 32 * 7 * 7
    assert first_linear.out_features == 64
