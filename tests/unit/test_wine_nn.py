"""Tests for the WineNet PyTorch model."""

import torch

from src.models.wine_nn import WineNet


def test_default_architecture():
    model = WineNet()
    assert isinstance(model.network, torch.nn.Sequential)


def test_forward_shape_default(sample_wine_tensor):
    X, _ = sample_wine_tensor
    model = WineNet()
    out = model(X)
    assert out.shape == (32, 2)


def test_forward_shape_with_custom_hidden(sample_wine_tensor):
    X, _ = sample_wine_tensor
    model = WineNet(input_size=11, hidden_sizes=[128, 64, 32], output_size=2, dropout=0.1)
    out = model(X)
    assert out.shape == (32, 2)


def test_dropout_inactive_in_eval_mode(sample_wine_tensor):
    X, _ = sample_wine_tensor
    model = WineNet(hidden_sizes=[32, 16], dropout=0.9)
    model.eval()
    with torch.no_grad():
        out1 = model(X)
        out2 = model(X)
    torch.testing.assert_close(out1, out2)
