"""Unit tests for the CIFAR-10 ResNet model.

Covers:
- Forward-pass output shape on the canonical 32x32 RGB input.
- Parameter-count sanity (CIFAR ResNet-18 is ~11M params).
- Stage channel widths and spatial downsampling pattern.
- Gradient flows back to every parameter (no frozen / disconnected layers).
- ``BasicBlock`` shape contract (preserves or halves spatial dims).
- Configurable ``base_channels`` and ``num_blocks_per_stage``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.cifar_resnet import BasicBlock, CifarResNet


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --- CifarResNet -----------------------------------------------------------


def test_cifar_resnet_default_forward_shape():
    """A batch of 4 images should yield logits of shape (4, 10)."""
    model = CifarResNet()
    model.eval()
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 10)


def test_cifar_resnet_default_param_count_in_range():
    """Default config should be in the 4M-7M param range (~5M in practice)."""
    model = CifarResNet()
    n = _count_params(model)
    # Allow generous tolerance; this is a shape-of-network sanity check,
    # not a regression pin. ``base_channels=64`` with 2 blocks per stage
    # gives ~5.2M params (not the 11M of an ImageNet ResNet-18, because
    # we use 64-128-256-256 widths instead of 64-128-256-512).
    assert 4_000_000 < n < 7_000_000, f"unexpected param count: {n}"


def test_cifar_resnet_channel_widths_double_per_stage():
    """The 2nd stage should double channels vs the 1st, 3rd vs 2nd, 4th stays."""
    model = CifarResNet(base_channels=64, num_blocks_per_stage=2)
    # Inspect actual layer widths by running a forward pass through each stage.
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        s = model.stem(x)
        assert s.shape[1] == 64
        s1 = model.stage1(s)
        assert s1.shape[1] == 64 and s1.shape[-1] == 32
        s2 = model.stage2(s1)
        assert s2.shape[1] == 128 and s2.shape[-1] == 16
        s3 = model.stage3(s2)
        assert s3.shape[1] == 256 and s3.shape[-1] == 8
        s4 = model.stage4(s3)
        assert s4.shape[1] == 256 and s4.shape[-1] == 4


def test_cifar_resnet_gradient_flows_to_all_params():
    """A backward pass should populate .grad on every trainable parameter."""
    model = CifarResNet()
    model.train()
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([0, 1])
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    params_without_grad = [
        name for name, p in model.named_parameters()
        if p.grad is None or torch.all(p.grad == 0)
    ]
    assert not params_without_grad, f"params without gradient: {params_without_grad}"


def test_cifar_resnet_dropout_zero_changes_param_count_minimally():
    """Dropout layers don't add trainable params; counts should match closely."""
    m1 = CifarResNet(dropout=0.0)
    m2 = CifarResNet(dropout=0.5)
    # No additional parameters; allow tiny BN-related differences.
    assert abs(_count_params(m1) - _count_params(m2)) < 1000


def test_cifar_resnet_supports_different_num_classes():
    """num_classes should change the output dimension."""
    model = CifarResNet(num_classes=100)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 100)


def test_cifar_resnet_supports_different_in_channels():
    """in_channels should change the stem input dimension."""
    model = CifarResNet(in_channels=1)
    model.eval()
    x = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10)


def test_cifar_resnet_smaller_channels_reduces_param_count():
    """base_channels=16 should give a noticeably smaller model."""
    big = CifarResNet(base_channels=64)
    small = CifarResNet(base_channels=16)
    assert _count_params(small) < _count_params(big) / 3


def test_cifar_resnet_more_blocks_per_stage_increases_params():
    """num_blocks_per_stage=3 should be larger than 2."""
    m2 = CifarResNet(num_blocks_per_stage=2)
    m3 = CifarResNet(num_blocks_per_stage=3)
    assert _count_params(m3) > _count_params(m2)


# --- BasicBlock -----------------------------------------------------------


def test_basic_block_identity_shape():
    """stride=1 + same channels should preserve shape exactly."""
    block = BasicBlock(64, 64, stride=1)
    block.eval()
    x = torch.randn(2, 64, 16, 16)
    with torch.no_grad():
        out = block(x)
    assert out.shape == x.shape


def test_basic_block_downsample_halves_spatial():
    """stride=2 should halve spatial dims and change channels."""
    block = BasicBlock(32, 64, stride=2)
    block.eval()
    x = torch.randn(2, 32, 16, 16)
    with torch.no_grad():
        out = block(x)
    assert out.shape == (2, 64, 8, 8)


def test_basic_block_channel_change_uses_projection_shortcut():
    """When channels differ, shortcut must use a 1x1 conv, not identity."""
    block = BasicBlock(32, 64, stride=1)
    assert not isinstance(block.shortcut, nn.Identity)
    # The projection should be a 1x1 conv.
    convs = [m for m in block.shortcut.modules() if isinstance(m, nn.Conv2d)]
    assert any(c.kernel_size == (1, 1) for c in convs)


@pytest.mark.parametrize("in_c,out_c,stride,hw", [
    (3, 64, 1, 32),
    (64, 64, 1, 32),
    (64, 128, 2, 32),
    (128, 128, 1, 16),
    (128, 256, 2, 16),
    (256, 256, 1, 8),
    (256, 256, 2, 8),
])
def test_basic_block_shape_contract(in_c: int, out_c: int, stride: int, hw: int):
    """Parametrized shape contract for the residual block."""
    block = BasicBlock(in_c, out_c, stride=stride)
    block.eval()
    x = torch.randn(1, in_c, hw, hw)
    with torch.no_grad():
        out = block(x)
    expected_hw = hw // stride
    assert out.shape == (1, out_c, expected_hw, expected_hw)
