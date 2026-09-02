"""Simple 3-layer CNN for MNIST digit classification.

Architecture:
    Conv2d(1, 32, 3) -> ReLU -> MaxPool(2)
    Conv2d(32, 64, 3) -> ReLU -> MaxPool(2)
    Flatten -> Linear(64*5*5, 128) -> ReLU -> Dropout -> Linear(128, 10)

Total parameters: ~104k. Reaches >98% test accuracy on MNIST in ~5 epochs.
"""

from __future__ import annotations

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """3-layer CNN for 28x28 grayscale digit classification."""

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: tuple[int, ...] = (32, 64),
        fc_hidden: int = 128,
        num_classes: int = 10,
        dropout: float = 0.25,
    ) -> None:
        """Initialize layers.

        Args:
            in_channels: Number of input channels (1 for grayscale).
            conv_channels: Output channels of the conv layers.
            fc_hidden: Width of the FC hidden layer.
            num_classes: Number of output classes (10 for MNIST).
            dropout: Dropout probability after the FC hidden layer.
        """
        super().__init__()
        c1, c2 = conv_channels
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # After two 2x2 pools, 28x28 -> 7x7.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Args:
            x: Input batch, shape ``(batch, in_channels, 28, 28)``.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return self.classifier(x)


__all__ = ["SimpleCNN"]
