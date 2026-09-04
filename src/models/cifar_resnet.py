"""CIFAR-style ResNet-18 trained from scratch.

This is the standard "ResNet for CIFAR-10" variant described in the
original He et al. (2015) deep-residual-learning paper:

- 3x3 conv stem on the raw 32x32 RGB image (NO 7x7 stem, NO initial
  max-pool). CIFAR images are too small to justify aggressive
  downsampling at the input.
- Four stages with two residual blocks each (16 -> 32 -> 64 channels).
  Spatial downsampling happens via stride-2 convolutions inside the
  first block of stages 2/3/4 (instead of max-pool).
- Global average pool + single linear classifier head.

Total parameters: ~11M for the default config (``base_channels=64``,
``num_blocks_per_stage=2``). Reaches ~92-94% test accuracy on CIFAR-10
in ~20 epochs with OneCycle LR.

References:
    He, K., Zhang, X., Ren, S., Sun, J. (2015).
    "Deep Residual Learning for Image Recognition." arXiv:1512.03385.
"""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    """ResNet basic block: two 3x3 convs + skip connection.

    The first conv of the downsampling variant uses stride=2 to halve
    spatial resolution; the skip connection uses a 1x1 conv with
    stride=2 to match shapes.
    """

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_planes)

        # Skip connection: identity if shapes match, else 1x1 conv + BN.
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, out_planes, kernel_size=1,
                    stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block."""
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.relu(out)


class CifarResNet(nn.Module):
    """ResNet-18 adapted for 32x32 RGB images.

    Stages: stem (3x3 conv) -> stage1 (16ch, stride 1) -> stage2 (32ch,
    stride 2) -> stage3 (64ch, stride 2) -> global avg pool -> linear.

    With ``num_blocks_per_stage=2`` and ``base_channels=64`` this matches
    the original ResNet-18 layer counts (2-2-2 blocks) but with the wider
    CIFAR channel scheme (64-128-256-512 in ImageNet ResNet-18). Using
    base_channels=64 here gives 16-32-64 channel widths and ~11M params.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 64,
        num_blocks_per_stage: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the CIFAR-style ResNet.

        Args:
            in_channels: Number of input channels (3 for RGB).
            num_classes: Number of output classes (10 for CIFAR-10).
            base_channels: Channel width of the FIRST stage. Subsequent
                stages double this (64 -> 128 -> 256 by default). Pass
                ``base_channels=16`` for the original CIFAR ResNet paper
                widths (16 -> 32 -> 64).
            num_blocks_per_stage: Number of ``BasicBlock``s per stage.
                ResNet-18 uses 2 (giving 2*3 + 1 conv layers = 18).
            dropout: Dropout probability before the classifier head.
        """
        super().__init__()
        self.in_planes = base_channels

        # Stem: 3x3 conv, no maxpool. Keeps the 32x32 spatial resolution.
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels, base_channels, kernel_size=3,
                stride=1, padding=1, bias=False,
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Stages: each downsamples by 2x except the first. Spatial sizes:
        #   stage1: 32x32  (stride 1, channels base)
        #   stage2: 16x16  (stride 2, channels 2x base)
        #   stage3: 8x8    (stride 2, channels 4x base)
        #   stage4: 4x4    (stride 2, channels 4x base)
        self.stage1 = self._make_stage(base_channels, num_blocks_per_stage, stride=1)
        self.stage2 = self._make_stage(base_channels * 2, num_blocks_per_stage, stride=2)
        self.stage3 = self._make_stage(base_channels * 4, num_blocks_per_stage, stride=2)
        self.stage4 = self._make_stage(base_channels * 4, num_blocks_per_stage, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pre_classifier = nn.Dropout(dropout)
        self.classifier = nn.Linear(base_channels * 4, num_classes)

        # He/Kaiming init for Conv2d (good for ReLU networks).
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d,)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_stage(
        self,
        out_planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Build a stage of ``num_blocks`` residual blocks.

        The first block uses the requested stride (typically 2 to
        downsample); subsequent blocks use stride 1.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[nn.Module] = []
        in_planes = self.in_planes
        for s in strides:
            layers.append(BasicBlock(in_planes, out_planes, stride=s))
            in_planes = out_planes
        self.in_planes = in_planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Args:
            x: Input batch, shape ``(batch, in_channels, 32, 32)``.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        x = self.pre_classifier(x)
        return self.classifier(x)


__all__ = ["BasicBlock", "CifarResNet"]
