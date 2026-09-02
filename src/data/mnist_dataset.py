"""MNIST dataset loader.

Uses torchvision to download and parse the canonical MNIST handwritten
digits dataset. Images are normalised to ``[0, 1]`` floats; labels are
``int64`` class indices 0-9.

The raw files live under ``data/raw/MNIST/`` (created on first download).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torchvision import datasets, transforms

from src.utils import get_logger, raw_data_path

logger = get_logger(__name__)


@dataclass(frozen=True)
class MnistData:
    """Container for MNIST train/test tensors."""

    X_train: torch.Tensor  # (60000, 1, 28, 28) float32 in [0, 1]
    y_train: torch.Tensor  # (60000,) int64
    X_test: torch.Tensor   # (10000, 1, 28, 28) float32 in [0, 1]
    y_test: torch.Tensor   # (10000,) int64
    class_names: tuple[str, ...]


def load_mnist_dataset(
    test_size: float = 0.0,  # MNIST already has a fixed 60k/10k split
    random_seed: int = 42,
) -> MnistData:
    """Download (if needed) and return MNIST train/test tensors.

    Args:
        test_size: Ignored for MNIST (uses canonical 60k/10k split).
        random_seed: Reserved for reproducibility hooks.

    Returns:
        ``MnistData`` with normalised images and integer labels.
    """
    del test_size, random_seed  # MNIST has fixed splits

    transform = transforms.Compose([transforms.ToTensor()])  # -> [0, 1] float
    data_root = raw_data_path("MNIST")
    os.makedirs(data_root, exist_ok=True)

    logger.info("Loading MNIST from %s", data_root)
    train_ds = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    # Stack into single tensors for DataLoader friendliness.
    X_train = torch.stack([img for img, _ in train_ds], dim=0)
    y_train = torch.tensor([label for _, label in train_ds], dtype=torch.int64)
    X_test = torch.stack([img for img, _ in test_ds], dim=0)
    y_test = torch.tensor([label for _, label in test_ds], dtype=torch.int64)

    class_names = tuple(str(i) for i in range(10))
    logger.info(
        "MNIST ready: train=%s test=%s classes=%s",
        tuple(X_train.shape), tuple(X_test.shape), class_names,
    )
    return MnistData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
    )


__all__ = ["MnistData", "load_mnist_dataset"]
