"""CIFAR-10 dataset loader and preprocessor.

Uses torchvision to download and parse the canonical CIFAR-10 image
classification dataset. Returns normalised tensors (mean / std of the
training split) split into train / val / test partitions.

The raw files live under ``data/raw/cifar-10-batches-py`` (created on
first download). The torchvision normalization constants are the standard
ones used for CIFAR-10:

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

Splitting strategy:
- Train: 45,000 images (90% of the 50k official train split).
- Val:    5,000 images (10% of the 50k official train split, stratified).
- Test:  10,000 images (the official test split).

The split is stratified on the integer label so each partition has the
same class distribution (each class has 4,500 / 500 / 1,000 samples).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

from src.utils import get_logger, raw_data_path

logger = get_logger(__name__)

# Canonical CIFAR-10 normalization. Computed on the 50k train split.
CIFAR_MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR_STD: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)
CIFAR_CLASSES: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass(frozen=True)
class CifarData:
    """Container for CIFAR-10 train/val/test tensors.

    All tensors are float32 in roughly ``[-2.1, 2.6]`` after normalization.
    Image layout is channels-first ``(N, 3, 32, 32)`` to match
    ``torchvision`` conventions and the ``SimpleCNN``/``CifarResNet`` models.
    """

    X_train: torch.Tensor  # (45000, 3, 32, 32)
    y_train: torch.Tensor  # (45000,) int64
    X_val: torch.Tensor    # (5000, 3, 32, 32)
    y_val: torch.Tensor    # (5000,) int64
    X_test: torch.Tensor   # (10000, 3, 32, 32)
    y_test: torch.Tensor   # (10000,) int64
    class_names: tuple[str, ...]


def _to_tensor_normalize() -> transforms.Compose:
    """No-augmentation transform: ToTensor + channel-wise normalisation."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def _stack_split(ds: datasets.CIFAR10) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialise a torchvision dataset into a single (X, y) tensor pair."""
    X = torch.stack([img for img, _ in ds], dim=0)
    y = torch.tensor([label for _, label in ds], dtype=torch.int64)
    return X, y


def load_cifar_dataset(
    val_size: float = 0.1,
    random_seed: int = 42,
) -> CifarData:
    """Download (if needed) and return CIFAR-10 train/val/test tensors.

    Args:
        val_size: Fraction of the official 50k train split to reserve for
            validation. The remaining ``1 - val_size`` becomes the training
            set. The official 10k test split is returned untouched.
        random_seed: Seed for the stratified train/val split.

    Returns:
        ``CifarData`` with normalized tensors and class names.
    """
    transform = _to_tensor_normalize()
    data_root = raw_data_path("cifar-10-batches-py")
    os.makedirs(data_root, exist_ok=True)

    logger.info("Loading CIFAR-10 from %s", data_root)
    train_ds = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform,
    )
    test_ds = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform,
    )

    # Materialise into tensors, then split into train/val.
    X_full, y_full = _stack_split(train_ds)
    X_test, y_test = _stack_split(test_ds)

    idx_train, idx_val = train_test_split(
        np.arange(len(y_full)),
        test_size=val_size,
        random_state=random_seed,
        stratify=y_full.numpy(),
    )
    X_train = X_full[idx_train]
    y_train = y_full[idx_train]
    X_val = X_full[idx_val]
    y_val = y_full[idx_val]

    logger.info(
        "CIFAR-10 ready: train=%s val=%s test=%s classes=%d",
        tuple(X_train.shape), tuple(X_val.shape), tuple(X_test.shape),
        len(CIFAR_CLASSES),
    )
    return CifarData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        class_names=CIFAR_CLASSES,
    )


__all__ = [
    "CIFAR_CLASSES",
    "CIFAR_MEAN",
    "CIFAR_STD",
    "CifarData",
    "load_cifar_dataset",
]
