"""Deterministic seeding for reproducible runs across Python, NumPy, and PyTorch."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy (and PyTorch if available) for reproducible runs.

    Args:
        seed: Integer seed value applied across all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # PyTorch is optional for the NumPy-only Iris project.
        pass


__all__ = ["set_seed"]
