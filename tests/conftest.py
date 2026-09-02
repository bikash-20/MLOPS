"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path for ``src`` imports when pytest is invoked
# from anywhere in the repo.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_iris_data():
    """Tiny Iris-like dataset for fast unit tests."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((4, 30))
    Y = np.eye(3)[rng.integers(0, 3, size=30)].T
    return X, Y


@pytest.fixture
def sample_wine_tensor():
    """Tiny wine-like tensor for fast PyTorch tests."""
    import torch

    rng = np.random.default_rng(42)
    X = rng.standard_normal((32, 11)).astype(np.float32)
    y = rng.integers(0, 2, size=32).astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


@pytest.fixture
def valid_wine_payload() -> dict:
    """Realistic but out-of-distribution wine payload for API tests."""
    return {
        "fixed_acidity": 7.0,
        "volatile_acidity": 0.27,
        "citric_acid": 0.36,
        "residual_sugar": 20.7,
        "chlorides": 0.045,
        "free_sulfur_dioxide": 45.0,
        "total_sulfur_dioxide": 170.0,
        "density": 1.001,
        "ph": 3.0,
        "sulphates": 0.45,
        "alcohol": 8.8,
    }
