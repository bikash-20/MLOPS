"""
Utility helpers shared by the neural-network projects.

These functions are deliberately tiny and dependency-light so they can be
imported from either ``iris_classifier.py`` (NumPy-only) or
``wine_quality.py`` (PyTorch) without pulling in extra packages.
"""

from __future__ import annotations

import os
import random

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Project root is one directory up from this file (…/neural-network/).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def ensure_dir(path: str) -> str:
    """Create ``path`` (including parents) if it does not exist and return it."""
    os.makedirs(path, exist_ok=True)
    return path


def models_path(filename: str) -> str:
    """Absolute path inside the project's ``models/`` folder."""
    return os.path.join(ensure_dir(MODELS_DIR), filename)


def plots_path(filename: str) -> str:
    """Absolute path inside the project's ``plots/`` folder."""
    return os.path.join(ensure_dir(PLOTS_DIR), filename)


def data_path(filename: str) -> str:
    """Absolute path inside the project's ``data/`` folder."""
    return os.path.join(ensure_dir(DATA_DIR), filename)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy (and PyTorch if available) for reproducible runs."""
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


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def banner(title: str, char: str = "=", width: int = 60) -> None:
    """Print a centred title surrounded by ``char`` characters."""
    print(char * width)
    print(title.center(width))
    print(char * width)


__all__ = [
    "PROJECT_ROOT",
    "MODELS_DIR",
    "PLOTS_DIR",
    "DATA_DIR",
    "ensure_dir",
    "models_path",
    "plots_path",
    "data_path",
    "set_seed",
    "banner",
]
