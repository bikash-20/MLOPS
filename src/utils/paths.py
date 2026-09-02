"""Filesystem path helpers for the neural-network project.

All paths are absolute and resolved from the project root, which is determined
by the location of ``src/utils/paths.py`` (one level up = ``neural-network/``).
"""

from __future__ import annotations

import os

# Project root is one directory up from src/.
PROJECT_ROOT: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
MODELS_DIR: str = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR: str = os.path.join(PROJECT_ROOT, "plots")
DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR: str = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, "processed")
EXTERNAL_DATA_DIR: str = os.path.join(DATA_DIR, "external")
REPORTS_DIR: str = os.path.join(PROJECT_ROOT, "reports")
MLRUNS_DIR: str = os.path.join(PROJECT_ROOT, "mlruns")


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
    """Absolute path inside the project's ``data/`` folder (legacy alias)."""
    return os.path.join(ensure_dir(DATA_DIR), filename)


def raw_data_path(filename: str) -> str:
    """Absolute path inside ``data/raw/`` (immutable original data)."""
    return os.path.join(ensure_dir(RAW_DATA_DIR), filename)


def processed_data_path(filename: str) -> str:
    """Absolute path inside ``data/processed/`` (transformed data, scalers)."""
    return os.path.join(ensure_dir(PROCESSED_DATA_DIR), filename)


def reports_path(filename: str) -> str:
    """Absolute path inside the ``reports/`` folder."""
    return os.path.join(ensure_dir(REPORTS_DIR), filename)


__all__ = [
    "DATA_DIR",
    "EXTERNAL_DATA_DIR",
    "MLRUNS_DIR",
    "MODELS_DIR",
    "PLOTS_DIR",
    "PROCESSED_DATA_DIR",
    "PROJECT_ROOT",
    "RAW_DATA_DIR",
    "REPORTS_DIR",
    "data_path",
    "ensure_dir",
    "models_path",
    "plots_path",
    "processed_data_path",
    "raw_data_path",
    "reports_path",
]
