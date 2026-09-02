"""Wine Quality dataset loader and preprocessor.

Downloads the UCI white-wine CSV if not present locally, applies binary
binarisation (good >= 7 vs not good), standardised scaling, and returns
PyTorch tensors ready for the model.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import get_logger, raw_data_path

logger = get_logger(__name__)

UCI_WHITE_WINE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-white.csv"
)


@dataclass(frozen=True)
class WineData:
    """Container for split Wine Quality arrays/tensors."""

    X_train: torch.Tensor
    X_test: torch.Tensor
    y_train: torch.Tensor
    y_test: torch.Tensor
    feature_names: tuple[str, ...]
    scaler: StandardScaler
    class_distribution: tuple[int, int]


def _download_wine_csv(target_path: str) -> None:
    """Download the UCI white-wine CSV to ``target_path`` if missing."""
    if os.path.exists(target_path):
        return
    logger.info("Downloading wine quality dataset from %s", UCI_WHITE_WINE_URL)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    urllib.request.urlretrieve(UCI_WHITE_WINE_URL, target_path)
    logger.info("Saved to %s", target_path)


def load_wine_dataset(
    binary: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    good_threshold: int = 7,
) -> WineData:
    """Load, split, and scale the Wine Quality dataset.

    Args:
        binary: If True, binarise target into "good" vs "not good".
        test_size: Fraction of data reserved for the test set.
        random_state: Seed for the train/test split.
        good_threshold: Quality score at or above which wine is "good".

    Returns:
        ``WineData`` containing train/test ``FloatTensor``/``LongTensor`` pairs.
    """
    filepath = raw_data_path("winequality-white.csv")
    _download_wine_csv(filepath)

    df = pd.read_csv(filepath, sep=";")
    logger.info("Wine dataset shape: %s, columns: %s", df.shape, df.columns.tolist())

    X = df.drop("quality", axis=1).values.astype(np.float64)
    y = df["quality"].values.astype(np.int64)
    feature_names = tuple(c for c in df.columns if c != "quality")

    if binary:
        y = (y >= good_threshold).astype(np.int64)
        distribution = tuple(int(v) for v in np.bincount(y))
        logger.info("Binary classification (>=%d): distribution=%s", good_threshold, distribution)
    else:
        distribution = tuple(int(v) for v in np.bincount(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info(
        "Wine dataset prepared: train=%d test=%d features=%d",
        X_train.shape[0], X_test.shape[0], X_train.shape[1],
    )

    return WineData(
        X_train=torch.from_numpy(X_train).float(),
        X_test=torch.from_numpy(X_test).float(),
        y_train=torch.from_numpy(y_train).long(),
        y_test=torch.from_numpy(y_test).long(),
        feature_names=feature_names,
        scaler=scaler,
        class_distribution=distribution,
    )


__all__ = ["WineData", "load_wine_dataset"]