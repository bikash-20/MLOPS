"""Iris dataset loader and preprocessor.

Loads the classic Iris dataset from scikit-learn, applies standardization,
and returns one-hot encoded labels in NumPy ``(features, samples)`` layout
to match the convention used by ``NeuralNetwork``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_iris as _load_iris_sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class IrisData:
    """Container for split Iris arrays in ``(features, N)`` layout."""

    X_train: np.ndarray
    X_test: np.ndarray
    Y_train: np.ndarray  # one-hot, shape (3, N_train)
    Y_test: np.ndarray   # one-hot, shape (3, N_test)
    y_train: np.ndarray  # integer labels, shape (N_train,)
    y_test: np.ndarray   # integer labels, shape (N_test,)
    class_names: tuple[str, ...]
    scaler: StandardScaler


def load_iris_dataset(
    test_size: float = 0.2,
    random_state: int = 42,
) -> IrisData:
    """Load, split, and scale the Iris dataset.

    Args:
        test_size: Fraction of data reserved for the test set.
        random_state: Seed for the train/test split.

    Returns:
        ``IrisData`` with train/test arrays in ``(features, N)`` layout.
    """
    iris = _load_iris_sklearn()
    X = iris.data.astype(np.float64)        # (150, 4)
    y = iris.target.astype(np.int64)        # (150,)
    class_names = tuple(iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # One-hot encode: y=0 -> [1,0,0], y=1 -> [0,1,0], y=2 -> [0,0,1].
    Y_train = np.eye(3)[y_train].T  # (3, N_train)
    Y_test = np.eye(3)[y_test].T    # (3, N_test)

    # Transpose to (features, N) layout.
    X_train = X_train.T
    X_test = X_test.T

    logger.info(
        "Iris dataset ready: train=%d test=%d features=%d classes=%s",
        X_train.shape[1], X_test.shape[1], X_train.shape[0], class_names,
    )
    return IrisData(
        X_train=X_train,
        X_test=X_test,
        Y_train=Y_train,
        Y_test=Y_test,
        y_train=y_train,
        y_test=y_test,
        class_names=class_names,
        scaler=scaler,
    )


__all__ = ["IrisData", "load_iris_dataset"]