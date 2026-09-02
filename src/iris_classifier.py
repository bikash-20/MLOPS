"""Backward-compat shim — delegates to the new Hydra entrypoint.

This file exists so ``python src/iris_classifier.py`` from older docs still
works. New code should use ``python -m src.training.train_iris``.

.. deprecated::
    Use ``python -m src.training.train_iris`` or ``make train-iris`` instead.
"""

from __future__ import annotations

import sys
import warnings


def main() -> None:
    """Forward to the new Iris training entrypoint."""
    warnings.warn(
        "src/iris_classifier.py is deprecated. "
        "Use `python -m src.training.train_iris` (or `make train-iris`) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    from src.training.train_iris import main as new_main

    new_main()


if __name__ == "__main__":
    sys.exit(main() or 0)
