"""Backward-compat shim — delegates to the new Hydra entrypoint.

This file exists so ``python src/wine_quality.py`` from older docs still
works. New code should use ``python -m src.training.train_wine``.

.. deprecated::
    Use ``python -m src.training.train_wine`` or ``make train-wine`` instead.
"""

from __future__ import annotations

import sys
import warnings


def main() -> None:
    """Forward to the new Wine training entrypoint."""
    warnings.warn(
        "src/wine_quality.py is deprecated. "
        "Use `python -m src.training.train_wine` (or `make train-wine`) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    from src.training.train_wine import main as new_main

    new_main()


if __name__ == "__main__":
    sys.exit(main() or 0)
