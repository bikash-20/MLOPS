"""Workaround for Hydra 1.3.x incompatibility with Python 3.14 argparse.

Hydra 1.3.x passes a non-string ``help`` object (``LazyCompletionHelp``)
to ``argparse.ArgumentParser.add_argument``. Python 3.14's argparse
validates help strings and raises ``ValueError: badly formed help string``.

Strategy: monkey-patch ``argparse._ActionsContainer.add_argument`` so that
any non-string ``help`` value is coerced to its ``str()`` form BEFORE
argparse's strict validation. This is a minimal, robust fix that survives
Hydra's internal imports and decorator application.

Import this BEFORE ``@hydra.main`` is invoked.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


def _patch_for_py314() -> None:
    if sys.version_info < (3, 14):
        return

    original_add_argument = argparse._ActionsContainer.add_argument

    def patched_add_argument(self, *args: Any, **kwargs: Any):
        # If ``help`` is not a string, coerce it.
        if "help" in kwargs and not isinstance(kwargs["help"], str):
            kwargs["help"] = str(kwargs["help"])
        # Also check positional help if provided (rare).
        return original_add_argument(self, *args, **kwargs)

    argparse._ActionsContainer.add_argument = patched_add_argument


_patch_for_py314()
