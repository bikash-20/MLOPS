"""Pretty-printing helpers for user-facing CLI output."""

from __future__ import annotations


def banner(title: str, char: str = "=", width: int = 60) -> None:
    """Print a centred title surrounded by ``char`` characters."""
    print(char * width)
    print(title.center(width))
    print(char * width)


__all__ = ["banner"]
