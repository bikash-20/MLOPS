"""Structured logging configuration.

Provides a single ``configure_logging`` entry point used by training
scripts and the API service. Supports JSON output for production via
``python-json-logger`` when available.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(
    name: str = "neural_network",
    level: str = "INFO",
    json_output: bool = False,
) -> logging.Logger:
    """Configure and return a logger.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        json_output: If True, attempt to emit JSON-formatted log records.
            Falls back to a plain formatter if ``python-json-logger`` is
            not installed.

    Returns:
        Configured stdlib ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Idempotent: don't double-add handlers if called twice.
        return logger

    logger.setLevel(level.upper())
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)

    if json_output:
        try:
            from pythonjsonlogger import jsonlogger

            formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s"
            )
        except ImportError:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Convenience wrapper that configures and returns a logger.

    Args:
        name: Logger name (use ``__name__`` from the caller).
        level: Log level string.

    Returns:
        Configured logger.
    """
    return configure_logging(name=name, level=level)


__all__ = ["configure_logging", "get_logger"]
