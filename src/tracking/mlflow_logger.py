"""Thin wrapper around MLflow for clean training-script integration.

Encapsulates the start/log/end lifecycle, env setup, and context-manager
behaviour so callers never touch ``mlflow.*`` directly. Falls back to a
no-op logger if MLflow is not installed.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class MlflowLogger:
    """High-level wrapper for parameter/metric/artifact tracking.

    If MLflow is not installed, all methods silently no-op so training
    scripts remain runnable without the dependency.
    """

    def __init__(
        self,
        experiment_name: str = "neural-network",
        tracking_uri: str | None = None,
        run_name: str | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        # MLflow 3.x requires a database backend; we default to SQLite at the
        # project root. Pass ``tracking_uri=None`` and the logger will use a
        # local SQLite DB next to ``mlruns/``.
        if tracking_uri is None:
            db_path = Path(__file__).resolve().parents[2] / "mlruns" / "mlflow.db"
            db_path.parent.mkdir(exist_ok=True)
            self.tracking_uri = f"sqlite:///{db_path}"
        else:
            self.tracking_uri = tracking_uri
        self.run_name = run_name
        self._active = False
        self._mlflow = self._import_mlflow()
        if self._mlflow is not None:
            self._mlflow.set_tracking_uri(self.tracking_uri)
            self._mlflow.set_experiment(experiment_name)

    @staticmethod
    def _import_mlflow():
        try:
            import mlflow

            return mlflow
        except ImportError:
            return None

    @property
    def available(self) -> bool:
        return self._mlflow is not None

    @contextmanager
    def start_run(self, run_name: str | None = None) -> Iterator[None]:
        """Context manager that opens and closes an MLflow run."""
        if not self.available:
            yield
            return

        name = run_name or self.run_name
        self._mlflow.start_run(run_name=name)
        self._active = True
        try:
            self._set_default_tags()
            yield
        finally:
            self._mlflow.end_run()
            self._active = False

    def _set_default_tags(self) -> None:
        if not self.available or not self._active:
            return
        tags: dict[str, str] = {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
        }
        try:
            tags["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        self._mlflow.set_tags(tags)

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.available or not self._active:
            return
        # MLflow only accepts str/bool/float/int.
        cleaned = {k: _coerce(v) for k, v in params.items()}
        self._mlflow.log_params(cleaned)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.available or not self._active:
            return
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        if not self.available or not self._active:
            return
        if os.path.exists(path):
            self._mlflow.log_artifact(path)

    def log_dict(self, data: dict[str, Any], filename: str) -> None:
        """Log a dict as a JSON artifact under ``filename``."""
        if not self.available or not self._active:
            return
        import json

        tmp_dir = Path(self.tracking_uri) / "_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = tmp_dir / filename
        tmp_file.write_text(json.dumps(data, indent=2, default=str))
        self.log_artifact(str(tmp_file))


def _coerce(value: Any) -> Any:
    """Coerce lists/dicts into strings for MLflow param logging."""
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


__all__ = ["MlflowLogger"]