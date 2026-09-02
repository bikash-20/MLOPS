"""Lazy, cached loader for the wine-quality model and its preprocessing.

Loads ``model.pth``, ``scaler.joblib``, and ``feature_names.json`` from a
versioned registry directory. Thread-safe singleton via module-level state.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from src.models.wine_nn import WineNet
from src.utils import get_logger, models_path

logger = get_logger(__name__)


class ModelRegistry:
    """Loads and serves a single versioned wine-quality model."""

    def __init__(
        self,
        project: str = "wine_quality",
        version: str = "v1",
        model_dir: str | None = None,
    ) -> None:
        self.project = project
        self.version = version
        self._model: WineNet | None = None
        self._scaler: Any | None = None
        self._feature_names: list[str] | None = None
        self._metrics: dict[str, Any] | None = None
        # If an explicit ``model_dir`` is provided (e.g. tests), use it; otherwise
        # resolve under ``models/<project>/<version>``.
        self._explicit_dir: str | None = model_dir

    @property
    def model_dir(self) -> str:
        if self._explicit_dir is not None:
            return self._explicit_dir
        return models_path(f"{self.project}/{self.version}")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._scaler is not None

    def load(self, model_cfg: dict[str, Any] | None = None) -> None:
        """Load weights, scaler, and metadata. ``model_cfg`` matches WineModelConfig."""
        if not Path(self.model_dir).exists():
            raise FileNotFoundError(
                f"Model registry not found: {self.model_dir}. "
                "Train first: `python -m src.training.train_wine`."
            )

        logger.info("Loading wine model from %s", self.model_dir)

        if model_cfg is None:
            # Try to read the frozen config the trainer saved.
            config_path = os.path.join(self.model_dir, "config.yaml")
            if Path(config_path).exists():
                try:
                    from omegaconf import OmegaConf

                    cfg = OmegaConf.load(config_path)
                    model_cfg = {
                        "input_size": int(cfg.model.input_size),
                        "hidden_sizes": list(cfg.model.hidden_sizes),
                        "output_size": int(cfg.model.output_size),
                        "dropout": float(cfg.model.dropout),
                    }
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to load config.yaml: %s", exc)
                    model_cfg = None
            if model_cfg is None:
                # Reasonable defaults; will fail if dropout was non-zero at train time.
                model_cfg = {
                    "input_size": 11,
                    "hidden_sizes": [64, 32],
                    "output_size": 2,
                    "dropout": 0.0,
                }

        self._model = WineNet(
            input_size=int(model_cfg.get("input_size", 11)),
            hidden_sizes=list(model_cfg.get("hidden_sizes", [64, 32])),
            output_size=int(model_cfg.get("output_size", 2)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
        state_path = os.path.join(self.model_dir, "model.pth")
        self._model.load_state_dict(torch.load(state_path, map_location="cpu"))
        self._model.eval()

        self._scaler = joblib.load(os.path.join(self.model_dir, "scaler.joblib"))

        feat_path = os.path.join(self.model_dir, "feature_names.json")
        if Path(feat_path).exists():
            self._feature_names = json.loads(Path(feat_path).read_text())
        else:
            self._feature_names = [
                "fixed acidity", "volatile acidity", "citric acid",
                "residual sugar", "chlorides", "free sulfur dioxide",
                "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
            ]

        metrics_path = os.path.join(self.model_dir, "metrics.json")
        if Path(metrics_path).exists():
            self._metrics = json.loads(Path(metrics_path).read_text())

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        """Run inference for one sample.

        Accepts either API-friendly underscored keys (``fixed_acidity``) or
        the original column names (``fixed acidity``) for backwards
        compatibility.

        Args:
            features: Dict mapping feature names to numeric values.

        Returns:
            Dict with ``label``, ``confidence``, ``probabilities``, ``model_version``.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call .load() first.")

        assert self._feature_names is not None
        ordered = np.array(
            [[float(self._lookup(features, name)) for name in self._feature_names]]
        )
        scaled = self._scaler.transform(ordered)
        with torch.no_grad():
            logits = self._model(torch.from_numpy(scaled).float())
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        classes = ["not_good", "good"]
        idx = int(np.argmax(probs))
        return {
            "label": classes[idx],
            "confidence": float(probs[idx]),
            "probabilities": {c: float(p) for c, p in zip(classes, probs)},
            "model_version": self.version,
        }

    @staticmethod
    def _lookup(features: dict[str, float], name: str) -> float:
        """Look up ``name`` in ``features``, trying several aliases."""
        candidates = [
            name,
            name.replace(" ", "_"),
            name.lower().replace(" ", "_"),
        ]
        # Special case: Pydantic schemas use ``ph`` for the column ``pH``.
        if name == "pH":
            candidates.append("ph")
            candidates.append("pH".replace(" ", "_"))
        for key in candidates:
            if key in features:
                return features[key]
        raise KeyError(
            f"Missing required feature: {name!r}. Tried: {candidates}"
        )


_singleton: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Return the process-wide model registry singleton (lazy-loaded)."""
    global _singleton
    if _singleton is None:
        _singleton = ModelRegistry()
        _singleton.load()
    return _singleton


__all__ = ["ModelRegistry", "get_registry"]
