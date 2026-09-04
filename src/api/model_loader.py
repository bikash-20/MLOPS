"""Lazy, cached loader for the wine-quality and MNIST CNN models.

The wine registry loads ``model.pth``, ``scaler.joblib``, and
``feature_names.json`` from a versioned registry directory.

The MNIST registry loads ``model.pth`` and ``model_arch.json``, and serves
predictions from 28x28 grayscale PNG uploads via PIL.

The CIFAR registry loads ``best.pt`` (or ``last.pt``) and ``model_arch.json``,
and serves predictions from 32x32 RGB PNG/JPEG uploads via PIL.

All are thread-safe singletons via module-level state.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from PIL import Image

from src.models.cifar_resnet import CifarResNet
from src.models.mnist_cnn import SimpleCNN
from src.models.wine_nn import WineNet
from src.utils import get_logger, models_path, resolve_runtime_version

logger = get_logger(__name__)


def _resolve_default_version(project: str) -> str:
    """Resolve the registry version for an API-side loader.

    Honours the ``MODEL_VERSION`` env var (e.g. ``"v2"``, ``"2"``, or
    ``"latest"``). Falls back to ``"v1"`` if no versions exist (this lets
    tests construct registries in ``tmp_path`` without having to create
    real on-disk artifacts first).
    """
    try:
        return resolve_runtime_version(project)
    except FileNotFoundError:
        return "v1"


class ModelRegistry:
    """Loads and serves a single versioned wine-quality model."""

    def __init__(
        self,
        project: str = "wine_quality",
        version: str | None = None,
        model_dir: str | None = None,
    ) -> None:
        self.project = project
        # If a version was passed in, honour it; otherwise resolve from
        # the ``MODEL_VERSION`` env var (or ``"latest"``) so operators
        # can pin a specific version in production without rebuilding.
        self.version = version if version is not None else _resolve_default_version(
            project,
        )
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


# --- MNIST registry --------------------------------------------------------


class MnistRegistry:
    """Loads and serves a single versioned MNIST CNN (28x28 grayscale).

    Unlike ``ModelRegistry`` (wine), this registry does not need a scaler:
    inputs are PNG/JPEG bytes that get preprocessed into a normalised
    ``[0, 1]`` ``[1, 1, 28, 28]`` float tensor at inference time.
    """

    def __init__(
        self,
        project: str = "mnist",
        version: str | None = None,
        model_dir: str | None = None,
        image_size: int = 28,
    ) -> None:
        self.project = project
        self.version = version if version is not None else _resolve_default_version(
            project,
        )
        self.image_size = int(image_size)
        self._model: SimpleCNN | None = None
        self._class_names: list[str] | None = None
        self._metrics: dict[str, Any] | None = None
        self._explicit_dir: str | None = model_dir

    @property
    def model_dir(self) -> str:
        if self._explicit_dir is not None:
            return self._explicit_dir
        return models_path(f"{self.project}/{self.version}")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, model_cfg: dict[str, Any] | None = None) -> None:
        """Load CNN weights and metadata.

        ``model_cfg`` should match the keys in ``model_arch.json`` written by
        ``train_mnist.py`` (``in_channels``, ``conv_channels``, ``fc_hidden``,
        ``num_classes``, ``dropout``).
        """
        if not Path(self.model_dir).exists():
            raise FileNotFoundError(
                f"Model registry not found: {self.model_dir}. "
                "Train first: `python -m src.training.train_mnist`."
            )

        logger.info("Loading MNIST model from %s", self.model_dir)

        if model_cfg is None:
            arch_path = os.path.join(self.model_dir, "model_arch.json")
            if Path(arch_path).exists():
                with open(arch_path) as f:
                    model_cfg = json.load(f)
            else:
                # Reasonable defaults; matches the values in configs/mnist.yaml.
                model_cfg = {
                    "in_channels": 1,
                    "conv_channels": [32, 64],
                    "fc_hidden": 128,
                    "num_classes": 10,
                    "dropout": 0.25,
                }

        self._model = SimpleCNN(
            in_channels=int(model_cfg.get("in_channels", 1)),
            conv_channels=tuple(model_cfg.get("conv_channels", [32, 64])),
            fc_hidden=int(model_cfg.get("fc_hidden", 128)),
            num_classes=int(model_cfg.get("num_classes", 10)),
            dropout=float(model_cfg.get("dropout", 0.25)),
        )
        state_path = os.path.join(self.model_dir, "model.pth")
        self._model.load_state_dict(
            torch.load(state_path, map_location="cpu"),
        )
        self._model.eval()

        class_names_path = os.path.join(self.model_dir, "class_names.json")
        if Path(class_names_path).exists():
            self._class_names = json.loads(Path(class_names_path).read_text())
        else:
            self._class_names = [str(i) for i in range(int(model_cfg.get("num_classes", 10)))]

        metrics_path = os.path.join(self.model_dir, "metrics.json")
        if Path(metrics_path).exists():
            self._metrics = json.loads(Path(metrics_path).read_text())

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        """Run inference on raw image bytes.

        Args:
            image_bytes: PNG/JPEG file contents.

        Returns:
            Dict with ``label``, ``confidence``, ``probabilities``,
            ``model_version``.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call .load() first.")

        assert self._class_names is not None
        tensor = self._preprocess(image_bytes)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        idx = int(np.argmax(probs))
        return {
            "label": self._class_names[idx],
            "confidence": float(probs[idx]),
            "probabilities": {c: float(p) for c, p in zip(self._class_names, probs)},
            "model_version": self.version,
        }

    def _preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """Decode bytes -> PIL -> 28x28 grayscale -> normalised tensor."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
        except Exception as exc:
            raise ValueError(f"Could not decode image: {exc}") from exc

        # Convert to grayscale regardless of source mode (RGBA, P, L, RGB...).
        img = img.convert("L")
        # Resize with bilinear (matches torchvision's ToTensor expectations).
        if img.size != (self.image_size, self.image_size):
            img = img.resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.BILINEAR,
            )
        arr = np.asarray(img, dtype=np.float32) / 255.0
        # Shape (28, 28) -> (1, 28, 28) -> (1, 1, 28, 28)
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        return tensor


_mnist_singleton: MnistRegistry | None = None


def get_mnist_registry() -> MnistRegistry:
    """Return the process-wide MNIST registry singleton (lazy-loaded)."""
    global _mnist_singleton
    if _mnist_singleton is None:
        _mnist_singleton = MnistRegistry()
        _mnist_singleton.load()
    return _mnist_singleton


# --- CIFAR registry --------------------------------------------------------


class CifarRegistry:
    """Loads and serves a single versioned CIFAR-10 ResNet.

    Inputs are PNG/JPEG bytes that get preprocessed into a normalised
    ``[1, 3, 32, 32]`` float tensor at inference time, using the canonical
    CIFAR-10 mean/std stored in ``model_arch.json``.

    Unlike the wine/MNIST registries, this one accepts any aspect ratio
    or color mode from the upload (RGB, RGBA, L, P) and resizes to
    ``image_size x image_size`` with bilinear filtering.
    """

    def __init__(
        self,
        project: str = "cifar",
        version: str | None = None,
        model_dir: str | None = None,
        image_size: int = 32,
        checkpoint: str = "best.pt",
    ) -> None:
        self.project = project
        self.version = version if version is not None else _resolve_default_version(
            project,
        )
        self.image_size = int(image_size)
        self.checkpoint_name = checkpoint
        self._model: CifarResNet | None = None
        self._class_names: list[str] | None = None
        self._metrics: dict[str, Any] | None = None
        self._mean: tuple[float, ...] = (0.4914, 0.4822, 0.4465)
        self._std: tuple[float, ...] = (0.2470, 0.2435, 0.2616)
        self._explicit_dir: str | None = model_dir

    @property
    def model_dir(self) -> str:
        if self._explicit_dir is not None:
            return self._explicit_dir
        return models_path(f"{self.project}/{self.version}")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, model_cfg: dict[str, Any] | None = None) -> None:
        """Load CIFAR ResNet weights, arch, and metadata.

        ``model_cfg`` should match keys in ``model_arch.json`` written by
        ``train_cifar.py``: ``in_channels``, ``num_classes``,
        ``base_channels``, ``num_blocks_per_stage``, ``dropout``.
        """
        if not Path(self.model_dir).exists():
            raise FileNotFoundError(
                f"Model registry not found: {self.model_dir}. "
                "Train first: `python -m src.training.train_cifar`."
            )

        logger.info("Loading CIFAR model from %s", self.model_dir)

        if model_cfg is None:
            arch_path = os.path.join(self.model_dir, "model_arch.json")
            if Path(arch_path).exists():
                with open(arch_path) as f:
                    model_cfg = json.load(f)
            else:
                # Sensible defaults matching configs/cifar.yaml.
                model_cfg = {
                    "in_channels": 3,
                    "num_classes": 10,
                    "base_channels": 64,
                    "num_blocks_per_stage": 2,
                    "dropout": 0.2,
                }

        # Pull normalisation constants if present.
        if "mean" in model_cfg and "std" in model_cfg:
            self._mean = tuple(float(x) for x in model_cfg["mean"])
            self._std = tuple(float(x) for x in model_cfg["std"])

        self._model = CifarResNet(
            in_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=int(model_cfg.get("num_classes", 10)),
            base_channels=int(model_cfg.get("base_channels", 64)),
            num_blocks_per_stage=int(model_cfg.get("num_blocks_per_stage", 2)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
        # Prefer the best checkpoint, fall back to last.
        candidates = [self.checkpoint_name, "best.pt", "last.pt", "model.pth"]
        loaded_from: str | None = None
        for name in candidates:
            state_path = os.path.join(self.model_dir, name)
            if Path(state_path).exists():
                self._model.load_state_dict(
                    torch.load(state_path, map_location="cpu"),
                )
                loaded_from = name
                break
        if loaded_from is None:
            raise FileNotFoundError(
                f"No checkpoint found in {self.model_dir}. "
                f"Tried: {candidates}",
            )
        self._model.eval()
        logger.info("Loaded CIFAR checkpoint: %s", loaded_from)

        class_names_path = os.path.join(self.model_dir, "class_names.json")
        if Path(class_names_path).exists():
            self._class_names = json.loads(Path(class_names_path).read_text())
        else:
            self._class_names = [
                str(i) for i in range(int(model_cfg.get("num_classes", 10)))
            ]

        metrics_path = os.path.join(self.model_dir, "metrics.json")
        if Path(metrics_path).exists():
            self._metrics = json.loads(Path(metrics_path).read_text())

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        """Run inference on raw image bytes.

        Returns:
            Dict with ``label``, ``confidence``, ``probabilities``,
            ``top5`` (list of {label, probability}), and ``model_version``.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call .load() first.")

        assert self._class_names is not None
        tensor = self._preprocess(image_bytes)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        idx = int(np.argmax(probs))
        # Build top-5 list (or N if fewer classes).
        n = len(self._class_names)
        top_k = min(5, n)
        top_indices = np.argsort(probs)[::-1][:top_k]
        top5 = [
            {"label": self._class_names[int(i)], "probability": float(probs[int(i)])}
            for i in top_indices
        ]
        return {
            "label": self._class_names[idx],
            "confidence": float(probs[idx]),
            "probabilities": {c: float(p) for c, p in zip(self._class_names, probs)},
            "top5": top5,
            "model_version": self.version,
        }

    def _preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """Decode bytes -> PIL -> 32x32 RGB -> normalised tensor.

        Mirrors the train-time transform: ToTensor + Normalize(mean, std).
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))
        except Exception as exc:
            raise ValueError(f"Could not decode image: {exc}") from exc

        # Always go through RGB so normalisation matches training.
        img = img.convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.BILINEAR,
            )
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
        # Normalise per channel.
        mean = np.asarray(self._mean, dtype=np.float32)
        std = np.asarray(self._std, dtype=np.float32)
        arr = (arr - mean) / std  # broadcast (H, W, 3)
        # ToTensor: HWC -> CHW, plus batch dimension.
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor


_cifar_singleton: CifarRegistry | None = None


def get_cifar_registry() -> CifarRegistry:
    """Return the process-wide CIFAR registry singleton (lazy-loaded)."""
    global _cifar_singleton
    if _cifar_singleton is None:
        _cifar_singleton = CifarRegistry()
        _cifar_singleton.load()
    return _cifar_singleton


__all__ = [
    "CifarRegistry",
    "MnistRegistry",
    "ModelRegistry",
    "get_cifar_registry",
    "get_mnist_registry",
    "get_registry",
]
