"""FastAPI app for the wine-quality, MNIST, and CIFAR-10 classifiers.

Endpoints:
    GET  /                       service info
    GET  /health                 liveness probe
    GET  /models                 registry listing (every project + every version)
    GET  /model/info             wine model metadata
    GET  /model/info/mnist       MNIST model metadata
    GET  /model/info/cifar       CIFAR-10 model metadata
    POST /predict                single wine sample inference (JSON)
    POST /predict/mnist          single 28x28 PNG inference (multipart upload)
    POST /predict/cifar          single 32x32 RGB PNG/JPEG inference (multipart)

Run locally:
    MODEL_VERSION=v2 uvicorn src.api.main:app --reload

The ``MODEL_VERSION`` env var pins the version each registry serves.
Accepts ``"latest"`` (default), ``"v2"``, or bare integer ``"2"``.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, status

from src.api.model_loader import get_cifar_registry, get_mnist_registry, get_registry
from src.api.schemas import (
    CifarPredictionResponse,
    HealthResponse,
    MnistPredictionResponse,
    ModelInfo,
    PredictionResponse,
    RegistryInfo,
    RegistryVersionInfo,
    WineFeatures,
)
from src.utils import get_logger, list_versions, models_path

logger = get_logger("neural_network.api", level=os.getenv("LOG_LEVEL", "INFO"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models on startup, log on shutdown."""
    logger.info("Starting neural-network API service")
    wine_loaded = False
    mnist_loaded = False
    cifar_loaded = False
    try:
        get_registry()
        wine_loaded = True
        logger.info("Wine model loaded successfully")
    except FileNotFoundError as exc:
        logger.error("Wine model registry missing: %s", exc)

    try:
        get_mnist_registry()
        mnist_loaded = True
        logger.info("MNIST model loaded successfully")
    except FileNotFoundError as exc:
        logger.warning("MNIST model registry missing: %s", exc)

    try:
        get_cifar_registry()
        cifar_loaded = True
        logger.info("CIFAR model loaded successfully")
    except FileNotFoundError as exc:
        logger.warning("CIFAR model registry missing: %s", exc)

    app.state.wine_loaded = wine_loaded
    app.state.mnist_loaded = mnist_loaded
    app.state.cifar_loaded = cifar_loaded
    yield
    logger.info("Shutting down neural-network API service")


app = FastAPI(
    title="Neural Network API",
    description=(
        "Wine-quality classifier (PyTorch MLP) + MNIST digit classifier "
        "(PyTorch CNN) + CIFAR-10 classifier (CIFAR-style ResNet-18)."
    ),
    version="2.2.0",
    lifespan=lifespan,
)


@app.get("/", response_model=dict)
def root() -> dict:
    """Service info."""
    return {
        "service": "neural-network-classifier",
        "version": app.version,
        "endpoints": [
            "/health",
            "/models",
            "/model/info",
            "/model/info/mnist",
            "/model/info/cifar",
            "/predict",
            "/predict/mnist",
            "/predict/cifar",
            "/docs",
        ],
    }


@app.get("/models", response_model=list[RegistryInfo])
def list_models() -> list[RegistryInfo]:
    """List every project + every available ``vN`` with its metrics.

    The currently active version (i.e. what the running API is serving)
    is highlighted via the ``active_version`` field on each project.
    """
    projects = ("wine_quality", "mnist", "cifar")

    def _active_for(project: str) -> str | None:
        try:
            if project == "wine_quality":
                return get_registry().version
            if project == "mnist":
                return get_mnist_registry().version
            if project == "cifar":
                return get_cifar_registry().version
        except (FileNotFoundError, RuntimeError):
            return None
        return None

    out: list[RegistryInfo] = []
    for project in projects:
        versions = list_versions(project)
        entries: list[RegistryVersionInfo] = []
        for version in versions:
            metrics_path = os.path.join(
                models_path(f"{project}/{version}"), "metrics.json",
            )
            metrics: dict = {}
            promoted = False
            reason = ""
            try:
                with open(metrics_path) as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    metrics = {
                        k: v for k, v in payload.items()
                        if not isinstance(v, (dict, list))
                    }
                    registry_section = payload.get("registry", {})
                    if isinstance(registry_section, dict):
                        promoted = bool(registry_section.get("promoted", False))
                        reason = str(registry_section.get("reason", ""))
            except (OSError, json.JSONDecodeError):
                metrics = {}
            entries.append(RegistryVersionInfo(
                version=version,
                available=True,
                metrics=metrics,
                promoted=promoted,
                reason=reason,
            ))
        out.append(RegistryInfo(
            project=project,
            active_version=_active_for(project),
            versions=entries,
        ))
    return out


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check.

    Reports overall health from the registries that are currently
    available. Uses ``app.state`` flags if the lifespan populated them;
    otherwise falls back to checking each registry directly (this lets
    the endpoint work in test contexts that bypass ``lifespan``).
    """
    wine_loaded = getattr(app.state, "wine_loaded", None)
    mnist_loaded = getattr(app.state, "mnist_loaded", None)
    cifar_loaded = getattr(app.state, "cifar_loaded", None)

    if wine_loaded is None:
        try:
            wine_loaded = get_registry().is_loaded
        except FileNotFoundError:
            wine_loaded = False
    if mnist_loaded is None:
        try:
            mnist_loaded = get_mnist_registry().is_loaded
        except FileNotFoundError:
            mnist_loaded = False
    if cifar_loaded is None:
        try:
            cifar_loaded = get_cifar_registry().is_loaded
        except FileNotFoundError:
            cifar_loaded = False

    overall = wine_loaded or mnist_loaded or cifar_loaded
    return HealthResponse(
        status="ok" if overall else "degraded",
        model_loaded=overall,
    )


@app.get("/model/info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    """Return version, metrics, and feature names of the loaded wine model."""
    registry = get_registry()
    if not registry.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Wine model not loaded",
        )
    return ModelInfo(
        version=registry.version,
        metrics=registry._metrics or {},
        feature_names=registry._feature_names or [],
        class_names=["not_good", "good"],
    )


@app.get("/model/info/mnist", response_model=ModelInfo)
def mnist_model_info() -> ModelInfo:
    """Return version, metrics, and class names of the loaded MNIST model."""
    registry = get_mnist_registry()
    if not registry.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MNIST model not loaded",
        )
    return ModelInfo(
        version=registry.version,
        metrics=registry._metrics or {},
        feature_names=[],
        class_names=registry._class_names or [],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: WineFeatures) -> PredictionResponse:
    """Predict the quality label for a single wine sample."""
    registry = get_registry()
    if not registry.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Wine model not loaded",
        )
    try:
        result = registry.predict(features.model_dump())
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("Wine prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc
    return PredictionResponse(**result)


@app.post("/predict/mnist", response_model=MnistPredictionResponse)
async def predict_mnist(file: UploadFile = File(...)) -> MnistPredictionResponse:  # noqa: B008
    """Predict the digit (0-9) for a single 28x28 grayscale PNG upload."""
    registry = get_mnist_registry()
    if not registry.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MNIST model not loaded",
        )
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty upload",
        )
    try:
        result = registry.predict(image_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("MNIST prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc
    return MnistPredictionResponse(**result)


@app.get("/model/info/cifar", response_model=ModelInfo)
def cifar_model_info() -> ModelInfo:
    """Return version, metrics, and class names of the loaded CIFAR model."""
    registry = get_cifar_registry()
    if not registry.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CIFAR model not loaded",
        )
    return ModelInfo(
        version=registry.version,
        metrics=registry._metrics or {},
        feature_names=[],
        class_names=registry._class_names or [],
    )


@app.post("/predict/cifar", response_model=CifarPredictionResponse)
async def predict_cifar(file: UploadFile = File(...)) -> CifarPredictionResponse:  # noqa: B008
    """Predict the CIFAR-10 class for a single 32x32 RGB image upload."""
    registry = get_cifar_registry()
    if not registry.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CIFAR model not loaded",
        )
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty upload",
        )
    try:
        result = registry.predict(image_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("CIFAR prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc
    return CifarPredictionResponse(**result)


__all__ = ["app"]
