"""FastAPI app for the wine-quality and MNIST CNN classifiers.

Endpoints:
    GET  /                    service info
    GET  /health              liveness probe
    GET  /model/info          wine model metadata
    GET  /model/info/mnist    MNIST model metadata
    POST /predict             single wine sample inference (JSON)
    POST /predict/mnist       single 28x28 PNG inference (multipart upload)

Run locally:
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, status

from src.api.model_loader import get_mnist_registry, get_registry
from src.api.schemas import (
    HealthResponse,
    MnistPredictionResponse,
    ModelInfo,
    PredictionResponse,
    WineFeatures,
)
from src.utils import get_logger

logger = get_logger("neural_network.api", level=os.getenv("LOG_LEVEL", "INFO"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models on startup, log on shutdown."""
    logger.info("Starting neural-network API service")
    try:
        get_registry()
        logger.info("Wine model loaded successfully")
    except FileNotFoundError as exc:
        logger.error("Wine model registry missing: %s", exc)

    try:
        get_mnist_registry()
        logger.info("MNIST model loaded successfully")
    except FileNotFoundError as exc:
        logger.warning("MNIST model registry missing: %s", exc)

    yield
    logger.info("Shutting down neural-network API service")


app = FastAPI(
    title="Neural Network API",
    description=(
        "Wine-quality classifier (PyTorch MLP) + MNIST digit classifier "
        "(PyTorch CNN)."
    ),
    version="2.1.0",
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
            "/model/info",
            "/model/info/mnist",
            "/predict",
            "/predict/mnist",
            "/docs",
        ],
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check."""
    wine_loaded = False
    mnist_loaded = False
    try:
        wine_loaded = get_registry().is_loaded
    except FileNotFoundError:
        pass
    try:
        mnist_loaded = get_mnist_registry().is_loaded
    except FileNotFoundError:
        pass
    overall = wine_loaded or mnist_loaded
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


__all__ = ["app"]
