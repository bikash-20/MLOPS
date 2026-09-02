"""FastAPI app for the wine-quality classifier.

Endpoints:
    GET  /            service info
    GET  /health      liveness probe
    GET  /model/info  metadata about the loaded model
    POST /predict     single-sample inference

Run locally:
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from src.api.model_loader import get_registry
from src.api.schemas import (
    HealthResponse,
    ModelInfo,
    PredictionResponse,
    WineFeatures,
)
from src.utils import get_logger

logger = get_logger("neural_network.api", level=os.getenv("LOG_LEVEL", "INFO"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model on startup, log on shutdown."""
    logger.info("Starting wine-quality API service")
    try:
        get_registry()
        logger.info("Model loaded successfully")
    except FileNotFoundError as exc:
        logger.error("Model registry missing: %s", exc)
    yield
    logger.info("Shutting down wine-quality API service")


app = FastAPI(
    title="Wine Quality Classifier API",
    description="Predict whether a white wine is 'good' (quality >= 7) using a PyTorch MLP.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/", response_model=dict)
def root() -> dict:
    """Service info."""
    return {
        "service": "wine-quality-classifier",
        "version": app.version,
        "endpoints": ["/health", "/model/info", "/predict", "/docs"],
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check."""
    registry = get_registry()
    return HealthResponse(status="ok", model_loaded=registry.is_loaded)


@app.get("/model/info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    """Return version, metrics, and feature names of the loaded model."""
    registry = get_registry()
    if not registry.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    return ModelInfo(
        version=registry.version,
        metrics=registry._metrics or {},
        feature_names=registry._feature_names or [],
        class_names=["not_good", "good"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: WineFeatures) -> PredictionResponse:
    """Predict the quality label for a single wine sample."""
    registry = get_registry()
    if not registry.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    try:
        result = registry.predict(features.model_dump())
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc
    return PredictionResponse(**result)


__all__ = ["app"]
