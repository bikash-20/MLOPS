"""Pydantic request/response schemas for the wine-quality API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WineFeatures(BaseModel):
    """11 chemical features of a white-wine sample."""

    fixed_acidity: float = Field(..., ge=0, le=20, description="g(tartaric acid)/dm^3")
    volatile_acidity: float = Field(..., ge=0, le=2, description="g(acetic acid)/dm^3")
    citric_acid: float = Field(..., ge=0, le=2, description="g/dm^3")
    residual_sugar: float = Field(..., ge=0, le=70, description="g/dm^3")
    chlorides: float = Field(..., ge=0, le=1, description="g(sodium chloride)/dm^3")
    free_sulfur_dioxide: float = Field(..., ge=0, le=300, description="mg/dm^3")
    total_sulfur_dioxide: float = Field(..., ge=0, le=500, description="mg/dm^3")
    density: float = Field(..., ge=0.98, le=1.05, description="g/cm^3")
    ph: float = Field(..., ge=2, le=5, description="pH")
    sulphates: float = Field(..., ge=0, le=3, description="g(potassium sulphate)/dm^3")
    alcohol: float = Field(..., ge=0, le=20, description="vol. %")


class PredictionResponse(BaseModel):
    """API response for a single wine-quality prediction."""

    label: str = Field(..., description="good or not_good")
    confidence: float = Field(..., ge=0, le=1)
    probabilities: dict[str, float]
    model_version: str


class HealthResponse(BaseModel):
    """Liveness response."""

    status: str
    model_loaded: bool


class ModelInfo(BaseModel):
    """Metadata about the loaded model."""

    version: str
    metrics: dict
    feature_names: list[str]
    class_names: list[str]


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str
