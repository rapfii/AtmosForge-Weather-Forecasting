"""Pydantic schemas for FastAPI serving endpoints.

Defines request and response models for all API endpoints
with full validation and documentation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Request body for POST /predict.

    Attributes:
        features: Time series input as list of lists.
            Shape: [seq_len, n_features] — each inner list is one timestep.
        horizon: Number of future steps to predict (default: 24).
        model: Model name to use for prediction (default: 'tft').
        return_quantiles: Whether to return quantile predictions.
    """

    features: list[list[float]] = Field(
        ...,
        description="Time series input: [seq_len][n_features]",
        min_length=1,
    )
    horizon: int = Field(
        default=24,
        description="Forecast horizon (steps ahead)",
        ge=1,
        le=168,
    )
    model: str = Field(
        default="tft",
        description="Model to use (cnn, lstm, gru, nhits, patchtst, tft)",
    )
    return_quantiles: bool = Field(
        default=True,
        description="Include quantile predictions (q10, q50, q90)",
    )

    model_config = {"json_schema_extra": {
        "examples": [{
            "features": [[0.1] * 14] * 168,
            "horizon": 24,
            "model": "lstm",
            "return_quantiles": True,
        }]
    }}


class ForecastResponse(BaseModel):
    """Response body for POST /predict.

    Attributes:
        forecast: Point forecast values.
        quantiles: Optional quantile predictions.
        model_used: Name of the model used.
        inference_time_ms: Inference time in milliseconds.
    """

    forecast: list[float] = Field(
        ..., description="Point forecast values"
    )
    quantiles: dict[str, list[float]] | None = Field(
        default=None,
        description="Quantile predictions: {'q10': [...], 'q50': [...], 'q90': [...]}",
    )
    model_used: str = Field(
        ..., description="Model used for prediction"
    )
    inference_time_ms: float = Field(
        ..., description="Inference time in milliseconds"
    )
    horizon: int = Field(
        ..., description="Forecast horizon"
    )


class BatchForecastRequest(BaseModel):
    """Request body for POST /predict/batch.

    Attributes:
        inputs: List of individual forecast requests.
    """

    inputs: list[ForecastRequest] = Field(
        ..., description="List of forecast requests", min_length=1
    )


class BatchForecastResponse(BaseModel):
    """Response for batch predictions.

    Attributes:
        predictions: List of individual forecast responses.
        total_time_ms: Total batch inference time.
    """

    predictions: list[ForecastResponse] = Field(
        ..., description="List of forecast responses"
    )
    total_time_ms: float = Field(
        ..., description="Total batch inference time in ms"
    )


class ModelInfo(BaseModel):
    """Model information for GET /models.

    Attributes:
        name: Model name.
        class_name: Python class name.
        parameters: Total parameter count.
        loaded: Whether model is currently loaded.
    """

    name: str
    class_name: str
    parameters: int
    loaded: bool


class ModelsResponse(BaseModel):
    """Response for GET /models."""

    models: list[ModelInfo]
    default_model: str


class HealthResponse(BaseModel):
    """Response for GET /health.

    Attributes:
        status: Service status ('healthy' or 'degraded').
        loaded_model: Currently loaded model name.
        mlflow_uri: MLflow tracking URI.
        version: API version.
        gpu_available: Whether GPU is available.
    """

    status: str = Field(..., description="Service status")
    loaded_model: str | None = Field(
        default=None, description="Currently loaded model"
    )
    mlflow_uri: str = Field(..., description="MLflow tracking URI")
    version: str = Field(..., description="API version")
    gpu_available: bool = Field(..., description="GPU availability")
    gpu_name: str | None = Field(default=None, description="GPU name")
