"""FastAPI async inference server for AtmosForge.

Endpoints:
    POST /predict      — Single forecast from time series input
    POST /predict/batch — Batch inference
    GET  /models       — List available models
    GET  /health       — Health check
    GET  /docs         — Swagger UI (auto-generated)

Authentication: API key via X-API-Key header (from env var API_KEY).
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import numpy as np
import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader

from src.serving.model_loader import ModelLoader
from src.serving.schemas import (
    BatchForecastRequest,
    BatchForecastResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
)
from src.utils.logger import setup_logger

logger = setup_logger("atmosforge.serving.api")

# Global state
model_loader: ModelLoader | None = None
API_KEY = os.environ.get("API_KEY", "dev-key-change-me")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

SUPPORTED_MODELS = ["cnn", "lstm", "gru", "nhits", "patchtst", "tft"]


# ─── Lifespan ────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: load models on startup."""
    global model_loader
    model_loader = ModelLoader()
    logger.info("AtmosForge API server starting...")
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    yield
    logger.info("AtmosForge API server shutting down...")


# ─── App ─────────────────────────────────────────────────────

app = FastAPI(
    title="AtmosForge — Weather Forecasting API",
    description=(
        "Production-grade inference endpoint for multivariate "
        "meteorological time series forecasting. Supports 6 deep "
        "learning architectures with uncertainty quantification."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ─── Auth ────────────────────────────────────────────────────


async def verify_api_key(
    api_key: str | None = Security(api_key_header),
) -> str:
    """Verify API key from X-API-Key header.

    Args:
        api_key: API key from header.

    Returns:
        Verified API key.

    Raises:
        HTTPException: If key is missing or invalid.
    """
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


# ─── Endpoints ───────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns service status, loaded model info, and GPU availability.
    """
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

    loaded_model = None
    if model_loader and model_loader._loaded_models:
        loaded_model = list(model_loader._loaded_models.keys())[0]

    return HealthResponse(
        status="healthy",
        loaded_model=loaded_model,
        mlflow_uri=os.environ.get("MLFLOW_TRACKING_URI", "./mlruns"),
        version="0.1.0",
        gpu_available=gpu_available,
        gpu_name=gpu_name,
    )


@app.get("/models", response_model=ModelsResponse, tags=["Models"])
async def list_models(
    _: str = Depends(verify_api_key),
) -> ModelsResponse:
    """List available models and their status."""
    from src.training.trainer import MODEL_REGISTRY, register_models
    register_models()

    models = []
    for name in SUPPORTED_MODELS:
        model_class = MODEL_REGISTRY.get(name)
        loaded = model_loader is not None and name in model_loader._loaded_models

        if model_class:
            # Create temp instance to get param count
            try:
                temp = model_class(input_size=14, horizon=24)
                param_count = temp.count_parameters()
            except Exception:
                param_count = 0

            models.append(ModelInfo(
                name=name,
                class_name=model_class.__name__,
                parameters=param_count,
                loaded=loaded,
            ))

    return ModelsResponse(
        models=models,
        default_model="tft",
    )


@app.post("/predict", response_model=ForecastResponse, tags=["Prediction"])
async def predict(
    request: ForecastRequest,
    _: str = Depends(verify_api_key),
) -> ForecastResponse:
    """Generate forecast from time series input.

    Accepts a sequence of feature vectors and returns point forecast
    with optional quantile predictions for uncertainty quantification.
    """
    if model_loader is None:
        raise HTTPException(
            status_code=503,
            detail="Model loader not initialized. Server is starting up.",
        )

    start_time = time.perf_counter()

    # Get or load model
    model = model_loader.get_loaded_model(request.model)

    if model is None:
        # Try to load from MLflow or checkpoint
        model = model_loader.load_from_mlflow(request.model)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model '{request.model}' is not available. "
                "Train a model first with: make train MODEL={request.model}"
            ),
        )

    try:
        # Prepare input tensor
        input_array = np.array(request.features, dtype=np.float32)
        input_tensor = torch.from_numpy(input_array).unsqueeze(0)  # Add batch dim
        input_tensor = input_tensor.to(model_loader.device)

        # Run inference
        with torch.no_grad():
            output = model(input_tensor)

        # Extract results
        forecast = output.point_forecast.squeeze(0).cpu().numpy().tolist()

        quantiles = None
        if request.return_quantiles and output.quantiles:
            quantiles = {
                f"q{int(q * 100)}": qv.squeeze(0).cpu().numpy().tolist()
                for q, qv in output.quantiles.items()
            }

        inference_time = (time.perf_counter() - start_time) * 1000

        return ForecastResponse(
            forecast=forecast[:request.horizon],
            quantiles=quantiles,
            model_used=request.model,
            inference_time_ms=round(inference_time, 2),
            horizon=request.horizon,
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post(
    "/predict/batch",
    response_model=BatchForecastResponse,
    tags=["Prediction"],
)
async def predict_batch(
    request: BatchForecastRequest,
    _: str = Depends(verify_api_key),
) -> BatchForecastResponse:
    """Batch inference for multiple inputs."""
    start_time = time.perf_counter()

    predictions = []
    for single_request in request.inputs:
        result = await predict(single_request, _)
        predictions.append(result)

    total_time = (time.perf_counter() - start_time) * 1000

    return BatchForecastResponse(
        predictions=predictions,
        total_time_ms=round(total_time, 2),
    )


# ─── Main ────────────────────────────────────────────────────


def main() -> None:
    """Run the FastAPI server."""
    uvicorn.run(
        "src.serving.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
