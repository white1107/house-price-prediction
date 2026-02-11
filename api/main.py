"""FastAPI application for House Price Prediction."""

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    HouseFeatures,
    PredictionResponse,
)
from house_prices.features.advanced_engineering import build_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global model state
model_state = {
    "model": None,
    "scaler": None,
    "feature_names": None,
    "model_name": None,
    "version": "1.0.0",
}


def load_model():
    """Load trained model and artifacts."""
    models_dir = PROJECT_ROOT / "models"
    try:
        model_state["model"] = joblib.load(models_dir / "best_model.joblib")
        model_state["scaler"] = joblib.load(models_dir / "scaler.joblib")
        model_state["feature_names"] = joblib.load(models_dir / "feature_names.joblib")
        model_state["model_name"] = type(model_state["model"]).__name__
        logger.info(
            f"Model loaded: {model_state['model_name']} "
            f"({len(model_state['feature_names'])} features)"
        )
    except FileNotFoundError as e:
        logger.warning(f"Model not found: {e}. Run 'make train' first.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    load_model()
    yield


app = FastAPI(
    title="House Price Prediction API",
    description="ML-powered house price prediction using ensemble models trained on Kaggle data",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_input(features: HouseFeatures) -> pd.DataFrame:
    """Convert input features to model-ready DataFrame."""
    data = features.model_dump(by_alias=True)

    # Handle None values (same as NA in original data)
    for key, val in data.items():
        if val is None:
            data[key] = np.nan

    df = pd.DataFrame([data])

    # Need a dummy second row for consistent preprocessing (get_dummies alignment)
    # We'll use the same row duplicated, then take only first prediction
    df_double = pd.concat([df, df], ignore_index=True)

    # Run feature engineering pipeline
    df_processed = build_features(df_double, encode_nominal=True, fix_skew=True)

    # Align columns with training features
    feature_names = model_state["feature_names"]
    missing_cols = set(feature_names) - set(df_processed.columns)
    for col in missing_cols:
        df_processed[col] = 0
    extra_cols = set(df_processed.columns) - set(feature_names)
    df_processed = df_processed.drop(columns=list(extra_cols), errors="ignore")
    df_processed = df_processed[feature_names]

    # Scale
    if model_state["scaler"] is not None:
        df_processed = pd.DataFrame(
            model_state["scaler"].transform(df_processed),
            columns=feature_names,
        )

    return df_processed.iloc[[0]]


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_state["model"] is not None,
        model_name=model_state["model_name"],
        features_count=len(model_state["feature_names"]) if model_state["feature_names"] else None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: HouseFeatures):
    """Predict house price for a single house."""
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run 'make train' first.")

    start_time = time.time()

    try:
        df = preprocess_input(features)
        prediction = model_state["model"].predict(df)[0]

        # Reverse log transform
        predicted_price = float(np.expm1(prediction))
        predicted_price = max(0, predicted_price)

        latency = time.time() - start_time
        logger.info(f"Prediction: ${predicted_price:,.0f} (latency: {latency:.3f}s)")

        return PredictionResponse(
            predicted_price=round(predicted_price, 2),
            model_name=model_state["model_name"],
            model_version=model_state["version"],
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """Predict house prices for multiple houses."""
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run 'make train' first.")

    if len(request.houses) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 houses per batch request")

    predictions = []
    for house in request.houses:
        try:
            df = preprocess_input(house)
            prediction = model_state["model"].predict(df)[0]
            predicted_price = float(np.expm1(prediction))
            predicted_price = max(0, predicted_price)

            predictions.append(PredictionResponse(
                predicted_price=round(predicted_price, 2),
                model_name=model_state["model_name"],
                model_version=model_state["version"],
            ))
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            predictions.append(PredictionResponse(
                predicted_price=-1.0,
                model_name="error",
                model_version=str(e),
            ))

    return BatchPredictionResponse(predictions=predictions, count=len(predictions))


@app.post("/model/reload", tags=["System"])
async def reload_model():
    """Reload model from disk (after retraining)."""
    load_model()
    if model_state["model"] is None:
        raise HTTPException(status_code=500, detail="Failed to reload model")
    return {
        "status": "reloaded",
        "model_name": model_state["model_name"],
        "features_count": len(model_state["feature_names"]),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
