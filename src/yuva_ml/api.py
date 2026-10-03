"""FastAPI service for the serialized customer-churn model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .schema import FEATURE_COLUMNS

ARTIFACT_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "churn_model.joblib"

app = FastAPI(
    title="Yuva Customer Churn Prediction API",
    version="1.0.0",
    description=(
        "Predicts churn probability from a customer record. The saved pipeline "
        "performs missing-value imputation, scaling, and category encoding."
    ),
)


class CustomerRecord(BaseModel):
    """Raw feature values; omitted or null values are imputed by the model."""

    model_config = ConfigDict(extra="ignore")

    customer_id: Optional[str] = None
    tenure_months: Optional[float] = None
    monthly_charges: Optional[float] = None
    total_charges: Optional[float] = None
    support_tickets: Optional[float] = None
    senior_citizen: Optional[int] = None
    contract: Optional[str] = None
    internet_service: Optional[str] = None
    payment_method: Optional[str] = None
    paperless_billing: Optional[str] = None
    partner: Optional[str] = None
    dependents: Optional[str] = None


class PredictionRequest(BaseModel):
    instances: list[CustomerRecord] = Field(min_length=1, max_length=100)


class PredictionItem(BaseModel):
    customer_id: Optional[str]
    prediction: str
    churn_probability: float


class PredictionResponse(BaseModel):
    predictions: list[PredictionItem]


@lru_cache(maxsize=1)
def get_model():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {ARTIFACT_PATH}. Run `python -m yuva_ml.train` first."
        )
    return joblib.load(ARTIFACT_PATH)


@app.get("/", tags=["service"])
def root() -> dict:
    return {
        "service": "Yuva Customer Churn Prediction API",
        "docs": "/docs",
        "prediction_endpoint": "POST /predict",
    }


@app.get("/health", tags=["service"])
def health() -> dict:
    return {
        "status": "ok" if ARTIFACT_PATH.exists() else "model_missing",
        "model_loaded": ARTIFACT_PATH.exists(),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        model = get_model()
    except (FileNotFoundError, OSError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    records = [
        record.model_dump(exclude={"customer_id"})
        for record in request.instances
    ]
    frame = pd.DataFrame(records, columns=FEATURE_COLUMNS)
    labels = model.predict(frame)
    probabilities = model.predict_proba(frame)[:, 1]
    result = [
        PredictionItem(
            customer_id=request.instances[index].customer_id,
            prediction="Yes" if int(labels[index]) == 1 else "No",
            churn_probability=round(float(probabilities[index]), 6),
        )
        for index in range(len(request.instances))
    ]
    return PredictionResponse(predictions=result)
