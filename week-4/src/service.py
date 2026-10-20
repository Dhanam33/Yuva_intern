#!/usr/bin/env python3
"""Week 4 - FastAPI prediction service.

The model is loaded once at startup and served from memory. Every request is
validated by Pydantic before any model code runs, so malformed input produces
a readable 422 instead of a stack trace from inside scikit-learn.

Endpoints
---------
    GET  /                Browser demo
    GET  /health          Liveness, readiness, uptime, request count
    GET  /model           Model card as JSON
    GET  /model/manifest  Artifact digests and load time
    POST /predict         Score one customer
    POST /predict/batch   Score up to 1,000 customers
    GET  /docs            Interactive OpenAPI documentation (FastAPI built in)

Run
---
    uvicorn service:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import (
    DEFAULT_THRESHOLD,
    MANIFEST_PATH,
    MODEL_CARD_PATH,
    MODEL_DESCRIPTION,
    MODEL_PATH,
    MODEL_VERSION,
    STATIC_DIR,
)
from explain import explain, risk_band
from schemas import (
    BatchRequest,
    BatchResponse,
    CustomerFeatures,
    HealthResponse,
    ModelInfo,
    PredictionResponse,
)

STATE: dict = {"model": None, "started_at": time.time(), "requests": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the artifact once, before the first request is served."""
    started = time.perf_counter()
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. Run `python src/train.py` first."
        )
    STATE["model"] = joblib.load(MODEL_PATH)
    STATE["load_ms"] = (time.perf_counter() - started) * 1000
    STATE["started_at"] = time.time()
    yield
    STATE["model"] = None


app = FastAPI(
    title="Customer Churn Prediction API",
    version=MODEL_VERSION,
    description=MODEL_DESCRIPTION,
    lifespan=lifespan,
)

# The demo page is same-origin, but permissive CORS keeps the API usable from
# a notebook, a separate front end or a hosted preview URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def count_requests(request: Request, call_next):
    STATE["requests"] += 1
    return await call_next(request)


def _predict_one(record: dict, threshold: float, index: int = 0,
                 with_explanation: bool = True) -> PredictionResponse:
    """Score one row.

    Explanations are optional because they cost roughly as much as the
    prediction itself: computing them clears the 5 ms budget under load.
    """
    model = STATE["model"]
    started = time.perf_counter()

    frame = pd.DataFrame([record])
    probability = float(model.predict_proba(frame)[0, 1])
    contributions = explain(model, record) if with_explanation else []
    latency_ms = (time.perf_counter() - started) * 1000

    return PredictionResponse(
        customer_index=index,
        churn_probability=round(probability, 4),
        churn_prediction="Yes" if probability >= threshold else "No",
        threshold=threshold,
        risk_band=risk_band(probability),
        top_contributions=contributions,
        model_version=MODEL_VERSION,
        latency_ms=round(latency_ms, 3),
    )


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    """Readiness probe: is the model actually in memory?"""
    loaded = STATE["model"] is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=MODEL_VERSION,
        uptime_seconds=round(time.time() - STATE["started_at"], 2),
        requests_served=STATE["requests"],
    )


@app.get("/model", response_model=ModelInfo, tags=["model"])
def model_info() -> ModelInfo:
    """The model card, so a client can see what it is calling."""
    if not MODEL_CARD_PATH.exists():
        raise HTTPException(status_code=503, detail="Model card not found.")
    card = json.loads(MODEL_CARD_PATH.read_text())
    return ModelInfo(
        model_name=card["model_name"],
        model_version=card["model_version"],
        algorithm=card["algorithm"],
        trained_at=card["trained_at"],
        training_rows=card["training_rows"],
        features=card["features"],
        default_threshold=card["default_threshold"],
        metrics={
            key: round(float(value), 4)
            for key, value in card["metrics"].items()
            if isinstance(value, (int, float))
        },
        intended_use=card["intended_use"],
        limitations=card["limitations"],
    )


@app.get("/model/manifest", tags=["model"])
def model_manifest() -> dict:
    """Artifact digests, so a deployment can prove which bytes it is serving."""
    if not MANIFEST_PATH.exists():
        raise HTTPException(status_code=503, detail="Manifest not found.")
    return json.loads(MANIFEST_PATH.read_text())


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(
    customer: CustomerFeatures,
    threshold: float = DEFAULT_THRESHOLD,
    explain_result: bool = True,
) -> PredictionResponse:
    """Score a single customer.

    ``threshold`` is a query parameter so a caller can trade precision for
    recall without retraining: 0.5 is neutral, 0.4 maximises F1 on this data.
    """
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    if not 0.0 < threshold < 1.0:
        raise HTTPException(status_code=422, detail="threshold must be between 0 and 1")
    return _predict_one(customer.as_record(), threshold, with_explanation=explain_result)


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(
    batch: BatchRequest,
    threshold: float = DEFAULT_THRESHOLD,
    include_explanations: bool = False,
) -> BatchResponse:
    """Score up to 1,000 customers in one call.

    Rows are transformed and scored in a single vectorised ``predict_proba``
    call rather than one at a time. Explanations are opt-in: computing them for
    every row in a 1,000-customer batch is what makes a bulk endpoint slow.
    """
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    if not 0.0 < threshold < 1.0:
        raise HTTPException(status_code=422, detail="threshold must be between 0 and 1")

    model = STATE["model"]
    started = time.perf_counter()

    records = [customer.as_record() for customer in batch.customers]
    frame = pd.DataFrame(records)

    # One transform, one predict for the whole batch.
    probabilities = model.predict_proba(frame)[:, 1]

    predictions = []
    for index, (record, probability) in enumerate(zip(records, probabilities)):
        probability = float(probability)
        predictions.append(
            PredictionResponse(
                customer_index=index,
                churn_probability=round(probability, 4),
                churn_prediction="Yes" if probability >= threshold else "No",
                threshold=threshold,
                risk_band=risk_band(probability),
                top_contributions=explain(model, record) if include_explanations else [],
                model_version=MODEL_VERSION,
                latency_ms=0.0,
            )
        )

    latency_ms = (time.perf_counter() - started) * 1000
    for prediction in predictions:
        prediction.latency_ms = round(latency_ms, 3)

    flagged = sum(1 for item in predictions if item.churn_prediction == "Yes")
    return BatchResponse(
        predictions=predictions,
        count=len(predictions),
        flagged=flagged,
        flagged_rate=round(flagged / len(predictions), 4),
        model_version=MODEL_VERSION,
        latency_ms=round(latency_ms, 3),
    )


@app.get("/", include_in_schema=False)
def index():
    """Serve the browser demo."""
    page = STATIC_DIR / "index.html"
    if page.exists():
        return FileResponse(page)
    return {"message": "Prediction API is running. See /docs."}


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
