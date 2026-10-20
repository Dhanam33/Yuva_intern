"""Pydantic request and response schemas.

Validation happens at the edge of the application, before any model code runs.
A malformed request therefore returns 422 with a readable explanation instead
of an exception inside scikit-learn.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from config import FEATURE_COLUMNS


class CustomerFeatures(BaseModel):
    """One customer record, as submitted to the prediction endpoint.

    ``extra="forbid"`` turns a typo such as ``tenure_month`` into a clear 422
    rather than a silently imputed median.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "tenure_months": 3.0,
                    "monthly_charges": 95.4,
                    "total_charges": 286.2,
                    "support_tickets": 4,
                    "senior_citizen": 0,
                    "contract": "Month-to-month",
                    "internet_service": "Fiber optic",
                    "payment_method": "Electronic check",
                    "paperless_billing": "Yes",
                    "partner": "No",
                    "dependents": "No",
                }
            ]
        },
    )

    tenure_months: float = Field(..., ge=0, le=120, description="Months as a customer")
    monthly_charges: float = Field(..., ge=0, le=500, description="Current monthly bill")
    total_charges: float | None = Field(
        None, ge=0, le=100000, description="Lifetime billed amount; may be omitted"
    )
    support_tickets: int = Field(..., ge=0, le=100, description="Support tickets opened")
    senior_citizen: int = Field(..., ge=0, le=1, description="1 if a senior citizen")

    contract: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    payment_method: Literal[
        "Electronic check", "Credit card", "Bank transfer", "Mailed check"
    ]
    paperless_billing: Literal["Yes", "No"]
    partner: Literal["Yes", "No"]
    dependents: Literal["Yes", "No"]

    def as_record(self) -> dict:
        """Flatten to the flat dict the pipeline expects."""
        record = self.model_dump()
        return {column: record[column] for column in FEATURE_COLUMNS}


class Contribution(BaseModel):
    """One feature's contribution to a single prediction, in log-odds."""

    feature: str
    value: float | str | None
    contribution: float
    direction: Literal["increases risk", "decreases risk"]


class PredictionResponse(BaseModel):
    """The result of scoring one customer."""

    customer_index: int = Field(0, description="Position of the row in the request")
    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: Literal["Yes", "No"]
    threshold: float
    risk_band: Literal["low", "medium", "high"]
    top_contributions: list[Contribution] = Field(
        default_factory=list, description="Empty unless explanations were requested"
    )
    model_version: str
    latency_ms: float


class BatchRequest(BaseModel):
    """A batch of customers for the bulk endpoint."""

    model_config = ConfigDict(extra="forbid")

    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    """Predictions for a batch, plus aggregate counts."""

    predictions: list[PredictionResponse]
    count: int
    flagged: int
    flagged_rate: float
    model_version: str
    latency_ms: float


class ModelInfo(BaseModel):
    """Everything a client needs to know about the deployed model."""

    model_name: str
    model_version: str
    algorithm: str
    trained_at: str
    training_rows: int
    features: list[str]
    default_threshold: float
    metrics: dict[str, float]
    intended_use: str
    limitations: list[str]


class HealthResponse(BaseModel):
    """Liveness and readiness in one payload."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    requests_served: int
