"""Contract tests for the prediction API.

These are written as a client would experience the service: through HTTP,
against the same artifact the deployment loads. Run from the repository root:

    python -m pytest week-4/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from service import app  # noqa: E402

VALID = {
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


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------
def test_health_reports_model_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_card_is_served(client):
    body = client.get("/model").json()
    assert body["model_version"] == "4.0.0"
    assert body["training_rows"] > 0
    assert "roc_auc" in body["metrics"]
    assert len(body["features"]) == 11


def test_manifest_lists_artifact_digest(client):
    body = client.get("/model/manifest").json()
    assert body["round_trip_identical"] is True
    assert any(item["sha256"] for item in body["files"])


def test_demo_page_is_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def test_predict_returns_valid_probability(client):
    body = client.post("/predict", json=VALID).json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in {"Yes", "No"}
    assert body["risk_band"] in {"low", "medium", "high"}
    assert body["model_version"] == "4.0.0"


def test_prediction_matches_probability_and_threshold(client):
    body = client.post("/predict", json=VALID).json()
    expected = "Yes" if body["churn_probability"] >= body["threshold"] else "No"
    assert body["churn_prediction"] == expected


def test_high_risk_profile_scores_above_low_risk_profile(client):
    low = {**VALID, "tenure_months": 64, "contract": "Two year",
           "payment_method": "Bank transfer", "support_tickets": 0,
           "internet_service": "DSL", "monthly_charges": 24.0}
    high = client.post("/predict", json=VALID).json()["churn_probability"]
    calm = client.post("/predict", json=low).json()["churn_probability"]
    assert high > calm


def test_threshold_parameter_changes_the_label(client):
    probability = client.post("/predict", json=VALID).json()["churn_probability"]
    strict = client.post("/predict?threshold=0.99", json=VALID).json()
    assert strict["churn_prediction"] == "No"
    assert strict["churn_probability"] == pytest.approx(probability, abs=1e-6)


def test_explanations_are_ranked_by_impact(client):
    contributions = client.post("/predict", json=VALID).json()["top_contributions"]
    assert contributions, "an explanation must not be empty"
    magnitudes = [abs(item["contribution"]) for item in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "field,value",
    [
        ("contract", "Weekly"),
        ("senior_citizen", 7),
        ("monthly_charges", -5),
        ("tenure_months", "not a number"),
    ],
)
def test_invalid_values_are_rejected(client, field, value):
    response = client.post("/predict", json={**VALID, field: value})
    assert response.status_code == 422


def test_unknown_field_is_rejected(client):
    response = client.post("/predict", json={**VALID, "tenure_month": 3})
    assert response.status_code == 422


def test_missing_required_field_is_rejected(client):
    payload = dict(VALID)
    payload.pop("contract")
    assert client.post("/predict", json=payload).status_code == 422


def test_null_total_charges_is_accepted(client):
    payload = {**VALID, "total_charges": None}
    assert client.post("/predict", json=payload).status_code == 200


# --------------------------------------------------------------------------
# Batch
# --------------------------------------------------------------------------
def test_batch_scores_every_row(client):
    body = client.post("/predict/batch", json={"customers": [VALID, VALID, VALID]}).json()
    assert body["count"] == 3
    assert len(body["predictions"]) == 3
    assert 0 <= body["flagged"] <= 3


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422
