from __future__ import annotations

import asyncio

import httpx

from yuva_ml.api import app


def request(method: str, path: str, **kwargs) -> httpx.Response:
    """Call the ASGI app in-process without opening a network port."""
    async def perform_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(perform_request())


def test_health_endpoint_reports_serialized_model():
    response = request("GET", "/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_prediction_endpoint_accepts_raw_and_missing_fields():
    response = request(
        "POST",
        "/predict",
        json={
            "instances": [
                {
                    "customer_id": "api-test-1",
                    "tenure_months": 4,
                    "monthly_charges": 105.0,
                    "support_tickets": 4,
                    "contract": "Month-to-month",
                    "internet_service": "Fiber optic",
                    "payment_method": "Electronic check",
                },
                {"customer_id": "api-test-2", "contract": "Two year"},
            ]
        },
    )
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert predictions[0]["customer_id"] == "api-test-1"
    assert predictions[0]["prediction"] in {"Yes", "No"}
    assert 0.0 <= predictions[0]["churn_probability"] <= 1.0
    assert predictions[1]["customer_id"] == "api-test-2"
    assert 0.0 <= predictions[1]["churn_probability"] <= 1.0


def test_prediction_request_rejects_empty_batch():
    response = request("POST", "/predict", json={"instances": []})
    assert response.status_code == 422
