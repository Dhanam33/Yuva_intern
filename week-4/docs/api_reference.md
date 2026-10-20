# API reference

Base URL in development: `http://localhost:8000`
Interactive documentation: [`/docs`](http://localhost:8000/docs) (generated from the Pydantic schemas)

---

## `GET /health`

Liveness **and** readiness. Returns `degraded` if the model object is not in memory,
which is the condition that actually matters — a container whose model failed to load
should not pass its probe.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "4.0.0",
  "uptime_seconds": 42.17,
  "requests_served": 19
}
```

---

## `GET /model`

The model card, so a client can see what it is calling without reading the source.

```json
{
  "model_name": "customer-churn-classifier",
  "model_version": "4.0.0",
  "algorithm": "LogisticRegression (scikit-learn Pipeline)",
  "trained_at": "2026-10-20T13:29:42+00:00",
  "training_rows": 1440,
  "features": ["tenure_months", "monthly_charges", "..."],
  "default_threshold": 0.5,
  "metrics": {"roc_auc": 0.7675, "recall": 0.6563, "precision": 0.4315, "...": 0},
  "intended_use": "Ranking existing customers by churn risk ...",
  "limitations": ["Trained on synthetic teaching data ...", "..."]
}
```

---

## `GET /model/manifest`

Provenance for the artifact bytes, so a deployment can prove which model it is serving.

```json
{
  "model_name": "customer-churn-classifier",
  "model_version": "4.0.0",
  "generated_at": "2026-10-20T13:29:42+00:00",
  "files": [
    {"path": "customer_churn_pipeline.joblib", "bytes": 5799,
     "sha256": "5f3a...c91e"},
    {"path": "customer_churn_pipeline.pkl", "bytes": 3809,
     "sha256": "b71d...04ac"}
  ],
  "load_time_ms": 1.021,
  "round_trip_identical": true,
  "max_abs_delta": 0.0
}
```

---

## `POST /predict`

Score one customer. Returns a probability, a decision at the requested threshold, a risk
band, the top contributions and the server-side latency.

### Query parameters

| Name | Type | Default | Effect |
|---|---|---|---|
| `threshold` | float in (0, 1) | `0.5` | Decision boundary. The probability is unaffected. |
| `explain_result` | bool | `true` | Set `false` to save roughly 5 ms. |

### Request body

| Field | Type | Required | Bounds |
|---|---|---|---|
| `tenure_months` | float | yes | 0–120 |
| `monthly_charges` | float | yes | 0–500 |
| `total_charges` | float or null | no | 0–100000 |
| `support_tickets` | int | yes | 0–100 |
| `senior_citizen` | int | yes | 0 or 1 |
| `contract` | enum | yes | `Month-to-month`, `One year`, `Two year` |
| `internet_service` | enum | yes | `DSL`, `Fiber optic`, `No` |
| `payment_method` | enum | yes | `Electronic check`, `Credit card`, `Bank transfer`, `Mailed check` |
| `paperless_billing` | enum | yes | `Yes`, `No` |
| `partner` | enum | yes | `Yes`, `No` |
| `dependents` | enum | yes | `Yes`, `No` |

Unknown fields are **rejected**, not ignored — a typo should be an error, not a silently
imputed median.

### Example

```bash
curl -X POST "http://localhost:8000/predict?threshold=0.4" \
  -H "Content-Type: application/json" \
  -d '{"tenure_months":3,"monthly_charges":95.4,"total_charges":286.2,
       "support_tickets":4,"senior_citizen":0,"contract":"Month-to-month",
       "internet_service":"Fiber optic","payment_method":"Electronic check",
       "paperless_billing":"Yes","partner":"No","dependents":"No"}'
```

```json
{
  "customer_index": 0,
  "churn_probability": 0.9753,
  "churn_prediction": "Yes",
  "threshold": 0.4,
  "risk_band": "high",
  "top_contributions": [
    {"feature": "contract = Month-to-month", "value": "Month-to-month",
     "contribution": 0.96, "direction": "increases risk"},
    {"feature": "tenure_months", "value": 3.0,
     "contribution": 0.55, "direction": "increases risk"}
  ],
  "model_version": "4.0.0",
  "latency_ms": 4.1
}
```

Risk bands: `low` below 0.15, `medium` below 0.40, `high` at or above 0.40.

### Errors

| Status | When | Body |
|---|---|---|
| 422 | Validation failure | `{"detail": [{"loc": ["body", "contract"], "msg": "Input should be ..."}]}` |
| 422 | Threshold outside (0, 1) | `{"detail": "threshold must be between 0 and 1"}` |
| 503 | Model not loaded | `{"detail": "Model is not loaded."}` |

---

## `POST /predict/batch`

Score up to 1,000 customers in one request. Rows are transformed and scored in a single
vectorised `predict_proba` call.

### Query parameters

| Name | Type | Default | Effect |
|---|---|---|---|
| `threshold` | float in (0, 1) | `0.5` | Applied to every row. |
| `include_explanations` | bool | `false` | Adds ~5 ms **per row**. Leave off for bulk scoring. |

### Example

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"customers":[{ ... }, { ... }]}'
```

```json
{
  "predictions": [{"customer_index": 0, "churn_probability": 0.9753, "...": 0}],
  "count": 2,
  "flagged": 1,
  "flagged_rate": 0.5,
  "model_version": "4.0.0",
  "latency_ms": 3.2
}
```

Empty batches are rejected with 422. Batches above 1,000 rows are rejected before any
allocation happens.

---

## Measured performance

300 sequential requests against a local Uvicorn server:

| Scenario | p50 | p95 | Throughput |
|---|---|---|---|
| `POST /predict` with explanation | 13.32 ms | 18.55 ms | 71 rows/s |
| `POST /predict?explain_result=false` | 7.63 ms | 9.54 ms | 124 rows/s |
| `POST /predict/batch` (100 rows) | 9.18 ms | 9.67 ms | 10,786 rows/s |
| `POST /predict/batch` (100 rows, explanations) | 535 ms | 545 ms | 186 rows/s |

All numbers are reproducible with `python src/benchmark.py`.
