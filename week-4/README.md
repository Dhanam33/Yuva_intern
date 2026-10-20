# Week 4 — AI Project Deployment & Capstone

An end-to-end machine-learning application: the churn classifier built in Weeks 2–3,
serialised, served over HTTP, tested, explained, measured and documented.

**Report:** [`reports/Week_4_AI_Project_Deployment_Report.docx`](reports/Week_4_AI_Project_Deployment_Report.docx)
**Presentation:** [`reports/Week_4_Capstone_Presentation.pptx`](reports/Week_4_Capstone_Presentation.pptx)
**Notebook:** [`notebooks/week4_deployment.ipynb`](notebooks/week4_deployment.ipynb)

> The model is trained on a reproducible **synthetic** dataset of 1,800 records. It is a
> teaching artifact, not a validated production system, and must not be used for credit,
> pricing, insurance or hiring decisions.

---

## Run it

```bash
cd week-4
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt

python src/train.py          # fit, evaluate, serialise, write model card
python src/benchmark.py      # measure the running service
python src/make_figures.py   # build the 10 figures
python docs/build_report.py  # build the Word report
python docs/build_deck.py    # build the presentation
python -m pytest tests -q    # 18 contract tests

# serve it
cd src && uvicorn service:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000> for the demo page, or `/docs` for the interactive API.

---

## What is in this folder

```text
week-4/
├── README.md
├── requirements.txt              everything, including notebook and docs
├── requirements-serve.txt        only what the container needs
├── Dockerfile
├── data/customer_churn_raw.csv
├── src/
│   ├── config.py                 paths, feature lists, model identity
│   ├── features.py               the shared preprocessor and pipeline
│   ├── schemas.py                Pydantic request/response models
│   ├── train.py                  fit → evaluate → serialise → model card
│   ├── explain.py                exact log-odds decomposition
│   ├── service.py                FastAPI application
│   ├── benchmark.py              drives real HTTP traffic and records latency
│   └── make_figures.py           10 figures from the outputs
├── static/index.html             browser demo
├── tests/test_service.py         18 contract tests
├── notebooks/
│   └── week4_deployment.ipynb    executed walkthrough (46 cells)
├── docs/
│   ├── build_report.py
│   ├── build_deck.py
│   ├── api_reference.md
│   ├── model_card.md
│   ├── deployment_notes.md
│   └── SUBMISSION_DESCRIPTION.md
├── reports/
│   ├── Week_4_AI_Project_Deployment_Report.docx
│   └── Week_4_Capstone_Presentation.pptx
└── outputs/
    ├── model/customer_churn_pipeline.joblib    5.7 KiB
    ├── model/customer_churn_pipeline.pkl       3.7 KiB
    ├── model_card.json
    ├── model_manifest.json                     SHA-256 per file
    ├── final_metrics.csv, confusion_matrix.csv
    ├── evaluation_comparison.csv
    ├── latency_benchmark.csv, latency_raw.csv
    ├── benchmark_summary.json
    ├── threshold_curve.csv
    └── figures/                                10 PNG figures
```

---

## The API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness **and** readiness — reports whether the model is loaded |
| `GET` | `/model` | Model card as JSON |
| `GET` | `/model/manifest` | Artifact digests and load time |
| `POST` | `/predict` | Score one customer, with explanation |
| `POST` | `/predict/batch` | Score up to 1,000, vectorised |
| `GET` | `/docs` | Interactive OpenAPI documentation |

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
  "churn_probability": 0.9753,
  "churn_prediction": "Yes",
  "threshold": 0.4,
  "risk_band": "high",
  "top_contributions": [
    {"feature": "contract = Month-to-month", "contribution": 0.96, "direction": "increases risk"},
    {"feature": "tenure_months", "contribution": 0.55, "direction": "increases risk"}
  ],
  "model_version": "4.0.0",
  "latency_ms": 4.1
}
```

---

## Measured performance

Against a live Uvicorn server, 300 sequential requests, p50:

| Scenario | p50 | Throughput |
|---|---|---|
| Single request with explanation | 13.3 ms | 71 rows/s |
| Single request, no explanation | 7.6 ms | 124 rows/s |
| Batch of 100, vectorised | 9.2 ms | **10,786 rows/s** |
| Model only, no HTTP | 4.6 ms | 203 rows/s |

**Cold start:** 1,356 ms from process launch to first healthy response — almost all of it
importing NumPy, pandas and scikit-learn rather than loading a 5.7 KiB file.

### The defect the benchmark found

The first batch endpoint looped over customers and scored them one at a time. It passed
all 18 tests and returned correct predictions, at **92 rows per second** — barely better
than sending 100 separate requests. Vectorising the `predict_proba` call took it to
**10,786 rows per second**, a 145× improvement, in three lines.

A correctness test cannot find this. Only a measurement can.

---

## Model quality

| Metric | Value | Reading |
|---|---|---|
| ROC-AUC | 0.768 | Ranking quality; useful for prioritising a call list |
| Recall | 0.656 | Catches two thirds of churners |
| Precision | 0.432 | Fewer than half of flagged customers actually churn |
| Accuracy | 0.678 | *Below* the 0.733 do-nothing baseline — by design |

Balanced class weights trade accuracy for recall deliberately. The dummy classifier that
never predicts churn scores higher on accuracy while catching nobody; reporting accuracy
alone would have rejected the most useful model.

---

## Design decisions worth knowing

| Decision | Reason |
|---|---|
| Serialise the whole `Pipeline` | One file, one hash, one version — a stale scaler becomes impossible |
| Validation before the model | A malformed request returns 422 with a sentence, not a 500 with a traceback |
| Explanations are opt-in on batch | They cost 5.1 ms per row, more than the prediction itself |
| Threshold is a query parameter | Choosing it is a business decision, not a modelling one |
| `total_charges` may be null | A new customer genuinely has none; the imputer handles it |
| Round-trip check after saving | Proves the artifact is the model that was evaluated |
| Health checks the model object | A container that lost its model should fail its probe |

---

## Limitations

- **Synthetic data.** Every pattern is a property of the generator.
- **No fairness audit.** Error rates can differ by subgroup and have not been measured.
- **No authentication or rate limiting.** A local demonstration, not a public service.
- **The model will drift.** The model card records the training date for this reason.

Full list in [`docs/model_card.md`](docs/model_card.md).
