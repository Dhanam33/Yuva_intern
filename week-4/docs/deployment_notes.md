# Deployment notes

Operational notes for anyone running this service. Companion to `api_reference.md` and
`model_card.md`.

## Starting it

```bash
cd week-4/src
uvicorn service:app --host 0.0.0.0 --port 8000
```

The startup hook loads `outputs/model/customer_churn_pipeline.joblib` **once**. If the
file is missing the process raises and exits rather than serving broken predictions —
which is what you want a container platform to see.

## Container

```bash
docker build -f week-4/Dockerfile -t churn-service .
docker run -p 8000:8000 churn-service
```

The image contains only runtime dependencies (`requirements-serve.txt`): no notebook
stack, no test tooling, no document libraries. It runs as a non-root user and declares a
`HEALTHCHECK` that queries `/health`.

## What breaks first, and what to do

| Symptom | Likely cause | First thing to check |
|---|---|---|
| `/health` reports `degraded` | Artifact missing or failed to load | Startup logs; the path in the error |
| 503 on `/predict` | Same as above | Same |
| Sudden 422s from a previously working client | Field renamed or enum changed upstream | The `detail` array names the field and the reason |
| Latency tripled | Explanations enabled on a large batch | `include_explanations=false` |
| Predictions drift but nothing errors | Real-world change | Compare live feature distributions to `model_card.json` |
| Everything slow after a deploy | Cold start, container not warmed | Route traffic only after `/health` is `ok` |

## Capacity planning

Measured on a single core, single worker:

- ~71 single predictions per second with explanations
- ~10,786 rows per second through the batch endpoint
- 3.0 ms of every request is HTTP and JSON, so scaling the model further is wasted effort

Scale out with more workers rather than up. The model is 5.7 KiB and holds no state, so
replicas are interchangeable.

## Versioning

Three things carry the version `4.0.0`:

1. `config.py` — `MODEL_VERSION`
2. `model_card.json` — written by `train.py`
3. every API response — `model_version` field

To release a new model: retrain, bump the constant, rebuild. The manifest's SHA-256
changes automatically and records exactly which bytes are live.

## What is deliberately missing

- **Authentication.** Add a gateway or API key middleware before exposing this publicly.
- **Rate limiting.** A single client can saturate a single worker.
- **Persistent logging.** Predictions are not stored; monitoring would need that.
- **Model registry.** Artifacts are copied into the image rather than pulled by version.

These are the four things separating a demonstration from a production service, and the
report says so rather than implying the work is finished.
