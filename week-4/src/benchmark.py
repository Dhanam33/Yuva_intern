#!/usr/bin/env python3
"""Week 4 - measure the deployed service.

Latency claims are only useful if they are measured against a running server,
so this script starts uvicorn on a free port, waits for the health check, drives
real HTTP traffic, and shuts the server down again.

Five scenarios are measured:

    cold start              process launch -> first successful health response
    predict + explanation   one customer per request, explanations on
    predict, no explanation one customer per request, explanations off
    batch of 100            vectorised scoring, explanations off
    model only              the pipeline's predict_proba, with no HTTP at all

Comparing the last one with the HTTP scenarios separates framework overhead
from model cost; comparing the explanation variants prices the explanation.

Usage
-----
    python src/benchmark.py
"""

from __future__ import annotations

import json
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx
import joblib
import pandas as pd

from config import MODEL_PATH, OUTPUT_DIR

SRC_DIR = Path(__file__).resolve().parent
SINGLE_REQUESTS = 300
BATCH_SIZE = 100
BATCH_REQUESTS = 25
BATCH_EXPLAINED_REQUESTS = 10

CUSTOMER = {
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


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def summarise(name: str, latencies: list[float], rows_per_request: int = 1) -> dict:
    ordered = sorted(latencies)
    count = len(ordered)
    total_rows = count * rows_per_request
    total_seconds = sum(ordered) / 1000
    return {
        "scenario": name,
        "requests": count,
        "rows_per_request": rows_per_request,
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[int(0.95 * (count - 1))], 3),
        "p99_ms": round(ordered[int(0.99 * (count - 1))], 3),
        "mean_ms": round(statistics.fmean(ordered), 3),
        "max_ms": round(max(ordered), 3),
        "ms_per_row": round(statistics.median(ordered) / rows_per_request, 3),
        "throughput_rows_per_s": round(total_rows / total_seconds, 1) if total_seconds else None,
    }


def wait_for_health(base_url: str, timeout: float = 40.0) -> float:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        try:
            response = httpx.get(f"{base_url}/health", timeout=1.0)
            if response.status_code == 200:
                return time.perf_counter() - started
        except httpx.HTTPError:
            pass
        time.sleep(0.02)
    raise RuntimeError("server did not become healthy in time")


def time_requests(client, method: str, url: str, repeats: int, **kwargs) -> list[float]:
    latencies = []
    for _ in range(repeats):
        started = time.perf_counter()
        response = client.request(method, url, **kwargs)
        latencies.append((time.perf_counter() - started) * 1000)
        response.raise_for_status()
    return latencies


def main() -> int:
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    print(f"starting uvicorn on {base_url}")

    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "service:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=SRC_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        cold_start = wait_for_health(base_url)
        print(f"cold start (launch -> healthy): {cold_start * 1000:.0f} ms")

        with httpx.Client(base_url=base_url, timeout=30.0) as client:
            for _ in range(20):  # warm-up
                client.post("/predict", json=CUSTOMER)

            single_explained = time_requests(
                client, "POST", "/predict", SINGLE_REQUESTS, json=CUSTOMER
            )
            single_plain = time_requests(
                client, "POST", "/predict?explain_result=false", SINGLE_REQUESTS, json=CUSTOMER
            )

            batch_payload = {"customers": [CUSTOMER] * BATCH_SIZE}
            batch_plain = time_requests(
                client, "POST", "/predict/batch", BATCH_REQUESTS, json=batch_payload
            )
            batch_explained = time_requests(
                client, "POST", "/predict/batch?include_explanations=true",
                BATCH_EXPLAINED_REQUESTS, json=batch_payload
            )

        # Model cost without HTTP, for comparison.
        pipeline = joblib.load(MODEL_PATH)
        frame = pd.DataFrame([CUSTOMER])
        pipeline.predict_proba(frame)
        model_only = []
        for _ in range(SINGLE_REQUESTS):
            started = time.perf_counter()
            pipeline.predict_proba(frame)
            model_only.append((time.perf_counter() - started) * 1000)

        results = [
            summarise("predict + explanation", single_explained, 1),
            summarise("predict, no explanation", single_plain, 1),
            summarise(f"batch of {BATCH_SIZE}", batch_plain, BATCH_SIZE),
            summarise(f"batch of {BATCH_SIZE} + explanations", batch_explained, BATCH_SIZE),
            summarise("model only (no HTTP)", model_only, 1),
        ]

        table = pd.DataFrame(results)
        table.to_csv(OUTPUT_DIR / "latency_benchmark.csv", index=False)

        raw = pd.DataFrame(
            [{"scenario": "predict + explanation", "latency_ms": v} for v in single_explained]
            + [{"scenario": "predict, no explanation", "latency_ms": v} for v in single_plain]
            + [{"scenario": "model only", "latency_ms": v} for v in model_only]
        )
        raw.to_csv(OUTPUT_DIR / "latency_raw.csv", index=False)

        by_name = {row["scenario"]: row for row in results}
        summary = {
            "cold_start_ms": round(cold_start * 1000, 1),
            "single_explained_p50_ms": by_name["predict + explanation"]["p50_ms"],
            "single_plain_p50_ms": by_name["predict, no explanation"]["p50_ms"],
            "single_explained_p95_ms": by_name["predict + explanation"]["p95_ms"],
            "single_plain_p95_ms": by_name["predict, no explanation"]["p95_ms"],
            "explanation_cost_ms": round(
                by_name["predict + explanation"]["p50_ms"]
                - by_name["predict, no explanation"]["p50_ms"], 3
            ),
            "model_only_p50_ms": by_name["model only (no HTTP)"]["p50_ms"],
            "http_overhead_ms": round(
                by_name["predict, no explanation"]["p50_ms"]
                - by_name["model only (no HTTP)"]["p50_ms"], 3
            ),
            "single_rps": by_name["predict + explanation"]["throughput_rows_per_s"],
            "batch_rows_per_s": by_name[f"batch of {BATCH_SIZE}"]["throughput_rows_per_s"],
            "batch_ms_per_row": by_name[f"batch of {BATCH_SIZE}"]["ms_per_row"],
            "batch_explained_rows_per_s": by_name[f"batch of {BATCH_SIZE} + explanations"]["throughput_rows_per_s"],
            "single_ms_per_row": by_name["predict + explanation"]["p50_ms"],
            "batch_size": BATCH_SIZE,
            "requests": SINGLE_REQUESTS,
        }
        summary["batch_speedup"] = round(
            summary["single_ms_per_row"] / summary["batch_ms_per_row"], 1
        )
        (OUTPUT_DIR / "benchmark_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )

        print()
        print(table.to_string(index=False))
        print()
        print(f"explanation cost : {summary['explanation_cost_ms']:.2f} ms/request")
        print(f"HTTP overhead    : {summary['http_overhead_ms']:.2f} ms/request")
        print(f"batch speed-up   : {summary['batch_speedup']:.1f}x per row")
        return 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    sys.exit(main())
