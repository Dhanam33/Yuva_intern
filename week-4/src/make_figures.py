#!/usr/bin/env python3
"""Week 4 - figures for the capstone report.

Every figure reads from ``outputs/`` rather than recomputing anything, so a
figure cannot disagree with the table it sits next to. The one exception is the
validation panel, which queries the live app through TestClient so the status
codes shown are the ones the service actually returns.

Usage
-----
    python src/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from config import DEFAULT_THRESHOLD, FIGURE_DIR, OUTPUT_DIR

sys.path.insert(0, str(Path(__file__).resolve().parent))

INK = "#12212E"
MUTED = "#5C6F82"
GRID = "#DDE4EB"
BLUE = "#2C6E9B"
ORANGE = "#D97B29"
GREEN = "#2E8B6F"
RED = "#B5443A"
PURPLE = "#6B4E9B"
PANEL = "#F4F7FA"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": "#B9C5D1",
        "axes.labelcolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


def save(fig, name: str, note: str | None = None) -> None:
    if note:
        # Reserve a strip at the bottom of the canvas, then write the note into
        # it. Without the reservation the note lands on the x-axis label.
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.text(0.012, 0.008, note, fontsize=7, color=MUTED, style="italic",
                 va="bottom", ha="left")
    path = FIGURE_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  {name}")


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / name)


def read_json(name: str) -> dict:
    return json.loads((OUTPUT_DIR / name).read_text())


# --------------------------------------------------------------------------
# 00 - System architecture
# --------------------------------------------------------------------------
def figure_architecture() -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.9)
    ax.axis("off")

    def card(x, y, w, h, title, detail, edge, fill):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.02,rounding_size=0.09",
                linewidth=1.3, edgecolor=edge, facecolor=fill, zorder=2,
            )
        )
        ax.text(x + 0.16, y + h - 0.15, title, fontsize=9.8, fontweight="bold",
                color=INK, va="top", zorder=3)
        ax.text(x + 0.16, y + h - 0.52, detail, fontsize=7.5, color=MUTED,
                va="top", linespacing=1.5, zorder=3)

    def arrow(x0, y0, x1, y1, colour=MUTED, dashed=False):
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                linewidth=1.3, color=colour, zorder=1,
                linestyle="--" if dashed else "-",
                shrinkA=2, shrinkB=2,
            )
        )

    # Row 1 - request path.
    card(0.15, 3.75, 2.35, 1.15, "1   Caller",
         "curl \u00b7 browser UI\nnotebook \u00b7 another service", BLUE, "#E9F1F8")
    card(3.05, 3.75, 2.35, 1.15, "2   Validate",
         "Pydantic schema\n422 before model code runs", BLUE, "#E9F1F8")
    card(5.95, 3.75, 3.9, 1.15, "3   Score",
         "impute \u2192 scale \u2192 one-hot \u2192 LogisticRegression\n"
         "one vectorised predict_proba call", BLUE, "#E9F1F8")

    # Row 2 - serpentine back.
    card(5.95, 2.05, 3.9, 1.2, "4   Explain",
         "log-odds decompose exactly for a linear model:\n"
         "logit(p) = intercept + \u03a3 coefficient \u00d7 feature", GREEN, "#E9F5F0")
    card(0.15, 2.05, 5.4, 1.2, "5   Respond",
         "probability \u00b7 Yes/No at the chosen threshold\n"
         "risk band \u00b7 top contributors \u00b7 latency", GREEN, "#E9F5F0")

    # Supporting artifacts.
    card(0.15, 0.35, 9.7, 1.15, "Artifacts loaded once, before the first request",
         "customer_churn_pipeline.joblib (5.7 KiB, 1 ms cold load, SHA-256 recorded)\n"
         "model_card.json (metrics, intended use, limitations)   \u00b7   "
         "model_manifest.json (digests)", ORANGE, "#FBF1E3")

    arrow(2.5, 4.32, 3.05, 4.32)
    arrow(5.4, 4.32, 5.95, 4.32)
    arrow(7.9, 3.75, 7.9, 3.25)
    arrow(5.7, 2.65, 5.55, 2.65)

    ax.text(0.15, 5.72, "One prediction request, end to end",
            fontsize=13.5, fontweight="bold", color=INK)
    ax.text(0.15, 5.44, "The model is loaded once at startup and served from memory; "
                        "nothing is re-fitted per request.",
            fontsize=8.3, color=MUTED)

    save(fig, "00_system_architecture.png")


# --------------------------------------------------------------------------
# 01 - Where the 12.5 ms goes
# --------------------------------------------------------------------------
def figure_latency_budget() -> None:
    summary = read_json("benchmark_summary.json")

    model = summary["model_only_p50_ms"]
    explain = summary["explanation_cost_ms"]
    http = summary["http_overhead_ms"]
    total = model + explain + http

    fig, ax = plt.subplots(figsize=(9, 2.0))
    segments = [
        ("Model inference", model, BLUE),
        ("Explanation", explain, PURPLE),
        ("HTTP + JSON", http, ORANGE),
    ]

    left = 0.0
    for label, value, colour in segments:
        ax.barh(0, value, left=left, height=0.62, color=colour, edgecolor="white",
                linewidth=1.4)
        share = value / total
        ax.text(left + value / 2, 0, f"{label}\n{value:.1f} ms \u00b7 {share:.0%}",
                ha="center", va="center", color="white", fontsize=8.6,
                fontweight="bold", linespacing=1.5)
        left += value

    ax.set_xlim(0, total * 1.01)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel(f"milliseconds per request   (median total {total:.1f} ms)")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_title("A single prediction spends more time explaining than predicting", pad=12)
    save(fig, "01_latency_budget.png",
         f"median of {summary['requests']} sequential requests against a local uvicorn server")


# --------------------------------------------------------------------------
# 02 - Latency distribution
# --------------------------------------------------------------------------
def figure_latency_distribution() -> None:
    raw = read_csv("latency_raw.csv")

    series = [
        ("Model inference only", "model only", BLUE),
        ("Full request, no explanation", "predict, no explanation", ORANGE),
        ("Full request with explanation", "predict + explanation", PURPLE),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    bins = np.linspace(0, 20, 45)

    for label, key, colour in series:
        values = raw.loc[raw["scenario"] == key, "latency_ms"]
        ax.hist(values, bins=bins, alpha=0.55, label=label, color=colour,
                edgecolor="white", linewidth=0.4)

    for _, key, colour in series:
        p50 = float(np.median(raw.loc[raw["scenario"] == key, "latency_ms"]))
        ax.axvline(p50, color=colour, linestyle="--", linewidth=1.2, alpha=0.9)
        ax.text(p50 - 0.08, ax.get_ylim()[1] * 0.975, f"{p50:.1f}", color=colour,
                fontsize=8, fontweight="bold", rotation=90, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.9))

    ax.set_xlim(0, 20)
    ax.set_xlabel("latency per request (ms)")
    ax.set_ylabel("number of requests")
    ax.set_title("Latency distribution across 300 sequential requests")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    save(fig, "02_latency_distribution.png",
         "dashed lines mark the median of each scenario; x-axis cropped at 20 ms, "
         "a handful of outliers sit beyond it")


# --------------------------------------------------------------------------
# 03 - Throughput
# --------------------------------------------------------------------------
def figure_throughput() -> None:
    table = read_csv("latency_benchmark.csv")
    order = [
        "predict + explanation",
        "predict, no explanation",
        "batch of 100 + explanations",
        "batch of 100",
    ]
    table = table.set_index("scenario").loc[order].reset_index()

    colours = [PURPLE, ORANGE, "#8FA8BD", GREEN]
    labels = ["Single request\nwith explanation", "Single request\nno explanation",
              "Batch of 100\nwith explanations", "Batch of 100\nvectorised"]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    positions = np.arange(len(table))
    values = table["throughput_rows_per_s"].to_numpy()

    bars = ax.barh(positions, values, color=colours, height=0.55, edgecolor="white")
    ax.set_ylim(-0.6, len(table) - 0.4)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.set_xscale("log")
    ax.set_xlim(50, 30000)
    ax.set_xlabel("rows scored per second (log scale)")

    for bar, value in zip(bars, values):
        ax.text(value * 1.15, bar.get_y() + bar.get_height() / 2, f"{value:,.0f}",
                va="center", fontsize=9, fontweight="bold", color=INK)

    ax.set_title("Vectorising the batch endpoint changed throughput by two orders of magnitude")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    save(fig, "03_throughput.png",
         "measured against a local uvicorn server; 100 customers per batch request")


# --------------------------------------------------------------------------
# 04 - Confusion matrix
# --------------------------------------------------------------------------
def figure_confusion_matrix() -> None:
    matrix = read_csv("confusion_matrix.csv")
    metrics = read_csv("final_metrics.csv").set_index("metric")["value"]

    tn = int(matrix.loc[0, "predicted No"])
    fp = int(matrix.loc[0, "predicted Yes"])
    fn = int(matrix.loc[1, "predicted No"])
    tp = int(matrix.loc[1, "predicted Yes"])

    counts = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.imshow(counts, cmap="Blues", alpha=0.88)

    for row in range(2):
        for column in range(2):
            value = counts[row, column]
            share = value / counts.sum()
            colour = "white" if value > counts.max() * 0.6 else INK
            ax.text(column, row - 0.06, f"{value}", ha="center", va="center",
                    fontsize=24, fontweight="bold", color=colour)
            ax.text(column, row + 0.24, f"{share:.1%} of rows", ha="center",
                    va="center", fontsize=8.5, color=colour)

    ax.set_xticks([0, 1], labels=["Predicted\nNo", "Predicted\nYes"])
    ax.set_yticks([0, 1], labels=["Actual\nNo", "Actual\nYes"])
    ax.tick_params(length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        f"Confusion matrix on the 360 held-out rows\n"
        f"ROC-AUC {metrics['roc_auc']:.3f} · recall {metrics['recall']:.3f} · "
        f"precision {metrics['precision']:.3f}",
        pad=14, fontsize=10,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    save(fig, "04_confusion_matrix.png",
         f"threshold {DEFAULT_THRESHOLD:.2f}; {fp} false alarms, {fn} missed churners")


# --------------------------------------------------------------------------
# 05 - ROC and precision-recall
# --------------------------------------------------------------------------
def figure_roc_pr() -> None:
    predictions = read_csv("test_predictions.csv")
    truth = predictions["y_true"].to_numpy()
    probability = predictions["y_probability"].to_numpy()

    fpr, tpr, _ = roc_curve(truth, probability)
    precision, recall, _ = precision_recall_curve(truth, probability)
    auc_value = roc_auc_score(truth, probability)
    ap_value = average_precision_score(truth, probability)
    baseline = truth.mean()

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))

    ax = axes[0]
    ax.plot(fpr, tpr, color=BLUE, linewidth=2.2, label=f"Churn model (AUC {auc_value:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1,
            label="No skill (AUC 0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    ax.plot(recall, precision, color=GREEN, linewidth=2.2,
            label=f"Churn model (AP {ap_value:.3f})")
    ax.axhline(baseline, color=MUTED, linestyle="--", linewidth=1,
               label=f"No skill (AP {baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.suptitle("Ranking quality of the deployed model", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "05_roc_pr.png",
         "the no-skill line on the right is the churn rate (26.7%), not 0.5")


# --------------------------------------------------------------------------
# 06 - Threshold trade-off
# --------------------------------------------------------------------------
def figure_thresholds() -> None:
    predictions = read_csv("test_predictions.csv")
    truth = predictions["y_true"].to_numpy()
    probability = predictions["y_probability"].to_numpy()

    thresholds = np.arange(0.05, 0.96, 0.01)
    rows = []
    for threshold in thresholds:
        predicted = (probability >= threshold).astype(int)
        tp = int(((predicted == 1) & (truth == 1)).sum())
        fp = int(((predicted == 1) & (truth == 0)).sum())
        fn = int(((predicted == 0) & (truth == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"threshold": threshold, "precision": precision, "recall": recall,
                     "f1": f1, "flagged": predicted.mean()})
    curve = pd.DataFrame(rows)
    curve.to_csv(OUTPUT_DIR / "threshold_curve.csv", index=False)

    best = curve.loc[curve["f1"].idxmax()]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(curve["threshold"], curve["precision"], color=BLUE, linewidth=2, label="Precision")
    ax.plot(curve["threshold"], curve["recall"], color=ORANGE, linewidth=2, label="Recall")
    ax.plot(curve["threshold"], curve["f1"], color=GREEN, linewidth=2.2, label="F1")

    ax.axvline(DEFAULT_THRESHOLD, color=MUTED, linestyle=":", linewidth=1.4)
    ax.text(DEFAULT_THRESHOLD + 0.012, 0.02, "default 0.50", fontsize=8,
            color=MUTED, va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.9))

    ax.scatter([best["threshold"]], [best["f1"]], s=70, color=GREEN, zorder=5,
               edgecolor="white", linewidth=1.4)
    ax.annotate(
        f"best F1 {best['f1']:.3f} at {best['threshold']:.2f}\n"
        f"recall {best['recall']:.2f} · precision {best['precision']:.2f}",
        xy=(best["threshold"], best["f1"]), xytext=(0.065, 0.72),
        fontsize=8.3, color=INK,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=GRID),
        arrowprops=dict(arrowstyle="-|>", color=GREEN, linewidth=1.2,
                        connectionstyle="arc3,rad=0.22"),
    )

    ax.set_xlabel("decision threshold")
    ax.set_ylabel("score")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1.02)
    ax.set_title("Where the operating point should sit is a business decision, not a modelling one")
    ax.legend(frameon=False, fontsize=8.8, ncol=3, loc="upper right")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    save(fig, "06_threshold_tradeoff.png",
         "computed from the 360 held-out predictions; nothing is refitted per threshold")


# --------------------------------------------------------------------------
# 07 - Probability distribution by class
# --------------------------------------------------------------------------
def figure_probability_distribution() -> None:
    predictions = read_csv("test_predictions.csv")
    churn = predictions.loc[predictions["y_true"] == 1, "y_probability"]
    retained = predictions.loc[predictions["y_true"] == 0, "y_probability"]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    bins = np.linspace(0, 1, 26)
    ax.hist(retained, bins=bins, color=BLUE, alpha=0.62, label="Actually stayed (267)",
            edgecolor="white", linewidth=0.4)
    ax.hist(churn, bins=bins, color=ORANGE, alpha=0.72, label="Actually churned (93)",
            edgecolor="white", linewidth=0.4)
    ax.axvline(DEFAULT_THRESHOLD, color=RED, linestyle="--", linewidth=1.5)
    ax.text(DEFAULT_THRESHOLD + 0.014, ax.get_ylim()[1] * 0.52, "threshold 0.50",
            fontsize=8.3, color=RED, rotation=90, va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.92))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    ax.set_xlabel("predicted churn probability")
    ax.set_ylabel("number of customers")
    ax.set_title("The classes overlap — which is why no threshold is perfect")
    ax.legend(frameon=False, fontsize=8.8, loc="upper center")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    save(fig, "07_probability_distribution.png",
         "a perfect model would show two separate humps; here they sit on top of each other")


# --------------------------------------------------------------------------
# 08 - Why one customer was flagged
# --------------------------------------------------------------------------
def figure_explanation() -> None:
    import joblib

    from config import MODEL_PATH
    from explain import explain, log_odds_breakdown

    pipeline = joblib.load(MODEL_PATH)
    record = {
        "tenure_months": 14.0,
        "monthly_charges": 72.0,
        "total_charges": 1008.0,
        "support_tickets": 2,
        "senior_citizen": 0,
        "contract": "Month-to-month",
        "internet_service": "DSL",
        "payment_method": "Credit card",
        "paperless_billing": "No",
        "partner": "Yes",
        "dependents": "No",
    }
    contributions = explain(pipeline, record, top_k=8)
    breakdown = log_odds_breakdown(pipeline, record)
    probability = float(pipeline.predict_proba(pd.DataFrame([record]))[0, 1])

    labels = [item["feature"] for item in contributions][::-1]
    values = [item["contribution"] for item in contributions][::-1]
    colours = [RED if value > 0 else GREEN for value in values]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    positions = np.arange(len(values))
    bars = ax.barh(positions, values, color=colours, height=0.62, alpha=0.88)
    ax.set_yticks(positions, labels=labels, fontsize=8.6)
    ax.axvline(0, color=INK, linewidth=1.1)

    for bar, value in zip(bars, values):
        # A negative bar's label goes just right of zero, where the row is
        # empty. Placing it at the bar's far end collides with the axis labels.
        x = value + 0.04 if value > 0 else 0.04
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:+.2f}",
                va="center", ha="left", fontsize=8.3, fontweight="bold",
                color=RED if value > 0 else GREEN)

    low = min(min(values), 0.0)
    high = max(max(values), 0.0)
    ax.set_xlim(low - abs(low) * 0.45 - 0.05, high + abs(high) * 0.28)
    ax.set_xlabel("effect on the log-odds of churn")
    ax.set_title(
        f"Why this customer was flagged: {probability:.1%} churn risk\n"
        f"intercept {breakdown['intercept']:+.2f} "
        f"+ contributions {breakdown['sum_of_contributions']:+.2f} "
        f"= log-odds {breakdown['log_odds']:+.2f}",
        fontsize=10,
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    save(fig, "08_explanation_example.png",
         "red pushes the prediction towards churn, green away from it; "
         "the arithmetic is exact for a linear model")


# --------------------------------------------------------------------------
# 09 - Artifact and validation summary
# --------------------------------------------------------------------------
def figure_artifacts_and_validation() -> None:
    artifacts = read_csv("artifacts.csv")
    manifest = read_json("model_manifest.json")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2),
                             gridspec_kw={"width_ratios": [1, 1.25]})

    ax = axes[0]
    names = artifacts["artifact"].tolist()
    sizes = artifacts["kib"].tolist()
    bars = ax.bar(names, sizes, color=[BLUE, PURPLE], width=0.5)
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, size + 0.15, f"{size:.1f} KiB",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("artifact size (KiB)")
    ax.set_ylim(0, max(sizes) * 1.28)
    ax.set_title("Serialised model size")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.text(0.03, 0.96,
            f"cold load {manifest['load_time_ms']:.1f} ms\n"
            f"round-trip identical: {manifest['round_trip_identical']}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.2, color=MUTED,
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="none", alpha=0.85))

    # Live validation against the app.
    from fastapi.testclient import TestClient

    from service import app

    valid = {
        "tenure_months": 3.0, "monthly_charges": 95.4, "total_charges": 286.2,
        "support_tickets": 4, "senior_citizen": 0, "contract": "Month-to-month",
        "internet_service": "Fiber optic", "payment_method": "Electronic check",
        "paperless_billing": "Yes", "partner": "No", "dependents": "No",
    }
    cases = [
        ("valid request", valid),
        ("unknown contract type", {**valid, "contract": "Weekly"}),
        ("senior_citizen = 7", {**valid, "senior_citizen": 7}),
        ("negative monthly charge", {**valid, "monthly_charges": -5}),
        ("misspelt field name", {**valid, "tenure_month": 3}),
        ("missing contract", {k: v for k, v in valid.items() if k != "contract"}),
        ("null total_charges", {**valid, "total_charges": None}),
    ]

    with TestClient(app) as client:
        outcomes = [
            (label, client.post("/predict", json=payload).status_code)
            for label, payload in cases
        ]

    ax = axes[1]
    ax.axis("off")
    ax.set_title("What the service accepts and rejects", fontsize=10,
                 fontweight="bold", loc="left")

    top = 0.88
    step = 0.115
    ax.text(0.02, top + 0.07, "request", fontsize=8, color=MUTED, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.98, top + 0.07, "status", fontsize=8, color=MUTED, fontweight="bold",
            ha="right", transform=ax.transAxes)

    for index, (label, status) in enumerate(outcomes):
        y = top - index * step
        ok = status == 200
        ax.text(0.02, y, label, fontsize=8.6, va="center", transform=ax.transAxes)
        ax.text(0.98, y, str(status), fontsize=8.8, ha="right", va="center",
                fontweight="bold", color=GREEN if ok else RED,
                transform=ax.transAxes)
        ax.plot([0.02, 0.98], [y - step * 0.42, y - step * 0.42], color=GRID,
                linewidth=0.8, transform=ax.transAxes, clip_on=False)

    save(fig, "09_artifacts_and_validation.png",
         "status codes captured by calling the running application, not written by hand")


def main() -> int:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    print("Build  figures")
    print("-" * 78)
    figure_architecture()
    figure_latency_budget()
    figure_latency_distribution()
    figure_throughput()
    figure_confusion_matrix()
    figure_roc_pr()
    figure_thresholds()
    figure_probability_distribution()
    figure_explanation()
    figure_artifacts_and_validation()
    print("-" * 78)
    print(f"{len(list(FIGURE_DIR.glob('*.png')))} figures written to {FIGURE_DIR.name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
