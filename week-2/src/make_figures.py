#!/usr/bin/env python3
"""Week 2 - build every figure from the saved training outputs.

Figures are generated from the CSV/JSON files written by ``train_models.py``
rather than from live model objects, so they can be rebuilt without retraining.

    python src/train_models.py     # first
    python src/make_figures.py     # then
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from sklearn.metrics import precision_recall_curve, roc_curve

WEEK_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = WEEK_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

NAVY = "#17324D"
TEAL = "#237A7A"
CORAL = "#D17A61"
BLUE = "#5577AA"
GREEN = "#5B8E7D"
AMBER = "#E0A458"
GREY = "#8A97A5"
PURPLE = "#7D6B9E"

MODEL_COLORS = {
    "Logistic Regression": "#237A7A",
    "Random Forest": "#5B8E7D",
    "Decision Tree": "#D17A61",
    "K-Nearest Neighbors": "#5577AA",
    "Dummy (most frequent)": "#B8C2CC",
    "Dummy (stratified)": "#D6DDE5",
    "Linear Regression": "#237A7A",
    "Random Forest Regressor": "#5B8E7D",
    "Decision Tree Regressor": "#D17A61",
    "K-Nearest Neighbors Regressor": "#5577AA",
    "Dummy (mean)": "#B8C2CC",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": NAVY,
        "axes.labelsize": 9,
        "axes.edgecolor": "#C9D3DC",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E3E9EF",
        "grid.linewidth": 0.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "xtick.color": "#44525F",
        "ytick.color": "#44525F",
        "legend.frameon": False,
    }
)


def save(fig: plt.Figure, name: str, *, tight: bool = True) -> Path:
    if tight:
        fig.tight_layout()
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"    {name}")
    return path


def load_summary() -> dict:
    return json.loads((OUTPUT_DIR / "week2_summary.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# 1. Workflow overview
# --------------------------------------------------------------------------
def figure_workflow() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 5.4))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    ax.text(0.4, 4.95, "Supervised learning workflow", fontsize=18, fontweight="bold", color=NAVY)
    ax.text(
        0.4, 4.55,
        "One held-out split · identical preprocessing for every model · one honest evaluation",
        fontsize=10, color=TEAL,
    )

    stages = [
        ("1  Split", "80% train / 20% test\nstratified on churn", BLUE, "#E9EFF6"),
        ("2  Fit", "impute → encode → model\nfitted on TRAIN only", TEAL, "#E6F1F1"),
        ("3  Predict", "apply the fitted pipeline\nto unseen test rows", AMBER, "#FBF1E3"),
        ("4  Evaluate", "accuracy, precision, recall,\nF1, ROC-AUC, R²", CORAL, "#F8ECE7"),
    ]

    width, height = 2.55, 1.75
    xs = [0.55, 3.55, 6.55, 9.55]
    y = 2.15

    for (title, detail, accent, fill), x in zip(stages, xs):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), width, height,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=fill, edgecolor=accent, linewidth=1.6,
            )
        )
        ax.text(x + 0.2, y + height - 0.35, title, fontsize=12, fontweight="bold", color=accent)
        ax.text(x + 0.2, y + height / 2 - 0.35, detail, fontsize=8.6, color="#3F4E5C", linespacing=1.5)
        if x != xs[-1]:
            ax.add_patch(
                FancyArrowPatch(
                    (x + width + 0.05, y + height / 2),
                    (x + width + 0.42, y + height / 2),
                    arrowstyle="-|>", mutation_scale=15, color=GREY, linewidth=1.6,
                )
            )

    ax.add_patch(
        Rectangle((0.55, 1.15), 11.55, 0.62, facecolor="#F3F7FA", edgecolor="none")
    )
    ax.text(
        0.75, 1.46,
        "Cross-validation runs inside the TRAIN box only.  The test split is scored once, at the end, "
        "so the reported numbers are not the numbers used to pick the model.",
        fontsize=8.8, color=NAVY, va="center", style="italic",
    )
    ax.text(
        0.55, 0.72,
        "Two tasks:  classification (churn: yes/no)  ·  regression (monthly charges: a number)",
        fontsize=9.6, color=TEAL, fontweight="bold",
    )
    ax.text(
        0.55, 0.35,
        "Fixed random seed 42 · every pipeline refits its preprocessing on training data only",
        fontsize=7.8, color=GREY,
    )
    save(fig, "00_workflow_overview.png")


# --------------------------------------------------------------------------
# 2. Class imbalance
# --------------------------------------------------------------------------
def figure_class_balance(summary: dict) -> None:
    data = pd.read_csv(WEEK_DIR / "data" / "customer_churn_raw.csv")
    counts = data["churn"].value_counts().reindex(["No", "Yes"]).astype(int)
    negative_rate = counts["No"] / counts.sum()

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})

    bars = axes[0].bar(
        counts.index, counts.values, color=[GREEN, CORAL], width=0.55, edgecolor="white"
    )
    axes[0].bar_label(
        bars,
        labels=[f"{value:,}\n({value / counts.sum():.1%})" for value in counts.values],
        padding=5, fontsize=10, fontweight="bold", color=NAVY,
    )
    axes[0].set_title("Target distribution")
    axes[0].set_ylabel("Customers")
    axes[0].set_ylim(0, counts.max() * 1.25)
    axes[0].grid(axis="x", visible=False)

    # Accuracy of a constant "No" prediction, versus a real model.
    real = [row for row in summary["classification"]["models"] if not row["model"].startswith("Dummy")]
    names = ["Always predict\n'No'"] + [row["model"] for row in real]
    accuracy = [negative_rate] + [row["accuracy"] for row in real]
    best_roc = max(real, key=lambda row: row["roc_auc"])["model"]
    colors = [GREY] + [
        CORAL if row["model"] == best_roc else BLUE for row in real
    ]

    bars = axes[1].bar(range(len(names)), accuracy, color=colors, width=0.62, edgecolor="white")
    axes[1].bar_label(bars, labels=[f"{value:.3f}" for value in accuracy], padding=3, fontsize=8.5)
    axes[1].axhline(negative_rate, color=GREY, linestyle="--", linewidth=1.3)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, fontsize=7.8, rotation=18, ha="right")
    axes[1].set_title("Accuracy alone is misleading here")
    axes[1].set_ylabel("Test accuracy")
    axes[1].set_ylim(0, 0.95)
    axes[1].grid(axis="x", visible=False)
    axes[1].text(
        0.02, 0.88,
        "A model that never predicts churn already scores "
        f"{negative_rate:.3f}.\nAnything below that line is worse than doing nothing.",
        transform=axes[1].transAxes, fontsize=8, color=CORAL, style="italic",
    )

    fig.suptitle(
        f"Class imbalance: {counts['Yes'] / counts.sum():.1%} of customers churn",
        fontsize=13, fontweight="bold", color=NAVY,
    )
    save(fig, "01_class_balance.png")


# --------------------------------------------------------------------------
# 3. Model comparison
# --------------------------------------------------------------------------
def figure_model_comparison(summary: dict) -> None:
    metrics = pd.read_csv(OUTPUT_DIR / "classification_metrics.csv")
    real = metrics[~metrics["model"].str.startswith("Dummy")].copy()
    real = real.sort_values("roc_auc", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9), gridspec_kw={"width_ratios": [1.35, 1]})

    # Left: the metrics that describe the ranking and the balanced view.
    core = ["balanced_accuracy", "f1", "roc_auc"]
    core_labels = ["Balanced accuracy", "F1", "ROC-AUC"]
    core_colors = [BLUE, PURPLE, TEAL]
    y = np.arange(len(real))
    height = 0.24
    for index, (column, label, color) in enumerate(zip(core, core_labels, core_colors)):
        offset = (index - (len(core) - 1) / 2) * height
        bars = axes[0].barh(y + offset, real[column], height, label=label, color=color,
                            edgecolor="white", linewidth=0.4)
        axes[0].bar_label(bars, labels=[f"{value:.2f}" for value in real[column]],
                          padding=2, fontsize=7.4, color="#33414F")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(real["model"])
    axes[0].set_xlim(0, 0.95)
    axes[0].set_xlabel("Score")
    axes[0].set_title("Ranking and balanced performance")
    axes[0].legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.20), fontsize=8.2)
    axes[0].grid(axis="y", visible=False)

    # Right: the trade-off pair that drives every real decision.
    width = 0.36
    positions = np.arange(len(real))
    bars_a = axes[1].bar(positions - width / 2, real["precision"], width, label="Precision", color=AMBER)
    bars_r = axes[1].bar(positions + width / 2, real["recall"], width, label="Recall", color=CORAL)
    axes[1].bar_label(bars_a, labels=[f"{value:.2f}" for value in real["precision"]], padding=2, fontsize=7.6)
    axes[1].bar_label(bars_r, labels=[f"{value:.2f}" for value in real["recall"]], padding=2, fontsize=7.6)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([name.replace(" ", "\n") for name in real["model"]], fontsize=7.8)
    axes[1].set_ylim(0, 0.85)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Precision and recall trade off")
    axes[1].legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.20), fontsize=8.2)
    axes[1].grid(axis="x", visible=False)

    fig.suptitle("Four classifiers on identical preprocessing — held-out test split",
                 fontsize=13, fontweight="bold", color=NAVY)
    save(fig, "02_model_comparison.png", tight=False)

    # Highlight where the "best" model differs by metric.
    best_accuracy = real.loc[real["accuracy"].idxmax(), "model"]
    best_roc = real.loc[real["roc_auc"].idxmax(), "model"]

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    width = 0.36
    positions = np.arange(len(real))
    ax.bar(positions - width / 2, real["accuracy"], width, label="Accuracy", color=GREY)
    ax.bar(positions + width / 2, real["recall"], width, label="Recall (churn caught)", color=CORAL)
    ax.set_xticks(positions)
    ax.set_xticklabels(real["model"], rotation=12, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.85)
    ax.set_title("Accuracy and recall disagree about which model is best")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    ax.text(
        0.02, 0.95,
        f"Highest accuracy: {best_accuracy}\nHighest ROC-AUC: {best_roc}\n"
        "No single metric identifies a winner.",
        transform=ax.transAxes, va="top", fontsize=8.4, color=NAVY,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F3F7FA", "edgecolor": "none"},
    )
    save(fig, "03_accuracy_vs_recall.png")


# --------------------------------------------------------------------------
# 4. Confusion matrices
# --------------------------------------------------------------------------
def figure_confusion_matrices(summary: dict) -> None:
    models = [row for row in summary["classification"]["models"] if not row["model"].startswith("Dummy")]

    fig, axes = plt.subplots(1, len(models), figsize=(13.4, 3.9))
    for ax, row in zip(axes, models):
        matrix = np.array(
            [
                [row["true_negative"], row["false_positive"]],
                [row["false_negative"], row["true_positive"]],
            ]
        )
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max() * 1.25)

        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                value = matrix[i, j]
                ax.text(
                    j, i - 0.13, f"{labels[i][j]}\n{value}",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if value > matrix.max() * 0.55 else NAVY,
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No", "Yes"], fontsize=8.5)
        ax.set_yticklabels(["No", "Yes"], fontsize=8.5)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)
        ax.set_title(
            f"{row['model']}\nacc {row['accuracy']:.3f} · recall {row['recall']:.3f}",
            fontsize=9.5,
        )
        ax.grid(False)

    fig.suptitle(
        "Confusion matrices on the 360 held-out customers",
        fontsize=13, fontweight="bold", color=NAVY, y=1.04,
    )
    fig.text(
        0.5, -0.06,
        "FP = retention offer wasted on a loyal customer.   FN = churner silently lost.   "
        "The two have different costs, so the 'best' matrix depends on the business.",
        ha="center", fontsize=8, color=GREY,
    )
    save(fig, "04_confusion_matrices.png", tight=False)


# --------------------------------------------------------------------------
# 5. ROC and precision-recall curves
# --------------------------------------------------------------------------
def figure_roc_curves(summary: dict) -> None:
    predictions = pd.read_csv(OUTPUT_DIR / "classification_predictions.csv")
    actual = predictions["actual"].to_numpy()

    models = [row for row in summary["classification"]["models"] if not row["model"].startswith("Dummy")]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))

    for row in models:
        column = f"prob::{row['model']}"
        if column not in predictions:
            continue
        probability = predictions[column].to_numpy()
        fpr, tpr, _ = roc_curve(actual, probability)
        axes[0].plot(
            fpr, tpr, linewidth=2.1,
            color=MODEL_COLORS.get(row["model"], BLUE),
            label=f"{row['model']} (AUC {row['roc_auc']:.3f})",
        )

        precision, recall, _ = precision_recall_curve(actual, probability)
        axes[1].plot(
            recall, precision, linewidth=2.1,
            color=MODEL_COLORS.get(row["model"], BLUE),
            label=f"{row['model']} (AP {row['pr_auc']:.3f})",
        )

    axes[0].plot([0, 1], [0, 1], linestyle="--", color=GREY, linewidth=1.3, label="Random (AUC 0.5)")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curves")
    axes[0].legend(loc="lower right", fontsize=8.2)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)

    baseline = actual.mean()
    axes[1].axhline(baseline, linestyle="--", color=GREY, linewidth=1.3, label=f"No-skill ({baseline:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–recall curves")
    axes[1].legend(loc="upper right", fontsize=8.2)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)

    fig.suptitle(
        "Ranking quality: ROC-AUC and the more informative PR-AUC",
        fontsize=13, fontweight="bold", color=NAVY,
    )
    save(fig, "05_roc_and_pr_curves.png")


# --------------------------------------------------------------------------
# 6. Threshold trade-off
# --------------------------------------------------------------------------
def figure_threshold(summary: dict) -> None:
    thresholds = pd.read_csv(OUTPUT_DIR / "threshold_analysis.csv")
    model = summary["classification"]["threshold_analysis"]["model"]

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.plot(thresholds["threshold"], thresholds["precision"], marker="o", markersize=3.4, color=BLUE, label="Precision")
    ax.plot(thresholds["threshold"], thresholds["recall"], marker="o", markersize=3.4, color=CORAL, label="Recall")
    ax.plot(thresholds["threshold"], thresholds["f1"], marker="o", markersize=3.4, color=TEAL, label="F1")
    ax.axvline(0.5, linestyle="--", color=GREY, linewidth=1.4, label="Default threshold 0.5")

    best_f1_row = thresholds.loc[thresholds["f1"].idxmax()]
    ax.scatter(
        [best_f1_row["threshold"]], [best_f1_row["f1"]],
        s=110, facecolor="none", edgecolor=NAVY, linewidth=1.8, zorder=5,
        label=f"Best F1 at {best_f1_row['threshold']:.2f}",
    )

    ax.set_xlabel("Decision threshold (probability above which a customer is predicted to churn)")
    ax.set_ylabel("Score")
    ax.set_title(f"Choosing a threshold — {model}")
    ax.set_ylim(0, 1.02)
    ax.legend(ncols=2, loc="lower left", fontsize=8.4)
    ax.text(
        0.02, 0.97,
        "0.5 is a convention, not an optimum. Moving left catches more churners\n"
        "and raises false alarms; moving right does the opposite.",
        transform=ax.transAxes, va="top", fontsize=8.2, color=NAVY,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F3F7FA", "edgecolor": "none"},
    )
    save(fig, "06_threshold_tradeoff.png")


# --------------------------------------------------------------------------
# 7. Class weighting
# --------------------------------------------------------------------------
def figure_class_weighting() -> None:
    table = pd.read_csv(OUTPUT_DIR / "class_weight_comparison.csv")

    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    positions = np.arange(len(table))
    width = 0.2
    series = [
        ("accuracy", "Accuracy", GREY),
        ("precision", "Precision", BLUE),
        ("recall", "Recall", CORAL),
        ("f1", "F1", TEAL),
    ]
    for index, (column, label, color) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        ax.bar(positions + offset, table[column], width, label=label, color=color)

    ax.set_xticks(positions)
    ax.set_xticklabels(table["setting"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.95)
    ax.set_title("What class weighting actually buys")
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8.4)
    ax.grid(axis="x", visible=False)

    unweighted = table.iloc[0]
    weighted = table.iloc[1]
    ax.annotate(
        "",
        xy=(1.0, weighted["recall"]),
        xytext=(0.0, unweighted["recall"]),
        arrowprops=dict(arrowstyle="->", color=CORAL, linewidth=1.6, connectionstyle="arc3,rad=-0.25"),
    )
    ax.text(
        0.5, 0.86,
        f"Recall rises {unweighted['recall']:.2f} → {weighted['recall']:.2f}, "
        f"accuracy falls {unweighted['accuracy']:.2f} → {weighted['accuracy']:.2f}",
        ha="center", fontsize=9, color=CORAL, fontweight="bold",
    )
    save(fig, "07_class_weighting.png")


# --------------------------------------------------------------------------
# 8. Feature importance
# --------------------------------------------------------------------------
def figure_feature_importance() -> None:
    importance = pd.read_csv(OUTPUT_DIR / "feature_importance.csv").head(12)
    importance = importance.sort_values("importance")

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    colors = [TEAL if value >= importance["importance"].median() else "#A8C4C4" for value in importance["importance"]]
    bars = ax.barh(importance["feature"], importance["importance"], color=colors, height=0.66)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in importance["importance"]], padding=4, fontsize=8.2)
    ax.set_xlabel("Mean decrease in impurity (Random Forest)")
    ax.set_title("Which features the forest actually uses")
    ax.set_xlim(0, importance["importance"].max() * 1.18)
    ax.grid(axis="y", visible=False)
    fig.text(
        0.01, -0.02,
        "Impurity importance is biased toward high-cardinality and continuous features. It ranks usefulness inside this model, "
        "not causal effect.",
        fontsize=7.4, color=GREY,
    )
    save(fig, "08_feature_importance.png")


# --------------------------------------------------------------------------
# 9. Cross-validation
# --------------------------------------------------------------------------
def figure_cross_validation() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))

    for ax, filename, metric, title, ylabel in [
        (axes[0], "classification_cv_results.csv", "cv_roc_auc", "Classification: 5-fold CV ROC-AUC", "ROC-AUC"),
        (axes[1], "regression_cv_results.csv", "cv_r2", "Regression: 5-fold CV R²", "R²"),
    ]:
        table = pd.read_csv(OUTPUT_DIR / filename).sort_values(f"{metric}_mean")
        bars = ax.barh(
            table["model"], table[f"{metric}_mean"],
            xerr=table[f"{metric}_std"], height=0.6,
            color=[MODEL_COLORS.get(name, BLUE) for name in table["model"]],
            error_kw={"ecolor": NAVY, "capsize": 4, "linewidth": 1.2},
        )
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in table[f"{metric}_mean"]], padding=8, fontsize=8.4)
        ax.set_xlabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(0, 1.25)
        ax.grid(axis="y", visible=False)
        ax.margins(y=0.05)

    fig.suptitle("Cross-validation on the training split — the numbers used to choose a model",
                 fontsize=12.5, fontweight="bold", color=NAVY)
    fig.text(
        0.5, -0.04,
        "Error bars are one standard deviation across folds. Overlapping bars mean the models are "
        "not reliably different on this sample size.",
        ha="center", fontsize=8, color=GREY,
    )
    save(fig, "09_cross_validation.png", tight=False)


# --------------------------------------------------------------------------
# 10. Regression
# --------------------------------------------------------------------------
def figure_regression(summary: dict) -> None:
    metrics = pd.read_csv(OUTPUT_DIR / "regression_metrics.csv")
    real = metrics[~metrics["model"].str.startswith("Dummy")].sort_values("r2")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))

    bars = axes[0].barh(
        real["model"], real["r2"], height=0.6,
        color=[MODEL_COLORS.get(name, BLUE) for name in real["model"]],
    )
    axes[0].bar_label(bars, labels=[f"{value:.3f}" for value in real["r2"]], padding=6, fontsize=9)
    axes[0].set_xlabel("R² on the held-out test split")
    axes[0].set_title("Variance explained")
    axes[0].set_xlim(0, 1.0)
    axes[0].grid(axis="y", visible=False)

    positions = np.arange(len(real))
    axes[1].bar(positions, real["mae"], width=0.55, color=CORAL)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(real["model"], rotation=14, ha="right", fontsize=8)
    axes[1].set_ylabel("Mean absolute error (currency units)")
    axes[1].set_title("Typical error in the prediction")
    axes[1].grid(axis="x", visible=False)
    for position, value in zip(positions, real["mae"]):
        axes[1].text(position, value + 0.2, f"{value:.2f}", ha="center", fontsize=9, fontweight="bold", color=NAVY)

    fig.suptitle(
        "Regression: predicting monthly charges",
        fontsize=13, fontweight="bold", color=NAVY,
    )
    save(fig, "10_regression_comparison.png")


def figure_regression_fit(summary: dict) -> None:
    predictions = pd.read_csv(OUTPUT_DIR / "regression_predictions.csv")
    linear = summary["regression"]["models"]
    linear_row = next(row for row in linear if row["model"] == "Linear Regression")

    actual = predictions["actual"].to_numpy()
    predicted = predictions["pred::Linear Regression"].to_numpy()
    residual = actual - predicted

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

    axes[0].scatter(actual, predicted, s=13, alpha=0.45, color=TEAL, edgecolor="none")
    low, high = float(min(actual.min(), predicted.min())), float(max(actual.max(), predicted.max()))
    axes[0].plot([low, high], [low, high], linestyle="--", color=CORAL, linewidth=1.6, label="Perfect prediction")
    axes[0].set_xlabel("Actual monthly charges")
    axes[0].set_ylabel("Predicted monthly charges")
    axes[0].set_title(
        f"Linear Regression — R² {linear_row['r2']:.3f}, MAE {linear_row['mae']:.2f}"
    )
    axes[0].legend(loc="lower right", fontsize=8.4)
    axes[0].text(
        0.03, 0.97,
        "Predictions form horizontal bands because\nthe inputs are almost entirely categorical:\n"
        "customers sharing a contract and service type\nreceive nearly the same predicted charge.",
        transform=axes[0].transAxes, va="top", fontsize=7.6, color=NAVY,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#F3F7FA", "edgecolor": "none"},
    )

    axes[1].hist(residual, bins=34, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.5)
    axes[1].axvline(0, color=CORAL, linewidth=1.7, label="Zero error")
    axes[1].set_xlabel("Residual (actual − predicted)")
    axes[1].set_ylabel("Customers")
    axes[1].set_title("Residual distribution")
    axes[1].legend(loc="upper right", fontsize=8.4)
    axes[1].text(
        0.02, 0.95,
        f"Mean residual: {residual.mean():+.2f}\n"
        f"Errors are centred near zero, so the\nmodel is not systematically biased.\n"
        f"Mean actual: {actual.mean():.1f}",
        transform=axes[1].transAxes, va="top", fontsize=8, color=NAVY,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#F3F7FA", "edgecolor": "none"},
    )

    fig.suptitle("Regression diagnostics on 345 held-out customers",
                 fontsize=12.5, fontweight="bold", color=NAVY)
    save(fig, "11_regression_diagnostics.png")


# --------------------------------------------------------------------------
# 11. Overfitting demonstration
# --------------------------------------------------------------------------
def figure_overfitting() -> None:
    """Show train vs test score as tree depth grows — the classic variance story."""
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score

    WEEK_NUMERIC = [
        "tenure_months", "monthly_charges", "total_charges", "support_tickets", "senior_citizen",
    ]
    WEEK_CATEGORICAL = [
        "contract", "internet_service", "payment_method", "paperless_billing", "partner", "dependents",
    ]

    data = pd.read_csv(WEEK_DIR / "data" / "customer_churn_raw.csv")
    features = data[WEEK_NUMERIC + WEEK_CATEGORICAL]
    target = data["churn"].map({"No": 0, "Yes": 1}).astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=42
    )

    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), WEEK_NUMERIC),
            ("cat", Pipeline([
                ("i", SimpleImputer(strategy="most_frequent")),
                ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), WEEK_CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )

    depths = list(range(1, 19))
    rows = []
    for depth in depths:
        pipeline = Pipeline(
            [
                ("pre", preprocessor),
                ("model", DecisionTreeClassifier(
                    max_depth=depth, class_weight="balanced", random_state=42
                )),
            ]
        )
        pipeline.fit(x_train, y_train)
        rows.append(
            {
                "depth": depth,
                "train_accuracy": accuracy_score(y_train, pipeline.predict(x_train)),
                "test_accuracy": accuracy_score(y_test, pipeline.predict(x_test)),
                "train_roc_auc": roc_auc_score(y_train, pipeline.predict_proba(x_train)[:, 1]),
                "test_roc_auc": roc_auc_score(y_test, pipeline.predict_proba(x_test)[:, 1]),
            }
        )

    curve = pd.DataFrame(rows)
    curve.to_csv(OUTPUT_DIR / "tree_depth_curve.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for ax, metric, title in [
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "roc_auc", "ROC-AUC"),
    ]:
        ax.plot(curve["depth"], curve[f"train_{metric}"], marker="o", markersize=3.6, color=TEAL, label="Training set")
        ax.plot(curve["depth"], curve[f"test_{metric}"], marker="o", markersize=3.6, color=CORAL, label="Held-out test set")
        ax.set_xlabel("Maximum tree depth")
        ax.set_ylabel(title)
        ax.set_title(f"Decision Tree — {title}")
        ax.legend(fontsize=8.4, loc="best")
        ax.set_xticks(range(1, 19, 2))
        ax.set_ylim(0.5, 1.02)

    fig.suptitle("Overfitting, made visible: training score climbs while test score stalls",
                 fontsize=12.5, fontweight="bold", color=NAVY)
    fig.text(
        0.5, -0.04,
        "A deep tree memorises the training rows. The widening gap is variance, and it is the reason "
        "max_depth is tuned on validation data rather than chosen freely.",
        ha="center", fontsize=8, color=GREY,
    )
    save(fig, "12_overfitting_tree_depth.png", tight=False)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()

    print("Building Week 2 figures")
    figure_workflow()
    figure_class_balance(summary)
    figure_model_comparison(summary)
    figure_confusion_matrices(summary)
    figure_roc_curves(summary)
    figure_threshold(summary)
    figure_class_weighting()
    figure_feature_importance()
    figure_cross_validation()
    figure_regression(summary)
    figure_regression_fit(summary)
    figure_overfitting()
    print("\nDone. Figures written to week-2/outputs/figures/")


if __name__ == "__main__":
    main()
