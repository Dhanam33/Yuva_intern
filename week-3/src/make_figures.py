#!/usr/bin/env python3
"""Week 3 - build every figure from the saved analysis outputs.

Figures are generated from the CSV/JSON files written by ``run_analysis.py``
rather than from live model objects, so they can be rebuilt without rerunning
the clustering or the grid search.

    python src/run_analysis.py     # first
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.cluster.hierarchy import dendrogram
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

CLUSTER_COLORS = ["#237A7A", "#5577AA", "#D17A61", "#7D6B9E", "#5B8E7D"]

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
    return json.loads((OUTPUT_DIR / "week3_summary.json").read_text(encoding="utf-8"))


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / name)


# --------------------------------------------------------------------------
# 1. Workflow overview
# --------------------------------------------------------------------------
def figure_workflow(summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(12.4, 5.8))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    ax.text(0.4, 5.35, "Unsupervised learning, then model evaluation", fontsize=18,
            fontweight="bold", color=NAVY)
    ax.text(0.4, 4.95, "Two halves of the brief, joined by one question: do the clusters explain churn?",
            fontsize=10, color=TEAL)

    # ---- Part A strip
    ax.text(0.4, 4.35, "PART A — UNSUPERVISED", fontsize=10.5, fontweight="bold", color=TEAL)
    stages_a = [
        ("1  Encode", "scale numeric\none-hot stays 0/1", BLUE, "#E9EFF6"),
        ("2  Reduce", "PCA: 21 → 10 dims\nfor 90% variance", AMBER, "#FBF1E3"),
        ("3  Cluster", "K-Means k=2..8\nWard, complete, average, single", TEAL, "#E6F1F1"),
        ("4  Validate", "silhouette · DB · CH\nthen ARI against churn", CORAL, "#F8ECE7"),
    ]
    width, height, y = 2.62, 1.42, 2.72
    xs = [0.5, 3.45, 6.40, 9.35]
    for (title, detail, accent, fill), x in zip(stages_a, xs):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    facecolor=fill, edgecolor=accent, linewidth=1.6))
        ax.text(x + 0.18, y + height - 0.30, title, fontsize=11.5, fontweight="bold", color=accent)
        ax.text(x + 0.18, y + height - 0.62, detail, fontsize=8.3, color="#3F4E5C",
                linespacing=1.5, va="top")
        if x != xs[-1]:
            ax.add_patch(FancyArrowPatch((x + width + 0.03, y + height / 2),
                                         (x + width + 0.30, y + height / 2),
                                         arrowstyle="-|>", mutation_scale=14,
                                         color=GREY, linewidth=1.5))

    # ---- Part B strip
    ax.text(0.4, 2.25, "PART B — SUPERVISED EVALUATION", fontsize=10.5, fontweight="bold", color=CORAL)
    stages_b = [
        ("1  Metric", "confusion matrix\nprecision · recall · F1", BLUE, "#E9EFF6"),
        ("2  Cross-validate", "5-fold stratified\non training data", TEAL, "#E6F1F1"),
        ("3  Tune", "GridSearchCV\n5 + 32 combinations", AMBER, "#FBF1E3"),
        ("4  Verify", "nested CV\nheld-out test, once", CORAL, "#F8ECE7"),
    ]
    y_b = 0.62
    for (title, detail, accent, fill), x in zip(stages_b, xs):
        ax.add_patch(FancyBboxPatch((x, y_b), width, height,
                                    boxstyle="round,pad=0.02,rounding_size=0.1",
                                    facecolor=fill, edgecolor=accent, linewidth=1.6))
        ax.text(x + 0.18, y_b + height - 0.30, title, fontsize=11.5, fontweight="bold", color=accent)
        ax.text(x + 0.18, y_b + height - 0.62, detail, fontsize=8.3, color="#3F4E5C",
                linespacing=1.5, va="top")
        if x != xs[-1]:
            ax.add_patch(FancyArrowPatch((x + width + 0.03, y_b + height / 2),
                                         (x + width + 0.30, y_b + height / 2),
                                         arrowstyle="-|>", mutation_scale=14,
                                         color=GREY, linewidth=1.5))

    ax.annotate("", xy=(9.35, y_b + height + 0.10), xytext=(9.35, y - 0.10),
                arrowprops=dict(arrowstyle="-|>", color=CORAL, linewidth=1.8,
                                connectionstyle="arc3,rad=0.35"))
    ax.text(10.55, 2.40,
            f"clusters vs churn\nARI = {summary['agreement']['ari_churn_kmeans']:+.3f}\n— nearly unrelated",
            fontsize=8.2, color=CORAL, fontstyle="italic", va="center")

    ax.text(0.5, 0.22,
            "Fixed random seed 42 · the churn label is never used to fit a cluster — only to check afterwards whether it happens to match",
            fontsize=7.8, color=GREY)
    save(fig, "00_workflow_overview.png")


# --------------------------------------------------------------------------
# 2. PCA variance
# --------------------------------------------------------------------------
def figure_pca_variance(summary: dict) -> None:
    variance = read_csv("pca_variance.csv")
    info = summary["pca"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.6, 4.1))

    left.bar(variance["component"], variance["explained_variance_ratio"] * 100,
             color=TEAL, edgecolor="white", linewidth=0.6)
    left.set_title("Each component's share of the variance")
    left.set_xlabel("Principal component")
    left.set_ylabel("Variance explained (%)")
    left.set_xticks(range(1, len(variance) + 1, 2))
    left.annotate(f"PC1 alone: {info['explained_first']:.1%}",
                  xy=(1, info["explained_first"] * 100), xytext=(4.2, info["explained_first"] * 100 - 1.4),
                  fontsize=8.6, color=NAVY,
                  arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2))

    right.plot(variance["component"], variance["cumulative_variance"] * 100,
               marker="o", markersize=4, color=CORAL, linewidth=2)
    for threshold, colour in ((90, AMBER), (95, BLUE)):
        right.axhline(threshold, color=colour, linestyle="--", linewidth=1.3, alpha=0.8)
        right.text(len(variance) + 0.2, threshold - 4.5, f"{threshold}%", fontsize=8,
                   color=colour, fontweight="bold", ha="right")
    right.axvline(info["n_components_90"], color=AMBER, linestyle=":", linewidth=1.4)
    right.scatter([info["n_components_90"]], [90], s=70, color=AMBER, zorder=5,
                  edgecolor="white", linewidth=1.4)
    right.annotate(f"{info['n_components_90']} components\nreach 90%",
                   xy=(info["n_components_90"], 90), xytext=(info["n_components_90"] + 1.4, 62),
                   fontsize=8.6, color=NAVY,
                   arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2))
    right.set_title("Cumulative variance")
    right.set_xlabel("Number of components kept")
    right.set_ylabel("Variance explained (%)")
    right.set_ylim(15, 103)
    right.set_xticks(range(1, len(variance) + 1, 2))

    save(fig, "01_pca_variance.png")


# --------------------------------------------------------------------------
# 3. PCA projection, coloured two ways
# --------------------------------------------------------------------------
def figure_pca_projection(summary: dict) -> None:
    projection = read_csv("pca_projection.csv")
    info = summary["pca"]
    profiles = {row["cluster"]: row["label"] for row in summary["kmeans"]["profiles"]}

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.9))

    for cluster in sorted(projection["kmeans_cluster"].unique()):
        block = projection[projection["kmeans_cluster"] == cluster]
        left.scatter(block["pc1"], block["pc2"], s=13, alpha=0.72,
                     color=CLUSTER_COLORS[int(cluster) % len(CLUSTER_COLORS)],
                     label=f"Cluster {int(cluster)} — {profiles[int(cluster)]}",
                     linewidths=0)
    left.set_title("Coloured by K-Means cluster")
    left.set_xlabel(f"PC1 — {info['explained_first']:.1%} of variance")
    left.set_ylabel(f"PC2 — {summary['pca']['explained_first_two'] - info['explained_first']:.1%}")
    left.legend(loc="upper right", fontsize=7.6, markerscale=1.9, handletextpad=0.4,
                frameon=True, facecolor="white", edgecolor="#D5E0E8", framealpha=0.92)

    for value, label, colour in ((0, "Stayed", GREEN), (1, "Churned", CORAL)):
        block = projection[projection["churn"] == value]
        right.scatter(block["pc1"], block["pc2"], s=13, alpha=0.62,
                      color=colour, label=label, linewidths=0)
    right.set_title("Coloured by actual churn")
    right.set_xlabel(f"PC1 — {info['explained_first']:.1%} of variance")
    right.set_ylabel(f"PC2 — {summary['pca']['explained_first_two'] - info['explained_first']:.1%}")
    right.legend(loc="upper right", fontsize=8, markerscale=1.9, handletextpad=0.4,
                 frameon=True, facecolor="white", edgecolor="#D5E0E8", framealpha=0.92)

    save(fig, "02_pca_projection.png")


# --------------------------------------------------------------------------
# 4. Choosing k
# --------------------------------------------------------------------------
def figure_kmeans_selection(summary: dict) -> None:
    sweep = read_csv("kmeans_sweep.csv")
    selected = summary["kmeans"]["selected_k"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.2))

    left.plot(sweep["k"], sweep["inertia"], marker="o", markersize=6,
              color=TEAL, linewidth=2.2)
    left.axvline(selected, color=CORAL, linestyle="--", linewidth=1.4)
    left.text(selected + 0.15, sweep["inertia"].max() * 0.96, f"chosen k = {selected}",
              color=CORAL, fontsize=9, fontweight="bold")
    left.set_title("Elbow: within-cluster sum of squares")
    left.set_xlabel("k — number of clusters")
    left.set_ylabel("Inertia (lower is tighter)")
    left.set_xticks(sweep["k"])

    ax2 = right
    ax2.plot(sweep["k"], sweep["silhouette"], marker="o", markersize=6,
             color=CORAL, linewidth=2.2, label="Silhouette (higher is better)")
    ax2.set_xlabel("k — number of clusters")
    ax2.set_ylabel("Silhouette score", color=CORAL)
    ax2.tick_params(axis="y", labelcolor=CORAL)
    ax2.set_xticks(sweep["k"])
    ax2.set_ylim(0.10, 0.19)

    peak = sweep.loc[sweep["silhouette"].idxmax()]
    ax2.scatter([peak["k"]], [peak["silhouette"]], s=130, facecolor="none",
                edgecolor=CORAL, linewidth=2.2, zorder=5)
    ax2.annotate(f"peak {peak['silhouette']:.3f}", xy=(peak["k"], peak["silhouette"]),
                 xytext=(peak["k"] + 1.2, peak["silhouette"] + 0.012),
                 fontsize=8.8, color=CORAL,
                 arrowprops=dict(arrowstyle="-|>", color=CORAL, lw=1.2))

    scale = 1 / 130
    ax3 = ax2.twinx()
    ax3.plot(sweep["k"], sweep["davies_bouldin"], marker="s", markersize=5,
             color=BLUE, linewidth=1.8, linestyle="--", label="Davies-Bouldin (lower is better)")
    ax3.set_ylabel("Davies-Bouldin index", color=BLUE)
    ax3.tick_params(axis="y", labelcolor=BLUE)
    ax3.grid(False)
    ax3.set_ylim(sweep["davies_bouldin"].min() - 0.12 * scale * 130,
                 sweep["davies_bouldin"].max() + 0.12 * scale * 130)

    handles = [ax2.lines[0], ax3.lines[0]]
    ax2.legend(handles, [h.get_label() for h in handles], loc="lower left", fontsize=8)
    ax2.set_title("Internal indices disagree")
    ax2.text(0.98, 0.04,
             f"silhouette → k={summary['kmeans']['preferred_k']['silhouette']}\n"
             f"Davies-Bouldin → k={summary['kmeans']['preferred_k']['davies_bouldin']}\n"
             f"Calinski-Harabasz → k={summary['kmeans']['preferred_k']['calinski_harabasz']}",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=7.8,
             color=NAVY, bbox=dict(boxstyle="round,pad=0.4", facecolor="#F3F7FA",
                                   edgecolor="#D5E0E8"))

    save(fig, "03_kmeans_selection.png")


# --------------------------------------------------------------------------
# 5. Cluster sizes and churn rate
# --------------------------------------------------------------------------
def figure_cluster_overview(summary: dict) -> None:
    profiles = pd.DataFrame(summary["kmeans"]["profiles"]).sort_values("size", ascending=False)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.2))

    colours = [CLUSTER_COLORS[int(c) % len(CLUSTER_COLORS)] for c in profiles["cluster"]]
    bars = left.barh(profiles["label"], profiles["size"], color=colours,
                     edgecolor="white", linewidth=1.2, height=0.62)
    left.bar_label(bars, labels=[f"{int(v):,}  ({s:.1%})" for v, s in
                                 zip(profiles["size"], profiles["share"])],
                   padding=6, fontsize=9, color=NAVY, fontweight="bold")
    left.set_xlim(0, profiles["size"].max() * 1.34)
    left.set_title("Cluster sizes")
    left.set_xlabel("Customers")
    left.invert_yaxis()
    left.grid(axis="y", visible=False)

    overall = summary["dataset"]["churn_rate"]
    bars = right.bar(profiles["label"], profiles["churn_rate"] * 100, color=colours,
                     edgecolor="white", linewidth=1.2, width=0.58)
    right.axhline(overall * 100, color=NAVY, linestyle="--", linewidth=1.6,
                  label=f"Overall churn rate {overall:.1%}")
    right.bar_label(bars, labels=[f"{v:.1%}" for v in profiles["churn_rate"]],
                    padding=4, fontsize=9, color=NAVY, fontweight="bold")
    right.set_ylim(0, max(profiles["churn_rate"].max() * 130, overall * 145))
    right.set_title("Churn rate per cluster")
    right.set_ylabel("Churn rate (%)")
    right.tick_params(axis="x", labelsize=7.8)
    right.legend(loc="upper right", fontsize=8.2)
    right.grid(axis="x", visible=False)

    save(fig, "04_cluster_overview.png")


# --------------------------------------------------------------------------
# 6. Dendrogram
# --------------------------------------------------------------------------
def figure_dendrogram(summary: dict) -> None:
    linkage_matrix = pd.read_csv(OUTPUT_DIR / "linkage_matrix.csv").to_numpy()
    linkage_name = summary["hierarchical"]["selected_linkage"]
    k = summary["hierarchical"]["selected_k"]

    fig, ax = plt.subplots(figsize=(11.8, 4.6))
    dendrogram(
        linkage_matrix,
        ax=ax,
        truncate_mode="lastp",
        p=24,
        show_contracted=True,
        leaf_rotation=90,
        leaf_font_size=7.5,
        color_threshold=None,
        above_threshold_color=TEAL,
    )
    ax.set_title(f"Dendrogram — {linkage_name} linkage, top 24 merges "
                 f"({summary['dataset']['rows']:,} customers collapsed into leaves)")
    ax.set_ylabel("Merge distance")
    ax.set_xlabel("Cluster (number in brackets = customers merged into that branch)")
    ax.grid(axis="x", visible=False)

    distances = np.sort(linkage_matrix[:, 2])[::-1]
    if len(distances) > k:
        cut = (distances[k - 1] + distances[k]) / 2
        ax.axhline(cut, color=CORAL, linestyle="--", linewidth=1.6)
        ax.text(0.012, cut, f"cut at k={k}", transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=8.8, color=CORAL, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="none", alpha=0.85))

    save(fig, "05_dendrogram.png")


# --------------------------------------------------------------------------
# 7. The silhouette trap
# --------------------------------------------------------------------------
def figure_linkage_trap(summary: dict) -> None:
    sweep = read_csv("hierarchical_sweep.csv")
    colours = {"ward": TEAL, "complete": BLUE, "average": AMBER, "single": CORAL}

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.3))

    for linkage, block in sweep.groupby("linkage"):
        block = block.sort_values("k")
        left.plot(block["k"], block["silhouette"], marker="o", markersize=6,
                  linewidth=2.0, color=colours[linkage], label=linkage)
        bad = block[block["degenerate"]]
        if len(bad):
            left.scatter(bad["k"], bad["silhouette"], marker="X", s=95,
                         color=colours[linkage], edgecolor="white", linewidth=1.2, zorder=5)
    left.set_title("Silhouette by linkage rule")
    left.set_xlabel("k — number of clusters")
    left.set_ylabel("Silhouette (higher is better)")
    left.set_xticks(sorted(sweep["k"].unique()))
    left.set_ylim(0.02, 0.325)
    left.legend(title="linkage", fontsize=8.2, title_fontsize=8.2, loc="upper right")
    left.text(0.02, 0.03, "X marks a degenerate solution\n(some cluster < 2% of the data)",
              transform=left.transAxes, fontsize=7.8, color=CORAL, va="bottom",
              bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                        edgecolor="#E4C3B8", alpha=0.95))

    for linkage, block in sweep.groupby("linkage"):
        block = block.sort_values("k")
        right.plot(block["k"], block["smallest_cluster"], marker="o", markersize=6,
                   linewidth=2.0, color=colours[linkage], label=linkage)
    right.axhline(len(read_csv("pca_projection.csv")) * 0.02, color=GREY,
                  linestyle=":", linewidth=1.4)
    right.text(4.9, 41, "2% of customers", fontsize=7.8, color=GREY, ha="right")
    right.set_yscale("log")
    right.set_title("Size of the smallest cluster")
    right.set_xlabel("k — number of clusters")
    right.set_ylabel("Customers (log scale)")
    right.set_xticks(sorted(sweep["k"].unique()))
    right.set_ylim(0.7, 1400)

    trap = summary["hierarchical"]["trap"]
    right.annotate(f"{trap['linkage']} k={trap['k']}: {trap['smallest_cluster']} customer\n"
                   f"but silhouette {trap['silhouette']:.3f} — the highest of all",
                   xy=(trap["k"], trap["smallest_cluster"]), xytext=(2.6, 6),
                   fontsize=8.2, color=CORAL,
                   arrowprops=dict(arrowstyle="-|>", color=CORAL, lw=1.2))

    save(fig, "06_linkage_trap.png")


# --------------------------------------------------------------------------
# 8. Cluster profiles, K-Means vs Ward
# --------------------------------------------------------------------------
def figure_cluster_profiles(summary: dict) -> None:
    metrics = ["tenure_months", "monthly_charges", "total_charges", "support_tickets"]
    pretty = ["Tenure\n(months)", "Monthly\ncharges", "Total\ncharges", "Support\ntickets"]

    def tidy(value: float) -> str:
        if value >= 1000:
            return f"{value:,.0f}"
        if value >= 100:
            return f"{value:.0f}"
        if value >= 10:
            return f"{value:.1f}"
        return f"{value:.2f}"

    kmeans = pd.DataFrame(summary["kmeans"]["profiles"]).sort_values("tenure_months")
    ward = pd.DataFrame(summary["hierarchical"]["profiles"]).sort_values("tenure_months")

    rows = []
    for algorithm, table in (("K-Means", kmeans), ("Ward", ward)):
        for row in table.itertuples():
            rows.append((algorithm, row.label, int(row.size), [getattr(row, m) for m in metrics]))

    values = np.array([row[3] for row in rows])
    z = (values - values.mean(axis=0)) / values.std(axis=0)

    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    image = ax.imshow(z, cmap="RdYlBu", vmin=-1.9, vmax=1.9, aspect="auto")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(pretty, fontsize=8.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{algorithm} · {label}\n({size:,} customers)"
                        for algorithm, label, size, _ in rows], fontsize=8.2)
    ax.grid(False)

    for row_index, (_, _, _, row_values) in enumerate(rows):
        for column, value in enumerate(row_values):
            # Dark cells need light text; the colormap extremes are navy and
            # deep red, where dark-on-dark is unreadable.
            ink = "white" if abs(z[row_index, column]) > 1.15 else "#1F2A33"
            ax.text(column, row_index, tidy(value), ha="center", va="center",
                    fontsize=9.6, fontweight="bold", color=ink)

    ax.axhline(2.5, color=NAVY, linewidth=2.0)

    colourbar = fig.colorbar(image, ax=ax, orientation="vertical",
                             fraction=0.028, pad=0.02)
    colourbar.set_label("Above / below the mean across all six segments (z-score)", fontsize=8)

    ax.set_title("Cluster profiles — two algorithms, the same three segments",
                 fontsize=11.5, pad=12)
    ax.text(0.5, -0.17,
            f"Rows are ordered by tenure so the two algorithms line up. "
            f"They agree on the partition (ARI {summary['agreement']['ari_models']:.3f}), "
            "which is evidence the segments are real rather than an artefact of one method.",
            transform=ax.transAxes, ha="center", fontsize=8.4, color=TEAL, fontstyle="italic")

    save(fig, "07_cluster_profiles.png")


# --------------------------------------------------------------------------
# 9. Do the clusters explain churn?
# --------------------------------------------------------------------------
def figure_clusters_vs_churn(summary: dict) -> None:
    agreement = summary["agreement"]
    crosstab = read_csv("cluster_churn_crosstab.csv")
    profiles = {row["cluster"]: row["label"] for row in summary["kmeans"]["profiles"]}

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.2, 4.5))

    # Stacked composition: how many churners hide in each cluster
    positions = np.arange(len(crosstab))
    left.bar(positions, crosstab["churn_No"], color="#DCE4EC", edgecolor="white",
             linewidth=1.2, label="Stayed", width=0.62)
    left.bar(positions, crosstab["churn_Yes"], bottom=crosstab["churn_No"],
             color=CORAL, edgecolor="white", linewidth=1.2,
             label="Churned", width=0.62)
    for position, row in enumerate(crosstab.itertuples()):
        total = row.churn_No + row.churn_Yes
        left.text(position, total + 22, f"{row.churn_rate:.1%}\nchurn", ha="center",
                  fontsize=8.6, color=CORAL, fontweight="bold")
    left.set_xticks(positions)
    left.set_xticklabels([profiles[int(c)] for c in crosstab["cluster"]], fontsize=8)
    left.set_ylim(0, (crosstab["churn_No"] + crosstab["churn_Yes"]).max() * 1.24)
    left.set_ylabel("Customers")
    left.set_title("Every cluster contains churners at a similar rate")
    left.legend(loc="upper right", fontsize=8.4)
    left.grid(axis="x", visible=False)

    labels = ["K-Means vs\nWard", "K-Means vs\nchurn", "Ward vs\nchurn"]
    values = [agreement["ari_models"], agreement["ari_churn_kmeans"],
              agreement["ari_churn_hierarchical"]]
    bar_colors = [TEAL, CORAL, CORAL]
    bars = right.bar(labels, values, color=bar_colors, edgecolor="white",
                     linewidth=1.2, width=0.55)
    right.bar_label(bars, labels=[f"{v:+.3f}" for v in values], padding=5,
                    fontsize=10, fontweight="bold", color=NAVY)
    right.axhline(0, color="#9AA7B4", linewidth=1.0)
    right.set_ylim(0, 0.86)
    right.set_ylabel("Adjusted Rand Index (1 = identical, 0 = random)")
    right.set_title("The two clusterings agree; the label does not")
    right.grid(axis="x", visible=False)
    right.text(0.5, 0.62,
               "Two algorithms independently find\nthe same segments — that is evidence\nthe structure is real. Churn simply\nlives somewhere else.",
               transform=right.transAxes, ha="center", fontsize=8.4, color=NAVY,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="#F3F7FA",
                         edgecolor="#D5E0E8"))

    save(fig, "08_clusters_vs_churn.png")


# --------------------------------------------------------------------------
# 10. Dimensionality reduction before clustering
# --------------------------------------------------------------------------
def figure_dimensionality(summary: dict) -> None:
    experiment = read_csv("dimensionality_experiment.csv")

    fig, ax = plt.subplots(figsize=(9.6, 4.5))
    ax.plot(experiment["components"], experiment["silhouette_reduced_space"],
            marker="o", markersize=7, linewidth=2.4, color=CORAL,
            label="Measured in the reduced space")
    ax.plot(experiment["components"], experiment["silhouette_original_space"],
            marker="s", markersize=6, linewidth=2.2, color=TEAL,
            label="Same labels, measured in the original 21-D space")
    ax.fill_between(experiment["components"],
                    experiment["silhouette_original_space"],
                    experiment["silhouette_reduced_space"],
                    color=CORAL, alpha=0.10)

    peak = experiment.loc[experiment["silhouette_reduced_space"].idxmax()]
    ax.annotate(f"{int(peak['components'])}D: looks like {peak['silhouette_reduced_space']:.3f}",
                xy=(peak["components"], peak["silhouette_reduced_space"]),
                xytext=(peak["components"] + 2.0, peak["silhouette_reduced_space"] - 0.030),
                fontsize=9, color=CORAL, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=CORAL, lw=1.3))
    ax.annotate(f"but the same clusters, measured in the real space: {peak['silhouette_original_space']:.3f}",
                xy=(peak["components"], peak["silhouette_original_space"]),
                xytext=(peak["components"] + 1.6, 0.098),
                fontsize=8.8, color=TEAL, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.3))

    ax.set_title("Projecting to 2-D flatters the clusters without improving them")
    ax.set_xlabel("Components kept before K-Means")
    ax.set_ylabel("Silhouette score")
    ax.set_xticks(experiment["components"])
    ax.set_ylim(0.075, 0.425)
    ax.legend(loc="upper right", fontsize=8.6)


    save(fig, "09_dimensionality_experiment.png")


# --------------------------------------------------------------------------
# 11. Confusion matrix and the metrics derived from it
# --------------------------------------------------------------------------
def figure_confusion_matrix(summary: dict) -> None:
    metrics = summary["tuning"]["metrics"]
    tn, fp = metrics["true_negative"], metrics["false_positive"]
    fn, tp = metrics["false_negative"], metrics["true_positive"]

    grid = np.array([[tn, fp], [fn, tp]])

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.6),
                                      gridspec_kw={"width_ratios": [1.0, 1.15]})

    left.imshow(grid, cmap="Blues", vmin=0, vmax=grid.max() * 1.15)
    labels = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
    for row in range(2):
        for column in range(2):
            value = grid[row, column]
            dark = value > grid.max() * 0.55
            left.text(column, row - 0.10, f"{value}", ha="center", va="center",
                      fontsize=21, fontweight="bold",
                      color="white" if dark else NAVY)
            left.text(column, row + 0.22, labels[row][column], ha="center", va="center",
                      fontsize=8, color="white" if dark else "#4A5865")
    left.set_xticks([0, 1])
    left.set_xticklabels(["Predicted: stayed", "Predicted: churned"], fontsize=8.6)
    left.set_yticks([0, 1])
    left.set_yticklabels(["Actually\nstayed", "Actually\nchurned"], fontsize=8.6)
    left.set_title(f"Confusion matrix — {summary['tuning']['winner']}, 360 held-out customers",
                   fontsize=10.2)
    left.grid(False)
    left.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    left.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    left.grid(which="minor", color="white", linewidth=2.5)

    formulas = [
        ("Accuracy", f"({tn} + {tp}) / 360", metrics["accuracy"], "how many answers were right"),
        ("Precision", f"{tp} / ({tp} + {fp})", metrics["precision"], "of the flags raised, how many were real"),
        ("Recall", f"{tp} / ({tp} + {fn})", metrics["recall"], "of the real churners, how many were caught"),
        ("F1", "2 · P · R / (P + R)", metrics["f1"], "harmonic mean of the two"),
        ("ROC-AUC", "area under the ROC curve", metrics["roc_auc"], "ranking quality, threshold-free"),
    ]
    right.axis("off")
    right.set_title("Every metric is a ratio of these four numbers", fontsize=10.2)
    for position, (name, formula, value, meaning) in enumerate(formulas):
        y = 0.86 - position * 0.185
        right.text(0.02, y, name, fontsize=10, fontweight="bold", color=TEAL,
                   transform=right.transAxes)
        right.text(0.02, y - 0.055, formula, fontsize=8.4, color="#4A5865",
                   fontfamily="DejaVu Sans Mono", transform=right.transAxes)
        right.text(0.02, y - 0.105, meaning, fontsize=7.8, color=GREY,
                   fontstyle="italic", transform=right.transAxes)
        right.text(0.97, y - 0.02, f"{value:.3f}", fontsize=15, fontweight="bold",
                   color=NAVY, ha="right", transform=right.transAxes)

    save(fig, "10_confusion_matrix.png")


# --------------------------------------------------------------------------
# 12. ROC, precision-recall and calibration
# --------------------------------------------------------------------------
def figure_curves(summary: dict) -> None:
    predictions = read_csv("final_predictions.csv")
    actual = predictions["actual"].to_numpy()
    probability = predictions["probability"].to_numpy()
    metrics = summary["tuning"]["metrics"]

    fpr, tpr, _ = roc_curve(actual, probability)
    precision, recall, _ = precision_recall_curve(actual, probability)

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3))

    axes[0].plot(fpr, tpr, color=TEAL, linewidth=2.4,
                 label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1.2,
                 label="No skill (0.500)")
    axes[0].fill_between(fpr, tpr, fpr, color=TEAL, alpha=0.12)
    axes[0].set_title("ROC curve")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend(loc="lower right", fontsize=8.4)
    axes[0].set_xlim(-0.01, 1.01)
    axes[0].set_ylim(-0.01, 1.01)

    baseline = float(actual.mean())
    axes[1].plot(recall, precision, color=CORAL, linewidth=2.4,
                 label=f"PR-AUC = {metrics['pr_auc']:.3f}")
    axes[1].axhline(baseline, color=GREY, linestyle="--", linewidth=1.2,
                    label=f"No skill ({baseline:.3f})")
    axes[1].set_title("Precision-recall curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="upper right", fontsize=8.4)
    axes[1].set_xlim(-0.01, 1.01)
    axes[1].set_ylim(0, 0.86)

    calibration = read_csv("final_calibration_curve.csv")
    axes[2].plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1.2,
                 label="Perfect calibration")
    axes[2].plot(calibration["mean_predicted_probability"], calibration["fraction_positive"],
                 marker="o", markersize=6, color=BLUE, linewidth=2.2,
                 label=f"Model (Brier {metrics['brier']:.3f})")
    axes[2].set_title("Calibration")
    axes[2].set_xlabel("Predicted churn probability")
    axes[2].set_ylabel("Observed churn rate")
    axes[2].legend(loc="upper left", fontsize=8.4)
    axes[2].set_xlim(-0.01, 1.01)
    axes[2].set_ylim(-0.01, 1.01)

    save(fig, "11_curves.png")


# --------------------------------------------------------------------------
# 13. What the grid search actually found
# --------------------------------------------------------------------------
def figure_tuning(summary: dict) -> None:
    logistic = read_csv("tuning_grid_logistic.csv").sort_values("param_model__C")
    forest = read_csv("tuning_grid_random_forest.csv")

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 4.5))

    left.errorbar(logistic["param_model__C"], logistic["mean_test_score"],
                  yerr=logistic["std_test_score"], marker="o", markersize=7,
                  color=TEAL, linewidth=2.2, capsize=4, capthick=1.4,
                  label="CV ROC-AUC ± 1 sd")
    left.plot(logistic["param_model__C"], logistic["mean_train_score"],
              marker="s", markersize=5, linestyle="--", color=BLUE,
              linewidth=1.6, label="Training ROC-AUC")
    best_c = float(summary["tuning"]["logistic_best"]["model__C"])
    left.scatter([best_c], [summary["tuning"]["logistic_best_score"]], s=150,
                 facecolor="none", edgecolor=CORAL, linewidth=2.2, zorder=5)
    left.set_xscale("log")
    left.set_title("Logistic Regression — one hyperparameter, no headroom")
    left.set_xlabel("C — inverse regularisation strength")
    left.set_ylabel("ROC-AUC")
    left.set_ylim(0.72, 0.87)
    left.legend(loc="lower right", fontsize=8.2)
    left.text(0.03, 0.96,
              f"best C = {best_c:g}  →  {summary['tuning']['logistic_best_score']:.3f}\n"
              "flat from C=0.1 to C=100: the default was already optimal",
              transform=left.transAxes, fontsize=8, color=NAVY, va="top",
              bbox=dict(boxstyle="round,pad=0.45", facecolor="#F3F7FA", edgecolor="#D5E0E8"))

    # Every one of the 32 Random Forest combinations, grouped by depth.
    depths = sorted(forest["param_model__max_depth"].dropna().unique())
    rng = np.random.default_rng(7)
    for position, depth in enumerate(depths):
        block = forest[forest["param_model__max_depth"] == depth]
        jitter = rng.uniform(-0.13, 0.13, len(block))
        right.scatter(block["mean_test_score"], position + jitter,
                      s=95, color=GREEN, alpha=0.75, edgecolor="white", linewidth=1.0)
    if forest["param_model__max_depth"].isna().any():
        block = forest[forest["param_model__max_depth"].isna()]
        jitter = rng.uniform(-0.13, 0.13, len(block))
        right.scatter(block["mean_test_score"], len(depths) + jitter,
                      s=95, color=AMBER, alpha=0.8, edgecolor="white", linewidth=1.0)

    best = forest.loc[forest["mean_test_score"].idxmax()]
    best_depth_position = (depths.index(best["param_model__max_depth"])
                           if pd.notna(best["param_model__max_depth"]) else len(depths))
    right.scatter([best["mean_test_score"]], [best_depth_position], s=320,
                  facecolor="none", edgecolor=CORAL, linewidth=2.4, zorder=6)

    low, high = forest["mean_test_score"].min(), forest["mean_test_score"].max()
    spread = high - low
    right.axvspan(low, high, color=TEAL, alpha=0.08)
    right.set_yticks(range(len(depths) + (1 if forest["param_model__max_depth"].isna().any() else 0)))
    labels = [f"depth {int(d)}" for d in depths]
    if forest["param_model__max_depth"].isna().any():
        labels.append("depth none")
    right.set_yticklabels(labels, fontsize=8.6)
    right.set_ylim(len(labels) - 0.35, -0.75)
    right.set_xlim(0.772, 0.806)
    right.set_title(f"Random Forest — all {len(forest)} combinations")
    right.set_xlabel("CV ROC-AUC")
    right.grid(axis="y", visible=False)
    right.text(0.5, 0.03,
               f"all {len(forest)} combinations land between {low:.3f} and {high:.3f}\n"
               f"a spread of {spread:.3f} against ±{forest['std_test_score'].mean():.3f} fold-to-fold noise",
               transform=right.transAxes, ha="center", va="bottom", fontsize=7.9, color=NAVY,
               bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                         edgecolor="#D5E0E8", alpha=0.95))

    save(fig, "12_tuning_results.png")


# --------------------------------------------------------------------------
# 14. Learning curve and nested CV
# --------------------------------------------------------------------------
def figure_learning_curve(summary: dict) -> None:
    curve = read_csv("learning_curve.csv")
    nested = read_csv("nested_cv_results.csv")

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.3),
                                      gridspec_kw={"width_ratios": [1.45, 1.0]})

    left.fill_between(curve["train_size"], curve["train_mean"] - curve["train_std"],
                      curve["train_mean"] + curve["train_std"], color=BLUE, alpha=0.15)
    left.plot(curve["train_size"], curve["train_mean"], marker="o", markersize=6,
              color=BLUE, linewidth=2.2, label="Training score")
    left.fill_between(curve["train_size"], curve["test_mean"] - curve["test_std"],
                      curve["test_mean"] + curve["test_std"], color=TEAL, alpha=0.15)
    left.plot(curve["train_size"], curve["test_mean"], marker="o", markersize=6,
              color=TEAL, linewidth=2.2, label="Validation score")

    gap_start = curve["train_mean"].iloc[0] - curve["test_mean"].iloc[0]
    gap_end = curve["train_mean"].iloc[-1] - curve["test_mean"].iloc[-1]
    left.annotate("", xy=(curve["train_size"].iloc[-1], curve["test_mean"].iloc[-1]),
                  xytext=(curve["train_size"].iloc[-1], curve["train_mean"].iloc[-1]),
                  arrowprops=dict(arrowstyle="<->", color=CORAL, lw=1.6))
    left.text(curve["train_size"].iloc[-1] * 0.94, curve["train_mean"].iloc[-1] + 0.004,
              f"gap {gap_end:.3f}", fontsize=8.6, color=CORAL, ha="right", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85))
    left.set_title("Learning curve — the model is not overfitting")
    left.set_xlabel("Training rows used")
    left.set_ylabel("ROC-AUC")
    left.legend(loc="lower right", fontsize=8.6)
    left.set_ylim(0.74, 0.88)
    left.text(0.03, 0.94,
              f"gap starts at {gap_start:.3f} and narrows to {gap_end:.3f}.\n"
              "The lines are converging, not diverging:\n"
              "the limit is the data, not the capacity.",
              transform=left.transAxes, fontsize=7.9, color=NAVY, va="top",
              bbox=dict(boxstyle="round,pad=0.45", facecolor="#F3F7FA", edgecolor="#D5E0E8"))

    positions = np.arange(1, len(nested) + 1)
    left2 = right
    left2.bar(positions, nested["roc_auc"], color=PURPLE, width=0.55,
              edgecolor="white", linewidth=1.2)
    left2.axhline(summary["nested_cv"]["mean"], color=NAVY, linestyle="--", linewidth=1.6,
                  label=f"nested mean {summary['nested_cv']['mean']:.3f}")
    left2.axhline(summary["tuning"]["best_cv_score"], color=CORAL, linestyle=":", linewidth=1.6,
                  label=f"tuned CV {summary['tuning']['best_cv_score']:.3f}")
    left2.bar_label(left2.containers[0], labels=[f"{v:.3f}" for v in nested["roc_auc"]],
                    padding=3, fontsize=8, color=NAVY)
    left2.set_xticks(positions)
    left2.set_xticklabels([f"fold {int(p)}" for p in positions], fontsize=8.2)
    left2.set_ylim(0.70, 0.89)
    left2.set_yticks([0.70, 0.74, 0.78, 0.82, 0.86])
    left2.set_ylabel("ROC-AUC")
    left2.set_title("Nested CV — the unbiased estimate")
    left2.grid(axis="x", visible=False)
    left2.legend(loc="upper right", fontsize=8, ncol=1)
    left2.text(0.02, 0.97,
               f"optimism of the tuned score\n{summary['nested_cv']['optimism']:+.3f}",
               transform=left2.transAxes, ha="left", va="top", fontsize=8.4,
               color=CORAL, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                         edgecolor="#E4C3B8", alpha=0.95))

    save(fig, "13_learning_curve.png")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    print(f"Building figures into {FIGURE_DIR.relative_to(WEEK_DIR.parent)}/")

    figure_workflow(summary)
    figure_pca_variance(summary)
    figure_pca_projection(summary)
    figure_kmeans_selection(summary)
    figure_cluster_overview(summary)
    figure_dendrogram(summary)
    figure_linkage_trap(summary)
    figure_cluster_profiles(summary)
    figure_clusters_vs_churn(summary)
    figure_dimensionality(summary)
    figure_confusion_matrix(summary)
    figure_curves(summary)
    figure_tuning(summary)
    figure_learning_curve(summary)

    print(f"Done — {len(list(FIGURE_DIR.glob('*.png')))} figures.")


if __name__ == "__main__":
    main()
