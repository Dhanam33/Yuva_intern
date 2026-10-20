#!/usr/bin/env python3
"""Create the Week 1 preprocessing-pipeline overview diagram.

The diagram is used in the README and in the Word report. It is generated
rather than hand-drawn so the project stays fully reproducible.

    python src/make_pipeline_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

WEEK_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = WEEK_DIR / "outputs" / "figures" / "00_pipeline_overview.png"

NAVY = "#17324D"
TEAL = "#237A7A"
CORAL = "#D17A61"
GREY = "#8FA3B5"

STAGES = [
    {
        "title": "STAGE 1  ·  UNDERSTAND & CLEAN",
        "accent": "#5577AA",
        "fill": "#E9EFF6",
        "steps": [
            ("1  Load", "Pandas read_csv", "1,800 × 13 table"),
            ("2  Audit", "dtypes, duplicates, IDs", "562 missing cells"),
            ("3  Clean", "trim text, fix invalid values", "0 duplicate rows"),
        ],
    },
    {
        "title": "STAGE 2  ·  PREPARE",
        "accent": "#237A7A",
        "fill": "#E6F1F1",
        "steps": [
            ("4  Impute", "median (numeric), mode (category)", "10 columns filled"),
            ("5  Explore", "6 EDA charts + IQR outlier audit", "patterns documented"),
            ("6  Encode", "one-hot, unseen-safe", "16 new columns"),
        ],
    },
    {
        "title": "STAGE 3  ·  MODEL-READY",
        "accent": "#C4664D",
        "fill": "#F8ECE7",
        "steps": [
            ("7  Scale", "StandardScaler on numeric inputs", "mean 0 · std 1"),
            ("8  Select", "mutual information ranking", "top 10 exported"),
            ("9  Export", "CSV · JSON · figures", "0 missing cells"),
        ],
    },
]


def _arrow(ax, start, end, color=GREY, linewidth=1.5, mutation=13):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation,
            color=color,
            linewidth=linewidth,
            shrinkA=0,
            shrinkB=0,
            zorder=1,
        )
    )


def _elbow(ax, points, color=GREY, linewidth=1.4):
    """Draw a rounded polyline with an arrowhead on the last segment."""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    ax.plot(xs[:-1] + [xs[-1]], ys[:-1] + [ys[-1]], color=color, linewidth=linewidth,
            solid_capstyle="round", solid_joinstyle="round", zorder=1)
    ax.annotate(
        "",
        xy=(xs[-1], ys[-1]),
        xytext=(xs[-2], ys[-2]),
        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=linewidth, mutation_scale=13),
        zorder=1,
    )


def draw() -> Path:
    column_width = 3.42
    card_height = 1.14
    card_gap = 0.26
    header_height = 0.44
    x_positions = [0.62, 4.62, 8.62]
    top_of_cards = 5.16

    fig, ax = plt.subplots(figsize=(12.6, 6.25))
    ax.set_xlim(0, 12.66)
    ax.set_ylim(0.42, 6.44)
    ax.axis("off")

    ax.text(0.62, 6.22, "Week 1 · Preprocessing pipeline", fontsize=19,
            fontweight="bold", color=NAVY, va="center")
    ax.text(
        0.62,
        5.85,
        "Raw CSV in  →  9 documented steps  →  clean, encoded, scaled, model-ready table out",
        fontsize=10.5,
        color=TEAL,
        va="center",
    )
    ax.plot([0.62, 12.04], [5.68, 5.68], color="#DCE4EC", linewidth=1.2, zorder=0)

    for column_index, stage in enumerate(STAGES):
        x = x_positions[column_index]

        # Stage header band
        ax.add_patch(
            Rectangle(
                (x, top_of_cards + 0.06),
                column_width,
                header_height,
                facecolor=stage["accent"],
                edgecolor="none",
                zorder=2,
            )
        )
        ax.text(
            x + 0.16,
            top_of_cards + 0.06 + header_height / 2,
            stage["title"],
            fontsize=9.4,
            fontweight="bold",
            color="white",
            va="center",
            zorder=3,
        )

        for step_index, (title, detail, badge) in enumerate(stage["steps"]):
            y = top_of_cards - (step_index + 1) * card_height - step_index * card_gap
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    column_width,
                    card_height,
                    boxstyle="round,pad=0.0,rounding_size=0.09",
                    facecolor=stage["fill"],
                    edgecolor="white",
                    linewidth=1.8,
                    zorder=2,
                )
            )
            ax.text(
                x + 0.20,
                y + card_height - 0.26,
                title,
                fontsize=11,
                fontweight="bold",
                color=NAVY,
                va="center",
                zorder=3,
            )
            ax.text(
                x + 0.20,
                y + 0.51,
                detail,
                fontsize=8.1,
                color="#4A5A69",
                va="center",
                zorder=3,
            )
            ax.text(
                x + 0.20,
                y + 0.21,
                badge,
                fontsize=7.8,
                color=stage["accent"],
                fontweight="bold",
                va="center",
                zorder=3,
            )

            # Vertical arrow between cards inside a stage.
            if step_index < 2:
                _arrow(
                    ax,
                    (x + column_width / 2, y - 0.02),
                    (x + column_width / 2, y - card_gap + 0.02),
                )

        # Elbow connector from a stage's last card into the next stage's first card.
        if column_index < 2:
            last_y = top_of_cards - 3 * card_height - 2 * card_gap
            elbow_x = x + column_width + 0.30
            target_y = top_of_cards - card_height / 2
            _elbow(
                ax,
                [
                    (x + column_width + 0.04, last_y + card_height / 2),
                    (elbow_x, last_y + card_height / 2),
                    (elbow_x, target_y),
                    (x_positions[column_index + 1] - 0.06, target_y),
                ],
            )

    ax.text(
        0.62,
        0.58,
        "Fixed random seed 42  ·  artifacts in week-1/outputs/  ·  "
        "for modelling, the same transformer is re-fitted inside training folds only",
        fontsize=8.2,
        color="#7A8794",
        va="center",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH.relative_to(WEEK_DIR.parent)}")
    return OUTPUT_PATH


if __name__ == "__main__":
    draw()
