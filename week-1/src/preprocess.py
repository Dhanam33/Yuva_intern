#!/usr/bin/env python3
"""Week 1 - Python for Machine Learning & Data Preprocessing.

A standalone, reproducible preprocessing pipeline for the customer-churn
sample. Running this file performs, in order:

    1. Data loading (Pandas)
    2. Data-quality audit (shape, dtypes, duplicates, invalid values)
    3. Cleaning (whitespace, empty strings, impossible values, duplicate rows)
    4. Missing-value audit and imputation strategy documentation
    5. Exploratory data analysis (six figures)
    6. Outlier audit (IQR rule; retained by design)
    7. Categorical encoding (one-hot, unknown-safe)
    8. Numeric scaling (StandardScaler / z-score)
    9. Feature selection (mutual information)
   10. Export of cleaned data, transformed data, audit tables, charts and a JSON summary

Usage
-----
    python src/preprocess.py

The script only needs numpy, pandas, scikit-learn and matplotlib. Every step
prints what it is doing, so it can be followed in a terminal or classroom.
All random operations use RANDOM_STATE for reproducibility.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------
WEEK_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = WEEK_DIR / "data" / "customer_churn_raw.csv"
OUTPUT_DIR = WEEK_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

ID_COLUMN = "customer_id"
TARGET_COLUMN = "churn"
TARGET_MAP = {"No": 0, "Yes": 1}

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets",
    "senior_citizen",
]
CATEGORICAL_FEATURES = [
    "contract",
    "internet_service",
    "payment_method",
    "paperless_billing",
    "partner",
    "dependents",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42

COLUMN_DESCRIPTIONS = {
    "customer_id": "Unique customer identifier. Used for traceability only, never as a model input.",
    "tenure_months": "Number of months the customer has been with the company.",
    "monthly_charges": "Current monthly bill in currency units.",
    "total_charges": "Cumulative amount billed to date.",
    "support_tickets": "Number of support tickets opened by the customer.",
    "senior_citizen": "Binary indicator: 1 if the customer is a senior citizen, otherwise 0.",
    "contract": "Contract type: Month-to-month, One year, or Two year.",
    "internet_service": "Internet service type: DSL, Fiber optic, or No.",
    "payment_method": "Billing method: Electronic check, Credit card, Bank transfer, or Mailed check.",
    "paperless_billing": "Whether the customer uses paperless billing (Yes/No).",
    "partner": "Whether the customer has a partner (Yes/No).",
    "dependents": "Whether the customer has dependents (Yes/No).",
    "churn": "Target label: Yes if the customer churned, No otherwise.",
}

COLORS = {
    "navy": "#17324D",
    "teal": "#237A7A",
    "coral": "#D17A61",
    "green": "#5B8E7D",
    "blue": "#5577AA",
    "amber": "#E0A458",
    "grey": "#8A97A5",
    "light": "#E8EEF3",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": COLORS["navy"],
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
    }
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def section(step: str, title: str) -> None:
    """Print a readable section banner in the terminal."""
    print(f"\n[{step}] {title}")
    print("-" * 78)


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Save a figure at report quality and close it."""
    path = FIGURE_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"    figure saved: {path.relative_to(WEEK_DIR.parent)}")
    return path


def write_csv(frame: pd.DataFrame, filename: str, **kwargs) -> Path:
    path = OUTPUT_DIR / filename
    frame.to_csv(path, **kwargs)
    print(f"    table saved:  {path.relative_to(WEEK_DIR.parent)}")
    return path


def _string_columns(frame: pd.DataFrame) -> list[str]:
    """Return text-like columns in a way that works with pandas 2 and 3."""
    return [column for column in frame.columns if is_string_dtype(frame[column])]


def _clean_category_series(series: pd.Series) -> pd.Series:
    """Trim surrounding whitespace and treat empty strings as missing."""
    cleaned = series.astype("object").map(lambda value: value.strip() if isinstance(value, str) else value)
    return cleaned.replace("", np.nan)


def _json_default(value):
    """Convert NumPy scalar types so the summary JSON keeps real numbers."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


# --------------------------------------------------------------------------
# Step 1 - Loading
# --------------------------------------------------------------------------
def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV and normalise column names.

    Pandas infers numeric columns automatically; text columns are kept as text
    so that the categorical encoding step can be explicit about them.
    """
    section("1/9", "Data loading")
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)
    frame.columns = [column.strip() for column in frame.columns]
    frame = frame.reset_index(drop=True)

    expected = set([ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS])
    missing_columns = expected.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    print(f"    file:    {path.relative_to(WEEK_DIR.parent)}")
    print(f"    shape:   {frame.shape[0]:,} rows x {frame.shape[1]} columns")
    print(f"    memory:  {frame.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"    columns: {', '.join(frame.columns)}")
    return frame


# --------------------------------------------------------------------------
# Step 2 - Quality audit
# --------------------------------------------------------------------------
def audit_raw_data(frame: pd.DataFrame) -> dict:
    """Audit the raw table before any modification.

    This step never changes data. It records the facts that justify every
    cleaning decision later in the pipeline.
    """
    section("2/9", "Data-quality audit")

    target_values = frame[TARGET_COLUMN].astype("object")
    unexpected_target = int((~target_values.isin(list(TARGET_MAP))).sum())
    if unexpected_target:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' contains {unexpected_target} value(s) "
            f"outside {sorted(TARGET_MAP)}."
        )

    whitespace_cells = 0
    empty_string_cells = 0
    for column in _string_columns(frame):
        text = frame[column].dropna().astype(str)
        whitespace_cells += int((text != text.str.strip()).sum())
        empty_string_cells += int((text.str.strip() == "").sum())

    negative_numeric_cells = {
        column: int((frame[column] < 0).sum()) for column in NUMERIC_FEATURES
    }
    invalid_senior_cells = int(
        (frame["senior_citizen"].notna() & ~frame["senior_citizen"].isin([0, 1])).sum()
    )

    per_column_missing = frame.isna().sum()
    audit = {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "memory_kb": round(float(frame.memory_usage(deep=True).sum() / 1024), 1),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_customer_ids": int(frame[ID_COLUMN].duplicated().sum()),
        "missing_cells_total": int(frame.isna().sum().sum()),
        "missing_cells_features": int(frame[FEATURE_COLUMNS].isna().sum().sum()),
        "missing_cells_percent": round(
            100.0 * float(frame[FEATURE_COLUMNS].isna().sum().sum()) / (len(frame) * len(FEATURE_COLUMNS)),
            2,
        ),
        "columns_with_missing": int((per_column_missing[FEATURE_COLUMNS] > 0).sum()),
        "missing_by_column": {
            column: int(per_column_missing[column]) for column in frame.columns
        },
        "whitespace_cells": whitespace_cells,
        "empty_string_cells": empty_string_cells,
        "negative_numeric_cells": negative_numeric_cells,
        "invalid_senior_citizen_cells": invalid_senior_cells,
        "target_counts": {
            label: int((target_values == label).sum()) for label in TARGET_MAP
        },
        "target_yes_rate": round(
            float((target_values == "Yes").mean()), 4
        ),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "numeric_summary": {
            column: {
                "count": int(frame[column].count()),
                "mean": round(float(frame[column].mean()), 3),
                "std": round(float(frame[column].std()), 3),
                "min": round(float(frame[column].min()), 3),
                "median": round(float(frame[column].median()), 3),
                "max": round(float(frame[column].max()), 3),
            }
            for column in NUMERIC_FEATURES
        },
        "categorical_levels": {
            column: int(frame[column].nunique(dropna=True))
            for column in CATEGORICAL_FEATURES
        },
        "checks": [
            {
                "check": "Duplicate rows",
                "result": int(frame.duplicated().sum()),
                "action": "Remove exact duplicates" if frame.duplicated().any() else "None required",
            },
            {
                "check": "Duplicate customer IDs",
                "result": int(frame[ID_COLUMN].duplicated().sum()),
                "action": "Keep first occurrence"
                if frame[ID_COLUMN].duplicated().any()
                else "None required",
            },
            {
                "check": "Missing feature cells",
                "result": int(frame[FEATURE_COLUMNS].isna().sum().sum()),
                "action": "Impute per column type",
            },
            {
                "check": "Leading/trailing whitespace cells",
                "result": whitespace_cells,
                "action": "Trim text values",
            },
            {
                "check": "Empty-string cells",
                "result": empty_string_cells,
                "action": "Treat as missing",
            },
            {
                "check": "Impossible negative numeric cells",
                "result": int(sum(negative_numeric_cells.values())),
                "action": "Treat as missing",
            },
            {
                "check": "Invalid senior_citizen codes",
                "result": invalid_senior_cells,
                "action": "Treat as missing",
            },
            {
                "check": "Target values outside {No, Yes}",
                "result": unexpected_target,
                "action": "Stop: target must be valid",
            },
        ],
    }

    print(f"    rows / columns:           {audit['rows']:,} / {audit['columns']}")
    print(f"    duplicate rows:           {audit['duplicate_rows']}")
    print(f"    duplicate customer IDs:   {audit['duplicate_customer_ids']}")
    print(f"    missing feature cells:    {audit['missing_cells_features']:,} "
          f"({audit['missing_cells_percent']}% of feature grid)")
    print(f"    whitespace / empty text:  {whitespace_cells} / {empty_string_cells}")
    print(f"    target:                   No={audit['target_counts']['No']:,}, "
          f"Yes={audit['target_counts']['Yes']:,} "
          f"({audit['target_yes_rate']:.1%} churn)")
    return audit


# --------------------------------------------------------------------------
# Step 3 - Cleaning
# --------------------------------------------------------------------------
def clean_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply deterministic cleaning rules and return a cleaning log.

    Rules
    -----
    * Trim whitespace from text and convert empty strings to missing.
    * Convert impossible values (negative charges/tenure/tickets and
      senior_citizen codes outside {0, 1}) to missing, then impute them later.
    * Drop rows with a missing or invalid target (labels cannot be imputed).
    * Drop exact duplicate rows and duplicate customer IDs.
    """
    section("3/9", "Data cleaning")

    clean = frame.copy()
    rows_start = len(clean)
    text_columns = _string_columns(clean)

    trimmed_cells = 0
    for column in text_columns:
        before = clean[column].dropna().astype(str)
        trimmed_cells += int((before != before.str.strip()).sum())
        clean[column] = _clean_category_series(clean[column])

    invalid_numeric_cells = 0
    for column in NUMERIC_FEATURES:
        invalid_mask = clean[column] < 0
        invalid_numeric_cells += int(invalid_mask.sum())
        clean.loc[invalid_mask, column] = np.nan

    invalid_senior_mask = clean["senior_citizen"].notna() & ~clean["senior_citizen"].isin([0, 1])
    invalid_senior_cells = int(invalid_senior_mask.sum())
    clean.loc[invalid_senior_mask, "senior_citizen"] = np.nan

    invalid_target_mask = ~clean[TARGET_COLUMN].isin(list(TARGET_MAP))
    invalid_target_rows = int(invalid_target_mask.sum())
    clean = clean.loc[~invalid_target_mask].copy()

    duplicate_rows = int(clean.duplicated().sum())
    if duplicate_rows:
        clean = clean.drop_duplicates().copy()

    duplicate_ids = int(clean[ID_COLUMN].duplicated().sum())
    if duplicate_ids:
        clean = clean.drop_duplicates(subset=ID_COLUMN, keep="first").copy()

    clean = clean.reset_index(drop=True)

    log = {
        "rows_input": int(rows_start),
        "rows_output": int(len(clean)),
        "rows_removed": int(rows_start - len(clean)),
        "invalid_target_rows_removed": invalid_target_rows,
        "duplicate_rows_removed": duplicate_rows,
        "duplicate_ids_removed": duplicate_ids,
        "whitespace_cells_trimmed": int(trimmed_cells),
        "empty_string_cells_set_missing": int(
            sum((frame[column].dropna().astype(str).str.strip() == "").sum() for column in text_columns)
        ),
        "invalid_numeric_cells_set_missing": int(invalid_numeric_cells),
        "invalid_senior_citizen_cells_set_missing": invalid_senior_cells,
        "rules": [
            "Trim whitespace and treat empty strings as missing.",
            "Negative numeric values and invalid senior_citizen codes become missing.",
            "Rows with missing or invalid target labels are removed.",
            "Exact duplicate rows are removed.",
            "Duplicate customer IDs keep the first occurrence.",
        ],
    }

    print(f"    rows before / after:      {rows_start:,} / {len(clean):,}")
    print(f"    duplicate rows removed:   {duplicate_rows}")
    print(f"    duplicate IDs removed:    {duplicate_ids}")
    print(f"    whitespace cells trimmed: {trimmed_cells}")
    print(f"    invalid values -> missing:{invalid_numeric_cells + invalid_senior_cells}")
    print(f"    invalid target rows:      {invalid_target_rows}")
    return clean, log


# --------------------------------------------------------------------------
# Step 4 - Missing-value audit
# --------------------------------------------------------------------------
def missing_value_audit(clean: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Document missingness and the fill value learned for each feature.

    Numeric columns use the median because it is robust to skew and outliers.
    Categorical columns use the most frequent value (mode). Missingness
    indicators are added by the scikit-learn imputer so that "was missing"
    remains available to a model.
    """
    section("4/9", "Missing-value audit and imputation strategy")

    rows: list[dict] = []
    fill_values: dict[str, object] = {}
    for column in FEATURE_COLUMNS:
        missing_count = int(clean[column].isna().sum())
        missing_percent = 100.0 * missing_count / len(clean)
        is_numeric = column in NUMERIC_FEATURES
        kind = "numeric" if is_numeric else "categorical"

        if is_numeric:
            learned_value = float(clean[column].median())
            strategy = "Median imputation + missing indicator"
            display_value: object = round(learned_value, 2)
        else:
            modes = clean[column].mode(dropna=True)
            learned_value = modes.iloc[0] if not modes.empty else "Unknown"
            strategy = "Most-frequent (mode) imputation"
            display_value = learned_value

        fill_values[column] = learned_value
        rows.append(
            {
                "feature": column,
                "kind": kind,
                "dtype": str(clean[column].dtype),
                "missing_count": missing_count,
                "missing_percent": round(missing_percent, 2),
                "strategy": strategy if missing_count else "No action required",
                "fill_value": display_value if missing_count else "-",
            }
        )

    audit = pd.DataFrame(rows).sort_values(
        ["missing_count", "feature"], ascending=[False, True]
    )
    total_filled = int(audit["missing_count"].sum())
    summary = {
        "columns_with_missing": int((audit["missing_count"] > 0).sum()),
        "total_cells_filled": total_filled,
        "highest_missing_feature": audit.iloc[0]["feature"],
        "highest_missing_count": int(audit.iloc[0]["missing_count"]),
        "numeric_strategy": "Median imputation with missing indicators",
        "categorical_strategy": "Most-frequent (mode) imputation",
    }

    print(f"    columns with missing:     {summary['columns_with_missing']}")
    print(f"    cells to fill:            {total_filled:,}")
    print(f"    highest:                  {summary['highest_missing_feature']} "
          f"({summary['highest_missing_count']} cells)")
    return audit, fill_values


# --------------------------------------------------------------------------
# Step 5 - Outlier audit
# --------------------------------------------------------------------------
BINARY_FEATURES = {"senior_citizen"}


def outlier_audit(clean: pd.DataFrame) -> pd.DataFrame:
    """Count outliers with the 1.5 x IQR rule on continuous features only.

    Outliers are *not* deleted. Extreme but plausible bills can carry real
    signal, and deleting them would lose rows. The audit documents them so the
    choice is explicit; later modelling can use robust or tree-based models.
    Binary 0/1 flags are excluded because every "1" would be flagged as an
    outlier by a rule designed for continuous measurements.
    """
    section("5/9", "Outlier audit (IQR rule)")

    rows = []
    for column in NUMERIC_FEATURES:
        values = clean[column].dropna()
        if column in BINARY_FEATURES:
            rows.append(
                {
                    "feature": column,
                    "q1": round(float(values.quantile(0.25)), 2),
                    "q3": round(float(values.quantile(0.75)), 2),
                    "iqr": round(float(values.quantile(0.75) - values.quantile(0.25)), 2),
                    "lower_fence": "-",
                    "upper_fence": "-",
                    "outlier_count": 0,
                    "outlier_percent": 0.0,
                    "action": "Not assessed (binary 0/1 flag)",
                }
            )
            continue

        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((values < lower) | (values > upper)).sum())
        rows.append(
            {
                "feature": column,
                "q1": round(q1, 2),
                "q3": round(q3, 2),
                "iqr": round(iqr, 2),
                "lower_fence": round(lower, 2),
                "upper_fence": round(upper, 2),
                "outlier_count": count,
                "outlier_percent": round(100.0 * count / len(clean), 2),
                "action": "Retained" if count else "No outliers",
            }
        )

    audit = pd.DataFrame(rows)
    assessed = audit[audit["action"] != "Not assessed (binary 0/1 flag)"]
    print(f"    outliers detected:        {int(assessed['outlier_count'].sum())} "
          f"across {int((assessed['outlier_count'] > 0).sum())} continuous feature(s); all retained")
    return audit


# --------------------------------------------------------------------------
# Step 6 - EDA figures
# --------------------------------------------------------------------------
def plot_target_distribution(clean: pd.DataFrame) -> Path:
    counts = clean[TARGET_COLUMN].value_counts().reindex(["No", "Yes"]).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[COLORS["green"], COLORS["coral"]],
        width=0.55,
        edgecolor="white",
    )
    labels = [f"{value:,}\n({value / len(clean):.1%})" for value in counts.values]
    ax.bar_label(bars, labels=labels, padding=5, fontsize=10, fontweight="bold", color=COLORS["navy"])
    ax.set_title("Churn target distribution")
    ax.set_xlabel("Churn label")
    ax.set_ylabel("Number of customers")
    ax.set_ylim(0, counts.max() * 1.20)
    ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.02, f"Source: customer_churn_raw.csv, n = {len(clean):,}", fontsize=7.5, color=COLORS["grey"])
    return save_figure(fig, "01_target_distribution.png")


def plot_missing_values(missing_audit: pd.DataFrame, n_rows: int) -> Path:
    data = missing_audit.sort_values("missing_count", ascending=True)
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    colors = [COLORS["coral"] if value > 0 else "#C9D3DC" for value in data["missing_count"]]
    bars = ax.barh(data["feature"], data["missing_count"], color=colors, height=0.62)
    labels = [
        f"{int(value):,} ({value / n_rows:.1%})" if value else "0"
        for value in data["missing_count"]
    ]
    ax.bar_label(bars, labels=labels, padding=4, fontsize=8.5, color=COLORS["navy"])
    ax.set_title("Missing values by feature (before imputation)")
    ax.set_xlabel("Missing cells")
    ax.set_ylabel("Feature")
    ax.set_xlim(0, max(data["missing_count"].max() * 1.18, 5))
    ax.grid(axis="y", visible=False)
    return save_figure(fig, "02_missing_values.png")


def plot_numeric_distributions(clean: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.2))
    axes = axes.ravel()

    continuous = ["tenure_months", "monthly_charges", "total_charges", "support_tickets"]
    for ax, column in zip(axes, continuous):
        values = clean[column].dropna()
        ax.hist(values, bins=28, color=COLORS["blue"], alpha=0.85, edgecolor="white", linewidth=0.6)
        median = float(values.median())
        mean = float(values.mean())
        ax.axvline(median, color=COLORS["coral"], linewidth=1.8, label=f"median {median:,.1f}")
        ax.axvline(mean, color=COLORS["amber"], linewidth=1.4, linestyle="--", label=f"mean {mean:,.1f}")
        ax.set_title(column.replace("_", " ").title())
        ax.set_ylabel("Customers")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        ax.grid(axis="x", visible=False)

    senior_counts = clean["senior_citizen"].value_counts().reindex([0, 1]).fillna(0).astype(int)
    bars = axes[4].bar(
        ["Not senior (0)", "Senior (1)"],
        senior_counts.values,
        color=[COLORS["blue"], COLORS["teal"]],
        width=0.55,
    )
    axes[4].bar_label(bars, labels=[f"{value:,}" for value in senior_counts.values], padding=3, fontsize=8)
    axes[4].set_title("Senior Citizen (binary)")
    axes[4].set_ylabel("Customers")
    axes[4].grid(axis="x", visible=False)

    axes[5].axis("off")
    axes[5].text(
        0.02,
        0.92,
        "Reading the plots\n\n"
        "• Histograms show shape, skew and\n  spread before scaling.\n"
        "• The median is robust to outliers;\n  the mean is pulled by extreme bills.\n"
        "• senior_citizen is a 0/1 flag, so a\n  count plot is more informative\n  than a histogram.",
        va="top",
        fontsize=8.4,
        color=COLORS["navy"],
        linespacing=1.5,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": COLORS["light"], "edgecolor": "none"},
    )
    fig.suptitle("Numeric feature distributions after cleaning", fontsize=13, fontweight="bold", color=COLORS["navy"])
    return save_figure(fig, "03_numeric_distributions.png")


def plot_correlation_matrix(clean: pd.DataFrame) -> Path:
    work = clean[NUMERIC_FEATURES].copy()
    work["churn (0/1)"] = clean[TARGET_COLUMN].map(TARGET_MAP).astype(int)
    correlation = work.corr()

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    image = ax.imshow(correlation.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.columns)))
    ax.set_xticklabels([column.replace("_", " ") for column in correlation.columns], rotation=35, ha="right")
    ax.set_yticklabels([column.replace("_", " ") for column in correlation.columns])
    for row in range(len(correlation)):
        for col in range(len(correlation)):
            value = correlation.iloc[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if abs(value) > 0.52 else COLORS["navy"],
                fontweight="bold" if row != col else "normal",
            )
    ax.set_title("Pearson correlation: numeric features and churn")
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Correlation coefficient", fontsize=8)
    fig.text(
        0.01,
        -0.02,
        "Pairwise-complete observations. Correlation measures linear association only; it is not causation.",
        fontsize=7.4,
        color=COLORS["grey"],
    )
    return save_figure(fig, "04_correlation_matrix.png")


def plot_churn_by_category(clean: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.6))
    axes = axes.ravel()
    churn_flag = clean[TARGET_COLUMN].map(TARGET_MAP).astype(int)

    for ax, column in zip(axes, CATEGORICAL_FEATURES):
        groups = clean[column].fillna("(missing)").astype(str)
        table = (
            pd.DataFrame({"group": groups, "churn": churn_flag})
            .groupby("group", observed=True)["churn"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
        )
        bars = ax.bar(
            range(len(table)),
            table["mean"].values,
            color=COLORS["teal"],
            width=0.62,
            edgecolor="white",
        )
        ax.set_xticks(range(len(table)))
        ax.set_xticklabels(
            [f"{label}\n(n={int(count):,})" for label, count in zip(table.index, table["count"])],
            fontsize=7.4,
        )
        for bar, value in zip(bars, table["mean"].values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.1%}",
                ha="center",
                fontsize=7.8,
                fontweight="bold",
                color=COLORS["navy"],
            )
        ax.set_title(column.replace("_", " ").title())
        ax.set_ylim(0, min(1.0, max(0.05, float(table["mean"].max()) * 1.25)))
        ax.set_ylabel("Churn rate")
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelrotation=12)

    fig.suptitle(
        "Observed churn rate by categorical level",
        fontsize=13,
        fontweight="bold",
        color=COLORS["navy"],
    )
    fig.text(
        0.01,
        -0.01,
        "Descriptive only: rates are not causal effects, and small groups have wider uncertainty. "
        "Missing categories are shown explicitly here; models impute them.",
        fontsize=7.4,
        color=COLORS["grey"],
    )
    return save_figure(fig, "05_churn_by_category.png")


CONTINUOUS_FEATURES = [column for column in NUMERIC_FEATURES if column not in BINARY_FEATURES]


def plot_outliers(clean: pd.DataFrame, outlier_table: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, len(CONTINUOUS_FEATURES), figsize=(12.0, 4.2))
    axes = np.atleast_1d(axes).ravel()
    for ax, column in zip(axes, CONTINUOUS_FEATURES):
        values = clean[column].dropna()
        ax.boxplot(
            values,
            patch_artist=True,
            widths=0.45,
            medianprops={"color": COLORS["coral"], "linewidth": 2},
            boxprops={"facecolor": COLORS["light"], "edgecolor": COLORS["blue"], "linewidth": 1.2},
            whiskerprops={"color": COLORS["blue"]},
            capprops={"color": COLORS["blue"]},
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": COLORS["coral"],
                "markeredgecolor": "none",
                "alpha": 0.55,
            },
        )
        count = int(outlier_table.loc[outlier_table["feature"] == column, "outlier_count"].iloc[0])
        ax.set_title(f"{column.replace('_', ' ').title()}\n({count} outliers)", fontsize=9)
        ax.set_xticks([])
        ax.grid(axis="x", visible=False)

    fig.suptitle(
        "Outliers by feature (1.5 × IQR fences) - retained in the dataset",
        fontsize=12,
        fontweight="bold",
        color=COLORS["navy"],
    )
    return save_figure(fig, "06_outliers_boxplot.png")


def make_eda_figures(
    clean: pd.DataFrame,
    missing_table: pd.DataFrame,
    outlier_table: pd.DataFrame,
) -> list[Path]:
    """Create the exploratory figures and return their paths."""
    section("6/9", "Exploratory data analysis")
    return [
        plot_target_distribution(clean),
        plot_missing_values(missing_table, len(clean)),
        plot_numeric_distributions(clean),
        plot_correlation_matrix(clean),
        plot_churn_by_category(clean),
        plot_outliers(clean, outlier_table),
    ]


# --------------------------------------------------------------------------
# Step 7 - Encoding + scaling (scikit-learn transformer)
# --------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Build the imputation -> encoding/scaling transformer.

    The transformer is a scikit-learn object, so if it is later placed inside a
    model pipeline it learns medians, modes, categories and scaling parameters
    from the training data only. That prevents information leakage.
    """
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ]
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("categorical", Pipeline(categorical_steps), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transform_features(
    clean: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], ColumnTransformer, pd.DataFrame, pd.DataFrame]:
    """Impute, encode and scale all features; return the transformed table."""
    section("7/9", "Categorical encoding and numeric scaling")

    features = clean[FEATURE_COLUMNS]
    target = clean[TARGET_COLUMN].map(TARGET_MAP).astype(int)

    preprocessor = build_preprocessor()
    matrix = preprocessor.fit_transform(features, target)
    feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
    transformed = pd.DataFrame(matrix, columns=feature_names, index=clean.index)

    if transformed.isna().any().any():
        raise RuntimeError("Transformed data still contains missing values")

    # Encoding audit -------------------------------------------------------
    encoder: OneHotEncoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    encoding_rows = []
    for column, categories in zip(CATEGORICAL_FEATURES, encoder.categories_):
        encoding_rows.append(
            {
                "feature": column,
                "levels": len(categories),
                "categories": ", ".join(map(str, categories)),
                "encoded_columns": len(categories),
            }
        )
    encoding_table = pd.DataFrame(encoding_rows)
    encoded_columns = int(sum(encoding_table["encoded_columns"]))
    indicators = int(sum(name.startswith("missingindicator_") for name in feature_names))

    # Scaling audit --------------------------------------------------------
    scaling_rows = []
    for column in NUMERIC_FEATURES:
        scaling_rows.append(
            {
                "feature": column,
                "mean_before": round(float(clean[column].mean()), 4),
                "std_before": round(float(clean[column].std()), 4),
                "mean_after": round(float(transformed[column].mean()), 4),
                "std_after": round(float(transformed[column].std()), 4),
            }
        )
    scaling_table = pd.DataFrame(scaling_rows)

    print(f"    numeric inputs:           {len(NUMERIC_FEATURES)} "
          f"({len(NUMERIC_FEATURES)} standardized + {indicators} missing indicators)")
    print(f"    categorical inputs:       {len(CATEGORICAL_FEATURES)} -> {encoded_columns} one-hot columns")
    print(f"    total model features:     {len(feature_names)}")
    print("    remaining missing cells:  0")
    print("    scaling check (mean/std): standardized numeric columns centred at 0 with unit variance")
    return transformed, feature_names, preprocessor, encoding_table, scaling_table


# --------------------------------------------------------------------------
# Step 8 - Feature selection
# --------------------------------------------------------------------------
def rank_features(
    transformed: pd.DataFrame,
    target: pd.Series,
    feature_names: list[str],
) -> pd.DataFrame:
    """Rank encoded features with mutual information and the churn target.

    Mutual information captures non-linear dependence as well as linear
    association. One-hot columns and missing indicators are declared discrete.
    This ranking is an explanatory Week 1 artifact; any feature selection used
    in modelling should be re-fitted inside training folds.
    """
    section("8/9", "Feature selection with mutual information")

    discrete_mask = np.array(
        [
            any(name.startswith(f"{column}_") for column in CATEGORICAL_FEATURES)
            or name.startswith("missingindicator_")
            or name in {"senior_citizen"}
            for name in feature_names
        ],
        dtype=bool,
    )
    # n_neighbors=5 (instead of the default 3) stabilises the k-NN based
    # estimator on this sample size; with 3 neighbours, weak signals such as
    # tenure_months collapse to exactly zero from sampling noise alone.
    scores = mutual_info_classif(
        transformed.to_numpy(),
        target.to_numpy(),
        discrete_features=discrete_mask,
        n_neighbors=5,
        random_state=RANDOM_STATE,
    )
    ranking = (
        pd.DataFrame({"feature": feature_names, "mutual_information": scores})
        .sort_values("mutual_information", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    ranking.index = ranking.index + 1
    ranking.index.name = "rank"

    print("    method: mutual_info_classif(n_neighbors=5, discrete one-hot + indicators)")
    print("    top five features:")
    for rank, row in ranking.head(5).iterrows():
        print(f"      {rank}. {row['feature']:<34} {row['mutual_information']:.4f}")
    return ranking


def plot_mutual_information(ranking: pd.DataFrame, top_n: int = 12) -> Path:
    data = ranking.head(top_n).sort_values("mutual_information", ascending=True)
    fig, ax = plt.subplots(figsize=(8.9, 5.6))
    colors = [COLORS["teal"]] * len(data)
    bars = ax.barh(data["feature"], data["mutual_information"], color=colors, height=0.62, alpha=0.9)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in data["mutual_information"]], padding=4, fontsize=8)
    ax.set_title(f"Top {top_n} encoded features by mutual information with churn")
    ax.set_xlabel("Mutual information score (nats)")
    ax.set_ylabel("Encoded feature")
    ax.set_xlim(0, max(float(data["mutual_information"].max()) * 1.18, 0.01))
    ax.grid(axis="y", visible=False)
    fig.text(
        0.01,
        -0.02,
        "Scores computed on the full cleaned sample for explanation. For modelling, re-fit selection inside training folds.",
        fontsize=7.4,
        color=COLORS["grey"],
    )
    return save_figure(fig, "07_mutual_information.png")


# --------------------------------------------------------------------------
# Step 9 - Human-readable export, scaling figure, model-readiness check
# --------------------------------------------------------------------------
def impute_for_export(clean: pd.DataFrame, fill_values: dict[str, object]) -> pd.DataFrame:
    """Create a readable, fully imputed table in the original column space."""
    readable = clean.copy()
    for column in NUMERIC_FEATURES:
        readable[column] = readable[column].astype(float).fillna(float(fill_values[column]))
    for column in CATEGORICAL_FEATURES:
        readable[column] = readable[column].astype("object").fillna(fill_values[column])
    return readable


def plot_scaling_effect(
    readable: pd.DataFrame,
    transformed: pd.DataFrame,
    columns: tuple[str, str] = ("monthly_charges", "total_charges"),
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.0))
    for index, column in enumerate(columns):
        before = readable[column].dropna()
        after = transformed[column]

        axes[index, 0].hist(before, bins=30, color=COLORS["blue"], alpha=0.85, edgecolor="white", linewidth=0.5)
        axes[index, 0].set_title(f"{column.replace('_', ' ').title()} - before scaling")
        axes[index, 0].set_xlabel("Original units")
        axes[index, 0].set_ylabel("Customers")
        axes[index, 0].axvline(before.median(), color=COLORS["coral"], linewidth=1.6, label="median")
        axes[index, 0].legend(fontsize=7, frameon=False)

        axes[index, 1].hist(after, bins=30, color=COLORS["teal"], alpha=0.85, edgecolor="white", linewidth=0.5)
        axes[index, 1].set_title(f"{column.replace('_', ' ').title()} - after StandardScaler")
        axes[index, 1].set_xlabel("z-score (standard deviations from mean)")
        axes[index, 1].set_ylabel("Customers")
        axes[index, 1].axvline(0, color=COLORS["coral"], linewidth=1.6, label="mean = 0")
        axes[index, 1].legend(fontsize=7, frameon=False)

    fig.suptitle(
        "Effect of standardization on numeric features",
        fontsize=12,
        fontweight="bold",
        color=COLORS["navy"],
    )
    return save_figure(fig, "08_scaling_effect.png")


def model_readiness_check(clean: pd.DataFrame) -> dict:
    """Fit a tiny baseline to prove the preprocessing pipeline is usable.

    This is a *smoke test*, not the Week 1 deliverable and not a tuned model.
    The whole pipeline (impute + encode + scale + classifier) is fitted on the
    training split only, then scored once on the held-out split.
    """
    section("9/9", "Model-readiness check (pipeline smoke test)")

    features = clean[FEATURE_COLUMNS]
    target = clean[TARGET_COLUMN].map(TARGET_MAP).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "model",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]

    result = {
        "purpose": "Smoke test only: confirms the preprocessing pipeline is model-ready; not a tuned modelling result.",
        "model": "LogisticRegression (class_weight='balanced')",
        "test_size": 0.20,
        "random_state": RANDOM_STATE,
        "test_rows": int(len(y_test)),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
    }
    print(f"    held-out accuracy:        {result['accuracy']:.4f}")
    print(f"    held-out ROC-AUC:         {result['roc_auc']:.4f}")
    return result


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_pipeline() -> dict:
    """Run every step, write artifacts and return the summary dictionary."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("YUVA INTERNSHIP - WEEK 1: PYTHON FOR ML & DATA PREPROCESSING")
    print("=" * 78)

    raw = load_raw_data()
    audit = audit_raw_data(raw)
    clean, cleaning_log = clean_data(raw)
    missing_table, fill_values = missing_value_audit(clean)
    outlier_table = outlier_audit(clean)

    write_csv(
        missing_table,
        "missing_value_audit.csv",
        index=False,
    )
    write_csv(
        pd.DataFrame(audit["checks"]),
        "data_quality_checks.csv",
        index=False,
    )
    write_csv(
        pd.DataFrame(audit["numeric_summary"]).T.rename_axis("feature").reset_index(),
        "numeric_summary.csv",
        index=False,
    )

    categorical_rows = []
    for column in CATEGORICAL_FEATURES:
        counts = clean[column].value_counts(dropna=False)
        for level, count in counts.items():
            categorical_rows.append(
                {
                    "feature": column,
                    "level": "(missing)" if pd.isna(level) else level,
                    "count": int(count),
                    "percent": round(100.0 * count / len(clean), 2),
                }
            )
    write_csv(pd.DataFrame(categorical_rows), "categorical_summary.csv", index=False)
    write_csv(outlier_table, "outlier_audit.csv", index=False)

    readable = impute_for_export(clean, fill_values)
    if readable[FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("Readable cleaned export still contains missing values")
    readable.to_csv(OUTPUT_DIR / "cleaned_customer_churn.csv", index=False)
    print("    table saved:  week-1/outputs/cleaned_customer_churn.csv")

    make_eda_figures(clean, missing_table, outlier_table)

    transformed, feature_names, preprocessor, encoding_table, scaling_table = transform_features(clean)
    write_csv(encoding_table, "encoding_summary.csv", index=False)
    write_csv(scaling_table, "scaling_summary.csv", index=False)

    target = clean[TARGET_COLUMN].map(TARGET_MAP).astype(int)
    ranking = rank_features(transformed, target, feature_names)
    write_csv(ranking.reset_index(), "feature_selection.csv", index=False)
    plot_mutual_information(ranking)

    top_features = ranking.head(10)["feature"].tolist()
    selected = pd.DataFrame(
        {
            ID_COLUMN: clean[ID_COLUMN].values,
            **{name: transformed[name].values for name in top_features},
            TARGET_COLUMN: target.values,
        }
    )
    selected.to_csv(OUTPUT_DIR / "selected_feature_dataset.csv", index=False)
    print("    table saved:  week-1/outputs/selected_feature_dataset.csv")

    export = transformed.copy()
    export.insert(0, ID_COLUMN, clean[ID_COLUMN].values)
    export[TARGET_COLUMN] = target.values
    export.to_csv(OUTPUT_DIR / "preprocessed_customer_churn.csv", index=False)
    print("    table saved:  week-1/outputs/preprocessed_customer_churn.csv")

    plot_scaling_effect(readable, transformed)
    readiness = model_readiness_check(clean)

    summary = {
        "project": "Yuva Internship - Week 1: Python for Machine Learning & Data Preprocessing",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "source_file": "week-1/data/customer_churn_raw.csv",
            "description": (
                "Reproducible synthetic customer-churn sample with numeric and categorical "
                "inputs, a binary churn label and deliberately introduced missing values. "
                "It is teaching data, not real customer information."
            ),
            "rows_raw": audit["rows"],
            "columns_raw": audit["columns"],
            "rows_clean": int(len(clean)),
            "target_counts": audit["target_counts"],
            "target_yes_rate": audit["target_yes_rate"],
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "id_column": ID_COLUMN,
            "target_column": TARGET_COLUMN,
            "column_dictionary": COLUMN_DESCRIPTIONS,
        },
        "quality_audit": audit,
        "cleaning": cleaning_log,
        "missing_values": {
            "columns_with_missing": int((missing_table["missing_count"] > 0).sum()),
            "total_cells_filled": int(missing_table["missing_count"].sum()),
            "highest_missing_feature": missing_table.iloc[0]["feature"],
            "highest_missing_count": int(missing_table.iloc[0]["missing_count"]),
            "numeric_strategy": "Median imputation with missing indicators",
            "categorical_strategy": "Most-frequent (mode) imputation",
            "remaining_missing_cells": int(transformed[feature_names].isna().sum().sum()),
        },
        "outliers": {
            "method": "1.5 x IQR fences computed per numeric feature",
            "total_outliers": int(outlier_table["outlier_count"].sum()),
            "features_with_outliers": int((outlier_table["outlier_count"] > 0).sum()),
            "action": "Retained; documented for transparency",
            "by_feature": outlier_table.to_dict(orient="records"),
        },
        "encoding": {
            "method": "OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
            "input_categorical_features": len(CATEGORICAL_FEATURES),
            "output_one_hot_columns": int(encoding_table["encoded_columns"].sum()),
            "by_feature": encoding_table.to_dict(orient="records"),
        },
        "scaling": {
            "method": "StandardScaler (z-score): x' = (x - mean) / standard deviation",
            "features_scaled": len(NUMERIC_FEATURES),
            "missing_indicators": int(sum(name.startswith("missingindicator_") for name in feature_names)),
            "by_feature": scaling_table.to_dict(orient="records"),
        },
        "feature_selection": {
            "method": (
                "Mutual information (mutual_info_classif, n_neighbors=5); one-hot columns, "
                "binary flags and missing indicators declared discrete"
            ),
            "top_10": top_features,
            "top_3": ranking.head(3)["feature"].tolist(),
            "ranking": ranking.reset_index().to_dict(orient="records"),
            "note": (
                "The ranking is an explanatory Week 1 artifact computed on the full cleaned "
                "sample. Any feature selection used for model training should be re-fitted "
                "inside training folds to avoid leakage. Mutual information scores each "
                "feature marginally, so a weak-but-real signal can score near zero and an "
                "individually useless feature can still matter in combination with others."
            ),
        },
        "final_dataset": {
            "rows": int(len(clean)),
            "model_features": int(len(feature_names)),
            "remaining_missing_cells": int(transformed.isna().sum().sum()),
            "target_encoding": "No = 0, Yes = 1",
            "features": feature_names,
        },
        "model_readiness_check": readiness,
        "artifacts": [
            "week-1/outputs/cleaned_customer_churn.csv",
            "week-1/outputs/preprocessed_customer_churn.csv",
            "week-1/outputs/selected_feature_dataset.csv",
            "week-1/outputs/missing_value_audit.csv",
            "week-1/outputs/data_quality_checks.csv",
            "week-1/outputs/numeric_summary.csv",
            "week-1/outputs/categorical_summary.csv",
            "week-1/outputs/outlier_audit.csv",
            "week-1/outputs/encoding_summary.csv",
            "week-1/outputs/scaling_summary.csv",
            "week-1/outputs/feature_selection.csv",
            "week-1/outputs/figures/",
        ],
    }

    write_csv(
        pd.DataFrame(
            [
                {"metric": "Rows (raw)", "value": audit["rows"]},
                {"metric": "Rows (clean)", "value": len(clean)},
                {"metric": "Raw feature columns", "value": len(FEATURE_COLUMNS)},
                {"metric": "Raw missing feature cells", "value": audit["missing_cells_features"]},
                {"metric": "Encoded/scaled model features", "value": len(feature_names)},
                {"metric": "Remaining missing cells", "value": int(transformed.isna().sum().sum())},
                {"metric": "Top feature (mutual information)", "value": ranking.iloc[0]["feature"]},
                {"metric": "Top mutual information score", "value": round(float(ranking.iloc[0]["mutual_information"]), 4)},
            ]
        ),
        "summary_metrics.csv",
        index=False,
    )

    summary_path = OUTPUT_DIR / "preprocessing_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    print("\n    summary saved: week-1/outputs/preprocessing_summary.json")

    print("\n" + "=" * 78)
    print("WEEK 1 PIPELINE COMPLETE")
    print("=" * 78)
    print(f"  raw rows .................. {audit['rows']:,}")
    print(f"  clean rows ................ {len(clean):,}")
    print(f"  missing cells filled ...... {int(missing_table['missing_count'].sum()):,}")
    print(f"  model-ready feature columns {len(feature_names)}")
    print(f"  top feature ............... {ranking.iloc[0]['feature']}")
    print("  outputs ................... week-1/outputs/")
    return summary


if __name__ == "__main__":
    run_pipeline()
