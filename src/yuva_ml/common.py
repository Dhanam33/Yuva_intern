"""Common paths, loading, and metrics helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .schema import (
    FEATURE_COLUMNS,
    OUTPUT_DIR,
    PLOT_DIR,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TARGET_MAP,
)


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the source CSV and validate the minimum expected columns."""
    data = pd.read_csv(path)
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN, "customer_id"])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return data


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model inputs and a numeric binary target (No=0, Yes=1)."""
    y = data[TARGET_COLUMN].map(TARGET_MAP)
    if y.isna().any():
        raise ValueError("Target must contain only 'No' and 'Yes' with no missing values")
    return data.loc[:, FEATURE_COLUMNS].copy(), y.astype(int)


def classification_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """Compute commonly used binary-classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")
