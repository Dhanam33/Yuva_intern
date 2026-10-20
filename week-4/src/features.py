"""Feature engineering and the shared scikit-learn pipeline.

The same `build_pipeline` function is used by the training script and by any
code that needs to rebuild the model, which is what guarantees that training
and serving cannot drift apart.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TARGET_MAP,
)


def build_preprocessor() -> ColumnTransformer:
    """Impute, scale and one-hot encode.

    Wrapping every step in a transformer means the medians, category lists and
    scaling parameters are learned from the training data only. At serving time
    the identical object is reused, so an unseen category cannot crash the API
    (``handle_unknown="ignore"``) and a missing value cannot reach the model.
    """
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(**logistic_params) -> Pipeline:
    """Full preprocessing + estimator pipeline."""
    params = {
        "C": 0.1,
        "penalty": "l2",
        "solver": "lbfgs",
        "class_weight": "balanced",
        "max_iter": 2000,
        "random_state": RANDOM_STATE,
    }
    params.update(logistic_params)
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", LogisticRegression(**params)),
        ]
    )


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Strip column names and drop the identifier and target."""
    frame = frame.copy()
    frame.columns = [column.strip() for column in frame.columns]
    missing = {*FEATURE_COLUMNS, TARGET_COLUMN}.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return frame


def split_features_target(frame: pd.DataFrame):
    """Return (X, y) with the target encoded as 0/1."""
    features = frame[FEATURE_COLUMNS].copy()
    target = frame[TARGET_COLUMN].map(TARGET_MAP)
    if target.isna().any():
        bad = frame.loc[target.isna(), TARGET_COLUMN].unique().tolist()
        raise ValueError(f"Unexpected target values: {bad}")
    return features, target.astype(int)


def feature_names_out(pipeline: Pipeline) -> list[str]:
    """Human-readable names for the transformed columns."""
    return list(pipeline.named_steps["preprocessor"].get_feature_names_out())
