"""Leakage-safe preprocessing pipelines shared by all weekly exercises."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    *,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build a transformer that imputes, scales, and one-hot encodes features.

    It is intentionally a scikit-learn transformer so imputation statistics,
    encoding categories, and scaling parameters are learned only from the
    training fold when used inside a model pipeline or cross-validation.
    """
    numeric_features = list(NUMERIC_FEATURES if numeric_features is None else numeric_features)
    categorical_features = list(
        CATEGORICAL_FEATURES if categorical_features is None else categorical_features
    )

    numeric_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_features),
            ("cat", Pipeline(categorical_steps), categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
