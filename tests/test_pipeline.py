from __future__ import annotations

import numpy as np

from yuva_ml.common import load_raw_data, split_features_target
from yuva_ml.preprocessing import build_preprocessor
from yuva_ml.schema import FEATURE_COLUMNS


def test_dataset_has_expected_schema_and_intentional_missing_values():
    data = load_raw_data()
    assert len(data) == 1800
    assert set(FEATURE_COLUMNS).issubset(data.columns)
    assert data["customer_id"].is_unique
    assert data["churn"].notna().all()
    assert data[FEATURE_COLUMNS].isna().sum().sum() > 0


def test_preprocessing_imputes_and_encodes_without_nan():
    data = load_raw_data()
    X, y = split_features_target(data)
    transformer = build_preprocessor()
    transformed = transformer.fit_transform(X, y)
    assert transformed.shape[0] == len(data)
    assert transformed.shape[1] > len(FEATURE_COLUMNS)
    assert np.isfinite(transformed).all()
    assert set(y.unique()) == {0, 1}
