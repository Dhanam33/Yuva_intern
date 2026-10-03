"""Fit and serialize the production-ready churn prediction pipeline."""

from __future__ import annotations

import platform

import joblib
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .common import load_raw_data, split_features_target, write_json
from .preprocessing import build_preprocessor
from .schema import ARTIFACT_DIR, FEATURE_COLUMNS


def train_final_model() -> dict:
    data = load_raw_data()
    X, y = split_features_target(data)

    tuned_path = ARTIFACT_DIR.parent / "outputs" / "tuned_hyperparameters.json"
    tuned_c = 1.0
    if tuned_path.exists():
        import json

        tuned = json.loads(tuned_path.read_text(encoding="utf-8"))
        tuned_c = float(tuned.get("best_params", {}).get("classifier__C", tuned_c))

    model = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    C=tuned_c,
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X, y)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / "churn_model.joblib"
    joblib.dump(model, model_path, compress=3)

    metadata = {
        "artifact": str(model_path.relative_to(ARTIFACT_DIR.parent)),
        "model_type": "scikit-learn Pipeline(LogisticRegression)",
        "target": "churn (No=0, Yes=1)",
        "feature_columns": FEATURE_COLUMNS,
        "training_rows": int(len(data)),
        "scikit_learn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "tuned_regularization_C": tuned_c,
        "training_data": "synthetic customer churn sample, generated with seed 2026",
        "preprocessing_included": [
            "median imputation plus missingness indicators for numeric fields",
            "most-frequent imputation for categorical fields",
            "standard scaling for numeric fields",
            "one-hot encoding with unknown-category tolerance",
        ],
        "deployment_note": "Artifact contains the full preprocessing and prediction pipeline; supply raw feature values.",
    }
    write_json(ARTIFACT_DIR / "model_metadata.json", metadata)
    print(f"Saved {model_path} (trained on {len(data):,} records; C={tuned_c:g})")
    return metadata


if __name__ == "__main__":
    train_final_model()
