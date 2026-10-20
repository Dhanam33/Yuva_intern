#!/usr/bin/env python3
"""Week 2 - Supervised Machine Learning Models.

Trains and compares supervised models with scikit-learn:

    Regression (numeric target: monthly_charges)
        Linear Regression, Decision Tree Regressor,
        Random Forest Regressor, K-Nearest Neighbors Regressor

    Classification (categorical target: churn)
        Logistic Regression, Decision Tree,
        Random Forest, K-Nearest Neighbors

Every model is wrapped in an identical pipeline, so imputation, scaling and
one-hot encoding are fitted inside the training data only. That makes the
comparison a comparison of algorithms rather than of data preparation, and it
prevents information leakage from the test split.

Model selection uses 5-fold stratified cross-validation on the training data.
The held-out test split is scored once, at the end, and is never used to choose
a model or a hyperparameter.

Usage
-----
    python src/train_models.py     # trains, evaluates and exports every table
    python src/make_figures.py     # builds the figures from those tables
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------
WEEK_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = WEEK_DIR / "data" / "customer_churn_raw.csv"
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

# Monthly charges is the regression target. total_charges is excluded because
# it is computed from monthly_charges and tenure, so including it would leak
# the answer into the inputs.
REGRESSION_TARGET = "monthly_charges"
REGRESSION_NUMERIC = ["tenure_months", "support_tickets", "senior_citizen"]
REGRESSION_CATEGORICAL = CATEGORICAL_FEATURES
REGRESSION_FEATURES = REGRESSION_NUMERIC + REGRESSION_CATEGORICAL

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


def section(step: str, title: str) -> None:
    print(f"\n[{step}] {title}")
    print("-" * 78)


def _json_default(value):
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
# Step 1 - Load
# --------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    section("1/7", "Loading the dataset")
    frame = pd.read_csv(DATA_PATH)
    frame.columns = [column.strip() for column in frame.columns]

    required = {ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    unexpected = ~frame[TARGET_COLUMN].isin(list(TARGET_MAP))
    if unexpected.any():
        raise ValueError(f"Target contains {int(unexpected.sum())} invalid value(s)")

    print(f"    file:  {DATA_PATH.relative_to(WEEK_DIR.parent)}")
    print(f"    shape: {frame.shape[0]:,} rows x {frame.shape[1]} columns")
    print(f"    churn: No {int((frame[TARGET_COLUMN] == 'No').sum()):,}, "
          f"Yes {int((frame[TARGET_COLUMN] == 'Yes').sum()):,}")
    return frame


# --------------------------------------------------------------------------
# Preprocessing pipelines
# --------------------------------------------------------------------------
def build_preprocessor(
    numeric: list[str],
    categorical: list[str],
    *,
    scale: bool = True,
) -> ColumnTransformer:
    """Impute, encode and (optionally) scale inside a single transformer.

    Because this object is fitted on training data only, medians, category
    lists and scaling parameters never see the test split.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", Pipeline(categorical_steps), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def classifier_pipeline(estimator, *, scale: bool = True) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, scale=scale)),
            ("model", estimator),
        ]
    )


def regressor_pipeline(estimator, *, scale: bool = True) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(REGRESSION_NUMERIC, REGRESSION_CATEGORICAL, scale=scale)),
            ("model", estimator),
        ]
    )


# --------------------------------------------------------------------------
# Step 2 - Classification
# --------------------------------------------------------------------------
def classification_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    """Every metric a reviewer would ask for, with a note on what each means."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def run_classification(data: pd.DataFrame) -> dict:
    section("2/7", "Classification — predicting churn")

    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN].map(TARGET_MAP).astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    print(f"    split: {len(x_train):,} train / {len(x_test):,} test (stratified, seed {RANDOM_STATE})")
    print(f"    train churn rate: {y_train.mean():.3f}   test churn rate: {y_test.mean():.3f}")

    # KNN measures distances, so it needs scaled inputs. The tree models and
    # logistic regression do not, but they are given the same pipeline anyway so
    # that the comparison isolates the algorithm rather than the preparation.
    candidates = {
        "Dummy (most frequent)": (
            DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
            True,
        ),
        "Dummy (stratified)": (
            DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
            True,
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            True,
        ),
        "Decision Tree": (
            DecisionTreeClassifier(
                max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE
            ),
            True,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            True,
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(n_neighbors=15, weights="distance"),
            True,
        ),
    }

    rows: list[dict] = []
    prediction_rows: list[dict] = []
    fitted: dict[str, Pipeline] = {}

    for name, (estimator, scale) in candidates.items():
        pipeline = classifier_pipeline(estimator, scale=scale)
        pipeline.fit(x_train, y_train)

        y_pred = pipeline.predict(x_test)
        y_prob = pipeline.predict_proba(x_test)[:, 1]

        scores = classification_metrics(y_test, y_pred, y_prob)
        rows.append({"model": name, **scores})

        for index, (truth, probability) in enumerate(zip(y_test.to_numpy(), y_prob)):
            prediction_rows.append(
                {
                    "row": int(index),
                    "actual": int(truth),
                    **{f"prob::{name}": float(probability)},
                }
            )

        if not name.startswith("Dummy"):
            fitted[name] = pipeline
            matrix = pd.DataFrame(
                confusion_matrix(y_test, y_pred, labels=[0, 1]),
                index=["actual_no", "actual_yes"],
                columns=["predicted_no", "predicted_yes"],
            )
            safe = name.lower().replace(" ", "_").replace("-", "")
            matrix.to_csv(OUTPUT_DIR / f"confusion_matrix_{safe}.csv")

    # Merge the per-model probability columns into one wide frame.
    probabilities = pd.DataFrame(prediction_rows)
    probabilities = probabilities.groupby(["row", "actual"], as_index=False).first()
    probabilities.to_csv(OUTPUT_DIR / "classification_predictions.csv", index=False)

    metrics = pd.DataFrame(rows)
    real_models = metrics[~metrics["model"].str.startswith("Dummy")].copy()
    metrics = pd.concat(
        [
            real_models.sort_values("roc_auc", ascending=False),
            metrics[metrics["model"].str.startswith("Dummy")],
        ]
    ).reset_index(drop=True)
    metrics.to_csv(OUTPUT_DIR / "classification_metrics.csv", index=False)

    print("\n    Held-out test performance (sorted by ROC-AUC):")
    display = metrics[["model", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]]
    print(display.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    # ---- Cross-validation on the training data only -----------------------
    section("3/7", "Cross-validation (training data only)")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    for name, (estimator, scale) in candidates.items():
        if name.startswith("Dummy"):
            continue
        pipeline = classifier_pipeline(estimator, scale=scale)
        auc_scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        f1_scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="f1", n_jobs=1)
        cv_rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": float(auc_scores.mean()),
                "cv_roc_auc_std": float(auc_scores.std()),
                "cv_f1_mean": float(f1_scores.mean()),
                "cv_f1_std": float(f1_scores.std()),
                **{f"fold_{index + 1}_roc_auc": float(score) for index, score in enumerate(auc_scores)},
            }
        )

    cv_results = pd.DataFrame(cv_rows).sort_values("cv_roc_auc_mean", ascending=False)
    cv_results.to_csv(OUTPUT_DIR / "classification_cv_results.csv", index=False)
    print(cv_results[["model", "cv_roc_auc_mean", "cv_roc_auc_std", "cv_f1_mean"]]
          .to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    selected = str(cv_results.iloc[0]["model"])
    print(f"\n    Selected by cross-validated ROC-AUC: {selected}")

    # ---- Threshold analysis for the selected model ------------------------
    selected_pipeline = fitted[selected]
    selected_prob = selected_pipeline.predict_proba(x_test)[:, 1]

    threshold_rows = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        prediction = (selected_prob >= threshold).astype(int)
        threshold_rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": float(precision_score(y_test, prediction, zero_division=0)),
                "recall": float(recall_score(y_test, prediction, zero_division=0)),
                "f1": float(f1_score(y_test, prediction, zero_division=0)),
                "accuracy": float(accuracy_score(y_test, prediction)),
                "predicted_positive_rate": float(prediction.mean()),
            }
        )
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(OUTPUT_DIR / "threshold_analysis.csv", index=False)

    # ---- Class weighting: the accuracy/recall trade-off, measured ---------
    section("4/7", "Class weighting and the accuracy–recall trade-off")

    weight_rows = []
    for label, weight in [("No class weight", None), ("class_weight='balanced'", "balanced")]:
        pipeline = classifier_pipeline(
            LogisticRegression(max_iter=2000, class_weight=weight, random_state=RANDOM_STATE)
        )
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_test)
        probability = pipeline.predict_proba(x_test)[:, 1]
        weight_rows.append(
            {
                "setting": label,
                "accuracy": float(accuracy_score(y_test, prediction)),
                "precision": float(precision_score(y_test, prediction, zero_division=0)),
                "recall": float(recall_score(y_test, prediction, zero_division=0)),
                "f1": float(f1_score(y_test, prediction, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, probability)),
                "predicted_positive_rate": float(prediction.mean()),
            }
        )
    weight_table = pd.DataFrame(weight_rows)
    weight_table.to_csv(OUTPUT_DIR / "class_weight_comparison.csv", index=False)
    print(weight_table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    # ---- Feature importance for the tree ensemble ------------------------
    section("5/7", "Feature importance (Random Forest)")

    forest = fitted["Random Forest"]
    feature_names = [str(name) for name in forest.named_steps["preprocessor"].get_feature_names_out()]
    importances = forest.named_steps["model"].feature_importances_

    importance = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    print(importance.head(10).to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    return {
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
            "positive_class": "Yes (churn = 1)",
        },
        "models": metrics.to_dict(orient="records"),
        "cross_validation": {
            "folds": CV_FOLDS,
            "scoring": "roc_auc and f1, stratified",
            "results": cv_results.to_dict(orient="records"),
            "selected_model": selected,
        },
        "threshold_analysis": {
            "model": selected,
            "rows": thresholds.to_dict(orient="records"),
            "note": (
                "The default 0.5 threshold is a convention, not a law. Lowering it raises "
                "recall (more churners caught) and lowers precision (more false alarms). "
                "The right threshold depends on the cost of each error."
            ),
        },
        "class_weighting": {
            "rows": weight_table.to_dict(orient="records"),
            "note": (
                "Balancing class weights trades accuracy for recall on the minority class. "
                "This is a deliberate choice for churn, where a missed churner usually costs "
                "more than a wasted retention offer."
            ),
        },
        "feature_importance": importance.head(15).to_dict(orient="records"),
        "fitted_models": fitted,
    }


# --------------------------------------------------------------------------
# Step 6 - Regression
# --------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "median_absolute_error": float(np.median(np.abs(np.asarray(y_true) - np.asarray(y_pred)))),
    }


def run_regression(data: pd.DataFrame) -> dict:
    section("6/7", f"Regression — predicting {REGRESSION_TARGET}")

    subset = data.loc[data[REGRESSION_TARGET].notna()].copy()
    features = subset[REGRESSION_FEATURES]
    target = subset[REGRESSION_TARGET].astype(float)

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"    usable rows: {len(subset):,}   train: {len(x_train):,}   test: {len(x_test):,}")
    print("    excluded feature: total_charges (derived from the target — would leak)")
    print(f"    target range: {target.min():.2f} to {target.max():.2f}  (mean {target.mean():.2f})")

    candidates = {
        "Dummy (mean)": (DummyRegressor(strategy="mean"), True),
        "Linear Regression": (LinearRegression(), True),
        "Decision Tree Regressor": (
            DecisionTreeRegressor(max_depth=6, min_samples_leaf=20, random_state=RANDOM_STATE),
            True,
        ),
        "Random Forest Regressor": (
            RandomForestRegressor(
                n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=1
            ),
            True,
        ),
        "K-Nearest Neighbors Regressor": (
            KNeighborsRegressor(n_neighbors=15, weights="distance"),
            True,
        ),
    }

    rows = []
    prediction_frame = pd.DataFrame({"actual": y_test.to_numpy()})
    for name, (estimator, scale) in candidates.items():
        pipeline = regressor_pipeline(estimator, scale=scale)
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_test)
        rows.append({"model": name, **regression_metrics(y_test, prediction)})
        prediction_frame[f"pred::{name}"] = prediction

    metrics = pd.DataFrame(rows)
    real = metrics[~metrics["model"].str.startswith("Dummy")].sort_values("r2", ascending=False)
    metrics = pd.concat([real, metrics[metrics["model"].str.startswith("Dummy")]]).reset_index(drop=True)
    metrics.to_csv(OUTPUT_DIR / "regression_metrics.csv", index=False)
    prediction_frame.to_csv(OUTPUT_DIR / "regression_predictions.csv", index=False)

    print("\n    Held-out test performance (sorted by R²):")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    # Cross-validation on the training data only.
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    for name, (estimator, scale) in candidates.items():
        if name.startswith("Dummy"):
            continue
        pipeline = regressor_pipeline(estimator, scale=scale)
        scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="r2", n_jobs=1)
        cv_rows.append(
            {
                "model": name,
                "cv_r2_mean": float(scores.mean()),
                "cv_r2_std": float(scores.std()),
                **{f"fold_{index + 1}_r2": float(score) for index, score in enumerate(scores)},
            }
        )
    cv_results = pd.DataFrame(cv_rows).sort_values("cv_r2_mean", ascending=False)
    cv_results.to_csv(OUTPUT_DIR / "regression_cv_results.csv", index=False)
    print("\n    5-fold cross-validated R² on the training split:")
    print(cv_results[["model", "cv_r2_mean", "cv_r2_std"]].to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    selected = str(cv_results.iloc[0]["model"])
    print(f"\n    Selected by cross-validated R²: {selected}")

    return {
        "target": REGRESSION_TARGET,
        "features_used": REGRESSION_FEATURES,
        "excluded_features": {
            "total_charges": "computed from monthly_charges and tenure, so it leaks the target"
        },
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
        },
        "models": metrics.to_dict(orient="records"),
        "cross_validation": {
            "folds": CV_FOLDS,
            "results": cv_results.to_dict(orient="records"),
            "selected_model": selected,
        },
    }


# --------------------------------------------------------------------------
# Step 7 - Summary
# --------------------------------------------------------------------------
def main() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("YUVA INTERNSHIP - WEEK 2: SUPERVISED MACHINE LEARNING MODELS")
    print("=" * 78)

    data = load_data()
    classification = run_classification(data)
    regression = run_regression(data)

    section("7/7", "Writing the summary")

    classification.pop("fitted_models")

    best_classifier = classification["models"][0]
    best_regressor = regression["models"][0]

    summary = {
        "project": "Yuva Internship - Week 2: Supervised Machine Learning Models",
        "author": "Peram Dhanam Srinivas",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "source_file": "week-2/data/customer_churn_raw.csv",
            "description": (
                "Reproducible synthetic customer-churn sample. Teaching data, not real "
                "customer information."
            ),
            "rows": int(len(data)),
            "columns": int(data.shape[1]),
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target": TARGET_COLUMN,
        },
        "tasks": {
            "classification": "Predict churn (binary: No = 0, Yes = 1)",
            "regression": f"Predict {REGRESSION_TARGET} (continuous)",
        },
        "classification": classification,
        "regression": regression,
        "headline": {
            "classification_best_roc_auc_model": best_classifier["model"],
            "classification_best_roc_auc": best_classifier["roc_auc"],
            "classification_best_accuracy_model": max(
                classification["models"], key=lambda row: row["accuracy"]
            )["model"],
            "classification_best_accuracy": max(
                row["accuracy"] for row in classification["models"]
            ),
            "regression_best_r2_model": best_regressor["model"],
            "regression_best_r2": best_regressor["r2"],
            "regression_linear_mae": next(
                row["mae"] for row in regression["models"] if row["model"] == "Linear Regression"
            ),
        },
        "methodology_note": (
            "Hyperparameters were fixed in advance. Model selection used 5-fold cross-validation "
            "on the training split; the held-out test split was scored once for reporting and was "
            "never used to choose a model."
        ),
        "limitations_note": (
            "The dataset is synthetic, so scores describe the generator rather than a real "
            "business. Accuracy on an imbalanced target is misleading on its own; precision, "
            "recall, F1 and ROC-AUC are reported alongside it."
        ),
    }

    path = OUTPUT_DIR / "week2_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print("    summary saved: week-2/outputs/week2_summary.json")

    print("\n" + "=" * 78)
    print("WEEK 2 TRAINING COMPLETE")
    print("=" * 78)
    print(f"  best classifier by ROC-AUC ... {summary['headline']['classification_best_roc_auc_model']} "
          f"({summary['headline']['classification_best_roc_auc']:.3f})")
    print(f"  best classifier by accuracy .. {summary['headline']['classification_best_accuracy_model']} "
          f"({summary['headline']['classification_best_accuracy']:.3f})")
    print(f"  best regressor by R2 ......... {summary['headline']['regression_best_r2_model']} "
          f"({summary['headline']['regression_best_r2']:.3f})")
    print("  outputs ...................... week-2/outputs/")
    return summary


if __name__ == "__main__":
    main()
