"""Week 2: fit and compare regression and classification models."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .common import (
    classification_metrics,
    ensure_output_dirs,
    load_raw_data,
    split_features_target,
    write_json,
)
from .preprocessing import build_preprocessor
from .schema import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    PLOT_DIR,
    REGRESSION_FEATURES,
    TARGET_COLUMN,
    TARGET_MAP,
)

RANDOM_STATE = 42


def _classifier_pipeline(classifier) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def run_week2() -> dict:
    data = load_raw_data()
    ensure_output_dirs()
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=8, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=220,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=17, weights="distance"),
    }

    metric_rows = []
    confusion_matrices = {}
    for name, estimator in models.items():
        model = _classifier_pipeline(estimator)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1]
        row = {"model": name, **classification_metrics(y_test, prediction, probability)}
        metric_rows.append(row)
        confusion_matrices[name] = confusion_matrix(y_test, prediction, labels=[0, 1])
        matrix_frame = pd.DataFrame(
            confusion_matrices[name],
            index=["actual_no", "actual_yes"],
            columns=["predicted_no", "predicted_yes"],
        )
        matrix_frame.to_csv(OUTPUT_DIR / f"confusion_matrix_{name.lower().replace(' ', '_').replace('-', '')}.csv")

    metrics = pd.DataFrame(metric_rows).sort_values("roc_auc", ascending=False)
    metrics.to_csv(OUTPUT_DIR / "supervised_classification_metrics.csv", index=False)

    # Linear Regression is used for a numeric target (monthly charges), rather
    # than misapplying it to churn labels. total_charges is excluded to prevent
    # target leakage because it is computed from monthly charges and tenure.
    regression_data = data.loc[data["monthly_charges"].notna()].copy()
    X_reg = regression_data[REGRESSION_FEATURES]
    y_reg = regression_data["monthly_charges"].astype(float)
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.20, random_state=RANDOM_STATE
    )
    numeric_regression = [column for column in REGRESSION_FEATURES if column in NUMERIC_FEATURES]
    categorical_regression = [column for column in REGRESSION_FEATURES if column in CATEGORICAL_FEATURES]
    regression_model = Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(numeric_regression, categorical_regression),
            ),
            ("regressor", LinearRegression()),
        ]
    )
    regression_model.fit(Xr_train, yr_train)
    regression_prediction = regression_model.predict(Xr_test)
    regression_scores = {
        "model": "Linear Regression",
        "target": "monthly_charges",
        "n_train": int(len(Xr_train)),
        "n_test": int(len(Xr_test)),
        "mae": float(mean_absolute_error(yr_test, regression_prediction)),
        "rmse": float(np.sqrt(mean_squared_error(yr_test, regression_prediction))),
        "r2": float(r2_score(yr_test, regression_prediction)),
        "leakage_control": "total_charges excluded because it is derived from monthly_charges",
    }
    write_json(OUTPUT_DIR / "supervised_regression_metrics.json", regression_scores)

    # Model comparison chart, with scale fixed to make model differences clear.
    plot_metrics = metrics.set_index("model")[["precision", "recall", "f1", "roc_auc"]]
    ax = plot_metrics.plot(kind="bar", figsize=(10.4, 5.8), color=["#52796f", "#e09f3e", "#9b5de5", "#4361ee"])
    ax.set_title("Classification model comparison on the held-out test split")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.22)
    ax.tick_params(axis="x", rotation=12)
    ax.legend(loc="lower right", ncols=2)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "supervised_model_comparison.png", dpi=160)
    plt.close()

    summary = {
        "split": "80/20 stratified test split; random_state=42",
        "n_train_classification": int(len(X_train)),
        "n_test_classification": int(len(X_test)),
        "positive_class": "Yes (1)",
        "classification_models": metric_rows,
        "best_by_test_roc_auc_for_reporting_only": str(metrics.iloc[0]["model"]),
        "regression": regression_scores,
        "important_note": "Reported test metrics are for comparison; model selection should use cross-validation on training data, not the test set.",
    }
    write_json(OUTPUT_DIR / "week2_summary.json", summary)
    print("Week 2 supervised learning complete")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(
        "Linear Regression: "
        f"MAE={regression_scores['mae']:.2f}, RMSE={regression_scores['rmse']:.2f}, "
        f"R2={regression_scores['r2']:.3f}"
    )
    return summary


if __name__ == "__main__":
    run_week2()
