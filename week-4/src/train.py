#!/usr/bin/env python3
"""Week 4 - train, evaluate and serialise the production model.

This is the only script that writes the deployable artifact. It performs the
full end-to-end path the brief asks for:

    load -> split -> fit -> evaluate -> serialise -> document

Two artifact formats are written so the report can compare them:

    outputs/model/customer_churn_pipeline.joblib   primary, used by the API
    outputs/model/customer_churn_pipeline.pkl      pickle, for portability

A ``model_card.json`` and a ``model_manifest.json`` (with SHA-256 digests)
travel with the artifact, so a deployment can prove which model it is serving.

Usage
-----
    python src/train.py                 # fit on the training split (default)
    python src/train.py --refit-full    # refit on all rows for production
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import statistics
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from config import (
    BEST_PARAMS,
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    DATA_PATH,
    DEFAULT_PARAMS,
    DEFAULT_THRESHOLD,
    FEATURE_COLUMNS,
    ID_COLUMN,
    MANIFEST_PATH,
    MODEL_CARD_PATH,
    MODEL_DESCRIPTION,
    MODEL_DIR,
    MODEL_NAME,
    MODEL_PATH,
    MODEL_PICKLE_PATH,
    MODEL_VERSION,
    NUMERIC_FEATURES,
    OBSERVED_RANGES,
    OUTPUT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from features import build_pipeline, clean_frame, split_features_target


def section(step: str, title: str) -> None:
    print(f"\n[{step}] {title}")
    print("-" * 78)


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(pipeline, features, target, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Score a fitted pipeline and return every metric the report cites."""
    probability = pipeline.predict_proba(features)[:, 1]
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, predicted, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(target, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "recall": float(recall_score(target, predicted, zero_division=0)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probability)),
        "pr_auc": float(average_precision_score(target, probability)),
        "mcc": float(matthews_corrcoef(target, predicted)),
        "log_loss": float(log_loss(target, probability)),
        "brier": float(brier_score_loss(target, probability)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "threshold": float(threshold),
    }


def load_time_seconds(path, repeats: int = 12) -> float:
    """Median cold-load time, so one slow read does not skew the figure."""
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        joblib.load(path)
        timings.append(time.perf_counter() - started)
    return float(statistics.median(timings))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refit-full",
        action="store_true",
        help="refit the selected pipeline on every row before serialising",
    )
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1 - Load
    # ------------------------------------------------------------------
    section("1/7", "Loading the dataset")
    frame = clean_frame(pd.read_csv(DATA_PATH))
    print(f"    file:  {DATA_PATH.relative_to(DATA_PATH.parents[1])}")
    print(f"    shape: {len(frame):,} rows x {len(frame.columns)} columns")

    features, target = split_features_target(frame)
    print(f"    churn: No {int((target == 0).sum()):,}, Yes {int((target == 1).sum()):,} "
          f"({target.mean():.1%})")

    # ------------------------------------------------------------------
    # 2 - Split
    # ------------------------------------------------------------------
    section("2/7", "Stratified train / test split")
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    print(f"    train: {len(x_train):,} rows  (churn {y_train.mean():.1%})")
    print(f"    test : {len(x_test):,} rows  (churn {y_test.mean():.1%})")

    # ------------------------------------------------------------------
    # 3 - Fit the selected model
    # ------------------------------------------------------------------
    section("3/7", "Fitting the selected model")
    print(f"    hyperparameters (GridSearchCV, Week 3): {BEST_PARAMS}")
    pipeline = build_pipeline(**BEST_PARAMS)

    started = time.perf_counter()
    pipeline.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started
    print(f"    fit time: {fit_seconds * 1000:.0f} ms")

    # ------------------------------------------------------------------
    # 4 - Evaluate on the held-out split
    # ------------------------------------------------------------------
    section("4/7", "Evaluating on the held-out test split")
    metrics = evaluate(pipeline, x_test, y_test)
    for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc",
                "pr_auc", "mcc", "log_loss", "brier"):
        print(f"    {key:<18} {metrics[key]:.3f}")
    print(f"    confusion        TN {metrics['true_negative']}  FP {metrics['false_positive']}  "
          f"FN {metrics['false_negative']}  TP {metrics['true_positive']}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(build_pipeline(**BEST_PARAMS), x_train, y_train,
                                cv=cv, scoring="roc_auc", n_jobs=1)
    metrics["cv_roc_auc_mean"] = float(cv_scores.mean())
    metrics["cv_roc_auc_std"] = float(cv_scores.std())
    print(f"    {CV_FOLDS}-fold CV ROC-AUC  {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # A second model with default hyperparameters, so the report can show what
    # the Week 3 tuning actually bought.
    default_pipeline = build_pipeline(**DEFAULT_PARAMS).fit(x_train, y_train)
    default_metrics = evaluate(default_pipeline, x_test, y_test)

    comparison = pd.DataFrame(
        [
            {"model": "Logistic Regression (C=1.0, default)", **default_metrics},
            {"model": "Logistic Regression (C=0.1, tuned)", **metrics},
        ]
    )
    comparison.to_csv(OUTPUT_DIR / "evaluation_comparison.csv", index=False)
    print(f"    tuning delta ROC-AUC: "
          f"{metrics['roc_auc'] - default_metrics['roc_auc']:+.3f}")

    # Per-row predictions, used by the figures and the notebook.
    predictions = pd.DataFrame(
        {
            ID_COLUMN: frame.loc[x_test.index, ID_COLUMN].to_numpy(),
            "y_true": y_test.to_numpy(),
            "y_probability": pipeline.predict_proba(x_test)[:, 1],
        }
    )
    predictions["y_predicted"] = (
        predictions["y_probability"] >= DEFAULT_THRESHOLD
    ).astype(int)
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    # ------------------------------------------------------------------
    # 5 - Serialise
    # ------------------------------------------------------------------
    section("5/7", "Serialising the model")
    if args.refit_full:
        print("    --refit-full: refitting on every row before serialising")
        deploy_pipeline = build_pipeline(**BEST_PARAMS).fit(features, target)
        training_rows = len(features)
    else:
        deploy_pipeline = pipeline
        training_rows = len(x_train)

    joblib.dump(deploy_pipeline, MODEL_PATH)
    with open(MODEL_PICKLE_PATH, "wb") as handle:
        pickle.dump(deploy_pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)

    joblib_bytes = MODEL_PATH.stat().st_size
    pickle_bytes = MODEL_PICKLE_PATH.stat().st_size
    load_ms = load_time_seconds(MODEL_PATH) * 1000
    print(f"    joblib: {joblib_bytes / 1024:,.1f} KiB  ->  {MODEL_PATH.name}")
    print(f"    pickle: {pickle_bytes / 1024:,.1f} KiB  ->  {MODEL_PICKLE_PATH.name}")
    print(f"    cold load (median of 12): {load_ms:.1f} ms")

    # Round-trip check: the reloaded artifact must reproduce the in-memory
    # predictions bit for bit, otherwise the deployment is not the model tested.
    reloaded = joblib.load(MODEL_PATH)
    in_memory = deploy_pipeline.predict_proba(x_test)[:, 1]
    from_disk = reloaded.predict_proba(x_test)[:, 1]
    max_delta = float(np.max(np.abs(in_memory - from_disk)))
    reproducible = bool(np.allclose(in_memory, from_disk, rtol=0, atol=1e-12))
    print(f"    round-trip identical: {reproducible}  (max delta {max_delta:.2e})")
    if not reproducible:
        print("    ERROR: serialised model diverges from the in-memory model")
        return 1

    # ------------------------------------------------------------------
    # 6 - Model card and manifest
    # ------------------------------------------------------------------
    section("6/7", "Writing the model card and manifest")
    model_card = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "description": MODEL_DESCRIPTION,
        "algorithm": "LogisticRegression (scikit-learn Pipeline)",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_rows": int(training_rows),
        "train_test_split": {
            "test_fraction": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
        },
        "hyperparameters": BEST_PARAMS,
        "hyperparameters_source": "GridSearchCV, 5-fold CV, selected in Week 3",
        "features": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "observed_ranges": {k: list(v) for k, v in OBSERVED_RANGES.items()},
        "default_threshold": DEFAULT_THRESHOLD,
        "metrics": {k: v for k, v in metrics.items() if not isinstance(v, int) or k.startswith("true") or k.startswith("false")},
        "confusion_matrix": {
            "true_negative": metrics["true_negative"],
            "false_positive": metrics["false_positive"],
            "false_negative": metrics["false_negative"],
            "true_positive": metrics["true_positive"],
        },
        "evaluated_on": "360 held-out rows, scored once" if not args.refit_full
                        else "training split metrics; artifact refitted on all rows",
        "intended_use": (
            "Ranking existing customers by churn risk so a retention team can "
            "prioritise outreach. Decision support only."
        ),
        "limitations": [
            "Trained on synthetic teaching data; not validated on real customers.",
            "Do not use for credit, pricing, hiring or any consequential decision.",
            "No subgroup fairness audit has been performed.",
            "Predictions are only as current as the features supplied.",
            "Performance drifts if customer behaviour changes.",
        ],
        "artifacts": {
            "joblib": MODEL_PATH.name,
            "pickle": MODEL_PICKLE_PATH.name,
            "joblib_bytes": joblib_bytes,
            "pickle_bytes": pickle_bytes,
        },
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    MODEL_CARD_PATH.write_text(json.dumps(model_card, indent=2) + "\n")
    print(f"    model card -> {MODEL_CARD_PATH.name}")

    manifest = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "generated_at": model_card["trained_at"],
        "files": [
            {
                "path": MODEL_PATH.name,
                "bytes": joblib_bytes,
                "sha256": sha256(MODEL_PATH),
            },
            {
                "path": MODEL_PICKLE_PATH.name,
                "bytes": pickle_bytes,
                "sha256": sha256(MODEL_PICKLE_PATH),
            },
        ],
        "load_time_ms": round(load_ms, 3),
        "round_trip_identical": reproducible,
        "max_abs_delta": max_delta,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"    manifest   -> {MANIFEST_PATH.name}")

    # ------------------------------------------------------------------
    # 7 - Tables for the report
    # ------------------------------------------------------------------
    section("7/7", "Exporting tables")
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    ).to_csv(OUTPUT_DIR / "final_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"actual": "No", "predicted No": metrics["true_negative"],
             "predicted Yes": metrics["false_positive"]},
            {"actual": "Yes", "predicted No": metrics["false_negative"],
             "predicted Yes": metrics["true_positive"]},
        ]
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv", index=False)
    pd.DataFrame(
        [
            {"artifact": "joblib", "bytes": joblib_bytes,
             "kib": round(joblib_bytes / 1024, 1), "load_ms": round(load_ms, 2)},
            {"artifact": "pickle", "bytes": pickle_bytes,
             "kib": round(pickle_bytes / 1024, 1), "load_ms": None},
        ]
    ).to_csv(OUTPUT_DIR / "artifacts.csv", index=False)

    print("\n" + "=" * 78)
    print("WEEK 4 MODEL READY")
    print("=" * 78)
    print(f"  artifact ......... {MODEL_PATH.relative_to(MODEL_PATH.parents[1])}")
    print(f"  version .......... {MODEL_VERSION}")
    print(f"  test ROC-AUC ..... {metrics['roc_auc']:.3f}")
    print(f"  test recall ...... {metrics['recall']:.3f}")
    print(f"  cold load ........ {load_ms:.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
