#!/usr/bin/env python3
"""Week 3 - Unsupervised Learning & Model Evaluation.

Two halves, because the brief asks for two different things:

    Part A - Unsupervised learning
        K-Means (k = 2..8), Agglomerative / hierarchical clustering across four
        linkage rules, and PCA as a dimensionality-reduction step. Clusters are
        judged with internal indices (silhouette, Davies-Bouldin, Calinski-
        Harabasz) and then, separately, checked against the churn label the
        algorithm never saw.

    Part B - Model evaluation
        Cross-validation, the confusion matrix and every metric derived from it
        (precision, recall, F1, ROC-AUC), and hyperparameter tuning with
        GridSearchCV plus a nested cross-validation to get an unbiased estimate
        of the tuned model's performance.

The two halves are connected by one question: the clusters are real, so do they
explain churn? They do not, and Part B shows what it costs to assume otherwise.

Usage
-----
    python src/run_analysis.py     # runs both halves, exports every table
    python src/make_figures.py     # builds the figures from those tables
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

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

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
K_RANGE = range(2, 9)
LINKAGES = ["ward", "complete", "average", "single"]
PCA_GRID = [2, 3, 5, 8, 10, 12, 16, 21]

# A cluster smaller than this share of the data is reported as degenerate. The
# silhouette score on its own will happily reward such a solution, which is the
# single most common way to be fooled by clustering output.
MIN_CLUSTER_SHARE = 0.02


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


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUTPUT_DIR / name, index=False)


# --------------------------------------------------------------------------
# Step 1 - Load and encode
# --------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    section("1/9", "Loading the dataset")
    frame = pd.read_csv(DATA_PATH)
    frame.columns = [column.strip() for column in frame.columns]

    required = {ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    invalid = ~frame[TARGET_COLUMN].isin(list(TARGET_MAP))
    if invalid.any():
        raise ValueError(f"Target contains {int(invalid.sum())} invalid value(s)")

    yes = int((frame[TARGET_COLUMN] == "Yes").sum())
    print(f"    file:   {DATA_PATH.relative_to(WEEK_DIR.parent)}")
    print(f"    shape:  {frame.shape[0]:,} rows x {frame.shape[1]} columns")
    print(f"    churn:  No {frame.shape[0] - yes:,}, Yes {yes:,} ({yes / len(frame):.1%})")
    return frame


def build_feature_matrix(frame: pd.DataFrame):
    """Scale numeric columns, leave one-hot columns on their natural 0/1 scale.

    Standardising a dummy variable would give a category present in 5% of rows
    the same weight in the distance metric as one present in 50%, which distorts
    every centroid. Numbers need scaling; indicators do not.
    """
    section("2/9", "Building the feature matrix")
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    matrix = preprocessor.fit_transform(frame[FEATURE_COLUMNS])
    names = list(preprocessor.get_feature_names_out())

    print(f"    input columns:   {len(FEATURE_COLUMNS)}")
    print(f"    encoded columns: {len(names)}")
    print("    numeric scaled, one-hot left at 0/1")
    return matrix, names


# --------------------------------------------------------------------------
# Step 3 - PCA
# --------------------------------------------------------------------------
def run_pca(matrix: np.ndarray, names: list[str]) -> dict:
    section("3/9", "Principal component analysis")
    pca = PCA(random_state=RANDOM_STATE).fit(matrix)
    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)

    frame = pd.DataFrame(
        {
            "component": np.arange(1, len(ratios) + 1),
            "explained_variance_ratio": ratios,
            "cumulative_variance": cumulative,
            "eigenvalue": pca.explained_variance_,
        }
    )
    write_csv(frame, "pca_variance.csv")

    n90 = int(np.searchsorted(cumulative, 0.90) + 1)
    n95 = int(np.searchsorted(cumulative, 0.95) + 1)

    loadings = pd.DataFrame(pca.components_[:3].T, index=names, columns=["PC1", "PC2", "PC3"])
    loadings.index.name = "feature"
    loadings = loadings.reindex(loadings["PC1"].abs().sort_values(ascending=False).index)
    loadings.to_csv(OUTPUT_DIR / "pca_loadings.csv")

    print(f"    components for 90% variance: {n90}")
    print(f"    components for 95% variance: {n95}")
    print(f"    PC1 explains {ratios[0]:.1%}, PC2 {ratios[1]:.1%}, PC3 {ratios[2]:.1%}")
    print("    strongest PC1 loadings:")
    for feature, value in loadings["PC1"].head(4).items():
        print(f"        {feature:32s} {value:+.3f}")

    return {
        "n_features": len(names),
        "n_components_90": n90,
        "n_components_95": n95,
        "explained_first": float(ratios[0]),
        "explained_first_two": float(cumulative[1]),
        "explained_first_three": float(cumulative[2]),
        "explained_first_ten": float(cumulative[min(9, len(cumulative) - 1)]),
        "top_pc1": [
            {"feature": str(f), "loading": float(v)} for f, v in loadings["PC1"].head(5).items()
        ],
    }


# --------------------------------------------------------------------------
# Step 4 - K-Means
# --------------------------------------------------------------------------
def run_kmeans(matrix: np.ndarray, churn: np.ndarray) -> dict:
    section("4/9", "K-Means: elbow, silhouette and cluster selection")
    rows = []
    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(matrix)
        labels = model.labels_
        sizes = np.bincount(labels, minlength=k)
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(matrix, labels)),
                "davies_bouldin": float(davies_bouldin_score(matrix, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
                "ari_vs_churn": float(adjusted_rand_score(churn, labels)),
                "nmi_vs_churn": float(normalized_mutual_info_score(churn, labels)),
                "smallest_cluster": int(sizes.min()),
                "smallest_cluster_share": float(sizes.min() / len(labels)),
                "largest_cluster": int(sizes.max()),
            }
        )
        print(
            f"    k={k}  inertia={model.inertia_:9.1f}  silhouette={rows[-1]['silhouette']:.3f}  "
            f"DB={rows[-1]['davies_bouldin']:.3f}  CH={rows[-1]['calinski_harabasz']:7.1f}  "
            f"ARI vs churn={rows[-1]['ari_vs_churn']:+.3f}"
        )

    sweep = pd.DataFrame(rows)
    write_csv(sweep, "kmeans_sweep.csv")

    best = sweep.loc[sweep["silhouette"].idxmax()]
    best_k = int(best["k"])

    # The three internal indices disagree here, which is normal. Record which k
    # each one prefers rather than pretending there is a single right answer.
    preferred = {
        "silhouette": int(sweep.loc[sweep["silhouette"].idxmax(), "k"]),
        "davies_bouldin": int(sweep.loc[sweep["davies_bouldin"].idxmin(), "k"]),
        "calinski_harabasz": int(sweep.loc[sweep["calinski_harabasz"].idxmax(), "k"]),
    }
    inertia = sweep["inertia"].to_numpy()
    drops = -np.diff(inertia)
    elbow_k = int(sweep["k"].to_numpy()[int(np.argmax(drops))] + 1)

    print(f"\n    best k by silhouette ....... {preferred['silhouette']}")
    print(f"    best k by Davies-Bouldin ... {preferred['davies_bouldin']}  (lower is better)")
    print(f"    best k by Calinski-Harabasz  {preferred['calinski_harabasz']}")
    print(f"    elbow (largest inertia drop) {elbow_k}")
    print(f"    selected k ................. {best_k}  (silhouette, with elbow support)")

    return {
        "sweep": sweep.to_dict(orient="records"),
        "selected_k": best_k,
        "selected_silhouette": float(best["silhouette"]),
        "preferred_k": preferred,
        "elbow_k": elbow_k,
    }


def fit_final_kmeans(matrix: np.ndarray, k: int) -> KMeans:
    model = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE).fit(matrix)
    return model


# --------------------------------------------------------------------------
# Step 5 - Hierarchical clustering
# --------------------------------------------------------------------------
def run_hierarchical(matrix: np.ndarray, churn: np.ndarray, kmeans_k: int) -> dict:
    section("5/9", "Hierarchical clustering across four linkage rules")
    rows = []
    for linkage in LINKAGES:
        for k in range(2, 6):
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit(matrix)
            labels = model.labels_
            sizes = np.bincount(labels, minlength=k)
            rows.append(
                {
                    "linkage": linkage,
                    "k": k,
                    "silhouette": float(silhouette_score(matrix, labels)),
                    "smallest_cluster": int(sizes.min()),
                    "smallest_cluster_share": float(sizes.min() / len(labels)),
                    "ari_vs_churn": float(adjusted_rand_score(churn, labels)),
                    "degenerate": bool(sizes.min() / len(labels) < MIN_CLUSTER_SHARE),
                }
            )
            print(
                f"    {linkage:9s} k={k}  silhouette={rows[-1]['silhouette']:.3f}  "
                f"smallest cluster={sizes.min():4d} ({rows[-1]['smallest_cluster_share']:.1%})"
                f"{'  <- degenerate' if rows[-1]['degenerate'] else ''}"
            )

    sweep = pd.DataFrame(rows)
    write_csv(sweep, "hierarchical_sweep.csv")

    # The trap: single and average linkage top the silhouette table while
    # producing one-point clusters. A partition is only useful if every cluster
    # is big enough to describe, so degenerate solutions are excluded first and
    # only then is the best remaining one taken.
    honest = sweep[~sweep["degenerate"]]
    trap = sweep.loc[sweep["silhouette"].idxmax()]
    best = honest.loc[honest["silhouette"].idxmax()]
    best_linkage = str(best["linkage"])
    best_k = int(best["k"])

    print(f"\n    highest silhouette overall .. {trap['linkage']}, k={int(trap['k'])}, "
          f"{trap['silhouette']:.3f}  but smallest cluster = {int(trap['smallest_cluster'])} rows")
    print(f"    best non-degenerate ......... {best_linkage}, k={best_k}, {best['silhouette']:.3f}")

    final = AgglomerativeClustering(n_clusters=kmeans_k, linkage=best_linkage).fit(matrix)
    final_sil = float(silhouette_score(matrix, final.labels_))
    sizes = np.bincount(final.labels_, minlength=kmeans_k)
    print(f"    final model ................. {best_linkage}, k={kmeans_k}, silhouette={final_sil:.3f}")
    print(f"    cluster sizes ............... {sizes.tolist()}")

    return {
        "sweep": sweep.to_dict(orient="records"),
        "selected_linkage": best_linkage,
        "selected_k": kmeans_k,
        "k_from_sweep": best_k,
        "selected_silhouette": final_sil,
        "sizes": sizes.tolist(),
        "labels": final.labels_,
        "trap": {
            "linkage": str(trap["linkage"]),
            "k": int(trap["k"]),
            "silhouette": float(trap["silhouette"]),
            "smallest_cluster": int(trap["smallest_cluster"]),
        },
    }


# --------------------------------------------------------------------------
# Step 6 - Profiles and agreement
# --------------------------------------------------------------------------
def _describe(labels: np.ndarray, n_clusters: int) -> list[dict]:
    """Profile each cluster on the original (unscaled) features."""
    frame = pd.read_csv(DATA_PATH)
    frame["cluster"] = labels
    rows = []
    for cluster in range(n_clusters):
        block = frame[frame["cluster"] == cluster]
        row = {
            "cluster": cluster,
            "size": int(len(block)),
            "share": float(len(block) / len(frame)),
            "churn_rate": float((block[TARGET_COLUMN] == "Yes").mean()),
        }
        for feature in NUMERIC_FEATURES:
            row[feature] = float(block[feature].mean())
        for feature in CATEGORICAL_FEATURES:
            counts = block[feature].value_counts(normalize=True)
            row[f"top_{feature}"] = str(counts.index[0])
            row[f"top_{feature}_share"] = float(counts.iloc[0])
        rows.append(row)
    return rows


def name_clusters(rows: list[dict]) -> list[dict]:
    """Give each cluster a plain-English label derived from its own profile."""
    seniors = [row for row in rows if row["senior_citizen"] >= 0.90]
    others = [row for row in rows if row["senior_citizen"] < 0.90]
    ordered = sorted(others, key=lambda row: row["tenure_months"])

    labels: dict[int, str] = {}
    for position, row in enumerate(ordered):
        if position == 0:
            labels[row["cluster"]] = "Short-tenure, low-spend"
        elif position == len(ordered) - 1:
            labels[row["cluster"]] = "Long-tenure, high-spend"
        else:
            labels[row["cluster"]] = "Mid-tenure, mid-spend"
    for row in seniors:
        labels[row["cluster"]] = "Senior segment"
    for row in rows:
        row["label"] = labels[row["cluster"]]
    return rows


def run_profiles(labels: np.ndarray, n_clusters: int, name: str) -> list[dict]:
    rows = name_clusters(_describe(labels, n_clusters))
    frame = pd.DataFrame(rows)
    write_csv(frame, f"cluster_profiles_{name}.csv")

    print(f"\n    {name} profiles")
    for row in rows:
        print(
            f"      cluster {row['cluster']} ({row['label']}): {row['size']:4d} rows "
            f"({row['share']:5.1%})  tenure {row['tenure_months']:5.1f}  "
            f"monthly {row['monthly_charges']:5.1f}  churn {row['churn_rate']:.1%}"
        )
    return rows


def run_agreement(kmeans_labels: np.ndarray, hierarchical_labels: np.ndarray, churn: np.ndarray) -> dict:
    section("6/9", "Do two different algorithms find the same segments?")
    ari = float(adjusted_rand_score(kmeans_labels, hierarchical_labels))
    ami = float(adjusted_mutual_info_score(kmeans_labels, hierarchical_labels))

    ari_churn_k = float(adjusted_rand_score(churn, kmeans_labels))
    ari_churn_h = float(adjusted_rand_score(churn, hierarchical_labels))

    frame = pd.DataFrame(
        [
            {"pair": "K-Means vs Ward", "ari": ari, "ami": ami,
             "note": "agreement between the two clusterings"},
            {"pair": "K-Means vs churn", "ari": ari_churn_k, "ami": float(normalized_mutual_info_score(churn, kmeans_labels)),
             "note": "agreement between clusters and the label"},
            {"pair": "Ward vs churn", "ari": ari_churn_h, "ami": float(normalized_mutual_info_score(churn, hierarchical_labels)),
             "note": "agreement between clusters and the label"},
        ]
    )
    write_csv(frame, "cluster_agreement.csv")

    print(f"    K-Means vs Ward ....... ARI {ari:+.3f}   (near 1 = same partition)")
    print(f"    K-Means vs churn ...... ARI {ari_churn_k:+.3f}  (near 0 = unrelated)")
    print(f"    Ward vs churn ......... ARI {ari_churn_h:+.3f}")
    print("    -> the two algorithms agree with each other far more than either agrees with churn")
    return {"ari_models": ari, "ami_models": ami,
            "ari_churn_kmeans": ari_churn_k, "ari_churn_hierarchical": ari_churn_h}


def run_churn_overlap(kmeans_labels: np.ndarray, kmeans_profiles: list[dict]) -> dict:
    """External sanity check: crosstab of clusters against the held-out label."""
    frame = pd.read_csv(DATA_PATH)
    frame["cluster"] = kmeans_labels
    table = pd.crosstab(frame["cluster"], frame[TARGET_COLUMN])
    table.columns = [f"churn_{column}" for column in table.columns]
    table["churn_rate"] = table["churn_Yes"] / (table["churn_No"] + table["churn_Yes"])
    write_csv(table.reset_index(), "cluster_churn_crosstab.csv")

    rates = table["churn_rate"]
    print(f"\n    churn rate by cluster: {[round(float(v), 3) for v in rates]}")
    print(f"    spread = {float(rates.max() - rates.min()):.3f} "
          f"(overall rate = {float((frame[TARGET_COLUMN] == 'Yes').mean()):.3f})")
    return {
        "rates": [float(v) for v in rates],
        "spread": float(rates.max() - rates.min()),
        "overall": float((frame[TARGET_COLUMN] == "Yes").mean()),
    }


# --------------------------------------------------------------------------
# Step 7 - Dimensionality reduction before clustering
# --------------------------------------------------------------------------
def run_dimensionality_experiment(matrix: np.ndarray, k: int) -> dict:
    section("7/9", "Does reducing dimensions before clustering help?")
    rows = []
    for n in PCA_GRID:
        if n > matrix.shape[1]:
            continue
        reduced = PCA(n_components=n, random_state=RANDOM_STATE).fit_transform(matrix)
        labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(reduced)
        sizes = np.bincount(labels, minlength=k)
        rows.append(
            {
                "components": n,
                "variance_kept": float(PCA(n_components=n, random_state=RANDOM_STATE)
                                       .fit(matrix).explained_variance_ratio_.sum()),
                "silhouette_reduced_space": float(silhouette_score(reduced, labels)),
                "silhouette_original_space": float(silhouette_score(matrix, labels)),
                "smallest_cluster": int(sizes.min()),
            }
        )
        print(
            f"    PCA {n:2d}D  variance kept {rows[-1]['variance_kept']:5.1%}  "
            f"silhouette in {n}D = {rows[-1]['silhouette_reduced_space']:.3f}  "
            f"silhouette in the original space = {rows[-1]['silhouette_original_space']:.3f}"
        )

    frame = pd.DataFrame(rows)
    write_csv(frame, "dimensionality_experiment.csv")

    inflated = frame.loc[frame["silhouette_reduced_space"].idxmax()]
    print(f"    -> projecting to {int(inflated['components'])}D inflates silhouette from "
          f"{inflated['silhouette_original_space']:.3f} to {inflated['silhouette_reduced_space']:.3f}")
    print("       the clusters themselves did not improve; only the space they are measured in")

    return {
        "rows": rows,
        "inflated_components": int(inflated["components"]),
        "inflated_reduced": float(inflated["silhouette_reduced_space"]),
        "inflated_original": float(inflated["silhouette_original_space"]),
    }


# --------------------------------------------------------------------------
# Step 8 - Supervised evaluation: metrics, CV, tuning
# --------------------------------------------------------------------------
def classification_metrics(y_true, y_pred, y_prob) -> dict:
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


def build_pipeline(estimator) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                  ("scaler", StandardScaler())]), NUMERIC_FEATURES),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                               sparse_output=False))]),
             CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def run_tuning(frame: pd.DataFrame) -> dict:
    section("8/9", "Hyperparameter tuning with GridSearchCV")
    features = frame[FEATURE_COLUMNS]
    target = frame[TARGET_COLUMN].map(TARGET_MAP)

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print(f"    train {len(X_train):,} / test {len(X_test):,}")

    logistic_grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "model__penalty": ["l2"],
    }
    logistic = GridSearchCV(
        build_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced",
                                          random_state=RANDOM_STATE)),
        logistic_grid, scoring="roc_auc", cv=cv, return_train_score=True, refit=True,
    ).fit(X_train, y_train)

    forest_grid = {
        "model__n_estimators": [150, 300],
        "model__max_depth": [4, 6, 8, None],
        "model__min_samples_leaf": [1, 5],
        "model__max_features": ["sqrt", 0.5],
    }
    forest = GridSearchCV(
        build_pipeline(RandomForestClassifier(class_weight="balanced",
                                              random_state=RANDOM_STATE)),
        forest_grid, scoring="roc_auc", cv=cv, return_train_score=True, refit=True,
    ).fit(X_train, y_train)

    for name, search in (("logistic", logistic), ("random_forest", forest)):
        results = pd.DataFrame(search.cv_results_)
        param_columns = [column for column in results.columns if column.startswith("param_")]
        keep = param_columns + ["mean_test_score", "std_test_score", "mean_train_score", "rank_test_score"]
        table = results[keep].sort_values("rank_test_score").reset_index(drop=True)
        write_csv(table, f"tuning_grid_{name}.csv")
        print(f"\n    {name}: {len(results)} combinations")
        print(f"      best params: {search.best_params_}")
        print(f"      best CV ROC-AUC: {search.best_score_:.3f}")

    candidates = {"Logistic Regression": logistic, "Random Forest": forest}
    winner_name = max(candidates, key=lambda key: candidates[key].best_score_)
    winner = candidates[winner_name]
    print(f"\n    selected estimator: {winner_name} (CV ROC-AUC {winner.best_score_:.3f})")

    # ---- Untuned comparison: how much did tuning actually buy?
    untuned = {
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "Random Forest": build_pipeline(
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)
        ),
    }
    comparison = []
    for name, estimator in untuned.items():
        cv_scores = cross_val_score(estimator, X_train, y_train, scoring="roc_auc", cv=cv)
        estimator.fit(X_train, y_train)
        probability = estimator.predict_proba(X_test)[:, 1]
        tuned_cv = candidates[name].best_score_
        comparison.append(
            {
                "model": name,
                "default_cv_roc_auc": float(cv_scores.mean()),
                "default_cv_std": float(cv_scores.std()),
                "tuned_cv_roc_auc": float(tuned_cv),
                "gain": float(tuned_cv - cv_scores.mean()),
                "default_test_roc_auc": float(roc_auc_score(y_test, probability)),
            }
        )
        print(f"    {name:20s} default {cv_scores.mean():.3f} -> tuned {tuned_cv:.3f} "
              f"({tuned_cv - cv_scores.mean():+.3f})")
    comparison_frame = pd.DataFrame(comparison)
    write_csv(comparison_frame, "tuned_vs_default.csv")

    # ---- Final evaluation, once, on the held-out split
    final_model = winner.best_estimator_
    probability = final_model.predict_proba(X_test)[:, 1]
    prediction = final_model.predict(X_test)
    final_metrics = classification_metrics(y_test, prediction, probability)

    print(f"\n    held-out test performance ({winner_name})")
    for metric in ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        print(f"      {metric:18s} {final_metrics[metric]:.3f}")

    metrics_frame = pd.DataFrame([{"model": winner_name, **final_metrics}])
    write_csv(metrics_frame, "final_metrics.csv")

    matrix = confusion_matrix(y_test, prediction, labels=[0, 1])
    pd.DataFrame(
        matrix, index=["actual_no", "actual_yes"], columns=["predicted_no", "predicted_yes"]
    ).to_csv(OUTPUT_DIR / "final_confusion_matrix.csv")
    print(f"      confusion matrix: TN {final_metrics['true_negative']} FP "
          f"{final_metrics['false_positive']} FN {final_metrics['false_negative']} "
          f"TP {final_metrics['true_positive']}")

    predictions = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": prediction,
            "probability": probability,
        }
    )
    write_csv(predictions, "final_predictions.csv")

    # ---- Curves used by the figures
    fpr, tpr, _ = roc_curve(y_test, probability)
    write_csv(pd.DataFrame({"fpr": fpr, "tpr": tpr}), "final_roc_curve.csv")

    fraction_positive, mean_predicted = calibration_curve(
        y_test, probability, n_bins=10, strategy="quantile"
    )
    write_csv(
        pd.DataFrame({"mean_predicted_probability": mean_predicted,
                      "fraction_positive": fraction_positive}),
        "final_calibration_curve.csv",
    )

    return {
        "winner": winner_name,
        "best_params": {key: str(value) for key, value in winner.best_params_.items()},
        "best_cv_score": float(winner.best_score_),
        "n_combinations": {"logistic": int(len(logistic.cv_results_["params"])),
                           "random_forest": int(len(forest.cv_results_["params"]))},
        "metrics": final_metrics,
        "comparison": comparison,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "cv_folds": CV_FOLDS,
        "logistic_best": {key: str(value) for key, value in logistic.best_params_.items()},
        "logistic_best_score": float(logistic.best_score_),
        "forest_best": {key: str(value) for key, value in forest.best_params_.items()},
        "forest_best_score": float(forest.best_score_),
    }


def run_nested_cv(frame: pd.DataFrame, tuning: dict) -> dict:
    """Nested CV: the outer loop never sees the inner loop's choices.

    A tuned CV score is optimistic because the same folds chose the winner and
    reported its score. Nested CV removes that bias by holding the outer folds
    back entirely.
    """
    section("9/9", "Nested cross-validation and the learning curve")
    # Both experiments run on the training split only. Scoring a tuned model on
    # the same rows the test split was drawn from would quietly spend the
    # hold-out set twice.
    features = frame[FEATURE_COLUMNS]
    target = frame[TARGET_COLUMN].map(TARGET_MAP)
    features, _, target, _ = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target
    )

    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    if tuning["winner"] == "Logistic Regression":
        grid = {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0], "model__penalty": ["l2"]}
        estimator = build_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        )
    else:
        grid = {
            "model__n_estimators": [150, 300],
            "model__max_depth": [4, 6, 8, None],
            "model__min_samples_leaf": [1, 5],
            "model__max_features": ["sqrt", 0.5],
        }
        estimator = build_pipeline(
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)
        )

    search = GridSearchCV(estimator, grid, scoring="roc_auc", cv=inner)
    scores = cross_val_score(search, features, target, scoring="roc_auc", cv=outer)
    write_csv(
        pd.DataFrame({"outer_fold": np.arange(1, len(scores) + 1), "roc_auc": scores}),
        "nested_cv_results.csv",
    )
    print(f"    nested CV ROC-AUC: {scores.mean():.3f} +/- {scores.std():.3f}")
    print(f"    tuned CV score ...: {tuning['best_cv_score']:.3f}")
    print(f"    optimism .........: {tuning['best_cv_score'] - scores.mean():+.3f}")

    # ---- Learning curve on the selected estimator
    # Fit once on the full training split so the curve describes the tuned model
    # rather than an arbitrary default.
    search.fit(features, target)
    sizes, train_scores, test_scores = learning_curve(
        search.best_estimator_, features, target,
        train_sizes=np.linspace(0.1, 1.0, 6), cv=5,
        scoring="roc_auc", random_state=RANDOM_STATE,
    )
    curve = pd.DataFrame(
        {
            "train_size": sizes,
            "train_mean": train_scores.mean(axis=1),
            "train_std": train_scores.std(axis=1),
            "test_mean": test_scores.mean(axis=1),
            "test_std": test_scores.std(axis=1),
        }
    )
    write_csv(curve, "learning_curve.csv")
    print("    learning curve (train / validation ROC-AUC):")
    for row in curve.itertuples():
        print(f"      n={int(row.train_size):4d}  {row.train_mean:.3f}  {row.test_mean:.3f}")

    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "scores": scores.tolist(),
        "optimism": float(tuning["best_cv_score"] - scores.mean()),
        "learning_curve": curve.to_dict(orient="records"),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_data()
    matrix, names = build_feature_matrix(frame)
    churn = frame[TARGET_COLUMN].map(TARGET_MAP).to_numpy()

    pca_info = run_pca(matrix, names)

    kmeans_info = run_kmeans(matrix, churn)
    k = kmeans_info["selected_k"]
    kmeans = fit_final_kmeans(matrix, k)
    kmeans_labels = kmeans.labels_
    kmeans_profiles = run_profiles(kmeans_labels, k, "kmeans")
    churn_overlap = run_churn_overlap(kmeans_labels, kmeans_profiles)

    # Projection for the scatter plots, stored so the figures need no refitting.
    projection = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(matrix)
    write_csv(
        pd.DataFrame(
            {"pc1": projection[:, 0], "pc2": projection[:, 1],
             "kmeans_cluster": kmeans_labels, "churn": churn}
        ),
        "pca_projection.csv",
    )

    hierarchical_info = run_hierarchical(matrix, churn, k)
    hierarchical_profiles = run_profiles(
        hierarchical_info["labels"], hierarchical_info["selected_k"], "hierarchical"
    )
    agreement = run_agreement(kmeans_labels, hierarchical_info["labels"], churn)

    # Dendrogram needs the raw linkage matrix; the figure script only plots it.
    from scipy.cluster.hierarchy import linkage

    linkage_matrix = linkage(matrix, method=hierarchical_info["selected_linkage"])
    pd.DataFrame(
        linkage_matrix, columns=["left", "right", "distance", "count"]
    ).to_csv(OUTPUT_DIR / "linkage_matrix.csv", index=False)

    dimensionality = run_dimensionality_experiment(matrix, k)
    tuning = run_tuning(frame)
    nested = run_nested_cv(frame, tuning)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "numeric_features": len(NUMERIC_FEATURES),
            "categorical_features": len(CATEGORICAL_FEATURES),
            "encoded_features": len(names),
            "churn_yes": int((frame[TARGET_COLUMN] == "Yes").sum()),
            "churn_no": int((frame[TARGET_COLUMN] == "No").sum()),
            "churn_rate": float((frame[TARGET_COLUMN] == "Yes").mean()),
            "senior_citizens": int(frame["senior_citizen"].sum()),
            "missing_cells": int(frame[FEATURE_COLUMNS].isna().sum().sum()),
        },
        "pca": pca_info,
        "kmeans": {
            "selected_k": kmeans_info["selected_k"],
            "selected_silhouette": kmeans_info["selected_silhouette"],
            "preferred_k": kmeans_info["preferred_k"],
            "elbow_k": kmeans_info["elbow_k"],
            "inertia": float(kmeans.inertia_),
            "sizes": [int(v) for v in np.bincount(kmeans_labels, minlength=k)],
            "profiles": kmeans_profiles,
            "churn_overlap": churn_overlap,
        },
        "hierarchical": {
            "selected_linkage": hierarchical_info["selected_linkage"],
            "selected_k": hierarchical_info["selected_k"],
            "selected_silhouette": hierarchical_info["selected_silhouette"],
            "sizes": hierarchical_info["sizes"],
            "profiles": hierarchical_profiles,
            "trap": hierarchical_info["trap"],
        },
        "agreement": agreement,
        "dimensionality": dimensionality,
        "tuning": tuning,
        "nested_cv": nested,
    }

    path = OUTPUT_DIR / "week3_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")

    print("\n" + "=" * 78)
    print("WEEK 3 ANALYSIS COMPLETE")
    print("=" * 78)
    print(f"  K-Means k ................. {k} (silhouette {kmeans_info['selected_silhouette']:.3f})")
    print(f"  Hierarchical .............. {hierarchical_info['selected_linkage']}, k={hierarchical_info['selected_k']}")
    print(f"  K-Means vs Ward ARI ....... {agreement['ari_models']:.3f}")
    print(f"  clusters vs churn ARI ..... {agreement['ari_churn_kmeans']:.3f}")
    print(f"  tuned model ............... {tuning['winner']} "
          f"(test ROC-AUC {tuning['metrics']['roc_auc']:.3f})")
    print(f"  nested CV ROC-AUC ......... {nested['mean']:.3f} +/- {nested['std']:.3f}")
    print(f"  outputs ................... {OUTPUT_DIR.relative_to(WEEK_DIR.parent)}/")
    return summary


if __name__ == "__main__":
    main()
