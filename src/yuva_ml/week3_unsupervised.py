"""Week 3: clustering, PCA, cross-validation, and hyperparameter tuning."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from .common import (
    classification_metrics,
    ensure_output_dirs,
    load_raw_data,
    split_features_target,
    write_json,
)
from .preprocessing import build_preprocessor
from .schema import OUTPUT_DIR, PLOT_DIR

RANDOM_STATE = 42


def run_week3() -> dict:
    data = load_raw_data()
    ensure_output_dirs()
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Hyperparameter tuning uses training folds only. The held-out test set is
    # evaluated once after the best regularization strength is selected.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tuning_pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE
                ),
            ),
        ]
    )
    search = GridSearchCV(
        estimator=tuning_pipeline,
        param_grid={"classifier__C": [0.1, 1.0, 10.0]},
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        return_train_score=True,
        refit=True,
    )
    search.fit(X_train, y_train)
    tuned_model = search.best_estimator_
    test_prediction = tuned_model.predict(X_test)
    test_probability = tuned_model.predict_proba(X_test)[:, 1]
    tuned_test_metrics = classification_metrics(y_test, test_prediction, test_probability)
    tuned_matrix = confusion_matrix(y_test, test_prediction, labels=[0, 1])
    pd.DataFrame(
        tuned_matrix,
        index=["actual_no", "actual_yes"],
        columns=["predicted_no", "predicted_yes"],
    ).to_csv(OUTPUT_DIR / "tuned_confusion_matrix.csv")

    cv_results = pd.DataFrame(search.cv_results_)
    compact_cv = cv_results[
        ["param_classifier__C", "mean_test_score", "std_test_score", "mean_train_score"]
    ].rename(
        columns={
            "param_classifier__C": "C",
            "mean_test_score": "mean_cv_roc_auc",
            "std_test_score": "std_cv_roc_auc",
            "mean_train_score": "mean_train_roc_auc",
        }
    )
    compact_cv.to_csv(OUTPUT_DIR / "cross_validation_results.csv", index=False)
    best_cv_mean = float(search.cv_results_["mean_test_score"][search.best_index_])
    best_cv_std = float(search.cv_results_["std_test_score"][search.best_index_])
    tuning_result = {
        "model": "Logistic Regression",
        "best_params": {key: float(value) for key, value in search.best_params_.items()},
        "best_cross_validation_roc_auc_mean": best_cv_mean,
        "best_cross_validation_roc_auc_std": best_cv_std,
        "cv_folds": 5,
        "cv_strategy": "StratifiedKFold(shuffle=True, random_state=42)",
        "test_metrics": tuned_test_metrics,
        "test_confusion_matrix": tuned_matrix.tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_set_used_for_tuning": False,
    }
    write_json(OUTPUT_DIR / "tuned_model_evaluation.json", tuning_result)
    write_json(OUTPUT_DIR / "tuned_hyperparameters.json", {"best_params": tuning_result["best_params"]})

    # Unsupervised exercise: the target is deliberately excluded from X. All
    # preprocessing is unsupervised here; churn labels are only used later to
    # describe cluster composition, not to create clusters.
    unsupervised_preprocessor = build_preprocessor()
    X_cluster = unsupervised_preprocessor.fit_transform(X)
    feature_names = unsupervised_preprocessor.get_feature_names_out()
    cluster_rows = []
    best = {"silhouette_score": -np.inf}
    sample_size = min(1000, len(X_cluster))
    for algorithm_name in ["K-Means", "Agglomerative"]:
        for k in range(2, 7):
            if algorithm_name == "K-Means":
                clusterer = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
            else:
                clusterer = AgglomerativeClustering(n_clusters=k, linkage="ward")
            labels = clusterer.fit_predict(X_cluster)
            score = float(
                silhouette_score(
                    X_cluster,
                    labels,
                    sample_size=sample_size,
                    random_state=RANDOM_STATE,
                )
            )
            cluster_rows.append(
                {
                    "algorithm": algorithm_name,
                    "n_clusters": k,
                    "silhouette_score": score,
                    "silhouette_sample_size": sample_size,
                }
            )
            if score > best["silhouette_score"]:
                best = {
                    "algorithm": algorithm_name,
                    "n_clusters": k,
                    "silhouette_score": score,
                    "labels": labels,
                }

    cluster_metrics = pd.DataFrame(cluster_rows).sort_values(
        "silhouette_score", ascending=False
    )
    cluster_metrics.to_csv(OUTPUT_DIR / "clustering_metrics.csv", index=False)
    best_labels = best.pop("labels")
    best["silhouette_sample_size"] = sample_size

    # PCA is fitted on transformed input features and reduced to two dimensions
    # only for visualization; clustering above uses the complete transformed data.
    pca = PCA(n_components=2, random_state=RANDOM_STATE, svd_solver="full")
    coordinates = pca.fit_transform(X_cluster)
    pca_summary = {
        "component_1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "component_2_explained_variance_ratio": float(pca.explained_variance_ratio_[1]),
        "two_component_cumulative_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "pca_role": "visualization only; clustering used all scaled/encoded features",
    }
    pca_frame = pd.DataFrame(coordinates, columns=["PC1", "PC2"])
    pca_frame["cluster"] = best_labels
    pca_frame["churn"] = y.to_numpy()
    pca_frame.to_csv(OUTPUT_DIR / "pca_cluster_coordinates.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.1, 6.0))
    scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=best_labels,
        cmap="viridis",
        s=18,
        alpha=0.70,
        edgecolors="none",
    )
    ax.set_title(f"PCA view of {best['algorithm']} clusters (k={best['n_clusters']})")
    ax.set_xlabel(f"PC1 ({pca_summary['component_1_explained_variance_ratio']:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca_summary['component_2_explained_variance_ratio']:.1%} variance)")
    ax.grid(alpha=0.18)
    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pca_cluster_view.png", dpi=160)
    plt.close(fig)

    cluster_composition = pd.DataFrame(
        {"cluster": best_labels, "churn_yes": y.to_numpy()}
    ).groupby("cluster", as_index=False).agg(
        customers=("churn_yes", "size"),
        churn_count=("churn_yes", "sum"),
        churn_rate=("churn_yes", "mean"),
    )
    cluster_composition.to_csv(OUTPUT_DIR / "selected_cluster_summary.csv", index=False)

    summary = {
        "selected_cluster_solution": best,
        "pca": pca_summary,
        "cluster_composition": cluster_composition.to_dict(orient="records"),
        "all_cluster_candidates": cluster_rows,
        "hyperparameter_tuning": tuning_result,
        "interpretation_note": (
            "Silhouette score measures separation in the transformed feature space; "
            "it is not proof that clusters are natural customer segments. PCA is a 2-D view only."
        ),
    }
    write_json(OUTPUT_DIR / "week3_summary.json", summary)
    print("Week 3 clustering and evaluation complete")
    print(cluster_metrics.head(5).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(
        f"Best CV ROC-AUC={best_cv_mean:.3f} +/- {best_cv_std:.3f}; "
        f"test ROC-AUC={tuned_test_metrics['roc_auc']:.3f}"
    )
    print(
        f"Selected clusters: {best['algorithm']} k={best['n_clusters']}, "
        f"silhouette={best['silhouette_score']:.3f}"
    )
    return summary


if __name__ == "__main__":
    run_week3()
