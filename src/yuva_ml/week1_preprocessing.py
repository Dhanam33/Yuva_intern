"""Week 1: data loading, cleaning, EDA, and preprocessing deliverables."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from .common import ensure_output_dirs, load_raw_data, split_features_target, write_json
from .preprocessing import build_preprocessor
from .schema import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    ID_COLUMN,
    OUTPUT_DIR,
    PLOT_DIR,
    TARGET_COLUMN,
    TARGET_MAP,
)


def run_week1() -> dict:
    data = load_raw_data()
    ensure_output_dirs()

    duplicate_rows = int(data.duplicated().sum())
    duplicate_ids = int(data[ID_COLUMN].duplicated().sum())
    if duplicate_rows:
        data = data.drop_duplicates().copy()
    if data[ID_COLUMN].duplicated().any():
        data = data.drop_duplicates(subset=ID_COLUMN, keep="first").copy()

    feature_data, y = split_features_target(data)
    missing = data[FEATURE_COLUMNS].isna().sum().rename("missing_count").to_frame()
    missing["missing_percent"] = 100.0 * missing["missing_count"] / len(data)
    missing["dtype"] = data[FEATURE_COLUMNS].dtypes.astype(str)
    missing.to_csv(OUTPUT_DIR / "missing_value_audit.csv", index_label="feature")

    # EDA chart 1: original-data missingness, before any imputation.
    missing_counts = missing.sort_values("missing_count")
    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    ax.barh(missing_counts.index, missing_counts["missing_count"], color="#5577aa")
    ax.set_title("Missing values by input feature (raw data)")
    ax.set_xlabel("Missing records")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "missing_values.png", dpi=160)
    plt.close(fig)

    # EDA chart 2: target balance.
    target_counts = data[TARGET_COLUMN].value_counts().reindex(["No", "Yes"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    bars = ax.bar(target_counts.index, target_counts.values, color=["#5b8e7d", "#d17a61"])
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_title("Customer churn target distribution")
    ax.set_ylabel("Customers")
    ax.set_ylim(0, target_counts.max() * 1.14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "target_distribution.png", dpi=160)
    plt.close(fig)

    # EDA chart 3: churn rate by contract. The missing category is retained as
    # an explicit group for visualization; model pipelines impute it instead.
    contract_view = data[["contract", TARGET_COLUMN]].copy()
    contract_view["contract"] = contract_view["contract"].fillna("Missing")
    churn_by_contract = (
        contract_view.assign(churn_flag=(contract_view[TARGET_COLUMN] == "Yes").astype(int))
        .groupby("contract", observed=False)["churn_flag"]
        .mean()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    bars = ax.bar(churn_by_contract.index, churn_by_contract.values, color="#d17a61")
    ax.bar_label(bars, fmt="%.1%", padding=3)
    ax.set_title("Observed churn rate by contract type")
    ax.set_ylabel("Churn rate")
    ax.set_ylim(0, min(1.0, float(churn_by_contract.max()) * 1.20))
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "churn_by_contract.png", dpi=160)
    plt.close(fig)

    # The transformer is fitted once here to create a readable educational
    # artifact. Model scoring later fits the same steps only on training folds.
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(feature_data, y)
    feature_names = preprocessor.get_feature_names_out()
    transformed_frame = pd.DataFrame(transformed, columns=feature_names, index=data.index)
    transformed_frame.insert(0, ID_COLUMN, data[ID_COLUMN].values)
    transformed_frame[TARGET_COLUMN] = y.values
    transformed_frame.to_csv(OUTPUT_DIR / "preprocessed_customer_churn.csv", index=False)

    discrete_features = np.array(
        [
            any(name.startswith(f"{column}_") for column in CATEGORICAL_FEATURES)
            for name in feature_names
        ],
        dtype=bool,
    )
    mutual_info = mutual_info_classif(
        transformed,
        y,
        discrete_features=discrete_features,
        random_state=42,
    )
    feature_ranking = pd.DataFrame(
        {"feature": feature_names, "mutual_information": mutual_info}
    ).sort_values("mutual_information", ascending=False, kind="stable")
    feature_ranking.to_csv(OUTPUT_DIR / "feature_selection.csv", index=False)
    selected_features = feature_ranking.head(10)["feature"].tolist()
    selected = transformed_frame[[ID_COLUMN, *selected_features, TARGET_COLUMN]]
    selected.to_csv(OUTPUT_DIR / "selected_feature_dataset.csv", index=False)

    clean_feature_cells = int(transformed_frame.drop(columns=[ID_COLUMN, TARGET_COLUMN]).isna().sum().sum())
    summary = {
        "source": "synthetic customer churn dataset; generated with seed 2026",
        "rows": int(len(data)),
        "raw_feature_columns": int(len(FEATURE_COLUMNS)),
        "raw_missing_feature_cells": int(data[FEATURE_COLUMNS].isna().sum().sum()),
        "duplicate_rows_removed": duplicate_rows,
        "duplicate_customer_ids_removed": duplicate_ids,
        "encoded_scaled_feature_columns": int(len(feature_names)),
        "processed_feature_cells_remaining_missing": clean_feature_cells,
        "churn_yes_count": int((data[TARGET_COLUMN] == "Yes").sum()),
        "churn_no_count": int((data[TARGET_COLUMN] == "No").sum()),
        "categorical_encoding": "OneHotEncoder(handle_unknown='ignore')",
        "numeric_imputation": "median; missingness indicators retained",
        "categorical_imputation": "most frequent value",
        "numeric_scaling": "StandardScaler",
        "feature_selection_method": "mutual information; top 10 encoded features exported for exploratory reporting",
        "top_10_features": selected_features,
        "model_evaluation_note": (
            "The all-row transformed CSV and mutual-information ranking are educational artifacts only, "
            "not inputs to the reported model evaluation. Evaluation scripts fit preprocessing inside "
            "the training split/CV folds; any feature selection used for modeling should also be fit there."
        ),
    }
    write_json(OUTPUT_DIR / "week1_summary.json", summary)
    print("Week 1 preprocessing complete")
    print(f"  rows: {len(data):,}; raw missing cells: {summary['raw_missing_feature_cells']:,}")
    print(f"  transformed columns: {len(feature_names)}; remaining missing cells: {clean_feature_cells}")
    print(f"  top feature: {feature_ranking.iloc[0]['feature']}")
    return summary


if __name__ == "__main__":
    run_week1()
