# Python and machine-learning fundamentals

These notes accompany the four weekly exercises and use the project’s customer-churn example.

## Python building blocks

- **Variables and types:** a variable names a value; Python values include integers, floats, strings, booleans, and `None`.
- **Lists and dictionaries:** lists store ordered values; dictionaries map keys such as feature names to values.
- **Functions:** a function packages reusable work, accepts arguments, and may return a result. The project’s `build_preprocessor()` and `classification_metrics()` functions each do one clearly defined job.
- **Modules and environments:** `.py` files are modules; `pyproject.toml` declares the package and dependencies. A virtual environment isolates project packages.
- **NumPy:** arrays provide compact numerical data and vectorized operations. For example, `probability = 1 / (1 + np.exp(-log_odds))` calculates many logistic probabilities at once.
- **Pandas:** a `DataFrame` is a labeled table and a `Series` is a labeled column. `read_csv`, `isna`, `value_counts`, `groupby`, and `to_csv` support loading, auditing, summarizing, and exporting data.

## Machine-learning workflow

1. Define the prediction target and what information is available at prediction time.
2. Inspect the data: dimensions, types, missingness, duplicates, label balance, and plausible values.
3. Split examples into training and test data. Fit transformations using training data only.
4. Impute missing values, encode categorical values, and scale numeric variables when the chosen algorithm benefits from it.
5. Train a baseline and compare alternatives using metrics appropriate to the task.
6. Tune settings with cross-validation on training data, then evaluate once on a held-out test set.
7. Bundle preprocessing and model steps together, serialize the fitted pipeline, and validate the prediction interface.

A **feature** is an input column and a **target** is the value to predict. Classification predicts a category; regression predicts a number. Supervised learning uses labeled examples. Unsupervised learning looks for structure without using the target to construct groups. In the clustering exercise, churn is used only after clustering to summarize the resulting groups.

## Metrics and interpretation

Accuracy is the fraction of correct predictions, but can hide poor performance for a minority class. Precision answers “of the predicted churners, how many actually churned?” Recall answers “of all churners, how many were found?” F1 combines precision and recall; ROC-AUC measures ranking across possible thresholds. Regression MAE and RMSE are charge units, while R-squared describes explained variance. The silhouette score compares within-cluster cohesion to separation from other clusters. None of these metrics removes the need to inspect data, threshold costs, fairness, and real-world relevance.

## Leakage and limitations

Data leakage occurs when test information or a proxy for the answer influences training. Here, imputation, scaling, and encoding are inside scikit-learn pipelines so cross-validation refits them on each training fold. For the monthly-charge regression, `total_charges` is excluded because it is derived from monthly charges and tenure. The dataset is synthetic, so results demonstrate workflow mechanics, not production performance.
