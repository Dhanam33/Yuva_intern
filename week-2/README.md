# Week 2 — Supervised Machine Learning Models

Complete Week 2 deliverable for the Yuva Internship: five supervised algorithms trained,
compared and evaluated on the customer-churn dataset prepared in Week 1.

**Dataset:** `data/customer_churn_raw.csv` — 1,800 synthetic customer records
**Report:** [`reports/Week_2_Supervised_Machine_Learning_Report.docx`](reports/Week_2_Supervised_Machine_Learning_Report.docx)
**Notebook:** [`notebooks/week2_supervised_learning.ipynb`](notebooks/week2_supervised_learning.ipynb)

> The dataset is reproducible **synthetic teaching data**, not real customer information.
> Every score in this folder describes a generator, not a business.

---

## What is in this folder

```text
week-2/
├── README.md                       This file
├── requirements.txt                Python dependencies
├── data/customer_churn_raw.csv     Raw sample dataset
├── src/
│   ├── train_models.py             Trains all 8 models + 3 baselines, exports every table
│   └── make_figures.py             Builds all 13 figures from the saved outputs
├── notebooks/
│   └── week2_supervised_learning.ipynb    Executed teaching notebook (52 cells)
├── docs/
│   ├── model_comparison.md         Metric-by-metric analysis
│   ├── SUBMISSION_DESCRIPTION.md   273-word description for the form
│   └── build_report.py             Generates the Word report from the JSON summary
├── reports/
│   └── Week_2_Supervised_Machine_Learning_Report.docx
└── outputs/
    ├── classification_metrics.csv          Ten metrics for 6 classifiers
    ├── regression_metrics.csv              Five metrics for 5 regressors
    ├── classification_cv_results.csv       Per-fold scores
    ├── regression_cv_results.csv           Per-fold scores
    ├── classification_predictions.csv      Per-row probabilities
    ├── regression_predictions.csv          Per-row predictions
    ├── confusion_matrix_*.csv              Four confusion matrices
    ├── threshold_analysis.csv              17 thresholds swept
    ├── class_weight_comparison.csv         Weighted vs unweighted
    ├── feature_importance.csv              Random Forest importance
    ├── tree_depth_curve.csv                Overfitting demonstration
    ├── week2_summary.json                  Machine-readable record
    └── figures/                            13 PNG figures
```

---

## Quick start

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r week-2/requirements.txt

python week-2/src/train_models.py    # trains everything, writes all tables
python week-2/src/make_figures.py    # builds the 13 figures
python week-2/docs/build_report.py   # rebuilds the Word report
```

Runs in about 15 seconds, needs no network, and is seeded at 42 throughout.

---

## Results

### Classification — predicting churn

1,440 training rows · 360 held-out test rows · stratified · seed 42

| Model | Accuracy | Balanced acc. | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.678 | 0.671 | 0.432 | 0.656 | 0.521 | **0.769** | **0.518** |
| Random Forest | 0.717 | **0.694** | 0.477 | 0.646 | **0.549** | 0.760 | 0.511 |
| K-Nearest Neighbors | **0.744** | 0.634 | **0.528** | 0.396 | 0.452 | 0.734 | 0.468 |
| Decision Tree | 0.653 | 0.661 | 0.409 | **0.677** | 0.510 | 0.725 | 0.486 |
| *Dummy (most frequent)* | *0.733* | *0.500* | *0.000* | *0.000* | *0.000* | *0.500* | *0.267* |
| *Dummy (stratified)* | *0.661* | *0.554* | *0.352* | *0.323* | *0.337* | *0.554* | *0.286* |

### Regression — predicting monthly charges

1,377 training rows · 345 test rows · `total_charges` excluded to prevent leakage

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression** | **7.59** | **9.65** | **0.818** |
| Random Forest Regressor | 7.80 | 9.96 | 0.806 |
| Decision Tree Regressor | 7.83 | 10.02 | 0.804 |
| KNN Regressor | 10.87 | 13.74 | 0.631 |
| *Dummy (mean)* | *18.50* | *22.80* | *-0.017* |

---

## Three findings worth reading the report for

### 1. Three models win three different metrics

KNN has the best **accuracy** but the worst **recall** — it misses 58 of 93 churners.
Logistic Regression has the best **ROC-AUC** and roughly double KNN's recall.
The Decision Tree has the highest **recall** but the lowest accuracy.

No single metric identifies a best model, because the metrics measure different things.

### 2. A do-nothing model beats two real models on accuracy

The `most_frequent` dummy scores **0.733** accuracy by never predicting churn.
Logistic Regression (0.678) and the Decision Tree (0.653) score lower.

That is not a bug. Both use `class_weight='balanced'`, which deliberately trades accuracy
for recall. They are *worse by one metric because they are better at the job* — and an
evaluation that reported only accuracy would have rejected the most useful model.

### 3. Scaling is not cosmetic — it was measured

The identical KNN, with and without `StandardScaler`:

| Setting | Accuracy | Recall | ROC-AUC |
|---|---|---|---|
| With scaling | 0.744 | **0.396** | **0.734** |
| Without scaling | 0.708 | **0.156** | **0.608** |

Recall falls by 24 percentage points, because distance computed on raw units is dominated
entirely by `total_charges` (thousands) versus `senior_citizen` (0 or 1). Nothing in the
training output warns you.

---

## Model selection discipline

| Rule | How it is enforced |
|---|---|
| Preprocessing fitted on training data only | Everything inside a scikit-learn `Pipeline` |
| Model chosen without touching the test set | 5-fold stratified CV on the training split |
| Test set scored once | Only after selection, at the end |
| Baselines always present | Two dummy classifiers, one dummy regressor |

Cross-validation showed the top two classifiers differ by **0.009 ROC-AUC** while the
fold-to-fold standard deviation is **0.025**. They are not reliably different, and the
report says so rather than declaring a winner.

---

## Thresholds and class weights

Two separate levers on the same precision-recall trade-off:

| Lever | Effect |
|---|---|
| **Decision threshold** (after training) | Moving 0.50 → 0.40 raises recall 0.45 → 0.82, cuts precision 0.61 → 0.42 |
| **Class weight** (during training) | Balanced raises recall 0.448 → 0.656, cuts precision 0.606 → 0.432 |

Which setting is correct depends on the cost of a missed churner versus a wasted retention
offer. **The data cannot answer that; only the business can.**

---

## Limitations

- **Synthetic data** — every pattern is a property of the generator.
- **360 test rows** — roughly ±2.5 percentage points of sampling noise on accuracy.
- **Hyperparameters fixed in advance** — tuning on the test set would invalidate the scores.
- **No fairness analysis** — error rates can differ by subgroup.
- **Regression is capped by its features** — monthly charges are a step function of
  categorical inputs, so predictions form bands no algorithm can smooth out.

---

## Related folders

`week-1/` builds the cleaned dataset this week consumes. Week 3 adds clustering,
PCA and cross-validated tuning. Week 4 serialises the pipeline behind an API.
