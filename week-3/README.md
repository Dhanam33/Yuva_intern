# Week 3 — Unsupervised Learning & Model Evaluation

Complete Week 3 deliverable for the Yuva Internship: clustering, dimensionality reduction, and a
full evaluation of a tuned classifier on the customer-churn dataset.

**Dataset:** `data/customer_churn_raw.csv` — 1,800 synthetic customer records
**Report:** [`reports/Week_3_Unsupervised_Learning_Model_Evaluation_Report.docx`](reports/Week_3_Unsupervised_Learning_Model_Evaluation_Report.docx)
**Notebook:** [`notebooks/week3_unsupervised_evaluation.ipynb`](notebooks/week3_unsupervised_evaluation.ipynb)

> The dataset is reproducible **synthetic teaching data**, not real customer information.
> Every pattern here is a property of the generator, not of customers.

---

## The short version

Two different clustering algorithms independently recover **the same three customer segments**
(ARI 0.732). That is strong evidence the segments are real.

Those segments carry **almost no information about churn** (ARI 0.015). Churn rate varies by only
9 percentage points across them.

Real structure and a useful predictor are different things — which is exactly why Part B exists.

---

## What is in this folder

```text
week-3/
├── README.md                       This file
├── requirements.txt                Python dependencies
├── data/customer_churn_raw.csv     Raw sample dataset
├── src/
│   ├── run_analysis.py             Clustering, PCA, tuning, nested CV — writes every table
│   └── make_figures.py             Builds all 14 figures from the saved outputs
├── notebooks/
│   └── week3_unsupervised_evaluation.ipynb     Executed teaching notebook (78 cells)
├── docs/
│   ├── clustering_and_evaluation.md            Metric-by-metric companion
│   ├── SUBMISSION_DESCRIPTION.md               230-word description for the form
│   └── build_report.py                         Generates the Word report from the JSON summary
├── reports/
│   └── Week_3_Unsupervised_Learning_Model_Evaluation_Report.docx
└── outputs/
    ├── kmeans_sweep.csv                   k = 2…8 across four indices
    ├── hierarchical_sweep.csv             Four linkages × four values of k
    ├── linkage_matrix.csv                 Ward merge history for the dendrogram
    ├── cluster_profiles_*.csv             Per-cluster means and category modes
    ├── cluster_churn_crosstab.csv         Clusters against the label
    ├── cluster_agreement.csv              ARI / AMI between partitions
    ├── pca_variance.csv                   Scree and cumulative variance
    ├── pca_loadings.csv                   Feature weights on PC1–PC3
    ├── pca_projection.csv                 2-D coordinates with cluster and label
    ├── dimensionality_experiment.csv      Silhouette in reduced vs original space
    ├── tuning_grid_logistic.csv           5 combinations
    ├── tuning_grid_random_forest.csv      32 combinations
    ├── tuned_vs_default.csv               Did tuning help?
    ├── nested_cv_results.csv              Outer fold scores
    ├── learning_curve.csv                 Train vs validation by training size
    ├── final_metrics.csv                  Held-out scores
    ├── final_confusion_matrix.csv         TN / FP / FN / TP
    ├── final_predictions.csv              Per-row probabilities
    ├── final_roc_curve.csv                ROC coordinates
    ├── final_calibration_curve.csv        Reliability diagram
    ├── week3_summary.json                 Machine-readable record
    └── figures/                           14 PNG figures
```

---

## Quick start

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r week-3/requirements.txt

python week-3/src/run_analysis.py    # clustering + tuning, writes all tables
python week-3/src/make_figures.py    # builds the 14 figures
python week-3/docs/build_report.py   # rebuilds the Word report
```

Runs in about 80 seconds, needs no network, and is seeded at 42 throughout.

---

## Part A — Unsupervised

### Dimensionality reduction

| Measure | Result |
|---|---|
| Encoded feature columns | 21 |
| Components for 90% of variance | **10** |
| Components for 95% of variance | 12 |
| Variance in PC1 / PC2 / PC3 | 24.4% / 14.9% / 12.1% |

PC1 loads on `total_charges` (+0.67), `monthly_charges` (+0.49) and `tenure_months` (+0.46) — not
three findings, but **one axis: how much a customer is worth and how long they have stayed.**

### Choosing k

| Index | Prefers | Better when |
|---|---|---|
| Silhouette | **k = 3** (0.162) | higher |
| Davies-Bouldin | k = 5 (1.814) | lower |
| Calinski-Harabasz | k = 2 (366.6) | higher |
| Elbow (largest inertia drop) | k = 3 | — |

They disagree. **k = 3** was chosen because silhouette peaks there, the elbow agrees, and three
segments are few enough to describe. At 0.162, the absolute silhouette is weak — below roughly 0.25
means the clusters overlap.

### The three segments

| Segment | Customers | Tenure | Monthly | Total | Churn |
|---|---|---|---|---|---|
| Short-tenure, low-spend | 797 (44.3%) | 27.8 | 48.8 | 1,172 | 21.8% |
| Long-tenure, high-spend | 685 (38.1%) | 48.3 | 73.5 | 3,459 | 30.4% |
| Senior segment | 318 (17.7%) | 37.2 | 62.8 | 2,256 | 30.8% |

The senior segment contains 318 of the dataset's 319 senior citizens. **K-Means was never told the
column existed.**

### Hierarchical clustering

| Linkage | Best silhouette | Smallest cluster | Verdict |
|---|---|---|---|
| ward | 0.145 (k=3) | 319 | **selected** |
| complete | 0.134 (k=2) | 268 | degenerate at k≥3 |
| average | 0.270 (k=2) | **5** | degenerate |
| single | **0.294** (k=2) | **1** | degenerate — and the highest score in the table |

**This is the trap.** Single linkage wins the silhouette contest by isolating a single customer. A
one-point cluster has a perfect silhouette by definition, so it inflates the average enormously.
Any solution with a cluster under 2% of the data is discarded before the best remaining one is chosen.

Ward at k=3 agrees with K-Means at **ARI 0.732**.

### Do the clusters explain churn?

| Comparison | ARI |
|---|---|
| K-Means vs Ward | **+0.732** |
| K-Means vs churn | **+0.015** |
| Ward vs churn | +0.010 |

Churn rate by cluster: 21.8%, 30.4%, 30.8% — a spread of 9.0 points around an overall 26.7%.

### The dimensionality trap

| Components kept | Variance retained | Silhouette in reduced space | Silhouette in **original** space |
|---|---|---|---|
| 2 | 39.2% | **0.389** | 0.135 |
| 3 | 51.3% | 0.359 | 0.163 |
| 5 | 68.5% | 0.256 | 0.162 |
| 21 (none) | 100% | 0.162 | 0.162 |

Projecting to 2-D appears to nearly triple the silhouette. It doesn't — **the ruler changed, not the
clusters.** Compare clusterings in a consistent space.

---

## Part B — Evaluation

### Held-out results

360 held-out test rows, scored once. Logistic Regression, C = 0.1, `class_weight='balanced'`.

| Metric | Value | Reading |
|---|---|---|
| Accuracy | 0.678 | **below** the 0.733 do-nothing baseline |
| Balanced accuracy | 0.671 | 0.500 for a do-nothing model |
| Precision | 0.432 | 57% of flags are false alarms |
| Recall | 0.656 | catches about two-thirds of churners |
| F1 | 0.521 | harmonic mean of the two |
| ROC-AUC | 0.768 | 0.500 for no skill |
| PR-AUC | 0.517 | 0.267 for no skill |
| Brier | 0.196 | mean squared error of probabilities |

Confusion matrix: **TN 181 · FP 83 · FN 33 · TP 63**

### The central paradox

A model that never predicts churn scores **0.733 accuracy** — higher than the classifier's 0.678.
It has recall of zero.

Accuracy is not broken; it is answering a question nobody asked. It rewards ignoring the minority
class, which is the only class the business cares about.

### Cross-validation and tuning

5-fold stratified CV on 1,440 training rows: **0.805 ± 0.025**.

| Model | Default CV | Tuned CV | Gain |
|---|---|---|---|
| Logistic Regression | 0.805 | 0.806 | **+0.001** |
| Random Forest | 0.779 | 0.799 | +0.020 |

| Grid | Combinations | Best |
|---|---|---|
| Logistic Regression | 5 (C) | C = 0.1 → 0.806 |
| Random Forest | 32 | depth 8, leaf 5, sqrt, 300 trees → 0.799 |

**Logistic Regression gained 0.001 across five values of C — twenty-five times smaller than the fold
noise.** The honest conclusion is not that C = 0.1 is best, but that C is not a lever that matters
here. The default was already optimal.

### Nested CV — is the tuned score optimistic?

| Estimate | ROC-AUC |
|---|---|
| Tuned CV (inner loop) | 0.806 |
| Nested CV (unbiased) | 0.805 ± 0.025 |
| Optimism | **+0.001** |

The tuned score was honest, because a five-value grid has almost nothing to overfit. With a wider
grid on a smaller dataset the gap can be several times larger — which is why the method matters even
when the number looks fine.

### Learning curve

Training and validation converge as data is added: the gap narrows from **0.081 to 0.007**.

That is variance, not bias. More data would help a little; more model capacity would not help at all.

---

## What this week actually demonstrates

| Claim | Evidence |
|---|---|
| Clustering found real structure | two algorithms agree at ARI 0.732 |
| The structure is weak | best silhouette 0.162 |
| Structure ≠ signal | clusters vs churn ARI 0.015 |
| Silhouette can be gamed | single linkage: 0.294 with a 1-customer cluster |
| Dimension reduction inflates scores | 0.135 → 0.389 by measuring in 2-D |
| Accuracy misleads on imbalance | baseline 0.733 beats the model's 0.678 |
| Tuning has a ceiling | +0.001 across five values of C |
| Tuned scores can be optimistic | nested CV exists to measure by how much |

---

## Where each brief objective is addressed

| Objective | Section | Notebook |
|---|---|---|
| K-Means and hierarchical clustering | 6, 8 | §3–6 |
| Dimensionality reduction | 5, 10 | §2, §7 |
| Cross-validation | 14 | §12 |
| Confusion matrix | 11 | §9 |
| Precision, recall, F1 | 12 | §10 |
| ROC-AUC | 13 | §11 |
| Hyperparameter tuning | 15 | §13 |
| Clustering deliverable | 5–10 | §2–7 |
| Evaluation report | 11–19 | §8–17 |

---

## Limitations

- **Synthetic data** — every pattern is a property of the generator.
- **1,800 records** — two percentage points of accuracy is roughly one standard error.
- **Silhouette 0.162** — the segmentation describes, it does not separate.
- **k chosen by judgement** — k = 2 or k = 5 are defensible alternatives.
- **No fairness analysis** — error rates can differ by subgroup.
- **Threshold left at 0.5** — a deployment would need one chosen against real costs.

---

## Related folders

`week-1/` builds the cleaned dataset. `week-2/` trains and compares the supervised models this week
tunes and evaluates. Week 4 serialises the pipeline behind an API.
