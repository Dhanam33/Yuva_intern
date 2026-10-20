# Week 3 — Clustering and evaluation, metric by metric

Companion to `src/run_analysis.py`. Every number here is produced by that script and stored in
`outputs/`; the report is generated from the same files.

- **Part A:** clustering and PCA, fitted without the churn label
- **Part B:** supervised evaluation, tuned on the training split
- **Seed:** 42 throughout · **Split:** 80/20 stratified · **CV:** 5-fold stratified

---

## Part A — Unsupervised

### 1 · The feature space

| | Count |
|---|---|
| Numeric features | 5 |
| Categorical features | 6 |
| Encoded columns | **21** |
| Missing feature cells | 562 |

**Numeric columns are standardised; one-hot columns are left at 0/1.** Scaling a dummy would give a
category present in 5% of rows the same weight in the distance metric as one present in 50%. Numbers
need scaling; indicators do not.

### 2 · PCA

| Component | Variance | Cumulative |
|---|---|---|
| 1 | 24.4% | 24.4% |
| 2 | 14.9% | 39.2% |
| 3 | 12.1% | 51.3% |
| 5 | — | 68.5% |
| 8 | — | 83.1% |
| **10** | — | **90.4%** |
| 12 | — | 95.9% |
| 21 | — | 100% |

Top PC1 loadings: `total_charges` +0.673, `monthly_charges` +0.492, `tenure_months` +0.463,
`internet_service_Fiber optic` +0.202.

**These are one axis, not three findings.** A customer paying a lot for a long time has high total
charges; PCA notices that the three columns move together.

### 3 · K-Means sweep

| k | Inertia | Silhouette | Davies-Bouldin | Calinski-Harabasz | ARI vs churn |
|---|---|---|---|---|---|
| 2 | 12,374 | 0.154 | 2.157 | 366.6 | +0.015 |
| **3** | **11,023** | **0.162** | 1.923 | 315.7 | +0.015 |
| 4 | 10,028 | 0.154 | 1.884 | 290.7 | +0.000 |
| 5 | 9,406 | 0.146 | **1.814** | 262.0 | +0.011 |
| 6 | 8,931 | 0.131 | 1.822 | 239.7 | +0.010 |
| 7 | 8,508 | 0.138 | 1.885 | 224.4 | +0.009 |
| 8 | 8,125 | 0.129 | 1.996 | 213.4 | +0.017 |

The three indices disagree — silhouette → 3, Davies-Bouldin → 5, Calinski-Harabasz → 2. No formula
arbitrates between them, because they measure different properties.

**k = 3** chosen: silhouette peaks there, the elbow agrees (largest inertia drop), and three segments
are describable.

**0.162 is a weak score.** Below roughly 0.25 means the clusters overlap. The ranking picks the
least-bad option; it does not identify a strong one.

### 4 · K-Means profiles (k = 3)

| # | Segment | n | Share | Tenure | Monthly | Total | Tickets | Churn |
|---|---|---|---|---|---|---|---|---|
| 0 | Long-tenure, high-spend | 685 | 38.1% | 48.3 | 73.5 | 3,459 | 1.14 | 30.4% |
| 1 | Short-tenure, low-spend | 797 | 44.3% | 27.8 | 48.8 | 1,172 | 0.88 | 21.8% |
| 2 | Senior segment | 318 | 17.7% | 37.2 | 62.8 | 2,256 | 1.02 | 30.8% |

Cluster 2 has `senior_citizen` mean = **1.00**. It holds 318 of the dataset's 319 seniors, and K-Means
was never told the column existed.

Clusters 0 and 1 form the tenure × spend axis — the same axis PC1 found in section 2.

### 5 · Hierarchical clustering

| Linkage | k=2 | k=3 | k=4 | k=5 | Smallest at best |
|---|---|---|---|---|---|
| ward | 0.116 | **0.145** | 0.136 | 0.111 | 319 |
| complete | 0.134 | 0.055 | 0.062 | 0.048 | 187 |
| average | 0.270 ⚠ | 0.179 ⚠ | 0.141 | 0.129 | **4** |
| single | **0.294** ⚠ | 0.271 ⚠ | 0.173 | 0.154 | **1** |

⚠ = degenerate (a cluster below 2% of the data).

**The trap:** single linkage has the highest silhouette in the entire report — and it is worthless.
A point alone in its cluster has a within-cluster distance of zero by definition, so its silhouette
is the maximum possible. A partition with singletons has a structurally inflated average.

Filter to non-degenerate solutions first, then take the best: **ward, k = 3, silhouette 0.145,
sizes [801, 680, 319]**.

### 6 · Do the clusters explain churn?

| Comparison | ARI | AMI |
|---|---|---|
| K-Means vs Ward | **+0.732** | 0.733 |
| K-Means vs churn | **+0.015** | — |
| Ward vs churn | +0.010 | — |

| Segment | Stayed | Churned | Churn rate |
|---|---|---|---|
| Long-tenure, high-spend | 477 | 209 | 30.4% |
| Short-tenure, low-spend | 623 | 174 | 21.8% |
| Senior segment | 220 | 98 | 30.8% |

Spread: **9.0 percentage points** around an overall 26.7%.

**Read this as two separate conclusions.**

1. *The segments are real.* Two algorithms with completely different logic recover almost the same
   partition (ARI 0.732), and the same senior group emerges in both. Structure that reproducible is
   not an artefact of one method's assumptions.
2. *The segments are not churn.* ARI 0.015 is indistinguishable from random. Customer segments and
   customer behaviour are different things.

### 7 · Clustering after dimensionality reduction

| Components | Variance kept | Silhouette in reduced space | Silhouette in original space | Inflation |
|---|---|---|---|---|
| **2** | 39.2% | **0.389** | 0.135 | +0.254 |
| 3 | 51.3% | 0.359 | 0.163 | +0.196 |
| 5 | 68.5% | 0.256 | 0.162 | +0.093 |
| 8 | 83.1% | 0.201 | 0.162 | +0.039 |
| 10 | 90.4% | 0.183 | 0.162 | +0.020 |
| 12 | 95.9% | 0.170 | 0.162 | +0.008 |
| 16 | 100% | 0.162 | 0.162 | 0.000 |
| 21 | 100% | 0.162 | 0.162 | 0.000 |

Silhouette appears to nearly triple when measured in 2-D. **The clusters did not change; the ruler
did.** Points in two dimensions are genuinely closer together, and silhouette rewards that.

The tell is the last column: at 16+ components the two columns converge, because there is no
compression left to exploit.

---

## Part B — Evaluation

### 8 · The confusion matrix

Of 360 held-out customers, 264 stayed and 96 churned.

| | Predicted stayed | Predicted churned |
|---|---|---|
| **Actually stayed** | 181 TN | 83 FP |
| **Actually churned** | 33 FN | 63 TP |

| Cell | Count | Cost when wrong |
|---|---|---|
| True negative | 181 | none |
| False positive | 83 | a wasted retention offer |
| False negative | 33 | a lost customer |
| True positive | 63 | none |

The model errs 2.5:1 toward false alarms. Whether that is correct depends on the ratio of an offer's
cost to a customer's value — **a business input the data cannot supply.**

### 9 · Every metric, derived

| Metric | Formula | Worked | Value |
|---|---|---|---|
| Accuracy | (TP+TN)/total | (63+181)/360 | 0.678 |
| Precision | TP/(TP+FP) | 63/146 | 0.432 |
| Recall | TP/(TP+FN) | 63/96 | 0.656 |
| F1 | 2PR/(P+R) | — | 0.521 |
| Balanced accuracy | mean of per-class recall | — | 0.671 |
| MCC | correlation with truth | — | 0.308 |
| ROC-AUC | area under ROC | — | 0.768 |
| PR-AUC | area under PR | — | 0.517 |
| Brier | MSE of probabilities | — | 0.196 |

### 10 · Why accuracy is the wrong headline

| Model | Accuracy | Recall |
|---|---|---|
| **Dummy (always "no")** | **0.733** | **0.000** |
| Logistic Regression | 0.678 | 0.656 |

The do-nothing model wins on accuracy by **5.5 points** and catches **zero** churners.

Accuracy is not broken — it is answering "how many were right?" when the question is "who is
leaving?". On a 27% positive class, the majority class is 73% of the answer by construction.

### 11 · ROC-AUC vs PR-AUC

| | ROC-AUC | PR-AUC |
|---|---|---|
| Model | 0.768 | 0.517 |
| No skill | 0.500 | **0.267** |

The no-skill baseline for PR-AUC is the positive rate, not 0.5. That difference is why PR-AUC is the
more honest summary on imbalanced data: it is what the business actually experiences.

ROC-AUC stays flattering because the false positive rate divides by 264 negatives — 83 false alarms
only move it by 0.31.

**Report both.**

### 12 · Cross-validation

| Fold | ROC-AUC | Deviation from mean |
|---|---|---|
| 1 | 0.801 | −0.004 |
| 2 | 0.757 | −0.048 |
| 3 | 0.823 | +0.018 |
| 4 | 0.820 | +0.015 |
| 5 | 0.824 | +0.019 |
| **Mean** | **0.805** | — |
| **Std. dev.** | **0.025** | — |

**The standard deviation is the point.** Two models scoring 0.800 and 0.810 are not distinguishable.
Every comparison in this report is read against that band.

### 13 · Hyperparameter tuning

**Logistic Regression — 5 combinations**

| C | CV ROC-AUC | CV std | Train ROC-AUC | Rank |
|---|---|---|---|---|
| 0.01 | 0.800 | 0.027 | 0.808 | 5 |
| **0.1** | **0.806** | 0.026 | 0.817 | **1** |
| 1.0 | 0.805 | 0.025 | 0.818 | 2 |
| 10 | 0.805 | 0.025 | 0.818 | 3 |
| 100 | 0.805 | 0.025 | 0.818 | 4 |

**The winner leads by 0.001 — twenty-five times smaller than the fold noise of 0.026.** The honest
conclusion is not "C = 0.1 is best" but **"C is not a lever that matters for this problem."**

**Random Forest — 32 combinations**

Best: `max_depth=8, min_samples_leaf=5, max_features='sqrt', n_estimators=300` → **0.799** CV.
All 32 combinations span 0.776–0.799 — a range of 0.023, still under one standard deviation.

Training score for the best combination: **0.919** vs CV 0.799. That gap is overfitting, and the grid
earns its keep by constraining it.

### 14 · Did tuning help?

| Model | Default CV | Tuned CV | Gain |
|---|---|---|---|
| Logistic Regression | 0.805 | 0.806 | **+0.001** |
| Random Forest | 0.779 | 0.799 | +0.020 |

Both inside the noise band. The forest gain is more real — an unconstrained forest overfits, and the
grid pushed it toward shallower trees.

**Tuning fixes overfitting. It cannot manufacture signal that is not in the data.**

### 15 · Nested cross-validation

| Estimate | ROC-AUC |
|---|---|
| Tuned CV (same folds chose and scored) | 0.806 |
| Nested CV (outer folds never saw the choice) | 0.805 ± 0.025 |
| **Optimism** | **+0.001** |

Nearly zero, because a five-value grid with one effective hyperparameter has almost nothing to
overfit. With a wide grid on a smaller dataset the gap can be several times larger — the method
matters even when this particular number looks fine.

### 16 · Learning curve

| Training rows | Train ROC-AUC | Validation ROC-AUC | Gap |
|---|---|---|---|
| 115 | 0.830 | 0.748 | +0.081 |
| 322 | 0.828 | 0.802 | +0.026 |
| 529 | 0.825 | 0.804 | +0.021 |
| 737 | 0.820 | 0.807 | +0.013 |
| 944 | 0.822 | 0.809 | +0.013 |
| 1,152 | 0.817 | 0.810 | +0.007 |

The gap narrows from **0.081 to 0.007** while validation rises. That is variance, not bias: the model
is not memorising, it lacks examples.

**More data would help a little. More capacity would not help at all.**

---

## 17 · Every decision, and its justification

| Decision | Choice | Justification |
|---|---|---|
| Numeric scaling | Standardise | Distance metrics require comparable magnitudes |
| One-hot scaling | **No** | Rare categories must not get equal weight to common ones |
| k selection | Silhouette + elbow + judgement | Three indices disagree; the reason is stated |
| Degenerate clusters | Reject below 2% | Otherwise single linkage "wins" with a 1-point cluster |
| Linkage | ward | Best non-degenerate silhouette; compact clusters |
| Dimensionality | PCA for analysis, not for clustering | Reduction inflates silhouette without improving labels |
| Train/test split | 80/20 stratified, seed 42 | Reproducible; preserves the 27% class balance |
| CV | 5-fold stratified | Every fold keeps the class balance |
| Tuning scoring | ROC-AUC, training data only | Threshold-independent; test set untouched |
| Model selection | Highest CV among tuned | With the noise band stated alongside |
| Final estimate | Nested CV | Removes selection bias from the tuned score |
| Threshold | 0.5 for reported metrics | Curve shown; a deployment needs a cost-based choice |

---

## Reproducing

```bash
python week-3/src/run_analysis.py
python week-3/src/make_figures.py
python week-3/docs/build_report.py
```
