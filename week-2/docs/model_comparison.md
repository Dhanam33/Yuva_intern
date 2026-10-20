# Week 2 — Model comparison, metric by metric

Companion to `src/train_models.py`. Every number here is produced by that script and
stored in `outputs/`; the report is generated from the same files.

- **Classification target:** `churn` (No = 0, Yes = 1)
- **Regression target:** `monthly_charges`
- **Split:** 80/20, stratified for classification, seed 42
- **Selection:** 5-fold stratified CV on the training split
- **Test set:** scored once, after selection

---

## 1 · The baseline comes first

Before any real model, two dummy classifiers establish what "worse than nothing" looks like.

| Baseline | Accuracy | Recall | ROC-AUC | Interpretation |
|---|---|---|---|---|
| Dummy (most frequent) | 0.733 | 0.000 | 0.500 | Always says "No". High accuracy, zero usefulness |
| Dummy (stratified) | 0.661 | 0.323 | 0.554 | Random guess respecting class frequencies |

**0.733 is the number every accuracy claim must beat.** A model scoring below it is
strictly worse than doing nothing — on that metric.

---

## 2 · Classification results

| Model | Accuracy | Balanced acc. | Precision | Recall | F1 | ROC-AUC | PR-AUC | MCC |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.678 | 0.671 | 0.432 | 0.656 | 0.521 | **0.769** | **0.518** | 0.308 |
| Random Forest | 0.717 | **0.694** | 0.477 | 0.646 | **0.549** | 0.760 | 0.511 | **0.357** |
| K-Nearest Neighbors | **0.744** | 0.634 | **0.528** | 0.396 | 0.452 | 0.734 | 0.468 | 0.295 |
| Decision Tree | 0.653 | 0.661 | 0.409 | **0.677** | 0.510 | 0.725 | 0.486 | 0.286 |

### Who wins what

| Metric | Winner | Score | What it rewards |
|---|---|---|---|
| Accuracy | K-Nearest Neighbors | 0.744 | Raw hit rate, including the 73% easy negatives |
| Balanced accuracy | Random Forest | 0.694 | Equal weighting of both classes |
| Precision | K-Nearest Neighbors | 0.528 | Being right when it flags a risk |
| Recall | Decision Tree | 0.677 | Catching actual churners |
| F1 | Random Forest | 0.549 | Balanced precision and recall |
| ROC-AUC | Logistic Regression | 0.769 | Ranking quality across all thresholds |
| PR-AUC | Logistic Regression | 0.518 | Ranking quality on the minority class |
| MCC | Random Forest | 0.357 | Correlation of prediction with truth |

**No model wins more than two metrics.** This is the finding, not a failure to find a winner.

---

## 3 · Error decomposition

Of 360 held-out customers, 267 did not churn and 93 did.

| Model | TN | FP | FN | TP | FP cost | FN cost |
|---|---|---|---|---|---|---|
| Logistic Regression | 181 | 83 | 33 | 63 | 83 wasted offers | 33 lost customers |
| Random Forest | 196 | 68 | 34 | 62 | 68 wasted offers | 34 lost customers |
| K-Nearest Neighbors | 230 | 34 | **58** | 38 | 34 wasted offers | **58 lost customers** |
| Decision Tree | 170 | **94** | 31 | 65 | **94 wasted offers** | 31 lost customers |

KNN makes the fewest false alarms and misses the most churners. The Decision Tree does the
opposite. Accuracy cannot distinguish these two situations — it counts both errors as
"wrong" and moves on.

**If a lost customer costs 50× a wasted offer**, KNN's 58 missed churners are far more
expensive than the Decision Tree's 94 false alarms, despite KNN having the best accuracy.

---

## 4 · Why accuracy alone misleads

I can show this numerically:

- Best accuracy: **0.744** (KNN) — recall 0.396
- Best recall: **0.677** (Decision Tree) — accuracy 0.653
- Do-nothing baseline: **0.733** accuracy — recall 0.000

A model that catches **two-thirds** of churners scores *lower* on accuracy than one that
catches **two-fifths**, which in turn scores barely above a model that catches **none**.

The metric is not broken — it is answering a question nobody asked. Accuracy measures
"how many were right?", but in churn prediction the question is "who is leaving?"

---

## 5 · Ranking quality

ROC-AUC and PR-AUC both ignore the threshold and measure only how well the model ranks.

| Model | ROC-AUC | PR-AUC | Gap |
|---|---|---|---|
| Logistic Regression | 0.769 | 0.518 | 0.251 |
| Random Forest | 0.760 | 0.511 | 0.249 |
| Decision Tree | 0.725 | 0.486 | 0.239 |
| K-Nearest Neighbors | 0.734 | 0.468 | 0.266 |

The rank order changes between the two metrics. ROC-AUC is dominated by the abundant
negative class; PR-AUC focuses on the minority class the business cares about.

**No-skill PR-AUC is 0.267** — the churn rate — not 0.5. That gap between expectation and
reality is why PR-AUC is often called the honest metric for imbalanced problems.

---

## 6 · Cross-validation

5-fold stratified CV, training split only. The test set plays no part.

| Model | CV ROC-AUC | Std. dev. | CV F1 | Std. dev. |
|---|---|---|---|---|
| Logistic Regression | 0.805 | 0.025 | 0.599 | — |
| Random Forest | 0.796 | 0.026 | 0.590 | — |
| Decision Tree | 0.767 | 0.027 | 0.559 | — |
| K-Nearest Neighbors | 0.766 | 0.030 | 0.451 | — |

**Selected: Logistic Regression.**

### The gap is smaller than the noise

Leader (0.805) minus runner-up (0.796) = **0.009**.
Standard deviation across folds = **0.025**.

The difference between the top two is **one third of a standard deviation**. They are not
reliably different on 1,440 rows. Reporting "Logistic Regression is the best model" would
over-claim.

What the comparison *does* support:

- Logistic Regression and Random Forest both perform well.
- Both are meaningfully better than KNN on F1 (0.599 / 0.590 vs 0.451).
- The ordering below the top two is not trustworthy either.

---

## 7 · Threshold sensitivity

The model outputs a probability. The threshold turns it into a label. This table sweeps
Threshold for Logistic Regression:

| Threshold | Precision | Recall | F1 | Accuracy | Share flagged |
|---|---|---|---|---|---|
| 0.10 | 0.31 | 0.98 | 0.47 | 0.40 | 86% |
| 0.20 | 0.34 | 0.93 | 0.49 | 0.49 | 73% |
| 0.30 | 0.38 | 0.91 | 0.53 | 0.58 | 64% |
| **0.40** | **0.42** | **0.82** | **0.56** | **0.65** | **52%** |
| 0.50 | 0.43 | 0.66 | 0.52 | 0.68 | 41% |
| 0.60 | 0.47 | 0.56 | 0.51 | 0.72 | 32% |
| 0.70 | 0.58 | 0.48 | 0.53 | 0.77 | 22% |
| 0.80 | 0.68 | 0.33 | 0.45 | 0.78 | 13% |
| 0.90 | 0.43 | 0.03 | 0.06 | 0.73 | 2% |

The F1 optimum is at **0.40**, not 0.50. At that threshold the model catches 82% of
churners while more than half its flags are false alarms.

Which row is correct is a **business decision**, not a modelling one:

- If a retention offer costs 10 and a lost customer is worth 500, use a low threshold.
- If offers are expensive or the customer base is huge, use a higher one.

---

## 8 · Class weighting

The second lever. This changes what the model optimises during training.

| Setting | Accuracy | Precision | Recall | F1 | Share flagged |
|---|---|---|---|---|---|
| No weighting | **0.775** | **0.606** | 0.448 | 0.515 | 19.7% |
| `balanced` | 0.678 | 0.432 | **0.656** | **0.521** | 40.6% |

**Accuracy falls by 9.7 points while recall rises by 20.8 points.**

If the evaluation had optimised accuracy, it would have selected the model that finds
fewer than half the churners. That single sentence justifies every extra metric in this
report.

---

## 9 · Feature importance

Random Forest impurity importance, top 8:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `monthly_charges` | 0.236 |
| 2 | `tenure_months` | 0.122 |
| 3 | `total_charges` | 0.109 |
| 4 | `contract_Month-to-month` | 0.106 |
| 5 | `internet_service_Fiber optic` | 0.088 |
| 6 | `contract_Two year` | 0.075 |
| 7 | `support_tickets` | 0.050 |
| 8 | `internet_service_No` | 0.031 |

**Caveat:** impurity importance is biased toward continuous and high-cardinality features
because they offer more candidate split points. It ranks usefulness inside this model — it
is not a causal effect.

Cross-check: Week 1's mutual-information ranking independently put `monthly_charges`,
Fiber-optic service and month-to-month contract at the top. Two different methods agreeing
is a useful sanity check.

---

## 10 · Scaling, measured rather than asserted

The identical KNN, both ways:

| Setting | Accuracy | Recall | ROC-AUC |
|---|---|---|---|
| With `StandardScaler` | 0.744 | 0.396 | 0.734 |
| Without scaling | 0.708 | **0.156** | 0.608 |

**Recall collapses by 24 percentage points.** Distance computed on raw units is dominated
entirely by `total_charges` (thousands) versus `senior_citizen` (0 or 1), so the model
effectively ignores every small-magnitude feature.

Nothing in the training output warns about this. It is a silent failure — which is why
preprocessing belongs inside a `Pipeline` rather than in a step someone might skip.

---

## 11 · Overfitting, measured

The same Decision Tree, trained at every depth from 1 to 18:

| Depth | Train accuracy | Test accuracy | Train ROC-AUC | Test ROC-AUC |
|---|---|---|---|---|
| 1 | 0.55 | 0.55 | 0.65 | 0.63 |
| 3 | 0.78 | **0.74** | 0.92 | **0.73** |
| 5 | 0.79 | 0.70 | 0.95 | 0.69 |
| 9 | 0.88 | 0.66 | 0.99 | 0.63 |
| 13 | 0.95 | 0.67 | 1.00 | 0.61 |
| 18 | **1.00** | 0.65 | **1.00** | 0.58 |

The training score reaches a perfect 1.00. The test score peaks at depth 3 and then
declines steadily.

**A perfect training score is a symptom, not an achievement.** This is variance, and it is
why `max_depth` is a hyperparameter chosen on validation data rather than freely.

---

## 12 · Regression

Predicting `monthly_charges`. `total_charges` is **excluded** because it is derived from
the target — including it would let the model recover the answer from the input.

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| **Linear Regression** | **7.59** | **9.65** | **0.818** | **15.3%** |
| Random Forest Regressor | 7.80 | 9.96 | 0.806 | 15.9% |
| Decision Tree Regressor | 7.83 | 10.02 | 0.804 | 15.8% |
| KNN Regressor | 10.87 | 13.74 | 0.631 | 29.5% |
| *Dummy (mean)* | *18.50* | *22.80* | *-0.017* | *53.1%* |

5-fold CV R² on the training split: Linear 0.789, Decision Tree 0.782, Random Forest 0.779,
KNN 0.601. **Selected: Linear Regression.**

### What the result means

- **MAE 7.59 on a mean charge of 60.73** is about 12% error — useful for triage and
  budgeting, not precise enough to bill a customer.
- **RMSE (9.65) > MAE (7.59)** means some predictions are badly wrong rather than all
  being mildly wrong.
- **The tree ensemble does not beat a linear model.** That is informative: when linear
  wins, the relationship is largely additive, and extra capacity buys only noise.

### The banded scatter plot

Predictions form horizontal stripes. This is not a bug — `monthly_charges` is determined
almost entirely by categorical inputs, so customers sharing a contract type and service
get nearly identical predictions.

**No model can smooth this.** When the target is a step function of categorical features,
the ceiling belongs to the data, not the algorithm.

---

## 13 · Summary table of every decision

| Decision | Choice | Justification |
|---|---|---|
| Task split | Classification AND regression | Different target types need different algorithms and metrics |
| Split | 80/20 stratified, seed 42 | Reproducible; preserves class balance |
| Preprocessing | Pipeline, fitted on train only | Prevents leakage structurally |
| Baselines | 2 dummies + 1 dummy regressor | Makes "better" meaningful |
| Model selection | 5-fold CV on train | Keeps the test set honest |
| Primary metric | ROC-AUC and PR-AUC | Threshold-independent; robust to imbalance |
| Reported metrics | 10 classification, 5 regression | No single metric is sufficient |
| Class weight | Balanced | A missed churner costs more than a false alarm |
| Threshold | Reported, not fixed at 0.5 | It is a business decision |
| Regression leakage | `total_charges` excluded | It is derived from the target |

---

## Reproducing

```bash
python week-2/src/train_models.py
python week-2/src/make_figures.py
python week-2/docs/build_report.py
```
