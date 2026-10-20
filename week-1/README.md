# Week 1 — Python for Machine Learning & Data Preprocessing

Complete Week 1 deliverable for the Yuva Internship: a reproducible, fully documented
preprocessing pipeline that turns a raw customer-churn CSV into a clean, encoded,
scaled and model-ready dataset.

**Dataset:** `data/customer_churn_raw.csv` — 1,800 synthetic customer records
**Report:** [`reports/Week_1_Python_ML_Data_Preprocessing_Report.docx`](reports/Week_1_Python_ML_Data_Preprocessing_Report.docx)
**Notebook:** [`notebooks/week1_python_ml_preprocessing.ipynb`](notebooks/week1_python_ml_preprocessing.ipynb)

> The dataset is reproducible **synthetic teaching data**, not real customer
> information. It deliberately contains missing values and categorical fields so every
> requested preprocessing technique can be demonstrated. No result in this folder is
> evidence about a real business.

---

## What is in this folder

```text
week-1/
├── README.md                          This file — overview and results
├── requirements.txt                   Python dependencies
├── data/
│   └── customer_churn_raw.csv         Raw sample dataset (1,800 × 13)
├── src/
│   ├── preprocess.py                  The nine-stage pipeline (produces every artifact)
│   └── make_pipeline_diagram.py       Generates the pipeline overview figure
├── notebooks/
│   └── week1_python_ml_preprocessing.ipynb
│                                      Teaching notebook: Python, NumPy, Pandas,
│                                      preprocessing — 64 executed code cells
├── docs/
│   ├── preprocessing_steps.md         Step-by-step documentation of every decision
│   ├── SUBMISSION_DESCRIPTION.md      The 307-word report description for the form
│   └── build_report.py                Generates the Word report from the artifacts
├── reports/
│   └── Week_1_Python_ML_Data_Preprocessing_Report.docx
└── outputs/
    ├── cleaned_customer_churn.csv     Readable clean table (gaps filled)
    ├── preprocessed_customer_churn.csv Model-ready numeric matrix
    ├── selected_feature_dataset.csv   Top-10 features + id + target
    ├── missing_value_audit.csv        Per-column missing counts and strategy
    ├── data_quality_checks.csv        The eight pre-flight checks
    ├── numeric_summary.csv            Descriptive statistics per numeric input
    ├── categorical_summary.csv        Level counts and shares
    ├── outlier_audit.csv              IQR fences and flagged counts
    ├── encoding_summary.csv           One-hot expansion per feature
    ├── scaling_summary.csv            Mean/std before and after standardisation
    ├── feature_selection.csv          Full mutual-information ranking
    ├── summary_metrics.csv            Headline metrics, one per row
    ├── preprocessing_summary.json     Machine-readable record of every decision
    └── figures/                       9 PNG figures at 160–180 DPI
```

---

## Quick start

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r week-1/requirements.txt

# 1. Run the pipeline — writes every table, figure and the JSON summary
python week-1/src/preprocess.py

# 2. Regenerate the pipeline diagram
python week-1/src/make_pipeline_diagram.py

# 3. Rebuild the Word report from the fresh artifacts
python week-1/docs/build_report.py
```

The pipeline completes in under a minute and needs no network access. All randomness
uses `RANDOM_STATE = 42`, so repeated runs produce identical output.

To explore the concepts interactively:

```bash
jupyter notebook week-1/notebooks/week1_python_ml_preprocessing.ipynb
```

---

## The nine preprocessing stages

| # | Stage | Technique | Result |
|---|---|---|---|
| 1 | Data loading | `pd.read_csv` with explicit `na_values`, schema assertion | 1,800 × 13 loaded |
| 2 | Quality audit | dtypes, duplicates, invalid values, target check | 562 missing feature cells found |
| 3 | Cleaning | trim text, repair impossible values, deduplicate | 1,800 rows retained |
| 4 | Missing values | median (numeric) + mode (categorical) + indicators | 10 columns filled, 0 cells left |
| 5 | Outlier audit | 1.5 × IQR fences, binary flags excluded | 3 flagged, 0 deleted |
| 6 | EDA | histograms, correlation, group churn rates, boxplots | 8 analysis figures |
| 7 | Encoding | `OneHotEncoder(handle_unknown="ignore")` | 6 inputs → 16 columns |
| 8 | Scaling | `StandardScaler` inside a `ColumnTransformer` | mean 0, std 1 |
| 9 | Feature selection | mutual information (`n_neighbors=5`) | ranking + top-10 export |

Full rationale for each decision: [`docs/preprocessing_steps.md`](docs/preprocessing_steps.md).

---

## Results

| Measure | Raw data | After preprocessing |
|---|---|---|
| Rows | 1,800 | 1,800 |
| Columns | 13 | 25 model features (+ id + target in exports) |
| Missing feature cells | 562 (2.84% of the feature grid) | **0** |
| Categorical columns | 6 | 0 (16 one-hot columns) |
| Numeric scale | mixed units | z-scores |
| Target | `"No"` / `"Yes"` | `0` / `1` |
| Unseen categories at prediction | would raise | tolerated |

**Class balance:** No 1,320 (73.3%) · Yes 480 (26.7%) — a 2.75 : 1 imbalance, which is
why plain accuracy is not used as a headline metric anywhere in this project.

**Top features by mutual information:** `monthly_charges` (0.059) →
`internet_service_Fiber optic` (0.042) → `contract_Month-to-month` (0.040) →
`contract_Two year` (0.030) → `internet_service_No` (0.021).

These agree with the descriptive group rates: month-to-month customers churn at ~39%
versus ~7% for two-year contracts, and Fiber-optic customers at ~41% versus ~17% for DSL.

---

## Documented decisions

Each of these is a choice with a stated reason, not a default:

1. **`customer_id` is excluded from model inputs.** A unique identifier can be memorised
   but cannot generalise. It is kept in every export for traceability.
2. **The target is never imputed.** Rows with an invalid label would be removed, because
   a guessed label teaches the model something false. (None were found here.)
3. **Median, not mean, for numeric imputation.** Every numeric column is skewed or
   contains extreme values; the mean would be dragged away from the typical customer.
4. **Missing indicators are retained.** Four extra 0/1 columns preserve the information
   contained in a value being absent.
5. **Rows are not deleted for missingness.** Listwise deletion would discard roughly a
   quarter of the dataset and bias the remainder.
6. **Outliers are flagged, not removed.** The few flagged values are plausible
   high-value customers, and removing them would bias every downstream statistic.
7. **One-hot, not ordinal, encoding.** The categorical levels have no order; assigning
   `Month-to-month = 0, One year = 1, Two year = 2` would invent a numeric distance.
8. **`handle_unknown="ignore"`.** A contract type that appears after deployment must not
   crash prediction; it encodes as an all-zero block instead.
9. **`StandardScaler`, not min–max.** The numeric features are skewed with long tails;
   min–max would squash the bulk of the data into a narrow band.
10. **Preprocessing lives in a `ColumnTransformer`.** Parameters are learned, stored and
    reapplied, so fitting on a training split cannot leak test information.
11. **The all-row feature ranking is explanatory only.** Any selection used for modelling
    is re-fitted inside training folds.

---

## Reproducibility and verification

The pipeline asserts its own invariants rather than assuming them:

- the expected columns exist at load time, otherwise it raises;
- the target contains only `No` and `Yes`;
- the transformed matrix contains no missing values, otherwise it raises;
- the readable export contains no missing values, otherwise it raises;
- each standardised column has mean ≈ 0 and std ≈ 1, verified in `scaling_summary.csv`;
- an unseen category encodes without error.

The Word report is generated from `outputs/preprocessing_summary.json` rather than typed
by hand, so a re-run cannot leave stale numbers in the document.

---

## Limitations

- **Synthetic data.** Every pattern is a property of the generator, not of customers.
- **Random missingness.** Real missingness is usually systematic; median imputation
  would then be biased. A real project must analyse the mechanism first.
- **Median/mode imputation understates variance.** The filled value creates a spike and
  narrows spread, which is measured and reported rather than hidden.
- **No fairness analysis.** Imputation rates and error rates should be audited by
  subgroup before any real deployment.
- **Association is not causation.** The churn-rate comparisons are descriptive.

---

## Related weeks

Week 2 trains and compares supervised models on this prepared data; Week 3 adds
clustering, PCA and cross-validated evaluation; Week 4 serialises the pipeline behind a
prediction API. Those folders are at the repository root.
