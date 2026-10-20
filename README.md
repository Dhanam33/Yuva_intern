# Yuva Internship — Four-Week Machine-Learning Project

This repository contains completed, runnable deliverables for the four-week Python and machine-learning task: data preprocessing, supervised learning, clustering/model evaluation, and an end-to-end prediction API.

**GitHub project URL:** https://github.com/Dhanam33/Yuva_intern/tree/arena/01a1004b-yuva-intern
**Working branch for this submission:** `arena/01a1004b-yuva-intern`

> **Dataset note:** the bundled 1,800-row customer-churn CSV is reproducible synthetic teaching data, not real customer information. It intentionally contains missing values and categorical fields so all requested preprocessing steps can be demonstrated. Model scores must not be interpreted as evidence of real-world business performance.

## Deliverables

| Week | Completed work | Report / artifacts |
|---|---|---|
| 1 — Python, NumPy, Pandas, preprocessing | Data loading, EDA, duplicate/ID audit, missing-value imputation, one-hot encoding, scaling, mutual-information feature selection | `reports/Week_1_Data_Preprocessing_Report.docx`; cleaned/encoded exports and charts in `outputs/` |
| 2 — supervised learning | Linear Regression for numeric monthly charges; Logistic Regression, Decision Tree, Random Forest, and KNN for churn; held-out comparison with metrics | `reports/Week_2_Supervised_Learning_Report.docx`; classification and regression metric files in `outputs/` |
| 3 — unsupervised learning and evaluation | K-Means and Agglomerative clustering; silhouette comparisons; PCA visualization; 5-fold stratified CV; regularization tuning; confusion matrix, precision, recall, F1, ROC-AUC | `reports/Week_3_Unsupervised_Learning_and_Evaluation_Report.docx`; evaluation artifacts in `outputs/` |
| 4 — deployment and capstone | Full preprocessing/model pipeline serialized with Joblib; FastAPI health and prediction endpoints; example request; project documentation and slide deck | `reports/Week_4_AI_Deployment_Capstone_Report.docx`; `reports/Capstone_Presentation.pptx`; `artifacts/churn_model.joblib` |

Each Word report includes its own **200+ word submission description**. The descriptions are also provided together in `reports/SUBMISSION_DESCRIPTIONS.md` for copying into the submission form. `docs/python_ml_fundamentals.md` summarizes the Python, NumPy, Pandas, and ML concepts used.

## Week 1 deliverable folder

`week-1/` is a self-contained, submission-ready copy of the Week 1 work. It contains the
reported dataset, a nine-stage preprocessing pipeline, a 64-cell teaching notebook on
Python, NumPy and Pandas, the step-by-step documentation, and the generated Word report.

| Item | Path |
|---|---|
| Word report | `week-1/reports/Week_1_Python_ML_Data_Preprocessing_Report.docx` |
| Submission description (307 words) | `week-1/docs/SUBMISSION_DESCRIPTION.md` |
| Documented preprocessing steps | `week-1/docs/preprocessing_steps.md` |
| Teaching notebook | `week-1/notebooks/week1_python_ml_preprocessing.ipynb` |
| Pipeline source | `week-1/src/preprocess.py` |
| Generated tables and figures | `week-1/outputs/` |

```bash
python week-1/src/preprocess.py             # regenerate all Week 1 artifacts
python week-1/src/make_pipeline_diagram.py
python week-1/docs/build_report.py          # rebuild the Week 1 report from those artifacts
```

## Week 2 deliverable folder

`week-2/` is the matching self-contained folder for the supervised-learning week. Four
classifiers (Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors) and
four regressors are trained on identical leakage-safe pipelines, with dummy baselines
included so that "better" has a meaning.

| Item | Path |
|---|---|
| Word report | `week-2/reports/Week_2_Supervised_Machine_Learning_Report.docx` |
| Submission description (273 words) | `week-2/docs/SUBMISSION_DESCRIPTION.md` |
| Metric-by-metric analysis | `week-2/docs/model_comparison.md` |
| Teaching notebook | `week-2/notebooks/week2_supervised_learning.ipynb` |
| Training source | `week-2/src/train_models.py` |
| Generated tables and figures | `week-2/outputs/` |

```bash
python week-2/src/train_models.py           # trains all models, writes every table
python week-2/src/make_figures.py           # builds the 13 figures
python week-2/docs/build_report.py          # rebuilds the Week 2 report
```

**Headline results.** K-Nearest Neighbors has the best accuracy (0.744) but the worst
recall (0.396); Logistic Regression has the best ROC-AUC (0.769) with roughly double that
recall. A do-nothing baseline scores 0.733 accuracy — higher than two of the four real
models — because balanced class weights deliberately trade accuracy for recall. Linear
Regression predicts monthly charges with R² 0.818 and MAE 7.59, with `total_charges`
excluded to prevent leakage.

The `week-1/` and `week-2/` folders are standalone: each has its own dataset copy,
dependency list and ignore rules, so either can be submitted on its own. The root-level
`src/yuva_ml/` package remains the source of truth for the shared Week 3–4 work.

## Run locally

Requires Python 3.11 or newer.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

# Reproduce the deliverables in order
python scripts/generate_dataset.py
python -m yuva_ml.week1_preprocessing
python -m yuva_ml.week2_supervised
python -m yuva_ml.week3_unsupervised
python -m yuva_ml.train
python scripts/build_reports.py
pytest
```

The generation seed, split seed, and preprocessing details are fixed for reproducibility. The Week 1 all-row transformed CSV and mutual-information ranking are learning artifacts only; Week 2/3 model evaluation fits preprocessing inside training data and CV folds to prevent information leakage. Feature selection, if used for modeling, should also be fitted within those folds.

## Run the API

After installing dependencies and training/including the artifact:

```bash
uvicorn yuva_ml.api:app --app-dir src --host 0.0.0.0 --port 8000
```

- Health check: `GET http://127.0.0.1:8000/health`
- Interactive API documentation: `http://127.0.0.1:8000/docs`
- Prediction: `POST /predict` with one or more customer records (up to 100 per request). Missing fields are accepted and imputed; unknown categories are tolerated.

Example:

```json
{
  "instances": [
    {
      "customer_id": "demo-001",
      "tenure_months": 4,
      "monthly_charges": 105.0,
      "support_tickets": 4,
      "contract": "Month-to-month",
      "internet_service": "Fiber optic",
      "payment_method": "Electronic check"
    }
  ]
}
```

Example response:

```json
{
  "predictions": [
    {
      "customer_id": "demo-001",
      "prediction": "Yes",
      "churn_probability": 0.964383
    }
  ]
}
```

The API is a local educational deployment, not a public production service. Before real use, replace the synthetic data, validate performance on representative data, choose a decision threshold with business costs in mind, add authentication/rate limiting/monitoring, and manage model-version compatibility. Only load Joblib artifacts from trusted sources.

## Repository layout

```text
src/yuva_ml/       Shared preprocessing, weekly exercises, training, and FastAPI app
scripts/           Dataset/report generation scripts
data/              Reproducible raw synthetic CSV and data notes
outputs/            Cleaned data, metrics, CV/clustering results, and plots
artifacts/          Serialized fitted pipeline and model metadata
reports/            Four .docx reports, descriptions, and capstone .pptx
tests/               Automated preprocessing and API tests
```
